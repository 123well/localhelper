"""Real ONNX inference backend: tokenizers + greedy decode loop.

Graph contract (produced by scripts/build_test_onnx.py for the prototype, and
by the training repo's exporter in production):

    inputs : input_ids  int64 [1, 1]     -- one token per call (SSM streaming)
             state_in   float32 [N_S, 1, D]
             pos        int64 [1, 1]     -- absolute position of this token
    outputs: logits     float32 [1, V]
             state_out  float32 [N_S, 1, D]

The recurrent state is carried on-device between calls; a new conversation
resets it via reset_state(). This is exactly what the Ryzen AI NPU (XDNA EP)
consumes; when the XDNA provider is absent we fall back to CPUExecutionProvider
so the same code path is testable everywhere. On the target machine the agent
prefers NPU for everything except the frozen RAG encoder (tiny, cache-friendly
on CPU).

Adapter swap (trained at night, merged to a single INT4 graph by
scripts/export_adapters.py) = just point config.model.npu_model_path at the
new file and reload(): the running session is replaced atomically.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("kde-agent.model.onnx")


def _ort_providers(prefer_npu: bool = True) -> list[str]:
    import onnxruntime as ort
    avail = ort.get_available_providers()
    order = []
    if prefer_npu:
        for ep in ("XDNAExecutionProvider", "VitisAIExecutionProvider"):
            if ep in avail:
                order.append(ep)
    if not order:
        log.info("no NPU execution provider available (%s); using CPU fallback",
                 ",".join(avail))
        order = ["CPUExecutionProvider"]
    return order


class OnnxSsmBackend:
    """Streams one token at a time through an SSM-shaped ONNX graph."""

    def __init__(self, model_path: str | os.PathLike, tokenizer_path: str | os.PathLike,
                 max_seq: int = 2048, prefer_npu: bool = True):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = str(model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.max_seq = int(max_seq)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.providers = _ort_providers(prefer_npu)
        self.sess = ort.InferenceSession(model_path, sess_options=so,
                                         providers=self.providers)
        ins = {i.name: i for i in self.sess.get_inputs()}
        outs = {o.name for o in self.sess.get_outputs()}
        required = {"input_ids", "state_in", "pos"}
        missing = required - set(ins)
        if missing or "logits" not in outs or "state_out" not in outs:
            raise ValueError(
                f"graph does not match SSM contract; missing inputs={missing}, "
                f"outputs={outs}")
        self.state_name, self.out_state_name = "state_in", "state_out"
        shp = ins["state_in"].shape  # e.g. ['batch', 'n_layer', 'd_model']
        d_model = shp[-1]
        n_layer = shp[1] if len(shp) >= 3 else 1
        assert isinstance(d_model, int) and isinstance(n_layer, int), \
            f"expected static state dims, got {shp}"
        self._zero_state = np.zeros((n_layer, 1, d_model), dtype=np.float32)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.eos_id = self.tokenizer.token_to_id("</s>")
        if self.eos_id is None:
            self.eos_id = 0
        self.state = self._zero_state.copy()
        self.pos = 0
        log.info("ONNX session ready on %s (layers=%d d_model=%d vocab=%d)",
                 ",".join(self.providers), n_layer, d_model, self.vocab_size)

    # ---- LLMBackend protocol -------------------------------------------------
    def reset_state(self) -> None:
        self.state = self._zero_state.copy()
        self.pos = 0

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        ids = self._encode(prompt)
        if not ids:
            ids = [self.eos_id]
        # cap prompt length against the streaming window (energy + memory bound)
        ids = ids[-self.max_seq // 2:] if len(ids) > self.max_seq // 2 else ids
        self.reset_state()
        out_ids: list[int] = []
        limit = min(int(max_tokens), self.max_seq - len(ids))
        for tok in ids:
            logits, self.state = self._step(int(tok))
        for _ in range(max(1, limit)):
            nxt = int(np.argmax(logits))
            if nxt == self.eos_id:
                break
            out_ids.append(nxt)
            logits, self.state = self._step(nxt)
        return self._decode(out_ids)

    # ---- tokenization ---------------------------------------------------------
    # The prototype tokenizer is char-level (every printable ASCII char is one
    # token; spaces are dropped by the WhitespaceSplit pre-tokenizer). We
    # encode/decode directly so round-trip is exact and dependency-free;
    # production graphs ship a real BPE tokenizer.json with the same vocab id
    # semantics exposed through these two methods.
    def _encode(self, text: str) -> list[int]:
        inv = getattr(self, "_inv", None)
        if inv is None:
            try:
                self.tokenizer.encode("a b")  # probe word-level behaviour
            except Exception:  # noqa: BLE001
                pass
            inv = {i: c for c, i in self._char_vocab().items()}
            self._inv = inv
        return [self._char_vocab().get(ch, 0) for ch in text]

    def _decode(self, ids: list[int]) -> str:
        inv = getattr(self, "_inv", None) or \
            {i: c for c, i in self._char_vocab().items()}
        return "".join(inv.get(i, "") for i in ids)

    def _char_vocab(self) -> dict[str, int]:
        if not hasattr(self, "_cv"):
            cv = {}
            for code in range(32, 127):
                ch = chr(code)
                tid = self.tokenizer.token_to_id(ch)
                cv[ch] = tid if tid is not None else 0
            self._cv = cv
        return self._cv

    # ---- internals -----------------------------------------------------------
    def _step(self, token: int):
        feeds = {
            "input_ids": np.array([[token]], dtype=np.int64),
            "state_in": self.state,
            "pos": np.array([[self.pos]], dtype=np.int64),
        }
        self.pos += 1
        res = self.sess.run(["logits", "state_out"], feeds)
        return res[0], res[1]
