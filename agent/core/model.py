"""Model abstraction over the frozen 7B SSM (Mamba-2, INT4).

Backends:
  * NPU  : ONNX Runtime with the AMD XDNA (Vitis AI) execution provider.
           Loads mamba2-7b-int4.onnx + trained LoRA adapters merged/exported
           by scripts/export_adapters.py. Streaming recurrent state is kept
           on-device; we feed tokens incrementally (SSM advantage: O(1) cache).
  * CPU/GPU (prototyping): llama.cpp GGUF via `llama-cpp-python`, or a tiny
           mock used in tests/CI so the whole agent runs without weights.

The system prompt enforces the contract: the model knows NO facts — every
fact must come from RAG tools; output must be structured JSON actions.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

log = logging.getLogger("kde-agent.model")


SYSTEM_PROMPT = """You are kde-agent, an autonomous assistant on Arch Linux + KDE Plasma 6.
RULES:
1. You have NO factual knowledge in your weights (no command syntax, no API docs).
   Before acting on anything you are unsure about, call rag_search / memory_search.
2. Respond ONLY with a JSON object: {"thought": str, "actions": [{"tool": str, "args": {}}], "reply": str}
   "actions" may be empty (pure reply). Max 3 actions per step.
3. Ask the user (tool=ask_user / confirm_dialog) before any destructive/risky action.
4. Prefer low-energy operation: request screen state only when necessary.
5. Be proactive but NEVER nagging; respect 'user said it was intentional' memories.
"""


@dataclass
class ModelResult:
    thought: str = ""
    actions: list[dict] = field(default_factory=list)
    reply: str = ""
    raw: str = ""
    ok: bool = True


class LLMBackend(Protocol):
    def generate(self, prompt: str, max_tokens: int) -> str: ...
    def reset_state(self) -> None: ...


class MockBackend:
    """Deterministic rule-based 'brain' for prototyping/tests without weights."""

    def reset_state(self) -> None:
        pass

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        p = prompt.lower()
        # crude intent extraction so the loop is testable end-to-end
        m = re.search(r"USER:\s*(.*)", prompt, re.S)
        user = (m.group(1).strip().splitlines() or [""])[-1] if m else ""
        if re.search(r"\b(list|покажи файлы|ls)\b", p):
            d = re.search(r"(~?/[\w./-]+|\w+/)", user)
            return json.dumps({
                "thought": "user wants a directory listing; verify path via RAG-free tool",
                "actions": [{"tool": "list_files", "args": {"path": (d.group(1) if d else ".")}}],
                "reply": "",
            })
        if re.search(r"\b(run|выполни|запусти|команд)\w*\b", p):
            cmd = re.sub(r"^.*?(run|выполни|запусти)\s+", "", user, flags=re.I).strip("`\".")
            return json.dumps({
                "thought": "terminal command requested; risk gate will vet it",
                "actions": [{"tool": "exec_command", "args": {"command": cmd}}],
                "reply": "",
            })
        if re.search(r"\b(read|прочитай|открой файл)\b", p):
            f = re.search(r"([\w./-]+\.[\w]+)", user)
            return json.dumps({
                "thought": "file read",
                "actions": [{"tool": "read_file", "args": {"path": f.group(1) if f else "."}}],
                "reply": "",
            })
        return json.dumps({
            "thought": "no tool needed",
            "actions": [],
            "reply": user[:200] or "Готов помочь.",
        })


class OnnxNpuBackend:
    """Thin wrapper kept for API compatibility; real work is in agent.core.onnx_model."""

    def __init__(self, model_path: str, tokenizer_path: str,
                 max_seq: int = 2048, prefer_npu: bool = True):
        from agent.core.onnx_model import OnnxSsmBackend
        self._b = OnnxSsmBackend(model_path, tokenizer_path,
                                 max_seq=max_seq, prefer_npu=prefer_npu)

    def reset_state(self) -> None:
        self._b.reset_state()

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        return self._b.generate(prompt, max_tokens)


class LlamaCppBackend:
    """llama.cpp GGUF (CPU/GPU) for debugging the prototype with real weights."""

    def __init__(self, model_path: str, n_ctx: int = 2048):
        from llama_cpp import Llama  # lazy
        self.ll = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)

    def reset_state(self) -> None:
        pass

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        out = self.ll(prompt, max_tokens=max_tokens, stop=["</s>"])
        return out["choices"][0]["text"]


def pick_backend(cfg) -> LLMBackend:
    want = cfg.model.backend
    npu = cfg.model.npu_model_path
    cpu = cfg.model.cpu_model_path
    tok = getattr(cfg.model, "tokenizer_path", "")
    if want in ("auto", "npu") and npu:
        try:
            tp = tok or str(Path(npu).with_name("tokenizer.json"))
            return OnnxNpuBackend(npu, tp, max_seq=cfg.model.context_budget_tokens,
                                  prefer_npu=(want != "cpu"))
        except Exception as e:  # noqa: BLE001
            log.warning("NPU/ONNX backend unavailable (%s); falling back", e)
            if want == "npu":
                raise
    if want in ("auto", "cpu", "llamacpp") and cpu:
        try:
            return LlamaCppBackend(cpu, cfg.model.context_budget_tokens)
        except Exception as e:  # noqa: BLE001
            log.warning("llama.cpp backend unavailable (%s); using mock", e)
    log.info("using MockBackend (no weights loaded)")
    return MockBackend()


class SSMModel:
    """Frozen base model + hot-swappable LoRA adapters (trained at night)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = pick_backend(cfg)
        self.adapter_version = "base"

    def load_adapter(self, path: str) -> None:
        """In the prototype adapters are applied at export time (merged INT4
        graph swap). Here we just switch the active graph/version tag."""
        self.adapter_version = Path(path).stem if path else "base"
        log.info("adapter switched: %s", self.adapter_version)

    def reload(self) -> None:
        """Re-pick the backend from config (e.g. after scripts/export_adapters.py
        wrote a merged adapter graph and config points at it). Atomic swap."""
        old = getattr(self.backend, "_b", self.backend)
        self.backend = pick_backend(self.cfg)
        try:
            del old
        except Exception:  # noqa: BLE001
            pass
        log.info("backend reloaded: %s", type(self.backend).__name__)

    def think(self, user_msg: str, context: str = "", observations: str = "") -> ModelResult:
        prompt = (
            f"{SYSTEM_PROMPT}\n\nACTIVE_ADAPTER: {self.adapter_version}\n"
            f"COMPRESSED CONTEXT:\n{context}\n"
            f"OBSERVATIONS:\n{observations}\n"
            f"USER: {user_msg}\nASSISTANT:"
        )
        raw = self.backend.generate(prompt, self.cfg.model.max_tokens)
        return parse_model_output(raw)


def parse_model_output(raw: str) -> ModelResult:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return ModelResult(reply=raw.strip(), ok=False)
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return ModelResult(reply=raw.strip(), ok=False)
    acts = [a for a in d.get("actions", []) if isinstance(a, dict) and a.get("tool")]
    return ModelResult(thought=d.get("thought", ""), actions=acts,
                       reply=d.get("reply", ""), raw=raw)
