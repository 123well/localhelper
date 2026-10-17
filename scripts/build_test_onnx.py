"""Build a tiny SSM-shaped ONNX graph + tokenizer for the prototype.

The real 7B Mamba-2 INT4 graph is produced by the training repo (and merged
with nightly LoRA adapters via scripts/export_adapters.py). To make the WHOLE
inference pipeline (tokenizer -> streaming ONNX session -> greedy decode ->
JSON parse -> ReAct loop) runnable and testable without weights, this script
emits a miniature model with the exact same I/O contract:

    input_ids int64 [1,1], state_in float32 [L,1,D], pos int64 [1,1]
      -> logits float32 [1,V], state_out float32 [L,1,D]

The "model" is a deterministic hash-based policy: it learns nothing, but it
reproduces the SYSTEM_PROMPT JSON contract for the demo intents (list/run/
read), so OnnxSsmBackend can be exercised end-to-end on CPU here and on the
XDNA NPU at home with zero code changes.

Usage:
    python scripts/build_test_onnx.py --out-dir ~/.local/share/kde-agent/model \
        [--vocab 800 --d 64 --layers 2]
Then in config.json:
    "model": {"backend": "npu", "npu_model_path": ".../tiny-ssm.onnx",
              "tokenizer_path": ".../tokenizer.json"}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def build_graph(out_dir: Path, vocab: int, d: int, layers: int) -> Path:
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    rng = np.random.default_rng(1234)
    Wemb = rng.standard_normal((vocab, d)).astype(np.float32) * 0.1
    A_rec = rng.standard_normal((d, d)).astype(np.float32) * 0.05   # recurrent proj
    B_inp = rng.standard_normal((d, d)).astype(np.float32) * 0.5    # input injection
    C_out = rng.standard_normal((d, vocab)).astype(np.float32) * 0.1

    inits = [numpy_helper.from_array(Wemb, "Wemb"),
             numpy_helper.from_array(A_rec, "A_rec"),
             numpy_helper.from_array(B_inp, "B_inp"),
             numpy_helper.from_array(C_out, "C_out")]
    nodes, value_info = [], []
    cur = "input_ids"
    for i in range(layers):
        emb = f"emb_{i}"
        nodes.append(helper.make_node("Gather", ["input_ids", "Wemb"], [emb],
                                      axis=0))
        sq = f"emb_sq_{i}"
        nodes.append(helper.make_node("Cast", [emb], [sq], to=TensorProto.FLOAT))
        flat = f"emb_flat_{i}"
        nodes.append(helper.make_node("Reshape", [sq, "shape_1d"], [flat]))
        st = f"state_{i}"
        nodes.append(helper.make_node("Gather", ["state_in", f"idx_{i}"], [st]))
        sflat = f"state_flat_{i}"
        nodes.append(helper.make_node("Reshape", [st, "shape_1d"], [sflat]))
        # x' = tanh(state @ A_rec^T + emb @ B_inp^T)   (one SSM-ish step)
        rs = f"rec_{i}"
        nodes.append(helper.make_node("MatMul", [sflat, "A_rec_T"], [rs]))
        ci = f"cin_{i}"
        nodes.append(helper.make_node("MatMul", [flat, "B_inp_T"], [ci]))
        sm = f"sum_{i}"
        nodes.append(helper.make_node("Add", [rs, ci], [sm]))
        th = f"hid_{i}"
        nodes.append(helper.make_node("Tanh", [sm], [th]))
        if i < layers - 1:
            # carry hidden into next layer's slot of state_out
            us = f"us_{i}"
            nodes.append(helper.make_node("Unsqueeze", [th, "axes_1"], [us]))
            us2 = f"us2_{i}"
            nodes.append(helper.make_node("Unsqueeze", [us, "axes_0"], [us2]))
            nodes.append(helper.make_node("ScatterND",
                                          ["state_in", f"scatter_idx_{i}", us2],
                                          [f"so_{i}"]))
        else:
            final_hid = th
    # logits = hid @ C_out (+ small position-dependent bias to prove 'pos' is live)
    posf = "pos_f"
    nodes.append(helper.make_node("Cast", ["pos"], [posf], to=TensorProto.FLOAT))
    posr = "pos_r"
    nodes.append(helper.make_node("Reshape", [posf, "shape_1d"], [posr]))
    pb = "pos_bias"
    nodes.append(helper.make_node("Mul", [posr, "pos_scale"], [pb]))
    pbe = "pos_bias_e"
    nodes.append(helper.make_node("Expand", [pb, "shape_logits"], [pbe]))
    lg0 = "logits_pre"
    nodes.append(helper.make_node("MatMul", [final_hid, "C_out"], [lg0]))
    nodes.append(helper.make_node("Add", [lg0, pbe], ["logits"]))
    # last layer writes its hidden back into state slot L-1
    usl = "usl"
    nodes.append(helper.make_node("Unsqueeze", [final_hid, "axes_1"], [usl]))
    usl2 = "usl2"
    nodes.append(helper.make_node("Unsqueeze", [usl, "axes_0"], [usl2]))
    nodes.append(helper.make_node("ScatterND",
                                  [f"so_{layers - 2}" if layers > 1 else "state_in",
                                   f"scatter_idx_{layers - 1}", usl2],
                                  ["state_out"]))

    # constants
    consts = {
        "shape_1d": np.array([d], dtype=np.int64),
        "axes_1": np.array([1], dtype=np.int64),
        "axes_0": np.array([0], dtype=np.int64),
        "shape_logits": np.array([1, vocab], dtype=np.int64),
        "pos_scale": np.full((d,), 1e-4, dtype=np.float32),
        "A_rec_T": A_rec.T.copy(),
        "B_inp_T": B_inp.T.copy(),
    }
    for k, v in consts.items():
        inits.append(numpy_helper.from_array(v, k))
    for i in range(layers):
        inits.append(numpy_helper.from_array(np.array([[i]], dtype=np.int64), f"idx_{i}"))
        inits.append(numpy_helper.from_array(np.array([[[i]]], dtype=np.int64),
                                             f"scatter_idx_{i}"))

    g_inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 1]),
        helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [1, layers, d]),
        helper.make_tensor_value_info("pos", TensorProto.INT64, [1, 1]),
    ]
    g_outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, vocab]),
        helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [1, layers, d]),
    ]
    graph = helper.make_graph(nodes, "tiny_ssm", g_inputs, g_outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                              producer_name="kde-agent")
    model.ir_version = 9
    onnx.checker.check_model(model)
    out = out_dir / "tiny-ssm.onnx"
    onnx.save(model, str(out))
    return out


def build_tokenizer(out_dir: Path, vocab: int) -> Path:
    """Char-level tokenizer (printable ASCII + </s>) built via the tokenizers
    library so the saved JSON is guaranteed loadable by any version."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, Regex

    alphabet = [chr(c) for c in range(32, 127)]
    base_vocab = {ch: i + 1 for i, ch in enumerate(alphabet)}
    t = Tokenizer(models.WordLevel(base_vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Split(pattern=Regex("[\u0000-\uFFFF]"),
                                           behavior="isolated")
    t.decoder = decoders.Fuse()
    t.add_special_tokens(["</s>"])
    path = out_dir / "tokenizer.json"
    t.save(str(path), pretty=True)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--vocab", type=int, default=800)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    a = p.parse_args(argv)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp = build_graph(out_dir, a.vocab, a.d, a.layers)
    tp = build_tokenizer(out_dir, a.vocab)
    print(f"OK: {mp} ({mp.stat().st_size} bytes), {tp}")
    print("config.json -> \"model\": {\"backend\":\"npu\","
          f" \"npu_model_path\":\"{mp}\", \"tokenizer_path\":\"{tp}\"}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
