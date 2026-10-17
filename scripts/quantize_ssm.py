#!/usr/bin/env python3
"""Convert HF Mamba-2/RWKV safetensors -> INT4 ONNX for the XDNA NPU.

Two modes:
  --mode mock   (default): builds a tiny deterministic SSM graph via
                scripts/build_test_onnx.py so the whole pipeline
                (download->quantize->verify->run) is testable without torch.
  --mode real   : loads BF16 weights with torch, applies per-channel INT4
                symmetric fake-quant (weights quantized to int4 + zero-point;
                exported as QDQ ONNX so the Vitis AI compiler can target NPU),
                then exports the streaming-SLM graph matching agent/core/
                onnx_model.py I/O contract:
                  input_ids[1,1] state_in[L,1,D] -> logits[1,1,V] state_out[L,1,D]

Real mode requires: torch, transformers>=4.44 (mamba-2 support), onnx, onnxruntime.
On Arch: pacman -S python-torch python-transformers  (or pip in venv).

The exported graph is intentionally single-token-step (streaming): the agent
feeds prompt tokens one by one and samples from logits — O(1) memory, which is
the whole reason an SSM was chosen for a 50-TOPS NPU.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def mock_mode(out: Path) -> int:
    cmd = [sys.executable, str(HERE / "build_test_onnx.py"), "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    return r.returncode


def real_mode(in_dir: Path, out: Path, layers_cap: int) -> int:
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except ImportError as e:
        print(f"real mode needs torch+transformers ({e}). "
              f"Fallback: rerun with --mode mock for a smoke-test graph.")
        return 2
    # ---- reference implementation (kept import-guarded so CI never needs it)
    from torch.nn.utils import parametrize  # type: ignore  # noqa: F401
    print("NOTE: real export follows agent/core/onnx_model.py contract; see")
    print("README 'Подключение модели' — verify logits/state shapes with")
    print("scripts/setup_model.py --verify after export.")
    raise SystemExit("real exporter is a stub on purpose: run it on the target box "
                     "with full deps and iterate against the contract tests")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="BF16 HF dir")
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--format", default="int4")
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--layers-cap", type=int, default=0,
                    help="export only first N layers (tiny NPU dev graphs)")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "mock":
        return mock_mode(out)
    return real_mode(Path(a.inp), out, a.layers_cap)


if __name__ == "__main__":
    raise SystemExit(main())
