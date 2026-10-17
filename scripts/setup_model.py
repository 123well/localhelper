#!/usr/bin/env python3
"""One-shot model provisioning: pick candidate -> plan -> (download+quantize) -> verify.

Examples:
  python scripts/setup_model.py --list
  python scripts/setup_model.py --plan-only                 # default candidate
  python scripts/setup_model.py --id neogrite-mamba2-7b --dest ~/models
  python scripts/setup_model.py --yes                        # actually download (~15GB!)
  python scripts/setup_model.py --verify models/mamba2-7b-int4.onnx

After a successful run, point config at the artifacts:
  "model": {"backend":"npu","npu_model_path":"...onnx","tokenizer_path":"...json"}
and re-run doctor.py — the LLM backend check should flip from Mock to ONNX.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.sourcing.candidates import CANDIDATES, recipe_for          # noqa: E402
from agent.sourcing.setup import plan, run, static_verify, smoke_test  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--id", default="", help="candidate id (see --list)")
    ap.add_argument("--dest", default="models")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="execute download/quantize (large disk+network impact)")
    ap.add_argument("--verify", default="", metavar="ONNX_PATH")
    a = ap.parse_args()

    if a.list:
        for c in CANDIDATES.values():
            print(f"{c.id:24s} {c.family:22s} {c.license:12s} int4≈{c.weight_gb_int4}GB  {c.reasoning[:60]}…")
        return 0

    if a.verify:
        tok = Path(a.verify).with_name("tokenizer.json")
        ok, msg = static_verify(a.verify, tok)
        print(f"static verify: {'OK' if ok else 'FAIL'} — {msg}")
        if ok:
            try:
                from agent.config import AgentConfig
                from agent.core.model import SSMModel
                cfg = AgentConfig()
                cfg.model.backend = "npu"
                cfg.model.npu_model_path = a.verify
                cfg.model.tokenizer_path = str(tok)
                m = SSMModel(cfg)
                sok, msgs = smoke_test(m)
                print("smoke:", "; ".join(msgs))
                return 0 if sok else 1
            except Exception as e:  # noqa: BLE001
                print(f"smoke skipped (backend init): {e}")
        return 0 if ok else 1

    p = plan(a.id, a.dest)
    print(json.dumps(p.to_dict(), indent=2, ensure_ascii=False))
    rec = recipe_for(p.candidate_id)
    print("\ntokenizer:", rec["tokenizer"])
    print("note     :", rec["note"])
    if a.plan_only or not a.yes:
        print("\nDRY RUN — ничего не скачивается. Добавьте --yes для выполнения.")
        return 0
    print("\nexecuting… (this can take a long time; ~15 GB download)")
    run(p)
    failed = [s for s in p.steps if s.ok is False and not s.optional]
    for s in p.steps:
        mark = "ok" if s.ok else ("skip" if s.ok is None else "FAIL")
        print(f"  [{mark}] {s.name}: {s.error[:200]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
