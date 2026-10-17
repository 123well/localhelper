"""CLI entry: python -m agent.cli {run|index|demo|export-adapters|doctor}"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kde-agent", description="Autonomous KDE Plasma 6 agent (Stage 1 prototype)")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("run", help="interactive terminal UI + event loop")

    pi = sub.add_parser("index", help="index documentation into RAG")
    pi.add_argument("path")

    sub.add_parser("demo", help="scripted end-to-end demo (initiative, RPE, RAG, sleep)")

    pe = sub.add_parser("export-adapters", help="merge trained adapter into ONNX INT4 graph (needs onnxruntime+weights)")
    pe.add_argument("--adapter", required=True)
    pe.add_argument("--model", required=True)
    pe.add_argument("--out", required=True)

    sub.add_parser("doctor", help="check environment readiness for Arch/KDE target")

    args = p.parse_args(argv)
    if args.cmd == "run":
        from agent.main import Agent
        try:
            asyncio.run(Agent().run())
        except KeyboardInterrupt:
            pass
        return 0
    if args.cmd == "index":
        from agent.config import AgentConfig
        from agent.main import index_docs
        from agent.memory.rag import Memory
        cfg = AgentConfig.load()
        mem = Memory(cfg.db_path, cfg.rag.dim)
        n = index_docs(mem, args.path)
        print(f"indexed {n} chunks into {cfg.db_path}")
        return 0
    if args.cmd == "demo":
        from agent.demo import run_demo
        logging.basicConfig(level=logging.WARNING)
        asyncio.run(run_demo())
        return 0
    if args.cmd == "export-adapters":
        from scripts.export_adapters import merge
        merge(args.adapter, args.model, args.out)
        return 0
    if args.cmd == "doctor":
        from scripts.doctor import report
        report()
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
