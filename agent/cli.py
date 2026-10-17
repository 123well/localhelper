"""CLI entry: python -m agent.cli {run|daemon|status|ctl|index|demo|export-adapters|doctor}

Daemon-first design (spec): `daemon` runs the agent in background with a
control socket; `status`/`ctl` talk to it without attaching to its terminal.
`run` stays for interactive debugging only."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys


def _socket_path():
    """Per-user control socket (0600). Shared by daemon and ctl clients so
    they can never cross-talk between users on multi-user systems."""
    import os
    from pathlib import Path
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime:
        runtime = f"/tmp/kde-agent-{os.getuid()}"
        Path(runtime).mkdir(parents=True, exist_ok=True, mode=0o700)
    return str(Path(runtime) / "kde-agent.ctl.sock")


def _send_ctl(command: str, arg: str = "") -> dict:
    """One-shot client request over the control socket; raises if no daemon."""
    import socket
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(120)
        s.connect(_socket_path())
        s.sendall((json.dumps({"cmd": command, "arg": arg}) + "\n").encode())
        buf = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode())
    finally:
        s.close()


class DaemonAlreadyRunning(Exception):
    """Single-instance guard tripped (second daemon must never corrupt state)."""


async def _serve_control(agent) -> None:
    """Control plane exposed as a local JSON-line protocol over UNIX socket.
    Same `agent.control()` API that the future settings window / D-Bus service
    will call — one implementation, many frontends."""
    import os
    path = _socket_path()
    try:
        os.unlink(path)
    except OSError:
        pass
    async def _cb(reader, writer):
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=5)
            if line:
                req = json.loads(line.decode())
                # async front door: heavy ops (study) run in worker threads
                resp = await agent.control_async(req.get("cmd", "status"),
                                                 req.get("arg", ""))
                writer.write((json.dumps(resp, ensure_ascii=False, default=str) + "\n").encode())
                await writer.drain()
        except Exception:  # noqa: BLE001  malformed client input is not fatal
            logging.getLogger("kde-agent.ctl").exception("control socket request failed")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass
    server = await asyncio.start_unix_server(_cb, path=path)
    os.chmod(path, 0o600)
    logging.getLogger("kde-agent").info("control socket: %s", path)
    async with server:
        await server.serve_forever()


async def _daemon_main(interactive: bool = False) -> int:
    """Daemon supervisor: run() + control socket as tasks; ANY crash or stop
    request tears everything down gracefully (no orphan background work)."""
    import signal
    from agent.main import Agent
    ag = Agent(interactive=interactive)   # `run` keeps the stdin REPL;
                                          # `daemon` must never attach to fd 0
    loop = asyncio.get_running_loop()

    # ---- single-instance guard ---------------------------------------------
    # flock is released automatically on ANY process exit (crash included),
    # so a killed daemon never leaves a stale lock behind.
    lock_path = ag.cfg.state_dir / "daemon.lock"
    lf = open(lock_path, "w")
    try:
        import fcntl
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lf.close()
        raise DaemonAlreadyRunning(str(lock_path))
    lf.write(str(__import__("os").getpid()))
    lf.flush()

    # ---- OS lifecycle signals (spec item 5): graceful safe shutdown --------
    def _sig(name):
        logging.getLogger("kde-agent").info("signal %s -> graceful shutdown", name)
        ag._bg_learning_paused = True          # freeze study/sleep instantly
        asyncio.ensure_future(ag.shutdown(f"signal:{name}"))
    for sig_name in ("SIGTERM", "SIGINT", "SIGHUP"):
        try:
            loop.add_signal_handler(getattr(signal, sig_name), _sig, sig_name)
        except (NotImplementedError, AttributeError, ValueError):
            pass
    # logind sleep-lock / reboot hooks can be wired to SIGUSR1/SIGUSR2 in the
    # unit file (ExecStopPost etc.). Lock => pause learning & save state only;
    # USR2 => full restart-ready shutdown.
    def _pause(reason):
        logging.getLogger("kde-agent").info("%s: pausing background learning, saving state", reason)
        ag._bg_learning_paused = True
        ag.initiative.save()
    try:
        loop.add_signal_handler(signal.SIGUSR1, _pause, "SIGUSR1(screen-lock/logind)")
    except (NotImplementedError, AttributeError, ValueError):
        pass

    async def _stop_watcher():
        # shutdown() flips ag.running -> end the control-socket server too
        while ag.running:
            await asyncio.sleep(0.5)
    tasks = [asyncio.create_task(ag.run(), name="run"),
             asyncio.create_task(_serve_control(ag), name="ctl-socket"),
             asyncio.create_task(_stop_watcher(), name="stop-watcher")]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        rc = 0
        for t in done:
            exc = t.exception()
            if exc and not isinstance(exc, asyncio.CancelledError):
                logging.getLogger("kde-agent").error("subsystem %s crashed: %r",
                                                     t.get_name(), exc)
                rc = 1
    finally:
        await ag.shutdown("exit")
        try:
            import os as _os
            _os.unlink(_socket_path())
        except OSError:
            pass
        lf.close()
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kde-agent", description="Autonomous KDE Plasma 6 agent (Stage 1 prototype)")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("run", help="interactive terminal UI + event loop (debug)")

    pd = sub.add_parser("daemon", help="background service mode: no tty needed, control via `ctl`")

    ps = sub.add_parser("status", help="daemon status (JSON) via control socket")

    pc = sub.add_parser("ctl", help="send a control command to the running daemon")
    pc.add_argument("command")
    pc.add_argument("arg", nargs="*", default=[])

    pi = sub.add_parser("index", help="index documentation into RAG")
    pi.add_argument("path")

    sub.add_parser("demo", help="scripted end-to-end demo (initiative, RPE, RAG, sleep)")

    pe = sub.add_parser("export-adapters", help="merge trained adapter into ONNX INT4 graph (needs onnxruntime+weights)")
    pe.add_argument("--adapter", required=True)
    pe.add_argument("--model", required=True)
    pe.add_argument("--out", required=True)

    sub.add_parser("doctor", help="check environment readiness for Arch/KDE target")

    args = p.parse_args(argv)
    if args.cmd in ("run", "daemon"):
        try:
            return asyncio.run(_daemon_main(interactive=(args.cmd == "run")))
        except DaemonAlreadyRunning as lk:
            print(f"демон уже запущен (блокировка: {lk})", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"fatal: {e}", file=sys.stderr)
            return 1
    if args.cmd == "status":
        try:
            print(json.dumps(_send_ctl("status"), ensure_ascii=False, indent=2))
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            print(json.dumps({"ok": False, "error": "демон не запущен (нет контрольного сокета)"}))
            return 1
        return 0
    if args.cmd == "ctl":
        try:
            resp = _send_ctl(args.command, " ".join(args.arg))
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            print(json.dumps({"ok": False,
                              "error": "демон не запущен: выполните `kde-agent daemon &`"}))
            return 1
        print(json.dumps(resp, ensure_ascii=False, indent=2, default=str))
        return 0 if resp.get("ok", "report" in resp or "log" in resp) else 1
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
        from agent.core.diagnostics import run_doctor
        rep = run_doctor(None)
        return {"ok": 0, "warn": 0, "fail": 1}[rep["worst"]]
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
