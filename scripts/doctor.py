"""Environment diagnostics for the Arch Linux + KDE Plasma 6 target.

Run:  python -m agent.cli doctor        (or)  python scripts/doctor.py
Checks config/XDG dirs, model backend availability, vector store, safety
gate, event sources and KDE integration — without ever requiring a display.
Exit code 0 = prototype is runnable (headless ok), 1 = blocking problems.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# allow running as a plain script: python scripts/doctor.py (repo root on path)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CHECKS: list[tuple[str, str]] = []          # (name, status)
BLOCKING = {"FAIL"}


def _line(name: str, ok: str, detail: str = "") -> None:
    CHECKS.append((name, ok))
    mark = {"OK": "✓", "WARN": "!", "FAIL": "✗"}[ok]
    print(f"  [{mark}] {name:<34} {ok:<5} {detail}")


def check_python() -> None:
    v = sys.version_info
    _line("python >= 3.10", "OK" if v >= (3, 10) else "FAIL", f"{v.major}.{v.minor}")


def check_numpy() -> None:
    try:
        import numpy  # noqa: F401
        _line("numpy (vectors/adapters)", "OK")
    except ImportError:
        _line("numpy (vectors/adapters)", "FAIL", "pip install numpy")


def check_config_xdg() -> None:
    from agent.config import AgentConfig
    cfg = AgentConfig.load()
    cfg.ensure_dirs()
    for name, p in (("config dir", cfg.config_dir), ("data dir", cfg.data_dir),
                    ("state dir", cfg.state_dir)):
        _line(name, "OK" if p.is_dir() else "FAIL", str(p))
    _line("log path writable", "OK" if _writable(cfg.log_path) else "FAIL", str(cfg.log_path))


def _writable(p: Path) -> bool:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a"):
            pass
        return True
    except OSError:
        return False


def check_model_backend() -> None:
    from agent.core.model import MockBackend, pick_backend
    from agent.config import AgentConfig
    cfg = AgentConfig.load()
    be = pick_backend(cfg)
    if isinstance(be, MockBackend):
        _line("LLM backend", "WARN",
              "MockBackend (no weights). Set model.npu_model_path/cpu_model_path in config.json")
    else:
        _line("LLM backend", "OK", type(be).__name__)


def check_npu() -> None:
    drv = Path("/dev/accel")
    has_ort = shutil.which("python") and subprocess.run(
        [sys.executable, "-c", "import onnxruntime"], capture_output=True).returncode == 0
    if drv.exists() and has_ort:
        _line("NPU (amdxdna + onnxruntime)", "OK", str(drv))
    elif drv.exists():
        _line("NPU (amdxdna + onnxruntime)", "WARN", "/dev/accel present, onnxruntime missing")
    else:
        _line("NPU (amdxdna + onnxruntime)", "WARN",
              "no /dev/accel — dev box? production target needs linux-zen + amdxdna")


def check_memory_rag() -> None:
    from agent.memory.rag import Memory
    tmp = Path(tempfile.mkdtemp(prefix="doctor-"))
    try:
        m = Memory(tmp / "memory.db", dim=64)
        i = m.add("docs", "pacman -Qdtq lists orphaned packages", "doc")
        hits = m.search("сиротские пакеты pacman", ("docs",), top_k=1)
        ok = bool(hits) and hits[0].id == i
        _line("vector store (sqlite+embed)", "OK" if ok else "FAIL",
              f"dim=64 test insert/search {'passed' if ok else 'FAILED'}")
        m.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_safety() -> None:
    from agent.config import SafetyConfig
    from agent.tools.safety import SafetyGate
    g = SafetyGate(SafetyConfig())
    probes = [("rm -rf /", "blocked"), ("sudo reboot", "needs_confirm"), ("ls ~", "safe")]
    bad = [c for c, exp in probes if g.assess_command(c).level != exp]
    _line("safety gate probes", "OK" if not bad else "FAIL", f"failed: {bad}" if bad else "3/3")


def check_initiative_imports() -> None:
    try:
        from agent.core.initiative import InitiativeEngine  # noqa: F401
        from agent.core.typo_model import make_analyzer     # noqa: F401
        _line("initiative + typo analyzer", "OK")
    except Exception as e:  # noqa: BLE001
        _line("initiative + typo analyzer", "FAIL", str(e))


def check_sleep_trainer() -> None:
    try:
        from agent.training.sleep import AdapterTrainer  # noqa: F401
        _line("sleep trainer (adapters)", "OK")
    except Exception as e:  # noqa: BLE001
        _line("sleep trainer (adapters)", "FAIL", str(e))


def check_kde_env() -> None:
    de = os.environ.get("XDG_CURRENT_DESKTOP", "")
    disp = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    tools = {t: shutil.which(t) for t in ("kdialog", "kdotool", "qdbus6", "plasmashell")}
    found = [t for t, p in tools.items() if p]
    if "KDE" in de.upper() and disp:
        _line("KDE Plasma session", "OK", f"{de}; helpers: {', '.join(found) or '-'}")
    elif found:
        _line("KDE Plasma session", "WARN", "helpers present but no session (headless)")
    else:
        _line("KDE Plasma session", "WARN",
              "headless mode — notifications degrade to log/stdin (fine for prototype)")


def check_dbus_py() -> None:
    try:
        import dbus_next  # noqa: F401
        _line("dbus-next (python D-Bus)", "OK")
    except ImportError:
        _line("dbus-next (python D-Bus)", "WARN", "pip install dbus-next (else qdbus fallback)")


def report(run_demo_check: bool = True) -> int:
    print("kde-agent doctor — диагностика прототипа\n")
    check_python(); check_numpy(); check_config_xdg(); check_model_backend()
    check_npu(); check_memory_rag(); check_safety(); check_initiative_imports()
    check_sleep_trainer(); check_kde_env(); check_dbus_py()
    if run_demo_check:
        try:
            import asyncio
            from agent.demo import run_demo
            import logging
            logging.disable(logging.CRITICAL)
            buf = __import__("io").StringIO()
            real_out, sys.stdout = sys.stdout, buf
            try:
                asyncio.run(run_demo())
            finally:
                sys.stdout = real_out
            ok = "DEMO OK" in buf.getvalue()
            _line("end-to-end demo", "OK" if ok else "FAIL", "python -m agent.cli demo")
            logging.disable(logging.NOTSET)
        except Exception as e:  # noqa: BLE001
            logging.disable(logging.NOTSET)
            _line("end-to-end demo", "FAIL", str(e)[:120])
    fails = [n for n, s in CHECKS if s == "FAIL"]
    warns = [n for n, s in CHECKS if s == "WARN"]
    print(f"\nИтог: {len(CHECKS)} проверок, FAIL={len(fails)}, WARN={len(warns)}")
    if fails:
        print("Блокирующие проблемы:", ", ".join(fails)); return 1
    if warns:
        print("Работоспособно в headless; для продакшена на железе закрыть:", ", ".join(warns))
    return 0


if __name__ == "__main__":
    sys.exit(report())
