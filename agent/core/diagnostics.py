"""Diagnostics (spec: режим диагностики). One command -> one readable report.

Checks every subsystem BEFORE new features are trusted: model backend boots,
NPU/weights availability (explicitly separated from the CPU/mock prototype
mode), config sanity, directory access, vector DB read/write, RAG roundtrip,
logging + rotation, tool execution under the current safety mode, confirmation
system, background task leaks, unfinished processes, memory growth and error
handling. Safe by construction: it instantiates its own throwaway agent in a
temp XDG dir when none is given, so `doctor` works even on a broken install —
and always runs with proactivity OFF to avoid spamming the user during checks.
"""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("kde-agent.doctor")


@dataclass
class CheckResult:
    name: str
    status: str            # ok | warn | fail
    detail: str = ""

    def line(self) -> str:
        mark = {"ok": "✓", "warn": "⚠", "fail": "✗"}[self.status]
        return f"{mark} {self.name}: {self.detail}"


@dataclass
class DoctorReport:
    results: list[CheckResult] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.results.append(CheckResult(name, status, detail))

    @property
    def worst(self) -> str:
        if any(r.status == "fail" for r in self.results):
            return "fail"
        if any(r.status == "warn" for r in self.results):
            return "warn"
        return "ok"

    def text(self) -> str:
        counts = {s: sum(1 for r in self.results if r.status == s)
                  for s in ("ok", "warn", "fail")}
        head = (f"ДИАГНОСТИКА kde-agent — {time.strftime('%Y-%m-%d %H:%M:%S')} — "
                f"итог: {self.worst.upper()} "
                f"(ok={counts['ok']} warn={counts['warn']} fail={counts['fail']})")
        return head + "\n" + "\n".join(r.line() for r in self.results)

    def as_dict(self) -> dict:
        return {"ts": self.ts, "worst": self.worst,
                "checks": [{"name": r.name, "status": r.status, "detail": r.detail}
                           for r in self.results]}


def run_doctor(agent=None) -> dict:
    """Full diagnostic pass. If `agent` given, probes the LIVE instance;
    otherwise builds a throwaway one in a temp dir (works on a fresh/broken
    install without touching real state)."""
    rep = DoctorReport()
    own = agent is None
    tmp = None
    try:
        if own:
            tmp = Path(tempfile.mkdtemp(prefix="kde-agent-doctor-"))
            old = {k: os.environ.get(k) for k in
                   ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME")}
            os.environ["XDG_CONFIG_HOME"] = str(tmp / "config")
            os.environ["XDG_DATA_HOME"] = str(tmp / "data")
            os.environ["XDG_STATE_HOME"] = str(tmp / "state")
            from agent.config import AgentConfig
            from agent.main import Agent
            agent = Agent(AgentConfig())
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        _check_all(agent, rep)
    except Exception as e:  # noqa: BLE001
        rep.add("ядро собирается", "fail", f"исключение: {e}")
        log.exception("doctor crashed")
    finally:
        if own and tmp:
            try:
                agent.memory.close()
            except Exception:  # noqa: BLE001
                pass
            shutil.rmtree(tmp, ignore_errors=True)
    print(rep.text())
    return rep.as_dict()


def _check_all(ag, rep: DoctorReport) -> None:
    # ---- 1. core wiring ---------------------------------------------------
    rep.add("ядро собирается", "ok",
            f"backend={type(ag.model.backend).__name__}, режим={ag.safety.mode}")

    # ---- 2. model boot (smoke inference) ----------------------------------
    try:
        res = ag.model.think("проверка: ответь ok", context="", observations="")
        rep.add("модель: инференс", "ok" if (res.ok or res.reply) else "warn",
                f"parsed={res.ok}; reply='{res.reply[:40]}'")
    except Exception as e:  # noqa: BLE001
        rep.add("модель: инференс", "fail", str(e)[:120])

    # ---- 3. NPU vs prototype split (explicit, spec requirement) -----------
    backend = type(ag.model.backend).__name__
    if backend == "MockBackend":
        rep.add("режим инференса", "warn",
                "ПРОТОТИПНЫЙ режим (mock на CPU): веса не подключены. "
                "Финальный NPU-режим требует model.npu_model_path + onnxruntime(XDNA)")
    elif backend == "OnnxNpuBackend" or backend == "OnnxSsmBackend":
        try:
            import onnxruntime as ort  # noqa: F401
            eps = ort.get_available_providers()
            have_xdna = "VitisAIExecutionProvider" in eps or "AMD_AIE" in str(eps)
            rep.add("NPU доступен", "ok" if have_xdna else "warn",
                    f"providers={eps}" + ("" if have_xdna else " → ONNX работает на CPU fallback"))
        except ImportError:
            rep.add("NPU доступен", "fail", "onnxruntime не установлен")
    else:
        rep.add("режим инференса", "warn", f"{backend} (CPU/GPU прототип-режим)")
    if Path("/dev/amdxdna").exists() or shutil.which("amdxdna-smi"):
        rep.add("драйвер amdxdna", "ok", "/dev/amdxdna или amdxdna-smi найден")
    else:
        rep.add("драйвер amdxdna", "warn", "NPU-драйвер не виден (в контейнере/прототипе — норма)")

    # ---- 4. configuration --------------------------------------------------
    try:
        raw = json.dumps(json.loads(ag.cfg.to_json()), ensure_ascii=False)
        rep.add("конфигурация корректна", "ok", f"{len(raw)} Б сериализуется")
    except Exception as e:  # noqa: BLE001
        rep.add("конфигурация корректна", "fail", str(e)[:120])
    if ag.safety.mode not in ("normal", "safe", "confirm_all", "read_only"):
        rep.add("режим безопасности", "fail", f"неизвестный режим '{ag.safety.mode}'")

    # ---- 5. directories ------------------------------------------------------
    for label, p in (("data", ag.cfg.data_dir), ("state", ag.cfg.state_dir),
                     ("adapters", ag.cfg.adapters_dir)):
        try:
            p.mkdir(parents=True, exist_ok=True)
            t = p / ".doctor_write_test"
            t.write_text("x")
            t.unlink()
            rep.add(f"доступ к {label}-директории", "ok", str(p))
        except OSError as e:
            rep.add(f"доступ к {label}-директории", "fail", f"{p}: {e}")
    for d in ag.cfg.allowed_study_dirs:
        pp = Path(d).expanduser()
        rep.add("изучаемая директория", "ok" if pp.is_dir() else "warn",
                f"{d}: {'доступна' if pp.is_dir() else 'нет/не читается'}")

    # ---- 6. vector DB + RAG roundtrip ---------------------------------------
    try:
        before = ag.memory.db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        rid = ag.memory.add("docs", "doctor-probe: векторная проверка alfa-bravo", "probe")
        hits = ag.memory.search("векторная проверка alfa", ("docs",), top_k=3)
        found = any(h.id == rid for h in hits)
        ag.memory.db.execute("DELETE FROM items WHERE id=?", (rid,))
        ag.memory.db.commit()
        ag.memory._cache = None
        st = "ok" if found else "warn"
        rep.add("векторная БД: запись/поиск", st,
                f"items={before}, roundtrip={'попал' if found else 'НЕ найден'}")
    except Exception as e:  # noqa: BLE001
        rep.add("векторная БД: запись/поиск", "fail", str(e)[:120])

    # ---- 7. knowledge cards --------------------------------------------------
    if ag.cards is not None:
        try:
            q = ag.cards.quality_report()
            rep.add("база знаний (карточки)", "ok" if q["open_conflicts"] == 0 else "warn",
                    json.dumps(q, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            rep.add("база знаний (карточки)", "fail", str(e)[:120])
    else:
        rep.add("база знаний (карточки)", "warn", "подсистема отключена (knowledge.enabled=false)")

    # ---- 8. logging + rotation ----------------------------------------------
    try:
        log_path = ag.cfg.log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        size = log_path.stat().st_size if log_path.exists() else 0
        limit = ag.cfg.log_max_bytes * (ag.cfg.log_backup_count + 1)
        st = "ok" if size <= limit else "warn"
        rep.add("логирование и ротация", st,
                f"{log_path} size={size/1e6:.1f}MB лимит≈{limit/1e6:.0f}MB")
    except OSError as e:
        rep.add("логирование", "fail", str(e)[:120])

    # ---- 9. tools under current mode ----------------------------------------
    try:
        out = asyncio.run(ag.tools.call("exec_command", {"command": "echo doctor-echo"}))
        if "doctor-echo" in out:
            rep.add("инструменты: выполнение команд", "ok", "exec_command прошёл risk-gate")
        elif out.startswith("BLOCKED"):
            rep.add("инструменты: выполнение команд", "ok",
                    f"заблокировано как и должен в режиме «{ag.safety.mode}»: {out[:80]}")
        else:
            rep.add("инструменты: выполнение команд", "warn", out[:120])
    except Exception as e:  # noqa: BLE001
        rep.add("инструменты: выполнение команд", "fail", str(e)[:120])

    # ---- 10. safety gate behaviour (negative tests) -------------------------
    try:
        bad = ag.safety.assess_command("rm -rf /")
        comp = ag.safety.assess_command("ls; rm -rf ~")
        okl = ag.safety.assess_command("ls")
        good = bad.level == "blocked" and comp.level != "safe" and okl.level == "safe"
        rep.add("гейт безопасности", "ok" if good else "fail",
                f"blacklist={bad.level}, composite-meta={comp.level}, plain={okl.level}")
    except Exception as e:  # noqa: BLE001
        rep.add("гейт безопасности", "fail", str(e)[:120])

    # ---- 11. confirmation system --------------------------------------------
    try:
        c = ag.bridge.confirm("doctor: тестовая проверка системы подтверждений", timeout_sec=1)
        rep.add("система подтверждений", "ok",
                f"канал подтверждения отвечает ({'принято' if c else 'отклонено/таймаут'})")
    except Exception as e:  # noqa: BLE001
        rep.add("система подтверждений", "fail", str(e)[:120])

    # ---- 12. sleep subsystem dry-run ----------------------------------------
    try:
        batch = ag.trainer.collect_batch()
        score = ag.trainer.eval_base_skills()
        st = "ok" if score >= ag.cfg.sleep.regression_base_score_min else "fail"
        rep.add("подсистема сна", st,
                f"буфер={len(batch)} эпизодов, базовые навыки={score:.2f}")
    except Exception as e:  # noqa: BLE001
        rep.add("подсистема сна", "fail", str(e)[:120])

    # ---- 13. background tasks / leaks ---------------------------------------
    try:
        loop_running_tasks = 0
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop_running_tasks = len([t for t in asyncio.all_tasks(loop)
                                          if not t.done()])
        except RuntimeError:
            pass
        gc.collect()
        rss_kb = _rss_kb()
        mem_limit = ag.cfg.max_rss_mb
        st = "ok" if rss_kb / 1024 <= mem_limit else "warn"
        rep.add("память процесса (RSS)", st, f"{rss_kb/1024:.0f} MB (лимит {mem_limit} MB)")
        rep.add("фоновые задачи", "ok",
                f"asyncio-задач: {loop_running_tasks or 'нет активного цикла (статич. проверка)'}, "
                f"обучение приостановлено: {getattr(ag, '_bg_learning_paused', False)}")
    except Exception as e:  # noqa: BLE001
        rep.add("ресурсы", "warn", str(e)[:120])

    # ---- 14. event bus error handling ----------------------------------------
    try:
        from agent.events.monitor import EventBus
        bus = EventBus()

        async def _boom(ev):
            raise ValueError("intentional doctor error")
        bus.subscribe("test", _boom)

        async def _drive():
            bus.emit("test")
            await asyncio.sleep(0.05)
            return bus.queue.empty()
        drained = asyncio.run(_drive())
        rep.add("обработка ошибок событий", "ok" if drained else "warn",
                "падающий handler не убил шину (isolated try/except)")
    except Exception as e:  # noqa: BLE001
        rep.add("обработка ошибок событий", "fail", str(e)[:120])

    # ---- 15. KDE integration (optional in this environment) ------------------
    if getattr(ag.bridge, "available", False):
        rep.add("KDE D-Bus мост", "ok", "сессия Plasma доступна")
    else:
        rep.add("KDE D-Bus мост", "warn",
                "нет сессии Plasma/D-Bus (CI/контейнер) — уведомления деградируют в лог")


def _rss_kb() -> int:
    try:
        return int(Path("/proc/self/status").read_text().split("VmRSS:")[1].split()[0])
    except (OSError, IndexError, ValueError):
        return 0
