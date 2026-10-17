"""Agent runtime core: state, control commands, lifecycle.

Channel-agnostic by design (spec): every input source (terminal stdin, shell
hook socket, D-Bus event, future voice input / GUI chat / hotkeys) funnels
into `handle_input(text, channel)`; every output goes through one of the
`OutputChannel`s (stdout text, notification, log; future TTS/GUI chat). The
kernel never knows which physical UI is attached — that is exactly the seam
where the C++ settings window plugs in later without touching this module.

Control plane (used now by CLI slash-commands & `kde-agent ctl`, later by the
settings UI / D-Bus service): a flat command dispatch `Agent.control(name, arg)`
returning a plain dict — JSON-friendly, testable, UI-ready.
"""
from __future__ import annotations

import asyncio
import re as _re
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class OutputMsg:
    kind: str          # answer | initiative | notify | error | status
    text: str
    channel_hint: str = "auto"   # auto -> cfg.output_channels
    meta: dict = field(default_factory=dict)


class Channels:
    """Output channels registry. Prototype wires stdout+log; KDE notifications
    are added when the bridge connects; voice/GUI are declared but inactive."""

    def __init__(self, agent):
        self.agent = agent
        self.registry: dict[str, Callable[[OutputMsg], None]] = {}
        self.declare("stdout", self._stdout)
        self.declare("log", self._log)
        self.declare("notification", self._notify)      # active once bridge up
        self.declare("voice_out", self._inactive)       # future TTS
        self.declare("gui_chat", self._inactive)        # future plasmoid/chat

    def declare(self, name: str, fn: Callable[[OutputMsg], None]) -> None:
        self.registry[name] = fn

    def _inactive(self, msg: OutputMsg) -> None:
        logging.getLogger("kde-agent.channel").debug(
            "channel %s not wired yet (planned): %s", msg.kind, msg.text[:80])

    def _stdout(self, msg: OutputMsg) -> None:
        prefix = {"initiative": "AGENT[инициатива]> ", "error": "AGENT[ошибка]> ",
                  "status": "", "notify": "AGENT> ", "answer": "AGENT> "}.get(msg.kind, "AGENT> ")
        print(f"\n{prefix}{msg.text}\n")

    def _log(self, msg: OutputMsg) -> None:
        logging.getLogger("kde-agent.out").info("[%s] %s", msg.kind, msg.text[:400])

    def _notify(self, msg: OutputMsg) -> None:
        b = getattr(self.agent, "bridge", None)
        if b and b.available:
            b.notify("KDE Agent", msg.text, urgency="normal" if msg.kind != "initiative" else "low")
        else:
            self._log(msg)

    def emit(self, msg: OutputMsg) -> None:
        allowed = self.agent.cfg.output_channels or ["stdout", "log"]
        for ch in allowed:
            fn = self.registry.get(ch)
            if fn:
                try:
                    fn(msg)
                except Exception:  # noqa: BLE001  a broken channel must not kill the loop
                    logging.getLogger("kde-agent.channel").exception("channel %s failed", ch)


# ---------------------------------------------------------------------------
CONTROL_HELP = {
    "status":            "состояние ядра, режим, каналы, очередь событий",
    "doctor":            "полная диагностика подсистем (отчёт ok/warn/fail)",
    "mode <нормальный|safe|confirm_all|read_only>": "режим работы безопасности",
    "proactivity on|off": "глобальное включение/выключение инициативы (демон продолжает обслуживать команды)",
    "sleep":             "принудительный цикл сна (обучение адаптеров + гейты)",
    "sleep stop":        "остановить сон/фоновое обучение немедленно",
    "sleep last":        "отчёт последнего сна",
    "study [бюджет_сек]": "принудительный прогон очереди фонового изучения",
    "study stop":        "принудительная остановка фонового обучения",
    "study queue":       "состояние очереди изучения",
    "study add <файл|dir>": "поставить файл или директорию в очередь изучения",
    "kb stats":          "статус базы знаний (карточки, конфликты, устаревшие)",
    "kb list":           "последние карточки знаний",
    "kb search <запрос>":"поиск по карточкам",
    "kb conflicts":      "список открытых конфликтов",
    "kb resolve <keep_id> <drop_id>": "разрешить конфликт вручную",
    "kb remove <id>":    "удалить карточку (устаревшее знание)",
    "kb reindex":        "принудительная переиндексация векторов памяти",
    "kb clear <docs|cards|episodes|prefs|all>": "очистка коллекции (требует подтверждения)",
    "logs [N]":          "последние N строк лога",
    "history [N]":       "последние действия агента (эпизоды)",
    "config [путь]":     "показать конфигурацию (без секретов)",
    "tasks":             "активные фоновые задачи (idle-гейт, планировщик)",
    "windows [list|probe|suggest <контекст>]": "оконный слой инициативы (снимки, профили)",
    "scenario [list|show <имя>|remove <имя>]": "сохранённые сценарии-навыки",
}


class ControlMixin:
    """Implements the control plane on top of Agent state. Mixed into Agent."""

    # ---- output helpers ---------------------------------------------------
    def say(self, text: str, kind: str = "answer") -> None:
        self.channels.emit(OutputMsg(kind, text))

    # ---- unified input funnel (channel-agnostic kernel entry) -------------
    async def handle_input(self, text: str, channel: str = "terminal") -> None:
        """Single entry point for ALL input sources: terminal line, shell hook,
        D-Bus event, future voice-in / GUI chat / hotkey. Slash/ctl commands
        bypass the model; natural language goes to the ReAct loop."""
        text = text.strip()
        if not text:
            return
        self.ctx.add("user", f"[{channel}] {text}", score=2.0)
        if text.startswith("/"):
            await self.slash(text)
            return
        await self.think_loop(user_msg=text)

    # ---- control plane ----------------------------------------------------
    async def control_async(self, name: str, arg: str = "") -> dict:
        """Async front door of the control plane (daemon socket server uses
        this). Heavy blocking ops (study batches) run in a worker thread so
        the event loop keeps serving user requests — the daemon must never
        freeze because a background job is draining its queue."""
        import asyncio as _aio
        if name.lower().strip() == "windows":
            return await self._ctl_windows(arg)
        if name.lower().strip() == "ask":
            q = (arg or "").strip()
            if not q:
                return {"ok": False, "error": "ask <вопрос>"}
            hits = self.memory.search(q, collections=("docs", "cards"), top_k=3)
            ctx = "\n---\n".join(h.text[:500] for h in hits)
            prompt = f"Вопрос: {q}\nКонтекст из базы знаний:\n{ctx}" if ctx else q
            reply = await self.think_loop(prompt)
            return {"ok": True, "answer": reply, "sources": len(hits)}
        if name.lower().strip() == "study":
            a = arg.strip()
            if not a.startswith("add") and a not in ("stop", "queue"):
                if self.study is not None:
                    # `study run --max N`: ограничить число документов за запуск
                    mmax = _re.search(r"--max[= ](\d+)", a)
                    max_docs = int(mmax.group(1)) if mmax else None
                    force = "--force" in a or "форс" in a
                    budget = float(a) if a.replace(".", "", 1).isdigit() else 20.0
                    self._bg_learning_paused = False
                    # NOTE: pass positionally — to_thread only forwards *args;
                    # a lambda would be picklable-safe but keyword form broke here.
                    reps = await _aio.to_thread(self.study.run_pending,
                                                budget, max_docs, force)
                    return {"ok": True, "processed": len(reps),
                            "queue": self.study.queue.stats()}
        return self.control(name, arg)

    def _study_kick(self, budget: float = 20.0) -> None:
        """Fire-and-forget background batch (never blocks the caller)."""
        if self.study is None or getattr(self, "_bg_learning_paused", False):
            return
        async def _run():
            try:
                reps = await asyncio.to_thread(self.study.run_pending, budget_sec=budget)
                if reps:
                    logging.getLogger("kde-agent.study").info(
                        "kick batch: %d sources processed", len(reps))
            except Exception:  # noqa: BLE001
                logging.getLogger("kde-agent.study").exception("kick batch failed")
        try:
            asyncio.get_running_loop().create_task(_run())
        except RuntimeError:
            pass   # no loop (sync tests / ctl cold path): queue persists anyway

    def control(self, name: str, arg: str = "") -> dict:
        """Programmatic management API (CLI `ctl`, D-Bus, future settings UI)."""
        name = name.lower().strip()
        try:
            if name == "status":
                return self._ctl_status()
            if name == "doctor":
                from agent.core.diagnostics import run_doctor
                return {"report": run_doctor(self)}
            if name == "mode":
                err = self.safety.set_mode(arg.strip())
                if err:
                    return {"ok": False, "error": err.reason}
                self.cfg.save()
                return {"ok": True, "mode": self.safety.mode}
            if name == "proactivity":
                on = arg.strip().lower() in ("on", "1", "true", "вкл")
                self.initiative.set_enabled(on)
                self.cfg.initiative.enabled = on
                self.cfg.save()
                return {"ok": True, "proactivity": on}
            if name == "sleep":
                a = arg.strip().lower()
                if a in ("stop", "останов"):
                    self._sleep_stop_event.set()
                    self._bg_learning_paused = True
                    return {"ok": True, "sleep": "stopped/paused"}
                if a in ("last", "report", "отчёт"):
                    # фикс парсинга: `ctl sleep last` раньше отдавал
                    # неизвестную команду, потому что аргумент не нормализовался
                    p = self.cfg.data_dir / "last_sleep.json"
                    return json.loads(p.read_text()) if p.exists() else {"skipped": "нет данных"}
                rep = self.trainer.sleep_cycle()
                (self.cfg.data_dir / "last_sleep.json").write_text(json.dumps(rep, ensure_ascii=False))
                if "saved" in rep:
                    self.model.load_adapter(str(self.trainer.active_path))
                return {"ok": True, "report": rep}
            if name == "study":
                if self.study is None:
                    return {"ok": False, "error": "knowledge subsystem disabled"}
                a = arg.strip()
                if a.startswith("add"):
                    # `study add <файл|директория>` — раньше путь считался
                    # только директорией; одиночный README/man-файл молча
                    # не попадал в очередь (rglob по файлу пустой).
                    path = a.split(maxsplit=1)[1].strip() if " " in a else ""
                    if not path:
                        return {"ok": False, "error": "study add <путь к файлу или директории>"}
                    from pathlib import Path as _P
                    pp = _P(path).expanduser()
                    if not pp.exists():
                        return {"ok": False, "error": f"путь не найден: {pp}"}
                    prio = self.cfg.knowledge.study_priority_default
                    if pp.is_dir():
                        n = self.study.ingest_dir(str(pp), priority=prio)
                    else:
                        from agent.knowledge.pipeline import Source
                        self.study.queue.enqueue(Source(path=str(pp), priority=prio))
                        n = 1
                    # never-idle policy: user explicitly asked to study this —
                    # kick a short background batch instead of waiting for idle
                    self._study_kick(20.0)
                    return {"ok": True, "queued": n, "path": str(pp)}
                if a == "stop":
                    # принудительная остановка фонового обучения (spec)
                    self._bg_learning_paused = True
                    return {"ok": True, "study": "paused"}
                if a == "queue":
                    return {"ok": True, "queue": self.study.queue.stats()}
                mmax = _re.search(r"--max[= ](\d+)", a)
                max_docs = int(mmax.group(1)) if mmax else None
                force = "--force" in a or "форс" in a
                budget = float(a) if a.replace(".", "", 1).isdigit() else 20.0
                self._bg_learning_paused = False
                reps = self.study.run_pending(budget_sec=budget, max_docs=max_docs,
                                              force=force)
                return {"ok": True, "processed": len(reps), "queue": self.study.queue.stats()}
            if name == "kb":
                return self._ctl_kb(arg)
            if name == "logs":
                n = int(arg or 30)
                lines = self.cfg.log_path.read_text(errors="replace").splitlines()[-n:] \
                    if self.cfg.log_path.exists() else []
                return {"ok": True, "log": "\\n".join(lines)[-4000:]}
            if name == "history":
                eps = self.memory.recent_episodes(int(arg or 15))
                return {"ok": True, "episodes": [{"kind": e.meta.get("kind", ""),
                                                   "text": e.text[:160], "ts": round(e.value, 2)} for e in eps]}
            if name == "config":
                d = json.loads(self.cfg.to_json())
                return {"ok": True, "config": d}
            if name == "tasks":
                return {"ok": True, "running": self.running,
                        "learning_paused": getattr(self, "_bg_learning_paused", False),
                        "pending_interventions": len(self.initiative.pending),
                        "study_queue": self.study.queue.stats() if self.study else {}}
            if name == "windows":
                return {"ok": False, "error": "async-only: используйте ctl windows через сокет демона"}
            if name == "scenario":
                return self._ctl_scenario(arg)
            if name == "help":
                return {"ok": True, "commands": CONTROL_HELP}
            return {"ok": False, "error": f"неизвестная команда '{name}'", "commands": list(CONTROL_HELP)}
        except Exception as e:  # noqa: BLE001  control plane must never crash the daemon
            logging.getLogger("kde-agent.ctl").exception("control %s failed", name)
            return {"ok": False, "error": str(e)}

    # ---- sub-handlers -----------------------------------------------------
    async def _ctl_windows(self, arg: str) -> dict:
        """ctl windows [list|probe|suggest <контекст>]"""
        parts = arg.split(maxsplit=1)
        op = parts[0] if parts else "list"
        if op == "probe":
            ch = await self.winprober.probe(force=True)
            return {"ok": True, "changes": len(ch),
                    "windows": len(self.winstate.windows)}
        if op == "suggest" and len(parts) > 1:
            return {"ok": True, "apps": self.winstate.suggest_for_context(parts[1]),
                    "note": "выученные спутники контекста; открытие — только через подтверждение"}
        if not self.winstate.windows and self.winprober.available():
            await self.winprober.probe(force=True)
        return {"ok": True, "available": self.winprober.available(),
                "windows": [{"wid": w.wid, "class": w.cls, "title": w.title[:60],
                             "active": w.active}
                            for w in self.winstate.windows.values()]}

    def _ctl_scenario(self, arg: str) -> dict:
        parts = arg.split(maxsplit=1)
        op = parts[0] if parts else "list"
        if op == "list":
            return {"ok": True, "scenarios": self.scenarios.list()}
        if op == "remove" and len(parts) > 1:
            ok = self.bridge.confirm(f"Удалить сценарий «{parts[1]}»?", dangerous=True)
            if not ok:
                return {"ok": False, "error": "отменено пользователем"}
            return {"ok": self.scenarios.remove(parts[1])}
        if op == "show" and len(parts) > 1:
            sc = self.scenarios.get(parts[1])
            return {"ok": bool(sc), "scenario": sc}
        return {"ok": False, "error": "scenario list|show <имя>|remove <имя>"}

    def _ctl_status(self) -> dict:
        return {"ok": True, "running": self.running,
                "mode": self.safety.mode,
                "proactivity": bool(getattr(self.cfg.initiative, "enabled", True)),
                "backend": type(self.model.backend).__name__,
                "adapter": self.model.adapter_version,
                "ctx_tokens": self.ctx.total(),
                "pending_interventions": list(self.initiative.pending)[:5],
                "channels": self.cfg.output_channels,
                "kb": self.cards.quality_report() if self.cards else "disabled",
                "sleep_last": (self.cfg.data_dir / "last_sleep.json").exists()}

    def _ctl_kb(self, arg: str) -> dict:
        if self.cards is None:
            return {"ok": False, "error": "knowledge subsystem disabled (knowledge.enabled=false)"}
        parts = arg.split()
        op = parts[0] if parts else "stats"
        if op == "stats":
            return {"ok": True, **self.cards.quality_report(),
                    "queue": self.study.queue.stats() if self.study else {}}
        if op == "list":
            return {"ok": True, "cards": [{"id": c.id, "v": c.version, "type": c.type,
                                            "summary": c.summary[:80], "value": round(c.value, 2),
                                            "usages": c.usages} for c in self.cards.list_cards(limit=25)]}
        if op == "search" and len(parts) > 1:
            q = arg.split(maxsplit=1)[1]
            found = self.cards.search(q, top_k=5)
            return {"ok": True, "hits": [{"id": c.id, "type": c.type, "summary": c.summary[:90]} for c in found]}
        if op == "conflicts":
            return {"ok": True, "conflicts": [{"card": c.id, "summary": c.summary[:70], "vs": other}
                                               for c, other in self.cards.pending_conflicts()]}
        if op == "resolve" and len(parts) == 3:
            self.cards.resolve_conflict(int(parts[1]), int(parts[2]))
            return {"ok": True, "resolved": f"kept #{parts[1]}"}
        if op == "remove" and len(parts) == 2:
            self.cards.remove(int(parts[1]))
            return {"ok": True, "removed": parts[1]}
        if op == "reindex":
            return self._kb_reindex()
        if op == "clear" and len(parts) >= 2:
            coll = parts[1]
            ok = self.bridge.confirm(f"Удалить коллекцию базы знаний '{coll}'? Это необратимо.", dangerous=True)
            if not ok:
                return {"ok": False, "error": "отменено пользователем"}
            return self._kb_clear(coll)
        return {"ok": False, "error": "kb stats|list|search <q>|conflicts|resolve <keep> <drop>|remove <id>|reindex|clear <coll>"}

    def _kb_reindex(self) -> dict:
        """Recompute vectors for every item with the current frozen encoder
        (needed after swapping the embedder implementation)."""
        rows = self.memory._exec("SELECT id,text FROM items").fetchall()
        for rid, text in rows:
            vec = self.memory.embedder.embed(text)
            self.memory._exec("UPDATE items SET vec=? WHERE id=?",
                              (vec.astype("float32").tobytes(), rid))
        self.memory._commit()
        self.memory.invalidate_cache()
        return {"ok": True, "reindexed": len(rows)}

    def _kb_clear(self, coll: str) -> dict:
        if coll == "all":
            q = "DELETE FROM items"; n = "items"
            self.memory._exec("DELETE FROM cards")
            self.memory._exec("DELETE FROM card_conflicts")
        else:
            q = "DELETE FROM items WHERE collection=?"; n = coll
            if coll == "cards":
                self.memory._exec("DELETE FROM cards")
        cur = self.memory._exec(q, () if coll == "all" else (coll,))
        self.memory._commit()
        self.memory.invalidate_cache()
        return {"ok": True, "deleted": cur.rowcount, "collection": n}
