"""Agent loop: wires model, context, tools, initiative, events, memory, sleep.

Energy profile: everything is event-driven; the process blocks on an asyncio
queue. The SSM wakes only for user requests / significant events; screen state
is fetched only when the model explicitly asks via a tool.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from agent.config import AgentConfig
from agent.core.context import ContextManager
from agent.core.initiative import InitiativeEngine
from agent.core.model import SSMModel
from agent.core.runtime import Channels, ControlMixin
from agent.events.monitor import EventBus, IdleWatcher, ShellSocketSource, StdinSource
from agent.kde.dbus import KDEBridge
from agent.memory.rag import Memory
from agent.tools.registry import ToolRegistry
from agent.tools.safety import SafetyGate
from agent.training.sleep import AdapterTrainer

log = logging.getLogger("kde-agent")


class Agent(ControlMixin):
    MAX_STEPS = 4   # bounded ReAct loop per request (energy cap)

    def __init__(self, cfg: AgentConfig | None = None):
        self.cfg = cfg or AgentConfig.load()
        self.cfg.ensure_dirs()
        self._setup_logging()

        self.memory = Memory(self.cfg.db_path, self.cfg.rag.dim)
        self.bridge = KDEBridge()
        self.safety = SafetyGate(self.cfg.safety)
        # --- knowledge subsystem: cards wired early (tools use them); the
        # study pipeline is wired after the event bus exists (idle gate input).
        # Whole subsystem is config-switchable (knowledge.enabled=false -> off).
        self.cards = None
        self.study = None
        if self.cfg.knowledge.enabled:
            from agent.knowledge.cards import CardStore
            self.cards = CardStore(self.memory, self.cfg.knowledge.dedup_high,
                                   self.cfg.knowledge.dedup_low)
        self.tools = ToolRegistry(self.cfg, self.memory, self.safety, self.bridge,
                                  cards=self.cards)
        self.model = SSMModel(self.cfg)
        self.ctx = ContextManager(budget_tokens=self.cfg.model.context_budget_tokens)
        # Typo judgement belongs to the model analyzer, not to string matching.
        # With real weights configured (backend != MockBackend) it is an
        # LLMModelAnalyzer wrapping the SSM; in prototype mode it is the
        # deterministic mock that mirrors the same evidence pipeline.
        from agent.core.model import MockBackend
        from agent.core.typo_model import make_analyzer
        typo_backend = None if isinstance(self.model.backend, MockBackend) else self.model.backend
        self.initiative = InitiativeEngine(self.cfg.initiative, self.memory,
                                           self.cfg.data_dir, self.bridge,
                                           typo_analyzer=make_analyzer(
                                               self.cfg.initiative, self.memory,
                                               backend=typo_backend))
        self.initiative.configure_safety(self.cfg.safety)
        self.trainer = AdapterTrainer(self.cfg, self.memory, self.memory.embedder,
                                      self.cfg.adapters_dir)
        # --- code assistant (diff + self-check ladder), config-switchable
        self.code = None
        if self.cfg.code_assist.enabled:
            from agent.tools.code_assist import CodeAssistant
            self.code = CodeAssistant(self.cfg, self.memory, self.safety,
                                      self.bridge, self.cfg.code_assist.gates)
            # /undo must be able to roll back applied diffs (text restore)
            self.code.undo_sink = self.tools.register_edit_undo
        self.bus = EventBus()
        if self.cards is not None:
            from agent.knowledge.pipeline import EnergyGate, StudyPipeline, StudyQueue
            queue = StudyQueue(self.cfg.data_dir / "study_queue.db")
            gate = EnergyGate(on_battery=self.bridge.on_battery,
                              idle_sec=self.bridge.idle_seconds)
            self.study = StudyPipeline(self.cards, queue, gate,
                                       max_segments_per_source=self.cfg.knowledge.max_segments_per_source)
        # auto-seed: first run with an EMPTY knowledge base must not sit idle —
        # the agent studies its own project docs (README + docs/) immediately.
        # User-authorized dirs (cfg.knowledge.auto_study_dirs) are seeded too.
        self._seeded = False
        if self.study is not None and self.cfg.knowledge.enabled:
            try:
                empty_kb = self.cards.quality_report().get("cards", 0) == 0
                n = 0
                if empty_kb:
                    n += self.study.ingest_dir(str(Path(__file__).resolve().parent.parent),
                                               project="agent-self", priority=3)
                for d in self.cfg.knowledge.auto_study_dirs:
                    n += self.study.ingest_dir(d, project="user-dir", priority=2)
                if n:
                    log.info("bootstrap study: %d docs queued", n)
                    self._seeded = True
            except Exception:  # noqa: BLE001
                log.exception("bootstrap study seeding failed (non-fatal)")
        # ---- runtime/control-plane state (spec: daemon + settings-ready API)
        self.channels = Channels(self)
        # initiative engine holds the InitiativeConfig dataclass; give it a
        # side-channel to the output registry so headless interventions reach
        # stdout/log instead of vanishing when no KDE notification bus exists.
        self.cfg.initiative._channels = self.channels
        self._sleep_stop_event = asyncio.Event()
        self._bg_learning_paused = False
        self.running = False

    def _setup_logging(self) -> None:
        self.cfg.state_dir.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        # Idempotent: re-running setup (tests, reloads) must not stack
        # handlers — every extra FileHandler keeps a duplicate log file
        # open (ResourceWarning) and multiplies every log line.
        for h in list(root.handlers):
            if getattr(h, "_kde_agent", False):
                root.removeHandler(h)
                try:
                    h.close()
                except Exception:  # noqa: BLE001
                    pass
        # Rotating file handler: daemon runs forever, logs must stay bounded
        # (spec: «логи не должны бесконтрольно разрастаться»).
        from logging.handlers import RotatingFileHandler
        handlers = [RotatingFileHandler(self.cfg.log_path,
                                        maxBytes=self.cfg.log_max_bytes,
                                        backupCount=self.cfg.log_backup_count),
                    logging.StreamHandler()]
        for h in handlers:
            h._kde_agent = True      # our handler tag (see above)
        logging.basicConfig(
            level=getattr(logging, self.cfg.log_level.upper(), logging.INFO),
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            handlers=handlers)

    # ---------------- wiring ----------------
    def subscribe(self) -> None:
        self.bus.subscribe("user_request", self.on_user_request)
        self.bus.subscribe("user_command", self.on_user_command)
        self.bus.subscribe("idle", self.on_idle)
        self.bus.subscribe("action_cancelled", self.on_cancelled)
        self.bus.subscribe("*", self.audit)

    async def run(self) -> None:
        self.running = True
        await self.bridge.connect()
        self.subscribe()
        self.tools.update_widget("online")
        tasks = [asyncio.create_task(c, name=n) for c, n in (
            (StdinSource(self.bus).run(), "stdin"),
            (ShellSocketSource(self.bus).run(), "shell-socket"),
            (IdleWatcher(self.bus, self.bridge,
                         self.cfg.initiative.idle_help_after_sec).run(), "idle"),
            (self.bus.run(), "event-bus"),
            (self.night_scheduler(), "night-scheduler"),
            (self.startup_study_burst(), "startup-study"))]
        # control socket task (created by the daemon entrypoint) is awaited
        # too — see cli._daemon_main; here we only own our source tasks.
        log.info("kde-agent up; data=%s", self.cfg.data_dir)
        try:
            done, pending = await asyncio.wait(
                tasks + [asyncio.create_task(self._wait_stop(), name="stop-watcher")],
                return_when=asyncio.FIRST_COMPLETED)
            # a source crashed or stop was requested — tear down the rest
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, asyncio.CancelledError):
                    log.error("source %s crashed: %r", t.get_name(), exc)
                    raise exc
        finally:
            self.running = False

    async def _wait_stop(self) -> None:
        """Poll-stop helper: shutdown() flips self.running; sources otherwise
        block forever on queues/pipes (daemon would never exit)."""
        while self.running:
            await asyncio.sleep(0.5)

    async def startup_study_burst(self) -> None:
        """Never-idle policy at boot: if bootstrap seeding queued material,
        process a short bounded batch right away (KB starts filling without
        waiting for the first idle window). Heavy work stays off the hot path:
        to_thread + 10s wall-clock budget; energy gate still applies inside."""
        if not self._seeded or self.study is None:
            return
        try:
            reps = await asyncio.to_thread(self.study.run_pending, 10.0)
            if reps:
                cards = sum(r.get("cards", 0) for r in reps)
                log.info("startup study burst: %d sources, %d cards", len(reps), cards)
                self.memory.episode("study", f"startup burst: {len(reps)} sources")
        except Exception:  # noqa: BLE001
            log.exception("startup study burst failed (non-fatal)")

    async def shutdown(self, reason: str = "stop") -> None:
        """Safe graceful stop (spec item 5: SIGTERM/logind events).
        Order matters: silence initiative first (no dangling offers), persist
        habits + context state, close background resources, then flip running.
        Active tool subprocesses are short-lived and guarded by timeouts."""
        if not self.running:
            return
        log.info("shutdown requested: %s", reason)
        try:
            self.initiative.set_enabled(False)      # no new interventions mid-stop
            self.initiative.save()                  # RPE/habits must survive restart
        except Exception:  # noqa: BLE001
            log.exception("initiative save on shutdown failed")
        try:
            self._sleep_stop_event.set()
            self._bg_learning_paused = True
        except Exception:  # noqa: BLE001
            pass
        for closer in ("close",):
            try:
                fn = getattr(self.study.queue, closer, None) if self.study else None
                if fn:
                    fn()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.memory.close()
        except Exception:  # noqa: BLE001
            pass
        self.running = False
        log.info("kde-agent stopped cleanly (%s)", reason)

    async def stop(self) -> None:
        await self.shutdown("stop")

    # ---------------- core reasoning loop ----------------
    async def on_user_request(self, ev) -> None:
        text = ev.payload["text"]
        channel = ev.payload.get("channel", "terminal")
        await self.handle_input(text, channel=channel)

    async def think_loop(self, user_msg: str = "") -> str:
        final = ""
        for step in range(self.MAX_STEPS):
            res = self.model.think(user_msg or final[:200] or "continue",
                                   context=self.ctx.render())
            self.ctx.add("assistant", res.thought or res.reply, score=1.5)
            if not res.actions:
                final = res.reply
                break
            obs = []
            for act in res.actions[:3]:
                out = await self.tools.call(act["tool"], act.get("args", {}))
                # screenshots/observations enter context only now — by demand
                score = 2.0 if out.startswith(("ERROR", "BLOCKED")) else 1.0
                self.ctx.add("tool", f"{act['tool']} -> {out[:600]}", score=score)
                obs.append(out)
            user_msg = ""   # subsequent steps work on observations
            final = obs[-1] if obs else final
        reply = final if final else (res.reply if res else "")
        self.say(reply, kind="answer")   # routed through output channels
        self.ctx.add("assistant", reply, score=1.5)
        self.memory.episode("answer", reply)
        self.tools.update_widget("idle", task="готов", pending="")
        return reply

    # ---------------- event handlers ----------------
    async def on_user_command(self, ev) -> None:
        cmd = ev.payload.get("cmd", "")
        ivs = self.initiative.on_user_line(cmd)
        if ivs:
            self.initiative.present(ivs)
        self.ctx.add("event", f"terminal command observed: {cmd[:120]}", score=0.8)

    async def on_idle(self, ev) -> None:
        # reminders first: they are "initiative with a timestamp" — due ones
        # fire even when typo/habit detectors stay silent (spec: проактивность)
        for rid, text in self.tools.due_reminders():
            self.say(f"напоминание: {text}", kind="initiative")
            self.tools.acknowledge_reminder(rid)
            self.memory.episode("reminder_fired", text)
        ivs = self.initiative.on_idle(ev.payload.get("idle_sec", 0))
        if ivs:
            self.initiative.present(ivs)
        # never-idle policy: idle is the natural window for background study
        # (energy gate + budget inside still protect battery/load/user work)
        if self.study is not None and not getattr(self, "_bg_learning_paused", False):
            try:
                done = await asyncio.to_thread(self.study.run_pending, 15.0)
                if done:
                    log.info("idle study batch: %d sources", len(done))
            except Exception:  # noqa: BLE001
                log.exception("idle study batch failed (non-fatal)")

    async def on_cancelled(self, ev) -> None:
        self.initiative.habits.utility.setdefault("idle", -0.2)
        self.memory.episode("cancel", "user cancelled agent action", value=1.2)

    async def audit(self, ev) -> None:
        log.debug("audit %s %s", ev.kind, ev.payload)

    # ---------------- slash commands (terminal UI) ----------------
    # Thin adapter over the control plane: every slash command maps 1:1 to a
    # `control()` call, so CLI / future settings window / D-Bus share logic.
    async def slash(self, line: str) -> None:
        parts = line.lstrip("/").split(maxsplit=1)
        cmd, arg = parts[0], (parts[1] if len(parts) > 1 else "")
        rep = lambda d: print(json.dumps(d, ensure_ascii=False)[:2000]
                              if isinstance(d, dict) and "report" not in d and "log" not in d
                              else (d.get("report", {}).get("text") if isinstance(d, dict) and "report" in d
                                    else d))  # noqa: E731
        if cmd == "feedback":           # /feedback <key> <reaction>
            keys = list(self.initiative.pending)
            key = next((k for k in keys if arg.startswith(k)), keys[0] if keys else None)
            if key:
                r = self.initiative.feedback(key, arg)
                print(f"[reward={r:+.2f}] записано в память")
        elif cmd == "missed":
            self.initiative.habits.penalize_missed(arg or "typo_cmd", {"note": line})
            print("пропущенная ошибка учтена (штраф)")
        elif cmd == "sleep":
            rep(await self.control_async("sleep", arg))
        elif cmd == "study":
            # /study add <файл|dir> -> control("study", "add ..."); иначе аргумент
            # передаётся как есть (число = бюджет секунд, stop/queue).
            rep(await self.control_async("study", arg))
        elif cmd == "index":
            if self.study is not None:
                n = self.study.ingest_dir(arg or str(Path.cwd()),
                                          priority=self.cfg.knowledge.study_priority_default)
                print(f"в очередь изучения поставлено файлов: {n} "
                      f"(обработка — по энергогейту, см. /study)")
            else:
                n = index_docs(self.memory, arg or str(Path.cwd()))
                print(f"проиндексировано фрагментов: {n}")
        elif cmd == "kb":
            rep(self.control("kb", arg))
        elif cmd == "mode":
            rep(self.control("mode", arg))
        elif cmd == "proactivity":
            rep(self.control("proactivity", arg))
        elif cmd == "doctor":
            from agent.core.diagnostics import run_doctor
            print(run_doctor(self)["checks"])       # live instance report
        elif cmd == "status":
            rep(self.control("status"))
        elif cmd == "tools":
            print(", ".join(self.tools.names()))
        elif cmd == "state":
            print(f"adapter={self.model.adapter_version} ctx={self.ctx.total()} tok "
                  f"pending={len(self.initiative.pending)} "
                  f"thresholds={self.initiative.habits.threshold}")
        elif cmd == "undo":
            print(await self.tools.call("undo_last", {}))
        elif cmd == "quit":
            await self.stop()
        else:
            print("/feedback /missed /sleep [stop|last] /study [add <dir>] "
                  "/index <dir> /kb <stats|list|search|conflicts|resolve|remove|reindex|clear> "
                  "/mode <normal|safe|confirm_all|read_only> /proactivity on|off "
                  "/doctor /status /tools /state /undo /quit")

    # ---------------- night scheduler (consolidation) ----------------
    async def night_scheduler(self) -> None:
        last_sleep_at = 0.0
        while self.running:
            try:
                await asyncio.wait_for(self._sleep_stop_event.wait(), timeout=300)
                # stop requested: idle until cleared (`ctl study` clears it)
                while self.running and self._sleep_stop_event.is_set():
                    await asyncio.sleep(5)
                self._sleep_stop_event.clear()
                continue
            except asyncio.TimeoutError:
                pass                       # normal path: 5-min wake, trivial cost
            hhmm = time.strftime("%H:%M")
            idle = self.bridge.idle_seconds()
            if getattr(self, "_bg_learning_paused", False):
                continue                   # explicit user pause (ctl sleep stop)
            should = (hhmm >= self.cfg.sleep.nightly_start and hhmm <= "05:00") \
                or (idle >= 0 and idle > self.cfg.sleep.train_at_idle_min * 60)
            if should and self.cfg.sleep.enabled \
                    and time.time() - last_sleep_at > 3600:   # debounce: at most 1/h
                last_sleep_at = time.time()
                report = self.trainer.sleep_cycle()
                (self.cfg.data_dir / "last_sleep.json").write_text(
                    json.dumps(report, ensure_ascii=False))
                if "saved" in report:
                    self.model.load_adapter(str(self.trainer.active_path))
                    log.info("night adapter activated: %s", report["saved"])
            # background study shares the same quiet windows (energy-gated);
            # idle==-1 means "no idle source" — only the nightly window works then
            if self.study is not None and ((idle >= 0 and idle > 60)
                                           or hhmm >= self.cfg.sleep.nightly_start):
                try:
                    done = self.study.run_pending(budget_sec=30)
                    if done:
                        log.info("study batch: %d sources", len(done))
                except Exception:  # noqa: BLE001
                    log.exception("study batch failed (non-fatal)")


def index_docs(memory: Memory, root: str | Path, chunk: int = 1200,
               patterns=("*.md", "*.txt", "*.rst", "*.man.txt")) -> int:
    """Feed documentation into RAG (facts live here, never in weights)."""
    n = 0
    root = Path(root).expanduser()
    for pat in patterns:
        for f in root.rglob(pat):
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            for i in range(0, len(text), chunk):
                memory.add("docs", text[i:i + chunk], "doc",
                           meta={"src": str(f), "part": i // chunk})
                n += 1
    return n
