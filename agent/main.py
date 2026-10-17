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
from agent.events.monitor import EventBus, IdleWatcher, ShellSocketSource, StdinSource
from agent.kde.dbus import KDEBridge
from agent.memory.rag import Memory
from agent.tools.registry import ToolRegistry
from agent.tools.safety import SafetyGate
from agent.training.sleep import AdapterTrainer

log = logging.getLogger("kde-agent")


class Agent:
    MAX_STEPS = 4   # bounded ReAct loop per request (energy cap)

    def __init__(self, cfg: AgentConfig | None = None):
        self.cfg = cfg or AgentConfig.load()
        self.cfg.ensure_dirs()
        self._setup_logging()

        self.memory = Memory(self.cfg.db_path, self.cfg.rag.dim)
        self.bridge = KDEBridge()
        self.safety = SafetyGate(self.cfg.safety)
        # --- knowledge subsystem (cards / background study), config-switchable
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
        # --- knowledge subsystem: cards wired early (tools use them), study
        # pipeline wired after the event bus exists (idle gate input).
        self.cards = None
        self.study = None
        if self.cfg.knowledge.enabled:
            from agent.knowledge.cards import CardStore
            self.cards = CardStore(self.memory, self.cfg.knowledge.dedup_high,
                                   self.cfg.knowledge.dedup_low)
        # --- code assistant (diff + self-check ladder), config-switchable
        self.code = None
        if self.cfg.code_assist.enabled:
            from agent.tools.code_assist import CodeAssistant
            self.code = CodeAssistant(self.cfg, self.memory, self.safety,
                                      self.bridge, self.cfg.code_assist.gates)
        self.bus = EventBus()
        if self.cards is not None:
            from agent.knowledge.pipeline import EnergyGate, StudyPipeline, StudyQueue
            queue = StudyQueue(self.cfg.data_dir / "study_queue.db")
            gate = EnergyGate(on_battery=self.bridge.on_battery,
                              idle_sec=self.bridge.idle_seconds)
            self.study = StudyPipeline(self.cards, queue, gate,
                                       max_segments_per_source=self.cfg.knowledge.max_segments_per_source)
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
        handlers = [logging.FileHandler(self.cfg.log_path),
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
        sources = [StdinSource(self.bus).run(), ShellSocketSource(self.bus).run(),
                   IdleWatcher(self.bus, self.bridge,
                               self.cfg.initiative.idle_help_after_sec).run(),
                   self.bus.run(), self.night_scheduler()]
        log.info("kde-agent up; data=%s", self.cfg.data_dir)
        await asyncio.gather(*sources)

    async def stop(self) -> None:
        self.running = False

    # ---------------- core reasoning loop ----------------
    async def on_user_request(self, ev) -> None:
        text = ev.payload["text"]
        if text.startswith("/"):
            await self.slash(text)
            return
        # pre-flight initiative check on the request itself (e.g. typo in it)
        self.ctx.add("user", text, score=2.0)
        await self.think_loop(user_msg=text)

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
        print(f"\nAGENT> {reply}\n")
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
        ivs = self.initiative.on_idle(ev.payload.get("idle_sec", 0))
        if ivs:
            self.initiative.present(ivs)

    async def on_cancelled(self, ev) -> None:
        self.initiative.habits.utility.setdefault("idle", -0.2)
        self.memory.episode("cancel", "user cancelled agent action", value=1.2)

    async def audit(self, ev) -> None:
        log.debug("audit %s %s", ev.kind, ev.payload)

    # ---------------- slash commands (terminal UI) ----------------
    async def slash(self, line: str) -> None:
        parts = line.split(maxsplit=1)
        cmd, arg = parts[0], (parts[1] if len(parts) > 1 else "")
        if cmd == "/feedback":          # /feedback <key> <reaction>
            keys = list(self.initiative.pending)
            key = next((k for k in keys if arg.startswith(k)), keys[0] if keys else None)
            if key:
                r = self.initiative.feedback(key, arg)
                print(f"[reward={r:+.2f}] записано в память")
        elif cmd == "/missed":
            self.initiative.habits.penalize_missed(arg or "typo_cmd", {"note": line})
            print("пропущенная ошибка учтена (штраф)")
        elif cmd == "/sleep":
            print("запуск цикла сна…", self.trainer.sleep_cycle())
        elif cmd == "/index":
            if self.study is not None:
                n = self.study.ingest_dir(arg or str(Path.cwd()),
                                          priority=self.cfg.knowledge.study_priority_default)
                print(f"в очередь изучения поставлено файлов: {n} "
                      f"(обработка — по энергогейту, см. /study)")
            else:
                n = index_docs(self.memory, arg or str(Path.cwd()))
                print(f"проиндексировано фрагментов: {n}")
        elif cmd == "/study":
            if self.study is None:
                print("подсистема знаний отключена (knowledge.enabled=false)")
            elif arg.startswith("cards"):
                for c in self.cards.list_cards(limit=30):
                    print(f"#{c.id} v{c.version} [{c.type}] {c.summary[:70]} "
                          f"value={c.value:.2f} use={c.usages}/{c.helpful}/{c.not_helpful}")
            elif arg.startswith("conflicts"):
                for card, other in self.cards.pending_conflicts():
                    print(f"конфликт: #{card.id} «{card.summary[:50]}» vs #{other}")
            elif arg.startswith("resolve") and len(arg.split()) == 3:
                _, keep, drop = arg.split()
                self.cards.resolve_conflict(int(keep), int(drop))
                print(f"конфликт решён: оставлен #{keep}")
            else:
                reps = self.study.run_pending(budget_sec=15)
                print(json.dumps([r for r in reps] or {"queued": self.study.queue.stats()},
                                 ensure_ascii=False)[:800])
        elif cmd == "/tools":
            print(", ".join(self.tools.names()))
        elif cmd == "/state":
            print(f"adapter={self.model.adapter_version} ctx={self.ctx.total()} tok "
                  f"pending={len(self.initiative.pending)} "
                  f"thresholds={self.initiative.habits.threshold}")
        elif cmd == "/undo":
            print(await self.tools.call("undo_last", {}))
        elif cmd == "/quit":
            await self.stop()
        else:
            print("/feedback /missed /sleep /index <dir> /tools /state /undo /quit")

    # ---------------- night scheduler (consolidation) ----------------
    async def night_scheduler(self) -> None:
        while self.running:
            await asyncio.sleep(300)               # wake every 5 min: trivial cost
            hhmm = time.strftime("%H:%M")
            idle = self.bridge.idle_seconds()
            should = (hhmm >= self.cfg.sleep.nightly_start and hhmm <= "05:00") \
                or idle > self.cfg.sleep.train_at_idle_min * 60
            if should and self.cfg.sleep.enabled:
                report = self.trainer.sleep_cycle()
                if "saved" in report:
                    self.model.load_adapter(str(self.trainer.active_path))
                    log.info("night adapter activated: %s", report["saved"])
            # background study shares the same quiet windows (energy-gated)
            if self.study is not None and (idle > 60 or hhmm >= self.cfg.sleep.nightly_start):
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
