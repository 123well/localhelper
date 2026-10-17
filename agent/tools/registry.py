"""Toolbelt exposed to the model. Every call: risk gate -> optional confirm
-> execute -> log to memory -> result string back into the context window."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import subprocess
import time
from pathlib import Path

log = logging.getLogger("kde-agent.tools")

CMD_TIMEOUT = 30

# Tools that change system state (execution/writes/GUI input). Everything else
# is observational (read/search/list) and stays allowed in safe/read_only modes.
MUTATING_TOOLS = {"exec_command", "write_file", "click", "type_text", "open_app",
                  "dbus_call", "undo_last", "propose_automation"}


class ToolRegistry:
    def __init__(self, cfg, memory, safety, bridge, cards=None):
        self.cfg, self.memory, self.safety, self.bridge = cfg, memory, safety, bridge
        self.cards = cards          # CardStore | None (knowledge subsystem)
        self.last_undo: tuple | None = None
        # Code assistant registers text-edit undo entries here (kind "edit").
        self._undo_edits: list[tuple] = []
        self._term_history: list[tuple[str, str]] = []
        self._tools = {
            "exec_command": self.exec_command,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_files": self.list_files,
            "screen_state": self.screen_state,
            "terminal_output": self.terminal_output,
            "active_window": self.active_window,
            "click": self.click,
            "type_text": self.type_text,
            "open_app": self.open_app,
            "ask_user": self.ask_user,
            "notify": self.notify,
            "confirm_dialog": self.confirm_dialog,
            "rag_search": self.rag_search,
            "memory_search": self.memory_search,
            "card_search": self.card_search,
            "set_reminder": self.set_reminder,
            "dbus_call": self.dbus_call,
            "update_widget": self.update_widget,
            "propose_automation": self.propose_automation,
            "undo_last": self.undo_last,
        }

    def names(self) -> list[str]:
        return sorted(self._tools)

    async def call(self, tool: str, args: dict) -> str:
        fn = self._tools.get(tool)
        if not fn:
            return f"ERROR: unknown tool '{tool}'. Available: {', '.join(self.names())}"
        # Operating-mode policy first (safe / read_only block all mutations;
        # confirm_all forces explicit user approval for every action).
        mode_decision = self.safety.assess_tool_call(tool, tool in MUTATING_TOOLS)
        if mode_decision.level == "blocked":
            out = f"BLOCKED: {mode_decision.reason} (режим «{self.safety.mode}»)"
            self.memory.episode("tool", f"{tool} blocked by mode")
            return out
        if mode_decision.level == "needs_confirm" and tool in MUTATING_TOOLS \
                and not args.get("confirmed"):
            if not self.bridge.confirm(f"Режим подтверждения всех действий: выполнить `{tool}` "
                                        f"с аргументами {json.dumps(args, ensure_ascii=False)[:200]}?"):
                out = "CANCELLED_BY_USER: пользователь отклонил операцию (режим confirm_all)."
                self.memory.episode("cancel", f"confirm_all refused {tool}")
                return out
            args = {**args, "confirmed": True}
        try:
            res = fn(**args)
            if asyncio.iscoroutine(res):
                res = await res
            out = res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
        except TypeError as e:
            out = f"ERROR: bad args for {tool}: {e}"
        except Exception as e:  # noqa: BLE001
            log.exception("tool %s failed", tool)
            out = f"ERROR: {tool} failed: {e}"
        self.memory.episode("tool", f"{tool}({json.dumps(args, ensure_ascii=False)[:200]}) -> {out[:300]}")
        return out

    # ---------------- terminal & files ------------------------------------
    async def exec_command(self, command: str, confirmed: bool = False) -> str:
        decision = self.safety.assess_command(command)
        if decision.level == "blocked":
            return f"BLOCKED: {decision.reason}. Спросите пользователя или найдите другой путь (rag_search)."
        if decision.level == "needs_confirm" and not confirmed:
            q = f"Разрешить выполнить: `{command}`? {decision.reason}"
            if decision.safer:
                q += f" Безопаснее: `{decision.safer}`."
            ok = self.bridge.confirm(q)
            if not ok:
                self.memory.episode("cancel", f"user refused: {command}")
                return "CANCELLED_BY_USER: пользователь отклонил операцию."
            confirmed = True
        snap = self.safety.snapshot_for(command)
        backup = self.safety.perform_backup(snap) if snap else None
        if snap and backup:
            self.last_undo = (snap, backup)
        shell = any(c in command for c in "|><&;$*")
        try:
            r = subprocess.run(command if shell else shlex.split(command),
                               shell=shell, capture_output=True, text=True,
                               timeout=CMD_TIMEOUT)
        except subprocess.TimeoutExpired:
            return f"TIMEOUT after {CMD_TIMEOUT}s: {command}"
        self._term_history.append((command, (r.stdout + r.stderr)[-4000:]))
        out = ((r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")).strip()
        status = f"[exit {r.returncode}]"
        if r.returncode != 0:
            self.memory.episode("error", f"{command} -> rc={r.returncode}: {out[:300]}", value=1.5)
        return f"{status}\n{out[:4000] or '(no output)'}"

    def terminal_output(self, last_n: int = 1) -> str:
        tail = self._term_history[-last_n:]
        return "\n\n".join(f"$ {c}\n{o}" for c, o in tail) or "(пусто)"

    def read_file(self, path: str, max_bytes: int = 8000) -> str:
        p = Path(os.path.expanduser(path))
        d = self.safety.assess_path(str(p))
        if d.level == "blocked":
            return f"BLOCKED: {d.reason}"
        try:
            data = p.read_text(errors="replace")
        except OSError as e:
            return f"ERROR: {e}"
        if len(data) > max_bytes:
            return data[:max_bytes] + f"\n…[truncated, total {len(data)} chars]"
        return data

    def write_file(self, path: str, content: str) -> str:
        p = Path(os.path.expanduser(path))
        d = self.safety.assess_path(str(p), write=True)
        if d.level == "blocked":
            return f"BLOCKED: {d.reason}"
        if d.level == "needs_confirm" and not self.bridge.confirm(
                f"Записать файл вне домашней папки: {p}?", dangerous=True):
            return "CANCELLED_BY_USER"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"OK: записано {len(content)} байт в {p}"

    def list_files(self, path: str = ".") -> str:
        p = Path(os.path.expanduser(path))
        if not p.is_dir():
            return f"ERROR: {p} — не директория"
        items = sorted(p.iterdir(), key=lambda x: (x.is_dir(), x.name.lower()))
        lines = [f"{'d' if x.is_dir() else '-'} {x.name}" for x in items[:200]]
        return "\n".join(lines) or "(пусто)"

    # ---------------- GUI ---------------------------------------------------
    def screen_state(self) -> str:
        return json.dumps(self.bridge.screen_state(), ensure_ascii=False)

    def active_window(self) -> str:
        return json.dumps(self.bridge.active_window(), ensure_ascii=False)

    def click(self, x: int, y: int) -> str:
        self.bridge.click(int(x), int(y))
        return f"OK: клик ({x},{y})"

    def type_text(self, text: str) -> str:
        self.bridge.type_text(text)
        return "OK: текст введён"

    def open_app(self, app: str) -> str:
        self.bridge.open_app(app)
        return f"OK: запущено {app}"

    # ---------------- human interaction -------------------------------------
    def ask_user(self, question: str) -> str:
        return self.bridge.ask(question) or "(пустой ответ)"

    def notify(self, title: str, body: str, urgency: str = "normal") -> str:
        self.bridge.notify(title, body, urgency=urgency)
        return "OK"

    def confirm_dialog(self, question: str) -> str:
        return "yes" if self.bridge.confirm(question) else "no"

    # ---------------- memory / RAG -------------------------------------------
    def rag_search(self, query: str, top_k: int | None = None) -> str:
        hits = self.memory.search(query, ("docs",), top_k or self.cfg.rag.top_k)
        if not hits:
            return "NO_RESULTS: в базе знаний пусто — уточните запрос или попросите документацию у пользователя."
        return "\n---\n".join(f"[{h.score:.2f}] {h.text[:600]}" for h in hits)

    def memory_search(self, query: str, top_k: int | None = None) -> str:
        hits = self.memory.search(query, ("episodes", "prefs"),
                                  top_k or self.cfg.rag.top_k)
        return "\n".join(f"({h.collection}/{h.meta.get('kind','')}) [{h.score:.2f}] {h.text[:200]}"
                         for h in hits) or "NO_RESULTS"

    def card_search(self, query: str, types: list[str] | None = None,
                    top_k: int = 4) -> str:
        """Knowledge-card retrieval with source attribution (spec: режим
        объяснения источника — each hit carries its provenance)."""
        cards = self.cards
        if cards is None:
            return "DISABLED: подсистема знаний выключена (knowledge.enabled=false)."
        found = cards.search(query, top_k=top_k,
                             types=tuple(types) if types else None)
        if not found:
            return "NO_CARDS: карточек по запросу нет — предложите пользователю источник (/index <dir>) или задайте вопрос напрямую."
        out = []
        for c in found:
            src = c.meta.get("source", "?")
            lines = c.meta.get("lines", [0, 0])
            out.append(f"#{c.id} v{c.version} [{c.type}] {c.summary}\n"
                       f"{c.body[:500]}\n"
                       f"(источник: {src}:{lines[0]}-{lines[1]}, "
                       f"value={c.value:.2f}, use={c.usages})")
            cards.record_usage(c.id)   # retrieval counts as a usage signal
        return "\n---\n".join(out)

    def set_reminder(self, text: str, minutes: int = 10) -> str:
        self.memory.add("reminders", text, "reminder", {"due": time.time() + minutes * 60})
        return f"OK: напоминание через {minutes} мин: {text}"

    def due_reminders(self) -> list[tuple[int, str]]:
        """Reminders past their due time (called by the scheduler tick).
        Fixes the write-only dead collection: previously nothing ever read them."""
        rows = self.memory.db.execute(
            "SELECT id,text,meta FROM items WHERE collection='reminders'").fetchall()
        now = time.time()
        out = []
        for rid, text, meta in rows:
            try:
                due = float(json.loads(meta or "{}").get("due", 0))
            except (ValueError, TypeError):
                continue
            if due and now >= due:
                out.append((rid, text))
        return out

    def acknowledge_reminder(self, reminder_id: int) -> None:
        self.memory.db.execute("DELETE FROM items WHERE id=? AND collection='reminders'",
                               (reminder_id,))
        self.memory.db.commit()
        self.memory._cache = None

    # ---------------- KDE plumbing --------------------------------------------
    def dbus_call(self, service: str, path: str, iface: str, method: str,
                  args: list | None = None) -> str:
        from shutil import which
        q = which("qdbus6") or which("qdbus") or which("busctl")
        if not q:
            return "ERROR: нет qdbus/busctl"
        cmd = ([q, service, path, method] + [str(a) for a in (args or [])]) if q != "busctl" \
            else ["busctl", "--user", "call", service, path, iface, method] + \
                 [f"{a}" for a in (args or [])]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        return (r.stdout + r.stderr).strip()[:2000] or "OK"

    def update_widget(self, status: str, task: str = "", pending: str = "") -> str:
        """Push state to the plasmoid via D-Bus (org.kde.plasma.applet custom
        signal) — falls back to local file the widget reads on repaint."""
        payload = {"status": status, "task": task, "pending": pending, "ts": time.time()}
        wfile = self.cfg.data_dir / "widget_state.json"
        wfile.write_text(json.dumps(payload, ensure_ascii=False))
        return f"OK: виджет обновлён ({status})"

    def propose_automation(self, pattern: str, script: str = "") -> str:
        self.memory.episode("automation_proposal", f"{pattern}\n{script}", value=1.5)
        self.bridge.notify("KDE Agent", f"Могу автоматизировать: {pattern[:80]}")
        return "OK: предложение автоматизации отправлено"

    def register_edit_undo(self, entry: tuple) -> None:
        """Undo sink for the code assistant: ('edit', path, old_content|None)."""
        if isinstance(entry, tuple) and len(entry) == 3 and entry[0] == "edit":
            self._undo_edits.append(entry)

    def undo_last(self) -> str:
        # text edits from the code assistant take precedence (LIFO), then the
        # shell-op snapshot journal (rm/mv backups) from SafetyGate
        while self._undo_edits:
            _, path, old = self._undo_edits.pop()
            target = Path(path)
            try:
                if old is None:
                    if target.exists():
                        target.unlink()
                    return f"отменено: {path} удалён (был создан агентом)"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(old)
                self.memory.episode("undo", f"restored {path}", value=1.2)
                return f"отменено: содержимое {path} восстановлено"
            except OSError as e:
                return f"ОТМЕНА НЕ УДАЛАСЬ: {e}"
        if not self.last_undo:
            return "Нечего отменять"
        snap, backup = self.last_undo
        msg = self.safety.undo(snap, backup)
        self.memory.episode("undo", msg, value=1.2)
        return msg
