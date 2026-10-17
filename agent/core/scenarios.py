"""Scenarios / saved skills (spec: «сценарии и высокоуровневые команды»).

A scenario is a decomposed high-level task stored as a reusable skill:
  name, description, steps (tool calls), params, precondition, per-step
  confirm flags, expected result, undo plan.

Execution model (safety-first):
  * every step goes through ToolRegistry.call -> SafetyGate already applies
    mode/blacklist/confirm policy; we add scenario-level gates on top:
  * a step marked `confirm` ALWAYS asks the user (even in normal mode);
  * dangerous steps are refused outright while safety.mode == safe/read_only;
  * successful runs reinforce the scenario (value++ -> better ranking when
    matching user requests to skills); failures record an episode so the
    night training can see it;
  * undo: each step may carry an inverse tool call; on abort/failure we run
    the recorded inverses in reverse order (best-effort, logged).

Storage: sqlite table inside the same Memory DB (single file, no new deps).
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid

log = logging.getLogger("kde-agent.scenario")


class ScenarioStore:
    def __init__(self, memory):
        self.mem = memory
        self.mem._exec("""CREATE TABLE IF NOT EXISTS scenarios(
            id TEXT PRIMARY KEY, name TEXT, description TEXT, spec TEXT,
            runs INTEGER DEFAULT 0, successes INTEGER DEFAULT 0,
            value REAL DEFAULT 1.0, ts REAL)""")
        self.mem._commit()

    # ---------- CRUD -------------------------------------------------------
    def save(self, name: str, description: str, steps: list[dict],
             params: list[str] | None = None, precondition: dict | None = None,
             expected: str = "", sid: str | None = None,
             known_tools=None) -> str:
        """steps: [{tool,args,confirm?,dangerous?,undo?:{tool,args}}]
        Validate BEFORE storing — a broken scenario must never be saved."""
        err = _validate_steps(steps, known_tools=known_tools)
        if err:
            raise ValueError(err)
        sid = sid or uuid.uuid4().hex[:10]
        spec = {"steps": steps, "params": params or [],
                "precondition": precondition or {}, "expected": expected}
        row = self.mem._exec("SELECT id FROM scenarios WHERE name=?", (name,)).fetchone()
        if row and row[0] != sid:
            sid = row[0]   # update-in-place: names are unique keys
        self.mem._exec(
            "INSERT OR REPLACE INTO scenarios(id,name,description,spec,runs,"
            "successes,value,ts) VALUES(?,?,?,?,?,?,?,?)",
            (sid, name, description, json.dumps(spec, ensure_ascii=False),
             *(lambda old: (old[0], old[1], old[2]) if old else (0, 0, 1.0))(
                 self.mem._exec("SELECT runs,successes,value FROM scenarios WHERE id=?",
                                (sid,)).fetchone()),
             time.time()))
        self.mem._commit()
        return sid

    def list(self) -> list[dict]:
        rows = self.mem._exec(
            "SELECT id,name,description,spec,runs,successes,value FROM scenarios"
            " ORDER BY value DESC, ts DESC").fetchall()
        return [{"id": r[0], "name": r[1], "description": r[2],
                 "steps": len(json.loads(r[3])["steps"]),
                 "runs": r[4], "successes": r[5], "value": r[6]} for r in rows]

    def get(self, name_or_id: str) -> dict | None:
        r = self.mem._exec(
            "SELECT id,name,description,spec,runs,successes FROM scenarios"
            " WHERE id=? OR name=?", (name_or_id, name_or_id)).fetchone()
        if not r:
            return None
        return {"id": r[0], "name": r[1], "description": r[2],
                "spec": json.loads(r[3]), "runs": r[4], "successes": r[5]}

    def remove(self, name_or_id: str) -> bool:
        cur = self.mem._exec("DELETE FROM scenarios WHERE id=? OR name=?",
                             (name_or_id, name_or_id))
        self.mem._commit()
        return cur.rowcount > 0

    def match(self, text: str, limit: int = 3) -> list[dict]:
        """Cheap trigger matcher (name/description tokens); the SSM decides
        finally — this only shortlists candidates for its context."""
        toks = set(re.findall(r"[\w-]{3,}", text.lower()))
        scored = []
        for s in self.list():
            hay = set(re.findall(r"[\w-]{3,}",
                                 (s["name"] + " " + s["description"]).lower()))
            ov = len(toks & hay)
            if ov:
                scored.append((ov * s["value"], s))
        scored.sort(key=lambda kv: -kv[0])
        return [s for _sc, s in scored[:limit]]

    def record_run(self, sid: str, ok: bool) -> None:
        r = self.mem._exec("SELECT runs,successes,value FROM scenarios WHERE id=?",
                           (sid,)).fetchone()
        if not r:
            return
        runs, succ, val = r
        val = round(val + (0.2 if ok else -0.3), 3)
        self.mem._exec("UPDATE scenarios SET runs=?, successes=?, value=? WHERE id=?",
                       (runs + 1, succ + (1 if ok else 0), max(0.1, val), sid))
        self.mem._commit()


def _validate_steps(steps: list[dict], known_tools=None) -> str | None:
    """Structural validation; when `known_tools` (a registry instance) is
    given, every step's tool name must actually exist there — a scenario
    referencing a typo'd tool must fail at SAVE time, not at run time."""
    if not isinstance(steps, list) or not steps:
        return "шаги пустой/не список"
    for i, st in enumerate(steps):
        if not isinstance(st, dict) or "tool" not in st:
            return f"шаг {i}: нет поля tool"
        if not isinstance(st.get("args", {}), dict):
            return f"шаг {i}: args должен быть объектом"
        if known_tools is not None and st["tool"] not in known_tools.names():
            return f"шаг {i}: неизвестный инструмент «{st['tool']}»"
    return None


class ScenarioRunner:
    """Executes one scenario with parameter substitution + gates + undo."""

    def __init__(self, store: ScenarioStore, tools, bridge, safety, memory):
        self.store, self.tools = store, tools
        self.bridge, self.safety, self.memory = bridge, safety, memory

    async def run(self, name_or_id: str, params: dict | None = None) -> dict:
        sc = self.store.get(name_or_id)
        if not sc:
            return {"ok": False, "error": f"сценарий не найден: {name_or_id}"}
        spec = sc["spec"]
        p = {**(spec.get("params") and {k: "" for k in spec["params"]}),
             **(params or {})}
        # precondition: simple key==value facts about current state
        pre = spec.get("precondition") or {}
        if pre.get("mode_not_safe") and self.safety.mode in ("safe", "read_only"):
            return {"ok": False, "error": "безопасный режим: исполнение запрещено"}
        done, undo_stack, outputs = 0, [], []
        for idx, st in enumerate(spec["steps"]):
            tool = st["tool"]
            args = _subst(st.get("args", {}), p)
            if self.safety.mode in ("safe", "read_only") and st.get("dangerous"):
                return {"ok": False, "error": f"шаг {idx+1} опасен, режим {self.safety.mode}",
                        "done": done, "outputs": outputs}
            if st.get("confirm"):
                q = f"Сценарий «{sc['name']}», шаг {idx+1}/{len(spec['steps'])}: " \
                    f"{tool} {json.dumps(args, ensure_ascii=False)[:160]}?"
                if not self.bridge.confirm(q, dangerous=bool(st.get("dangerous"))):
                    await self._undo(undo_stack)
                    self.store.record_run(sc["id"], False)
                    return {"ok": False, "cancelled_at": idx + 1,
                            "done": done, "outputs": outputs}
            out = await self.tools.call(tool, args)
            outputs.append(f"[{tool}] {str(out)[:300]}")
            done += 1
            if st.get("undo"):
                undo_stack.append(st["undo"])
            if str(out).startswith(("ERROR", "BLOCKED", "CANCELLED")):
                await self._undo(undo_stack)
                self.memory.episode("scenario_fail",
                                    f"{sc['name']} step {idx+1}: {str(out)[:200]}",
                                    value=1.5)
                self.store.record_run(sc["id"], False)
                return {"ok": False, "failed_at": idx + 1, "error": str(out)[:300],
                        "done": done, "outputs": outputs}
        self.store.record_run(sc["id"], True)
        self.memory.episode("scenario_ok", f"{sc['name']} выполнен ({done} шагов)")
        return {"ok": True, "done": done, "expected": spec.get("expected", ""),
                "outputs": outputs}

    async def _undo(self, undo_stack: list[dict]) -> None:
        for st in reversed(undo_stack):
            try:
                await self.tools.call(st["tool"], st.get("args", {}))
            except Exception as e:  # noqa: BLE001
                log.warning("scenario undo step failed: %s", e)


def _subst(obj, params: dict):
    """Recursively replace {param} placeholders in strings inside args dicts."""
    if isinstance(obj, str):
        def rep(m):
            k = m.group(1)
            return str(params.get(k, m.group(0)))
        return re.sub(r"\{(\w+)\}", rep, obj)
    if isinstance(obj, dict):
        return {k: _subst(v, params) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_subst(v, params) for v in obj]
    return obj
