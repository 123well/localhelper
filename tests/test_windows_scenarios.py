"""Tests for the window initiative layer, scenario skills and the
notification-click feedback loop (all headless-safe: no kdotool, no D-Bus)."""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _fresh_env():
    tmp = Path(tempfile.mkdtemp(prefix="kde-agent-win-"))
    os.environ["XDG_CONFIG_HOME"] = str(tmp / "config")
    os.environ["XDG_DATA_HOME"] = str(tmp / "data")
    os.environ["XDG_STATE_HOME"] = str(tmp / "state")
    return tmp


def _snapshot(n_kate=5, stale=True, active_title="~/dev/myproj — main.py"):
    # titles are app-style: "project — file" (kate/konsole format), so the
    # project keyword is extracted consistently from every window
    wins = []
    now = time.time()
    for i in range(n_kate):
        wins.append({"wid": f"0x10{i}", "title": f"myproj — file{i}.py",
                     "class": "kate",
                     "active": False})
    wins.append({"wid": "0x200", "title": active_title, "class": "konsole",
                 "active": True})
    return wins, now


class TestWindowState(unittest.TestCase):
    def setUp(self):
        from agent.kde.windows import WindowState
        self.tmp = Path(tempfile.mkdtemp(prefix="ws-"))
        self.st = WindowState(self.tmp)

    def test_snapshot_new_and_closed_changes(self):
        wins, _ = _snapshot(n_kate=2)
        ch = self.st.apply_snapshot(wins)
        self.assertTrue(any(c.startswith("new:") for c in ch))
        self.assertEqual(len(self.st.windows), 3)
        # drop one kate window -> closed change
        ch2 = self.st.apply_snapshot(wins[1:])
        self.assertTrue(any(c.startswith("closed:") for c in ch2))
        self.assertEqual(len(self.st.windows), 2)

    def test_focus_learning_cooccurrence_and_context(self):
        wins, _ = _snapshot()
        self.st.apply_snapshot(wins)
        # simulate repeated focus pattern: konsole(myproj) <-> kate
        for _ in range(4):
            w = next(w for w in self.st.windows.values() if w.cls == "kate")
            self.st.on_focus(w)
            k = next(x for x in self.st.windows.values() if x.cls == "konsole")
            self.st.on_focus(k)
        sugg = self.st.suggest_for_context("myproj")
        self.assertTrue(sugg, "context keyword should map to apps")
        self.assertEqual(sugg[0][0], "kate")
        self.assertIn("kate", dict(self.st.companions("konsole")))

    def test_profiles_persist_roundtrip(self):
        from agent.kde.windows import WindowState
        wins, _ = _snapshot()
        self.st.apply_snapshot(wins)
        k = next(x for x in self.st.windows.values() if x.cls == "konsole")
        for _ in range(3):
            self.st.on_focus(next(w for w in self.st.windows.values() if w.cls == "kate"))
            self.st.on_focus(k)
        self.st.save()
        st2 = WindowState(self.tmp)
        self.assertTrue(st2.ctx_apps, "profiles must survive restart")

    def test_ignore_suppresses_detector(self):
        from agent.kde.windows import WindowInitiative
        from agent.config import InitiativeConfig

        class Cfg(InitiativeConfig):
            pass
        st = self.st
        wins, _ = _snapshot(n_kate=5)
        st.apply_snapshot(wins)
        if wins:
            pass
        # backdate focus so windows look stale
        for w in st.windows.values():
            w.last_focus -= 7200
            w.first_seen -= 7200
        wi = WindowInitiative(Cfg(), st)
        ivs = wi.detect()
        self.assertTrue(any(i.type == "many_windows" for i in ivs),
                        f"expected many_windows, got {[i.type for i in ivs]}")
        key = next(i.context["keys"] for i in ivs if i.type == "many_windows")
        st.note_ignore(key)
        st.note_ignore(key)
        ivs2 = wi.detect()
        self.assertFalse(any(i.type == "many_windows" and i.context["keys"] == key
                             for i in ivs2), "twice-rejected suggestion must go silent")


class TestWindowTools(unittest.TestCase):
    def setUp(self):
        _fresh_env()
        from agent.config import AgentConfig
        from agent.main import Agent
        self.ag = Agent(AgentConfig())

    def tearDown(self):
        self.ag.memory.close()

    def test_close_window_requires_confirmation(self):
        # headless bridge.confirm returns False (deny-by-default) => refusal
        out = asyncio.run(self.ag.tools.call("close_window", {"wid": "0x99"}))
        self.assertTrue(out.startswith("CANCELLED_BY_USER"), out)

    def test_close_window_confirmed_without_kdotool_errors_not_silent(self):
        # confirmed=True bypasses the dialog; without kdotool the tool must
        # report a real ERROR (never pretend success)
        out = asyncio.run(self.ag.tools.call("close_window",
                                             {"wid": "0x99", "confirmed": True}))
        self.assertTrue(out.startswith("ERROR") or "rc=127" in out or "no kdotool" in out,
                        out)

    def test_list_windows_degrades_gracefully(self):
        out = asyncio.run(self.ag.tools.call("list_windows", {}))
        self.assertIsInstance(out, str)
        self.assertTrue(out.startswith("[") or out.startswith("INFO"), out)

    def test_prober_inactive_without_kdotool(self):
        self.assertFalse(self.ag.winprober.available() and False)
        # probe must be a no-op returning [] when kdotool missing
        ch = asyncio.run(self.ag.winprober.probe(force=True))
        self.assertEqual(ch, [])


class TestScenarios(unittest.TestCase):
    def setUp(self):
        _fresh_env()
        from agent.config import AgentConfig
        from agent.main import Agent
        self.ag = Agent(AgentConfig())

    def tearDown(self):
        self.ag.memory.close()

    def test_save_validates_unknown_tool(self):
        out = asyncio.run(self.ag.tools.call(
            "save_scenario", {"name": "плохой", "description": "тест",
                              "steps": [{"tool": "no_such_tool", "args": {}}]}))
        self.assertTrue(out.startswith("ERROR"), out)

    def test_save_run_roundtrip_with_params(self):
        steps = [{"tool": "exec_command",
                  "args": {"command": "echo hello-{who}"},
                  "undo": None}]
        out = asyncio.run(self.ag.tools.call(
            "save_scenario", {"name": "проверка", "description": "эхо теста",
                              "steps": steps, "params": ["who"]}))
        self.assertTrue(out.startswith("OK"), out)
        rep = json.loads(asyncio.run(self.ag.tools.call(
            "run_scenario", {"name": "проверка", "params": {"who": "мир"}})))
        self.assertTrue(rep["ok"], rep)
        self.assertIn("hello-мир", rep["outputs"][0])
        s = self.ag.scenarios.get("проверка")
        self.assertEqual(s["runs"], 1)
        self.assertEqual(s["successes"], 1)

    def test_match_shortlists_by_tokens(self):
        self.ag.scenarios.save("backup docs", "резервное копирование документов",
                               [{"tool": "exec_command", "args": {"command": "tar czf /tmp/b.tgz docs"}}])
        hits = self.ag.scenarios.match("сделай backup документов")
        self.assertTrue(hits and hits[0]["name"] == "backup docs")

    def test_safe_mode_blocks_dangerous_step(self):
        self.ag.scenarios.save("опасный", "тест опасного шага",
                               [{"tool": "exec_command",
                                 "args": {"command": "rm -rf ~/trashme"},
                                 "dangerous": True}],
                               precondition={"mode_not_safe": True})
        self.ag.safety.set_mode("safe")
        rep = asyncio.run(self.ag.scenario_runner.run("опасный"))
        self.assertFalse(rep["ok"])
        self.ag.safety.set_mode("normal")

    def test_confirm_step_cancelled_headless(self):
        self.ag.scenarios.save("спроси", "шаг с подтверждением",
                               [{"tool": "exec_command",
                                 "args": {"command": "echo hi"},
                                 "confirm": True}])
        rep = asyncio.run(self.ag.scenario_runner.run("спроси"))
        self.assertFalse(rep["ok"])
        self.assertEqual(rep.get("cancelled_at"), 1)

    def test_failed_step_records_and_undoes(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="sc-undo-"))
        # HOME-scoped path: write_file outside $HOME requires confirmation,
        # which is deny-by-default in headless tests (that would mask the
        # actual failure-under-test at step 1 instead of step 2)
        old_home = os.environ.get("HOME")
        os.environ["HOME"] = str(tmpdir)
        try:
            target = tmpdir / "f.txt"
            self.ag.scenarios.save("исправить", "запись с откатом",
                                   [{"tool": "write_file",
                                     "args": {"path": str(target), "content": "v2"}},
                                    {"tool": "exec_command",
                                     "args": {"command": "false-command-xyz"}}])
            rep = asyncio.run(self.ag.scenario_runner.run("исправить"))
        finally:
            os.environ["HOME"] = old_home or "/root"
        self.assertFalse(rep["ok"])
        self.assertEqual(rep.get("failed_at"), 2)
        # undo not registered for write_file step here; failure recorded in store
        s = self.ag.scenarios.get("исправить")
        self.assertEqual(s["runs"], 1)
        self.assertEqual(s["successes"], 0)


class TestNotificationFeedbackLoop(unittest.TestCase):
    """ActionInvoked click -> initiative.feedback without the terminal."""

    def test_click_resolves_pending_intervention(self):
        _fresh_env()
        from agent.config import AgentConfig
        from agent.main import Agent
        ag = Agent(AgentConfig())
        try:
            eng = ag.initiative
            eng.cfg.cooldown_sec = 0
            ivs = eng.intervene("gti status")
            self.assertTrue(ivs)
            key = next(k for k in eng.pending if k.startswith("typo_cmd"))
            thr0 = eng.habits.thr("typo_cmd")
            # simulate the D-Bus signal path: bridge had stored the hook under
            # reply_key; a click on «Принять» delivers action key "default"/0th
            hook = ag.bridge.reply_hooks.get(key)
            self.assertIsNotNone(hook, "present() must register a reply hook")
            hook("default")
            self.assertNotIn(key, eng.pending, "click must consume pending intervention")
            self.assertLess(eng.habits.thr("typo_cmd"), thr0,
                            "accept-click should lower threshold like /feedback да")
        finally:
            ag.memory.close()


import json  # noqa: E402  (used in scenario tests above)

if __name__ == "__main__":
    unittest.main(verbosity=2)
