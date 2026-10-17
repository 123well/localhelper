"""Unit tests for the kde-agent prototype (stdlib unittest, no pytest needed).

Run:  python -m unittest discover -s tests -v
Covers: typo analyzer (evidence pipeline + no self-legitimization + intentional
veto), safety gate, initiative learning loop, RAG memory, sleep trainer math.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _fresh_env():
    tmp = Path(tempfile.mkdtemp(prefix="kde-agent-test-"))
    os.environ["XDG_CONFIG_HOME"] = str(tmp / "config")
    os.environ["XDG_DATA_HOME"] = str(tmp / "data")
    os.environ["XDG_STATE_HOME"] = str(tmp / "state")
    return tmp


class TestTypoAnalyzer(unittest.TestCase):
    def setUp(self):
        from agent.core.typo_model import Lexicon, ProposalEngine, MockModelAnalyzer
        from agent.core.initiative import KNOWN_CMDS
        self.lex = Lexicon(None)
        self.an = MockModelAnalyzer(ProposalEngine(set(KNOWN_CMDS)))

    def test_transposition_detected(self):
        v = self.an.analyze_line("gti status", self.lex)
        self.assertTrue(any(x.kind == "command" and x.intended.startswith("git") for x in v))

    def test_no_self_legitimization(self):
        # repeated observation of a typo must NEVER make it 'known'
        for _ in range(6):
            self.lex.observe("gti status")
            self.assertFalse(self.an.analyze_line("gti status", self.lex) == [])

    def test_real_command_not_flagged(self):
        self.assertEqual(self.an.analyze_line("git status", self.lex), [])
        self.assertEqual(self.an.analyze_line("ls -la ~/projects", self.lex), [])

    def test_path_typo_with_existing_sibling(self):
        home = Path.home()
        real = next((p.name for p in home.iterdir() if p.is_dir() and len(p.name) > 4), None)
        if not real:
            self.skipTest("no suitable home dir")
        bad = real[:-2]  # delete two trailing chars -> distance 2, sibling exists
        v = self.an.analyze_line(f"cd ~/{bad}", self.lex)
        self.assertTrue(any(x.kind == "path_segment" for x in v), f"expected flag for ~/{bad}")

    def test_new_file_not_a_typo(self):
        # brand-new filename inside an existing dir is normal usage
        v = self.an.analyze_line("touch ~/brand_new_file_xyz.txt", self.lex)
        self.assertEqual([x for x in v if x.kind == "path_segment"], [])


class TestSafety(unittest.TestCase):
    def setUp(self):
        from agent.config import SafetyConfig
        from agent.tools.safety import SafetyGate
        self.g = SafetyGate(SafetyConfig())

    def test_levels(self):
        cases = [("rm -rf /", "blocked"), ("mkfs.ext4 /dev/sda1", "blocked"),
                 ("sudo reboot now", "needs_confirm"), ("rm file.txt", "needs_confirm"),
                 ("ls ~", "safe"), ("cat /etc/hostname", "safe")]
        for cmd, exp in cases:
            with self.subTest(cmd=cmd):
                self.assertEqual(self.g.assess_command(cmd).level, exp)

    def test_whitelist_never_blocked(self):
        for w in ["ls", "echo hi", "pwd"]:
            self.assertNotEqual(self.g.assess_command(w).level, "blocked")


class TestInitiativeLearning(unittest.TestCase):
    def setUp(self):
        _fresh_env()
        from agent.config import AgentConfig
        from agent.main import Agent
        self.ag = Agent(AgentConfig())
        self.eng = self.ag.initiative
        self.eng.cfg.cooldown_sec = 0

    def tearDown(self):
        self.ag.memory.close()

    def test_accept_lowers_threshold(self):
        ivs = self.eng.intervene("gti status")   # detect + present -> pending
        self.assertTrue(ivs)
        thr0 = self.eng.habits.thr("typo_cmd")
        key = next(k for k in self.eng.pending if k.startswith("typo_cmd"))
        r = self.eng.feedback(key, "да, спасибо!")
        self.assertGreater(r, 0)
        self.assertLess(self.eng.habits.thr("typo_cmd"), thr0)

    def test_intentional_veto_persists(self):
        self.eng.intervene("cd /hme/projects")
        keys = [k for k in self.eng.pending if k.startswith("typo_path")]
        self.assertTrue(keys, "/hme should have been flagged first")
        self.eng.feedback(keys[0], "это намеренно")
        self.eng.habits.last_seen.clear()
        again = self.eng.on_user_line("cd /hme/projects && ls")
        self.assertFalse([i for i in again if i.type == "typo_path"])

    def test_annoyed_raises_threshold(self):
        self.eng.intervene("gti status")
        thr0 = self.eng.habits.thr("typo_cmd")
        key = next(k for k in self.eng.pending if k.startswith("typo_cmd"))
        r = self.eng.feedback(key, "не мешай мне")
        self.assertLess(r, 0)
        self.assertGreater(self.eng.habits.thr("typo_cmd"), thr0)

    def test_hourly_budget_caps_nagging(self):
        self.eng.cfg.max_interventions_per_hour = 2
        presented = 0
        for i in range(5):
            self.eng.habits.last_seen.clear()
            ivs = self.eng.on_user_line(f"gti push origin f{i}")
            self.eng.present(ivs)
            presented += len(ivs)
        self.assertLessEqual(presented, 2)


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = _fresh_env()
        from agent.memory.rag import Memory
        self.m = Memory(self.tmp / "mem.db", dim=128)

    def tearDown(self):
        self.m.close()

    def test_add_search_roundtrip(self):
        i = self.m.add("docs", "journalctl -u pacman reports package transactions", "doc")
        hits = self.m.search("pacman logs journalctl", ("docs",), top_k=3)
        self.assertIn(i, [h.id for h in hits])

    def test_collection_filter(self):
        self.m.add("docs", "alpha text about widgets", "doc")
        self.m.episode("answer", "beta answer about widgets")
        hits = self.m.search("widgets", ("episodes",), top_k=5)
        self.assertTrue(all(h.collection == "episodes" for h in hits))

    def test_preference_value_kwarg(self):
        self.m.preference("intentional-token:typo_path:foo", {"verdict": "intentional"}, value=3.0)
        hits = self.m.search("intentional-token:typo_path:foo", ("prefs",), top_k=1)
        self.assertTrue(hits and hits[0].meta["verdict"] == "intentional")


class TestSleepTrainer(unittest.TestCase):
    def setUp(self):
        self.tmp = _fresh_env()
        from agent.config import AgentConfig
        from agent.main import Agent
        self.ag = Agent(AgentConfig())
        self.tr = self.ag.trainer

    def tearDown(self):
        self.ag.memory.close()

    def test_train_step_shapes_and_moves(self):
        batch = [{"id": 1, "text": "good help accepted", "kind": "initiative:x:accepted",
                  "meta": {"reward": 0.8}, "value": 1.5},
                 {"id": 2, "text": "annoying suggestion", "kind": "initiative:x:rejected",
                  "meta": {"reward": -0.5}, "value": 1.5}]
        a0, b0 = self.tr.A.copy(), self.tr.B.copy()
        loss = self.tr.train_step(batch, lr=1e-3)
        self.assertGreater(loss, 0)
        self.assertEqual(self.tr.A.shape, a0.shape)
        self.assertEqual(self.tr.B.shape, b0.shape)
        self.assertFalse(np_allclose(a0, self.tr.A) and np_allclose(b0, self.tr.B))

    def test_sleep_cycle_saves_adapter(self):
        m = self.ag.memory
        m.episode("initiative:typo_cmd:accepted", "helpful fix", {"reward": 0.6}, value=1.6)
        m.episode("initiative:typo_cmd:rejected", "not helpful", {"reward": -0.3}, value=1.4)
        m.episode("error", "cmd -> ERROR fail", {}, value=1.5)
        rep = self.tr.sleep_cycle()
        self.assertIn("saved", rep, f"gates failed: {rep}")
        self.assertTrue((self.tr.adapters_dir / rep["saved"]).exists())

    def test_gates_block_bad_adapter(self):
        # sabotage: huge delta must trip the nagging-drift gate & roll back
        import numpy as np
        self.tr.A += np.full_like(self.tr.A, 5.0)
        self.tr.B += np.full_like(self.tr.B, 5.0)
        ok, why = self.tr.run_gates()
        self.assertFalse(ok)
        self.assertIn("nagging", why)


def np_allclose(a, b):
    import numpy as np
    return bool(np.allclose(a, b))


if __name__ == "__main__":
    unittest.main(verbosity=2)
