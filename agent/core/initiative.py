"""Initiative engine (Component 2).

Watches user input & system events, detects situations where help is possible,
decides whether to intervene (adaptive threshold + cooldowns + hourly budget),
and learns from reactions through explicit rewards.

Detectors: typos in paths/commands, dangerous commands, deviation from habit,
repetitive sequences -> automation proposal, long idle, unsaved-file reminder,
risky command with safer alternative.

Reward model (dopamine-like RPE stored per intervention type):
  accepted/thankful      -> +R (type utility up, threshold down)
  'intentional'          -> 0/-small, pattern memorized (don't repeat!)
  annoyed/dismissed      -> -R (threshold up => less intrusive)
  missed obvious error   -> -R (detector sensitivity up)
Sparse representations: each situation is encoded as a small binary feature
vector; learning happens on the few active features only.
"""
from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field

from agent.core.typo_model import Lexicon, ModelTypoAnalyzer, make_analyzer


# ---------------------------------------------------------------------------
@dataclass
class Intervention:
    type: str            # typo_path | typo_cmd | danger | repetition | idle | unsaved | safer_alt | deviation
    message: str
    confidence: float
    context: dict
    payload: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    resolved: bool = False
    reward: float = 0.0


# Seed vocabulary ONLY to bootstrap the proposal engine before the user has
# any history; the authoritative vocab is externalized facts: PATH binaries,
# live filesystem, learned dictionary, RAG docs (see typo_model.Lexicon).
KNOWN_CMDS = {
    "git", "ls", "cd", "cat", "grep", "sed", "awk", "tar", "zip", "unzip", "pacman",
    "yay", "systemctl", "journalctl", "chmod", "chown", "mv", "cp", "rm", "mkdir",
    "touch", "nano", "vim", "nvim", "code", "firefox", "konsole", "dolphin", "ps",
    "kill", "top", "htop", "df", "du", "ping", "ssh", "scp", "curl", "wget", "make",
    "cmake", "python", "python3", "pip", "npm", "node", "docker", "kubectl", "gpg",
    "echo", "pwd", "whoami", "date", "uname", "trash-put", "kdotool", "qdbus6",
}
INTENTIONAL_MARKERS = ["намеренно", "так задумано", "intentio", "deliberate", "special",
                       "свой алиас", "my alias", "не лезь", "not a typo", "всё верно", "намеренн"]
ANNOYED_MARKERS = ["не мешай", "отстань", "достал", "не надо", "не предлагай", "irritat",
                   "stop asking", "хватит", "молчи", "не назой"]


class TypoDetector:
    """Thin adapter: delegates ALL judgement to the model analyzer.

    The program does not decide 'this is a typo' by string matching anymore.
    It hands the line + evidence packet to the analyzer (mock during
    prototyping, the frozen SSM on NPU in production), which returns
    calibrated verdicts with reasons; we only translate verdicts into
    Interventions and enforce habit-based gating (threshold/cooldown/veto)."""

    def __init__(self, analyzer: ModelTypoAnalyzer | None = None, memory=None):
        self.lex = Lexicon(memory)
        self.analyzer = analyzer

    TYPE_BY_KIND = {"command": "typo_cmd", "path_segment": "typo_path"}

    def analyze(self, line: str, rag_context: str = "") -> list[Intervention]:
        if not self.analyzer:
            return []
        out = []
        for v in self.analyzer.analyze_line(line, self.lex, rag_context):
            itype = self.TYPE_BY_KIND.get(v.kind, "typo_cmd")
            out.append(Intervention(
                itype, v.reason.split(" [")[0], v.confidence,
                {"line": line},
                {"suggest": v.intended, "actual": v.bad_token,
                 "candidates": [{"variant": c.variant, "source": c.source,
                                 "evidence": c.evidence} for c in v.candidates[:4]],
                 "model_reason": v.reason}))
        return out


class DangerDetector:
    def __init__(self, safety_cfg):
        self.cfg = safety_cfg

    def classify(self, cmd: str) -> str:
        c = cmd.strip()
        if any(bl in c for bl in self.cfg.blacklist):
            return "blocked"
        destructive = ("rm " in c or "sudo " in c or c.startswith("dd ") or "mkfs" in c
                       or "| sh" in c or "| bash" in c or "chmod -R" in c
                       or "git push --force" in c or "git reset --hard" in c
                       or ">" in c and "/etc" in c)
        return "dangerous" if destructive else "safe"

    def check(self, cmd: str) -> Intervention | None:
        risk = self.classify(cmd)
        if risk == "blocked":
            return Intervention("danger", f"Команда в чёрном списке: «{cmd}». Выполнять нельзя.",
                                0.98, {"cmd": cmd}, {"risk": risk})
        if risk == "dangerous":
            safer = _safer_alternative(cmd)
            msg = f"Опасная команда «{cmd}»."
            if safer:
                msg += f" Безопаснее: «{safer}»."
            return Intervention("danger" if not safer else "safer_alt", msg,
                                0.85, {"cmd": cmd}, {"risk": risk, "safer": safer})
        return None


def _safer_alternative(cmd: str) -> str | None:
    if re.match(r"^rm\s+(-[rf]*\s+)*(/|\.\.)", cmd.strip()):
        return "trash-put " + cmd.strip()[3:]  # recoverable deletion via trash
    if cmd.strip().startswith("dd "):
        return "проверьте of= и добавьте бэкап; либо use cp с --reflink"
    return None


class RepetitionDetector:
    """Sliding window over executed commands; finds loops worth automating."""

    def __init__(self, window: int = 12, min_count: int = 3):
        self.window = window
        self.min_count = min_count
        self.hist: deque[str] = deque(maxlen=window)

    def observe(self, cmd: str) -> Intervention | None:
        self.hist.append(cmd.strip())
        h = list(self.hist)
        seqs: dict[tuple, int] = {}
        for k in (1, 2, 3):
            for i in range(len(h) - k + 1):
                s = tuple(h[i:i + k])
                seqs[s] = seqs.get(s, 0) + 1
        best = max(seqs.items(), key=lambda kv: kv[1] * len(kv[0]), default=((), 0))
        if best[1] >= self.min_count and len(best[0]) >= 1:
            pat = "; ".join(best[0])
            return Intervention("repetition",
                                f"Вы повторяете «{pat}» ({best[1]} раз). Предложить скрипт-автоматизацию?",
                                0.7, {"seq": list(best[0]), "count": best[1]},
                                {"pattern": pat, "count": best[1]})
        return None


class HabitModel:
    """Tracks per-type utility (EMA of rewards) and adaptive thresholds.
    Also stores 'intentional' contexts so we never nag twice about the same thing."""

    def __init__(self, cfg, memory=None):
        self.cfg = cfg
        self.memory = memory
        self.utility: dict[str, float] = {}      # EMA reward per intervention type
        self.threshold: dict[str, float] = {}    # adaptive per type
        self.last_seen: dict[str, float] = {}
        self.hour_log: deque[float] = deque(maxlen=64)
        self._data_dir = None                    # set by InitiativeEngine before load()

    def thr(self, itype: str) -> float:
        return self.threshold.setdefault(itype, self.cfg.base_threshold)

    def should_intervene(self, iv: Intervention) -> bool:
        now = time.time()
        if now - self.last_seen.get(iv.type, 0) < self.cfg.cooldown_sec:
            return False
        # NOTE: hourly budget is consumed only when an intervention is actually
        # presented (see InitiativeEngine.present), not during candidate evaluation.
        if sum(1 for t in self.hour_log if now - t < 3600) >= self.cfg.max_interventions_per_hour:
            return False
        conf = iv.confidence * (1.0 + 0.3 * self.utility.get(iv.type, 0.0))
        thr = self.thr(iv.type)
        # intentional-pattern veto: exact context previously marked deliberate.
        # A previously-marked pattern must be SKIPPED even if the adaptive
        # threshold has drifted below its confidence — the veto outranks score.
        if self.memory:
            # exact ctx-key match first (cheap string hit)
            for h in self.memory.search(_ctx_key(iv), ("prefs",), top_k=2):
                if h.meta.get("verdict") == "intentional" and h.meta.get("type") == iv.type \
                        and _same_context(h.meta.get("ctx", {}), iv.context):
                    # compensate the raised threshold so the veto path never
                    # accidentally re-enables nagging via the score check
                    return False
            # fallback: token-level veto saved by MockModelAnalyzer as
            # "intentional-token:<token>" — covers analyzer paths that don't
            # go through resolve() payload wiring
            bad = iv.payload.get("actual") or ""
            if bad:
                try:
                    for h in self.memory.search(f"intentional-token:{bad}",
                                                ("prefs",), top_k=3):
                        if h.meta.get("verdict") == "intentional" \
                                and h.meta.get("token") == bad.lower():
                            return False
                except Exception:  # noqa: BLE001
                    pass
            # last resort: scan prefs meta directly. Hash embeddings give poor
            # recall for short structured keys, so a deterministic lookup must
            # back them up — an explicit "intentional" veto may NEVER be missed.
            tok = (iv.payload.get("actual") or "").lower()
            try:
                for h in self.memory._all():
                    if h.collection != "prefs" or not isinstance(h.meta, dict):
                        continue
                    if h.meta.get("verdict") != "intentional":
                        continue
                    mtype = h.meta.get("type")
                    if mtype and mtype != iv.type:
                        continue
                    # token-level veto: analyzer saved the exact bad token
                    if tok and str(h.meta.get("token", "")).lower() == tok:
                        return False
                    # context-level veto: same normalized command line
                    mctx = h.meta.get("ctx") or {}
                    if _same_context(mctx, iv.context):
                        return False
            except Exception:  # noqa: BLE001
                pass
        if conf >= thr:
            # pre-arm cooldown for vetoed-but-scored cases is unnecessary;
            # last_seen/hour_log are armed at presentation time
            return True
        # The model was confident enough BEFORE we raised the threshold because
        # of a previous 'intentional' veto of a DIFFERENT context — keep silent.
        return False

    def resolve(self, iv: Intervention, reaction: str) -> float:
        """Map user reaction to reward, update thresholds/utility, persist."""
        r = reaction.lower()
        if any(m in r for m in INTENTIONAL_MARKERS):
            reward, verdict = -0.1, "intentional"
            self.threshold[iv.type] = min(self.cfg.max_threshold, self.thr(iv.type) + 0.12)
            if self.memory:
                self.memory.preference(_ctx_key(iv),
                                       {"verdict": "intentional", "type": iv.type,
                                        "ctx": iv.context})
                # reinforce the exact tokens the user declared intentional so the
                # analyzer's lexicon stops proposing corrections for them
                actual = iv.payload.get("actual")
                if actual:
                    self.memory.preference(f"intentional-token:{iv.type}:{actual}",
                                           {"verdict": "intentional", "token": actual,
                                            "type": iv.type, "ctx": iv.context},
                                           value=3.0)
        elif any(m in r for m in ANNOYED_MARKERS) or r in ("no", "нет", "отмена", "dismiss"):
            reward = -0.3 if any(m in r for m in ANNOYED_MARKERS) else -0.15
            verdict = "rejected"
            self.threshold[iv.type] = min(self.cfg.max_threshold, self.thr(iv.type) + 0.08)
        elif any(m in r for m in ("спасибо", "thanks", "thank")):
            reward, verdict = 1.0, "accepted_thanks"
            self.threshold[iv.type] = max(self.cfg.min_threshold, self.thr(iv.type) - 0.10)
        elif r in ("yes", "да", "ок", "accept", "apply", "+"):
            reward, verdict = 0.6, "accepted"
            self.threshold[iv.type] = max(self.cfg.min_threshold, self.thr(iv.type) - 0.05)
        else:
            reward, verdict = 0.0, "neutral"
        iv.reward, iv.resolved = reward, True
        u = self.utility.get(iv.type, 0.0)
        self.utility[iv.type] = round(u + 0.2 * (reward - u), 4)  # EMA
        if self.memory:
            self.memory.episode(f"initiative:{iv.type}:{verdict}",
                                iv.message, {"ctx": iv.context, "reward": reward},
                                value=max(0.2, 1 + reward))
        self.save()
        return reward

    def penalize_missed(self, itype: str, ctx: dict) -> None:
        """User found an obvious error we missed -> negative signal, lower threshold."""
        self.threshold[itype] = max(self.cfg.min_threshold, self.thr(itype) - 0.10)
        u = self.utility.get(itype, 0.0)
        self.utility[itype] = u - 0.1
        if self.memory:
            self.memory.episode(f"initiative:{itype}:missed", "missed obvious issue",
                                {"ctx": ctx, "reward": -0.5}, value=1.5)
        self.save()

    STATE_FILE_NAME = "habit_state.json"

    def state_path(self, data_dir):
        return data_dir / self.STATE_FILE_NAME

    def save_to(self, data_dir) -> None:
        self.state_path(data_dir).write_text(json.dumps(
            {"utility": self.utility, "threshold": self.threshold}))

    def load_from(self, data_dir) -> None:
        p = self.state_path(data_dir)
        if p.exists():
            d = json.loads(p.read_text())
            self.utility.update(d.get("utility", {}))
            self.threshold.update(d.get("threshold", {}))

    # convenience when constructed with data dir via cfg path monkey
    def save(self) -> None:
        if getattr(self, "_data_dir", None):
            self.save_to(self._data_dir)

    def load(self) -> None:
        if getattr(self, "_data_dir", None):
            self.load_from(self._data_dir)


def _ctx_key(iv: Intervention) -> str:
    return f"{iv.type}:" + json.dumps(iv.context, sort_keys=True, ensure_ascii=False)[:200]


def _same_context(a: dict, b: dict) -> bool:
    """Generalizing veto: 'intentional' memory covers the same situation class.
    For command contexts we compare the normalized command line (first segment
    before '&&' / ';' pipes), so 'cd /hme' marks also vetoes 'cd /hme && ls'."""
    if not a or not b:
        return False
    la, lb = str(a.get("line", "")), str(b.get("line", ""))
    if la and lb:
        na = _norm_line(la)
        nb = _norm_line(lb)
        if na == nb:
            return True
        # prefix-subsumption: one is a continuation of the other
        return na.startswith(nb + " ") or nb.startswith(na + " ")
    ka, kb = set(a), set(b)
    common = ka & kb
    return any(str(a[k]) == str(b[k]) for k in common)


def _norm_line(line: str) -> str:
    first = re.split(r"&&|;", line)[0].strip()
    toks = first.split()
    return " ".join(t.lower() for t in toks[:3])


class InitiativeEngine:
    def __init__(self, cfg, memory, data_dir, notifier=None, typo_analyzer=None):
        self.cfg = cfg
        self.habits = HabitModel(cfg, memory)
        self.habits._data_dir = data_dir
        self.habits.load()
        self.memory = memory
        self.typer = TypoDetector(typo_analyzer, memory)
        self.danger = DangerDetector(cfg)          # replaced via configure_safety
        self.reps = RepetitionDetector(cfg.repetition_window, cfg.repetition_min_count)
        self.notifier = notifier
        self.pending: dict[str, Intervention] = {}

    def configure_safety(self, safety_cfg) -> None:
        """DangerDetector needs the SAFETY config (blacklist/prefixes)."""
        self.danger = DangerDetector(safety_cfg)

    def attach_model_backend(self, backend) -> None:
        """Stage-2 hook: route typo judgement through the real SSM backend."""
        self.typer.analyzer = make_analyzer(self.cfg, self.memory, backend=backend) \
            if not hasattr(backend, "analyze_line") else backend

    # ---- event entry points (called by EventMonitor / loop) -------------
    def on_user_line(self, line: str) -> list[Intervention]:
        out = []
        # 1. model-based typo verdicts FIRST — before the lexicon has seen this
        #    line. Observing first would let a single sighting of 'gti'
        #    legitimize the typo (self-legitimization bug).
        rag_ctx = ""
        if self.memory:
            try:
                hits = self.memory.search(line, ("prefs", "docs"), top_k=3)
                rag_ctx = "\n" + "\n".join(
                    f"{h.text} | reward={h.meta.get('reward')}"
                    if isinstance(h.meta, dict) and "reward" in h.meta else h.text[:200]
                    for h in hits if h.score > 0.35)
            except Exception:  # noqa: BLE001
                rag_ctx = ""
        for iv in self.typer.analyze(line, rag_ctx):
            if self.habits.should_intervene(iv):
                out.append(iv)
        # 2. deterministic risk gate (NOT a language judgement — policy check)
        dv = self.danger.check(line)
        if dv and self.habits.should_intervene(dv):
            out.append(dv)
        rep = self.reps.observe(line)
        if rep and self.habits.should_intervene(rep):
            out.append(rep)
        # 3. NOW train the externalized lexicon from this line (cheap, no model).
        #    Repeated user usage raises future evidence; first sightings never
        #    suppress their own analysis.
        self.typer.lex.observe(line)
        return out

    def on_idle(self, idle_sec: float) -> list[Intervention]:
        if idle_sec >= self.cfg.idle_help_after_sec:
            iv = Intervention("idle", "Похоже, вы застряли — нужна помощь?", 0.5,
                              {"idle_sec": idle_sec})
            if self.habits.should_intervene(iv):
                return [iv]
        return []

    def on_unsaved(self, filename: str, minutes: float) -> list[Intervention]:
        if minutes * 60 >= self.cfg.unsaved_file_remind_after_sec:
            iv = Intervention("unsaved", f"Файл {filename} открыт {int(minutes)} мин без сохранений.",
                              0.6, {"file": filename})
            if self.habits.should_intervene(iv):
                return [iv]
        return []

    def on_deviation(self, cmd: str, habits_seq: list[str]) -> list[Intervention]:
        """Unusual pattern vs user's historical sequence for this app/time."""
        if not self.memory or not habits_seq:
            return []
        near = self.memory.search(f"routine {' '.join(habits_seq[-3:])}", ("episodes",), 3)
        expected = [h for h in near if h.score > 0.85]
        if expected and cmd not in expected[0].text:
            iv = Intervention("deviation",
                              f"Необычный шаг «{cmd}»; обычно дальше идёт «{expected[0].text[:60]}». Всё ок?",
                              0.55, {"cmd": cmd})
            if self.habits.should_intervene(iv):
                return [iv]
        return []

    # ---- presentation & feedback ----------------------------------------
    def present(self, ivs: list[Intervention]) -> None:
        for iv in ivs:
            key = iv.type + ":" + str(iv.ts)
            self.pending[key] = iv
            self.habits.last_seen[iv.type] = time.time()   # start cooldown
            self.habits.hour_log.append(time.time())       # consume hourly budget
            if self.notifier:
                self.notifier.notify("KDE Agent", iv.message,
                                     actions=["Принять", "Отклонить"], reply_key=key)

    def feedback(self, key: str, reaction: str) -> float:
        iv = self.pending.pop(key, None)
        if not iv:
            return 0.0
        return self.habits.resolve(iv, reaction)

    def intervene(self, line: str) -> list[Intervention]:
        """Detect + present in one call. Used by the event loop and tests so
        that interventions actually reach `pending` (on_user_line alone only
        evaluates candidates; presentation is a separate, explicit step)."""
        ivs = self.on_user_line(line)
        if ivs:
            self.present(ivs)
        return ivs
