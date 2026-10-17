"""Model-based typo analyzer — the "AI" half of typo detection.

Design contract (matches the system architecture):
  * The program NEVER makes the final call by itself. Cheap heuristics only
    PROPOSE candidates ("this token looks unusual, here are nearby variants").
  * The judgement belongs to the model: it receives the line, the proposed
    candidates with their evidence, and external facts (live filesystem
    state, PATH binaries, user history/RAG hits) — never vocabulary baked
    into weights — and returns a structured verdict {is_typo, confidence,
    intended, reason}.
  * Because the model evaluates EVIDENCE rather than doing raw string
    matching, this generalizes to complex words, multi-segment paths,
    wrong keyboard layout (йцукен <-> qwerty), transpositions and domain
    vocabulary. A naive difflib pass cannot.

Backends (same pattern as agent.core.model):
  * MockModelAnalyzer — deterministic stand-in used while prototyping / in
    tests without weights. It mirrors the real pipeline (propose -> weigh
    evidence -> calibrated verdict) so swapping in the SSM changes nothing
    else in the system.
  * LLMModelAnalyzer  — sends the same evidence packet to the frozen SSM
    backend and parses its JSON verdict; falls back to the mock's score if
    the model is unavailable.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Keyboard-layout decoding (fact about hardware layout, not about language)
# ---------------------------------------------------------------------------
QWERTY = "qwertyuiopasdfghjklzxcvbnm-[]"
JOHANNESON = "йцукенгшщзхъфывапролджэячсмитьбю.-"   # ru layout on the same keys
LAYOUT_MAP = dict(zip(QWERTY + QWERTY.upper(), JOHANNESON + JOHANNESON.upper()))
LAYOUT_REV = {v: k for k, v in LAYOUT_MAP.items()}


def layout_switch(word: str) -> str:
    """'lfydb' -> 'city', 'gthrfcbdf' -> 'python' (typed on wrong layout)."""
    return "".join(LAYOUT_MAP.get(c, c) for c in word)


def layout_switch_rev(word: str) -> str:
    """Reverse decode: latin-looking text typed while RU layout was active."""
    return "".join(LAYOUT_REV.get(c, c) for c in word)


# ---------------------------------------------------------------------------
# Distances used as EVIDENCE features (the scoring decision lives elsewhere)
# ---------------------------------------------------------------------------
_KB_ROWS = ["`1234567890-", "qwertyuiop[]", "asdfghjkl;'\\", "zxcvbnm,./"]
_KEY_POS: dict[str, tuple[int, int]] = {}
for _r, _row in enumerate(_KB_ROWS):
    for _c, _ch in enumerate(_row):
        _KEY_POS[_ch] = (_r, _c)


def key_proximity_cost(a: str, b: str) -> float:
    """Substitution cost scaled by physical distance between keys."""
    pa, pb = _KEY_POS.get(a.lower()), _KEY_POS.get(b.lower())
    if pa and pb:
        dr, dc = abs(pa[0] - pb[0]), abs(pa[1] - pb[1])
        return min(1.0, 0.45 + 0.15 * dr + 0.08 * dc)
    return 1.0


def levenshtein(a: str, b: str) -> int:
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for i, cb in enumerate(b, 1):
        cur = [i]
        for j, ca in enumerate(a):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ca != cb)))
        prev = cur
    return prev[-1]


def damerau(a: str, b: str) -> int:
    """Levenshtein + adjacent transposition ('gti'->'git' costs 1)."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def weighted_distance(a: str, b: str) -> float:
    """DP edit distance where substitutions are cheaper for neighbouring keys
    and transpositions always cost 1 — models finger slips, not logic."""
    la, lb = len(a), len(b)
    dp = [[0.0] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        dp[i][0] = i
    for j in range(1, lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            sub = (dp[i - 1][j - 1] if a[i - 1] == b[j - 1]
                   else dp[i - 1][j - 1] + key_proximity_cost(a[i - 1], b[j - 1]))
            dp[i][j] = min(sub, dp[i - 1][j] + 1.0, dp[i][j - 1] + 1.0)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1.0)
    return dp[la][lb]


def char_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower()), set(b.lower())
    return len(sa & sb) / len(sa | sb) if sa or sb else 0.0


# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    variant: str
    kind: str                 # command | path_segment
    source: str               # builtin | path | filesystem | learned_dict | layout | rag
    evidence: dict = field(default_factory=dict)


@dataclass
class TypoVerdict:
    is_typo: bool
    confidence: float
    intended: str | None      # reconstructed correct line
    reason: str               # short rationale (logged, shown to user, fed to memory)
    bad_token: str = ""
    kind: str = ""
    candidates: list[Candidate] = field(default_factory=list)


class Lexicon:
    """Externalized vocabulary — facts live OUTSIDE the weights:
    user's own command/file history, live filesystem, PATH binaries.
    Cheap to maintain; continuously updated from observed lines."""

    def __init__(self, memory=None):
        self.memory = memory
        self.user_words: dict[str, int] = {}   # token -> observed count

    def observe(self, line: str) -> None:
        for tok in re.findall(r"[\w][\w._-]{1,}", line):
            base = tok.split("/")[-1].lower()
            if base:
                self.user_words[base] = self.user_words.get(base, 0) + 1

    def known(self, word: str) -> int:
        """How often the USER has actually used this token (personal history).
        PATH binaries and seed lists must not leak in here — otherwise a single
        observed typo would 'legitimize' it forever."""
        return self.user_words.get(word.lower(), 0)

    @staticmethod
    def path_exists(p: str) -> bool:
        try:
            return Path(os.path.expanduser(p)).exists()
        except OSError:
            return False


# ---------------------------------------------------------------------------
class ProposalEngine:
    """Cheap candidate generator shared by all backends. Deliberately NOT a
    detector: it only says 'unknown token X, nearby variants A,B,C with these
    evidence features'. All thresholding/judgement happens in the analyzer."""

    def __init__(self, builtin_cmds: set[str]):
        self.builtin = builtin_cmds
        self.vocab_hook = None  # optional callable() -> extra terms from RAG

    # ---- commands ----------------------------------------------------------
    def command_vocab(self, lex: Lexicon) -> set[str]:
        vocab = set(self.builtin) | set(lex.user_words)
        for d in os.environ.get("PATH", "").split(os.pathsep):
            try:
                vocab |= {p.name for p in Path(d).iterdir()
                          if p.is_file() and os.access(p, os.X_OK)}
            except OSError:
                continue
        if self.vocab_hook:
            try:
                vocab |= set(self.vocab_hook())
            except Exception:  # noqa: BLE001
                pass
        return vocab

    def cmd_is_known(self, cmd: str, lex: Lexicon) -> bool:
        """A command is 'known' if it's a seed/builtin or a real PATH binary.
        Personal history alone must NEVER legitimize an unknown token — the
        user may repeat a typo (or a habit of typos) many times; only an
        EXPLICIT 'intentional' verdict (memory veto in MockModelAnalyzer)
        silences suggestions for personal tokens."""
        return cmd in self.builtin or bool(shutil.which(cmd))

    def propose_command(self, cmd: str, lex: Lexicon) -> list[Candidate]:
        vocab = self.command_vocab(lex)
        out: list[Candidate] = []
        low = cmd.lower()
        for v in vocab:
            if not v or v.lower() == low:
                continue
            wd = weighted_distance(low, v.lower())
            rel = wd / max(1, len(low))
            if rel <= 0.66 or damerau(low, v.lower()) <= 2:
                src = ("learned_dict" if v in lex.user_words else
                       "builtin" if v in self.builtin else "path")
                out.append(Candidate(v, "command", src,
                                     {"wdist": round(wd, 2), "rel": round(rel, 3),
                                      "transposition": damerau(low, v.lower()) == 1,
                                      "freq": lex.user_words.get(v, 0)}))
        dec = layout_switch_rev(cmd)          # RU keys pressed with QWERTY shape
        if dec != cmd and dec.lower() in vocab:
            out.append(Candidate(dec.lower(), "command", "layout",
                                 {"decoded": True, "wdist": 0.0, "rel": 0.0}))
        out.sort(key=lambda c: c.evidence["rel"])
        return out[:6]

    # ---- paths ---------------------------------------------------------------
    def check_path(self, p: str, lex: Lexicon) -> tuple[str, list[Candidate]] | None:
        """Walk segments; first non-existent segment whose siblings/dictionary
        contain a close EXISTING name => propose it. New-file names at the end
        of an existing directory are normal (touch foo.txt) -> no proposal."""
        expanded = os.path.expanduser(p)
        if Path(expanded).exists():
            return None
        # walk the path with '~' already resolved; a literal '~' segment would
        # otherwise be treated as a home dir name and never exist under '/'
        parts = expanded.split("/")
        acc = ""                      # last verified prefix (expanded form)
        bad_idx = None
        for i, seg in enumerate(parts):
            if seg == ".":
                continue
            if i == 0 and seg == "":          # leading '/'
                acc = "/"
                continue
            nxt = f"{acc}{'' if acc.endswith('/') else '/'}{seg}"
            if not Path(nxt).exists():
                bad_idx = i
                break
            acc = nxt
        if bad_idx is None:
            return None               # every prefix exists -> brand-new file: fine
        seg = parts[bad_idx]
        parent = Path(acc) if acc else Path.cwd()
        siblings: set[str] = set()
        try:
            siblings = {e.name for e in parent.iterdir()}
        except OSError:
            pass
        cands: list[Candidate] = []
        low = seg.lower()
        for s in siblings:
            if s.lower() == low:
                continue
            wd = weighted_distance(low, s.lower())
            if wd <= max(2.0, len(low) * 0.55):
                # target_exists: verified live fact — the sibling is present on
                # disk right now (checked again at judgement time below)
                cands.append(Candidate(s, "path_segment", "filesystem",
                                       {"wdist": round(wd, 2),
                                        "rel": round(wd / max(1, len(low)), 3),
                                        "target_exists": True}))
        for v, freq in lex.user_words.items():
            if v == low or v in siblings:
                continue
            wd = weighted_distance(low, v)
            if wd <= 2.0:
                cands.append(Candidate(v, "path_segment", "learned_dict",
                                       {"wdist": round(wd, 2), "freq": freq,
                                        "rel": round(wd / max(1, len(low)), 3)}))
        dec = layout_switch_rev(seg)
        if dec != seg and (dec.lower() in {s.lower() for s in siblings}
                           or lex.known(dec)):
            cands.append(Candidate(dec, "path_segment", "layout",
                                   {"decoded": True, "wdist": 0.0, "rel": 0.0}))
        if not cands:
            return None
        cands.sort(key=lambda c: c.evidence["rel"])
        return seg, cands[:6]


PATH_RE = re.compile(r"(?:~?/|\./)[\w.@+\-/]*")


class ModelTypoAnalyzer:
    """Interface: judge typos in one user line. Returns verdicts that already
    ARE model judgements (confidence + reason), not raw string matches."""

    def analyze_line(self, line: str, lex: Lexicon, rag_context: str = "") -> list[TypoVerdict]:
        raise NotImplementedError


class MockModelAnalyzer(ModelTypoAnalyzer):
    """Deterministic stand-in for the SSM during prototyping/tests.

    Mirrors the production pipeline exactly: ProposalEngine generates
    evidence packets; this class then WEIGHS them (dictionary hit, fs
    existence, layout decode, proximity, frequency, ambiguity penalty) and
    emits a calibrated verdict with a human-readable reason — the same
    interface the trained model will implement on the NPU."""

    ABSTAIN_BELOW = 0.45   # model stays silent instead of nagging

    def __init__(self, proposals: ProposalEngine):
        self.prop = proposals

    def analyze_line(self, line: str, lex: Lexicon, rag_context: str = "") -> list[TypoVerdict]:
        verdicts: list[TypoVerdict] = []
        toks = line.strip().split()
        if not toks:
            return verdicts

        # --- command slot (skip env-prefix words like sudo/env) --------------
        idx = 0
        while idx < len(toks) - 1 and toks[idx] in ("sudo", "env", "time"):
            idx += 1
        cmd = toks[idx].split("/")[-1]
        if (re.fullmatch(r"[\w][\w.-]*", cmd) and not cmd.startswith("-")
                and "." not in cmd and not cmd.startswith("./")
                and not self._declared_intentional(cmd, lex)
                and not self.prop.cmd_is_known(cmd, lex)):
            cands = self.prop.propose_command(cmd, lex)
            v = self._judge(cmd, cands, "command", original=line)
            if v:
                verdicts.append(v)

        # --- path tokens ------------------------------------------------------
        for m in PATH_RE.finditer(line):
            raw = m.group(0).rstrip(".,;:!?")
            res = self.prop.check_path(raw, lex)
            if not res:
                continue
            seg, cands = res
            if self._declared_intentional(seg, lex):
                continue                     # user said: intentional — never nag twice
            v = self._judge(seg, cands, "path_segment", original=raw)
            if v:
                verdicts.append(v)

        seen: set[tuple[str, str]] = set()
        uniq = []
        for v in verdicts:
            k = (v.bad_token, v.kind)
            if k not in seen:
                seen.add(k)
                uniq.append(v)
        return uniq

    @staticmethod
    def _declared_intentional(token: str, lex: Lexicon) -> bool:
        """Hard veto from explicit user feedback stored in prefs memory."""
        if lex.memory is None or not token:
            return False
        try:
            hits = lex.memory.search(f"intentional-token:{token}", ("prefs",), top_k=3)
            return any(h.meta.get("verdict") == "intentional"
                       and h.meta.get("token") == token.lower() for h in hits)
        except Exception:  # noqa: BLE001
            return False

    def _judge(self, token: str, cands: list[Candidate], kind: str,
               original: str) -> TypoVerdict | None:
        """Evidence weighting = the 'thinking' step. Every factor is
        explainable; ambiguity lowers confidence (unsure -> stay silent)."""
        if not cands:
            return None
        best = cands[0]
        ev = best.evidence
        conf = 0.30
        rel = ev.get("rel", 1.0)
        if rel <= 0.34:
            conf += 0.30
        elif rel <= 0.5:
            conf += 0.18
        elif rel <= 0.66:
            conf += 0.08
        if ev.get("transposition"):
            conf += 0.10                     # classic finger slip
        if kind == "path_segment" and ev.get("target_exists") \
                and ev.get("wdist", 9) <= 2.0:
            # verified filesystem evidence: the proposed target really exists
            # as a sibling of the broken segment — strongest possible signal,
            # independent of candidate ranking (a truncated name like '.cac'
            # can rank behind generic dictionary words yet still be the answer)
            conf += 0.30
        if best.source in ("filesystem", "learned_dict"):
            conf += 0.15                     # target verifiably exists / user uses it
        if best.source == "layout":
            conf += 0.25                     # exact layout decode is strong evidence
        if ev.get("freq", 0) >= 3:
            conf += 0.05
        if len(cands) > 1:                   # near-tie ambiguity penalty
            second = cands[1].evidence.get("rel", 1.0)
            if second - rel < 0.05 and cands[1].source != "layout":
                conf -= 0.15
        if kind == "path_segment" and len(token) >= 8 and best.source == "learned_dict":
            conf -= 0.10                     # long unknown names are often legit
        conf = max(0.0, min(conf, 0.97))
        if conf < self.ABSTAIN_BELOW:
            return None
        # for path segments always correct AGAINST an existing sibling: pick
        # the best fs-verified candidate even if a learned-dict word ranked
        # closer by raw distance (fs existence beats fuzzy dictionary match)
        if kind == "path_segment":
            fs_verified = [c for c in cands
                           if c.source == "filesystem" and c.evidence.get("target_exists")]
            if fs_verified:
                best = fs_verified[0]
        if kind == "path_segment":
            fixed = _replace_segment(original, token, best.variant)
            msg = (f"Путь «{original}» не существует: «{token}» похоже на "
                   f"«{best.variant}». Исправить?")
        else:
            fixed = re.sub(rf"(?<![\w./-]){re.escape(token)}(?![\w./-])",
                           best.variant, original, count=1)
            msg = f"Команда «{token}» не найдена. Возможно, вы имели в виду «{best.variant}»?"
        reason = (f"признаки: источник={best.source}, расст.≈{ev.get('wdist')}, "
                  f"транспозиция={bool(ev.get('transposition'))}, частота={ev.get('freq', 0)}")
        return TypoVerdict(True, round(conf, 2), fixed, msg + " [" + reason + "]",
                           token, kind, cands)


def _replace_segment(path: str, bad: str, good: str) -> str:
    parts = path.split("/")
    for i, s in enumerate(parts):
        if s == bad:
            parts[i] = good
            break
    return "/".join(parts)


# ---------------------------------------------------------------------------
class LLMModelAnalyzer(ModelTypoAnalyzer):
    """Production path: ship the evidence packet to the frozen 7B SSM and let
    IT decide. Facts arrive via prompt (filesystem listing, RAG context) —
    never from weights. Falls back to the mock verdict if the backend errors."""

    PROMPT = """TASK: does USER_LINE contain a typo (unknown command or wrong path)?
You have NO built-in vocabulary. Decide ONLY from EVIDENCE and RAG_CONTEXT.
EVIDENCE items were generated by a cheap proposer: distances/sources are hints, not truth.
Reply ONLY JSON: {"is_typo": bool, "confidence": 0..1, "intended": str|null, "reason": str}
Rules: plausible NEW filenames are not typos; ambiguous candidates -> lower confidence;
if RAG_CONTEXT says the user marked this pattern intentional -> is_typo=false.
TOKEN: {token} (kind: {kind})
USER_LINE: {line}
EVIDENCE: {cands}
RAG_CONTEXT: {rag}
ASSISTANT:"""

    def __init__(self, backend, fallback: MockModelAnalyzer):
        self.backend = backend
        self.fallback = fallback

    def analyze_line(self, line: str, lex: Lexicon, rag_context: str = "") -> list[TypoVerdict]:
        base = self.fallback.analyze_line(line, lex, rag_context)
        for v in base:
            prompt = self.PROMPT.format(
                token=v.bad_token, kind=v.kind, line=line[:400],
                cands=json.dumps([{"variant": c.variant, "source": c.source,
                                   "evidence": c.evidence} for c in v.candidates],
                                  ensure_ascii=False)[:900],
                rag=(rag_context[:600] or "(none)"))
            try:
                raw = self.backend.generate(prompt, max_tokens=160)
                m = re.search(r"\{.*\}", raw, re.S)
                d = json.loads(m.group(0)) if m else None
                if isinstance(d, dict) and isinstance(d.get("is_typo"), bool):
                    conf = float(d.get("confidence", v.confidence))
                    v.is_typo = bool(d["is_typo"]) and conf >= 0.4
                    v.confidence = round(max(0.0, min(conf, 1.0)), 2)
                    if d.get("intended"):
                        v.intended = str(d["intended"])[:300]
                    if d.get("reason"):
                        v.reason = str(d["reason"])[:300]
            except Exception:  # noqa: BLE001 — model down: heuristic verdict stands
                pass
        return [v for v in base if v.is_typo]


def make_analyzer(cfg, memory, backend=None) -> ModelTypoAnalyzer:
    """Factory honoring the config: real model when a backend is provided,
    deterministic mock otherwise (prototype mode)."""
    from agent.core.initiative import KNOWN_CMDS
    prop = ProposalEngine(set(KNOWN_CMDS))
    if memory is not None:
        prop.vocab_hook = lambda: {h.text.split()[0]
                                   for h in memory.search("alias command", ("docs",), top_k=20)
                                   if h.text}
    mock = MockModelAnalyzer(prop)
    if backend is not None and getattr(cfg, "typo_use_model", True):
        return LLMModelAnalyzer(backend, mock)
    return mock
