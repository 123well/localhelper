"""Context manager: bounded token budget, event compression, relevance pruning.

Layers (hierarchy of processing, biological principle #3):
  L0 raw events      : kept only for the current turn
  L1 recent window   : verbatim last N turns
  L2 summaries       : older turns compressed by an extractive summarizer
                       (in production: the SSM itself summarizes on NPU)
  eviction           : low-relevance items dropped entirely; user prefs and
                       'intentional' notes are pinned and survive.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


@dataclass
class Turn:
    role: str          # user | assistant | tool | event | system_note
    text: str
    score: float = 1.0     # relevance heuristic
    pinned: bool = False


@dataclass
class ContextManager:
    budget_tokens: int = 2048
    recent_keep: int = 8
    turns: deque = field(default_factory=deque)
    summaries: list[str] = field(default_factory=list)

    def add(self, role: str, text: str, score: float = 1.0, pinned: bool = False) -> None:
        self.turns.append(Turn(role, text, score, pinned))
        self._fit()

    def _summarize(self, batch: list[Turn]) -> str:
        """Cheap extractive summary: first clause per turn, deduped.
        Stage-2 replaces this with one NPU call ('сожми диалог до 40 слов')."""
        parts = []
        for t in batch:
            line = re.sub(r"\s+", " ", t.text).strip()
            parts.append(f"{t.role}: {line[:90]}")
        s = " || ".join(parts)
        return s[:400]

    def _fit(self) -> None:
        while self.total() > self.budget_tokens and len(self.turns) > self.recent_keep:
            # evict lowest-score non-pinned oldest first
            cand = min(((i, t) for i, t in enumerate(self.turns[:-self.recent_keep])
                        if not t.pinned), key=lambda it: (it[1].score, it[0]), default=None)
            if cand is None:
                break
            old = self.turns[cand[0]]
            del self.turns[cand[0]]
            if self.summaries and _approx_tokens(self.summaries[-1]) < 60:
                self.summaries[-1] = self._summarize(
                    [Turn("summary", self.summaries[-1])] + [old])
            else:
                self.summaries.append(self._summarize([old]))
            while len(self.summaries) > 6:
                head = self.summaries[:3]
                self.summaries = [self._summarize([Turn("s", " || ".join(head))])] + self.summaries[3:]

    def total(self) -> int:
        return (_approx_tokens(" ".join(self.summaries))
                + sum(_approx_tokens(t.text) for t in self.turns))

    def render(self) -> str:
        s = ""
        if self.summaries:
            s += "SUMMARY OF OLDER EVENTS:\n" + "\n".join(f"- {x}" for x in self.summaries) + "\n"
        s += "RECENT:\n" + "\n".join(f"[{t.role}] {t.text}" for t in self.turns)
        return s

    def prune_irrelevant(self, keywords: list[str]) -> int:
        """Drop unpinned low-score turns that share no keyword with now."""
        kw = {k.lower() for k in keywords}
        removed = 0
        keep = deque()
        for t in self.turns:
            tl = t.text.lower()
            if t.pinned or t.score >= 1.5 or any(k in tl for k in kw):
                keep.append(t)
            else:
                removed += 1
        self.turns = keep
        return removed
