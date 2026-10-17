"""Knowledge cards: the unit of curated external memory (spec 6.2).

Contract per card (ТЗ): type / summary / body / examples / search phrases /
metadata / versioning. Cards live in sqlite as a thin index; the *payload*
(embedding + text) is stored through Memory under collection "cards", so the
frozen encoder, decay/prune and hit counters are shared with the rest of RAG.

Dedup & conflicts (quality control before anything reaches the model):
  * cosine >= dedup_high                     -> near-duplicate: merge into the
    existing card (bump version, union examples/phrases, reinforce value);
  * dedup_low <= cosine < dedup_high         -> same topic, different content:
    if summaries differ -> record a conflict for human review (the program
    proposes, the user decides — never silently overwrite knowledge);
  * otherwise -> new card.

Usefulness counters drive the sleep curriculum: frequently retrieved and
marked-helpful cards get reinforced; dead ones decay via Memory.prune.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("kde-agent.cards")

COLLECTION = "cards"


@dataclass
class Card:
    id: int
    version: int
    type: str                      # concept | api | command | safety_rule | snippet
    summary: str
    body: str
    examples: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    item_id: int | None = None
    value: float = 1.0
    usages: int = 0
    helpful: int = 0
    not_helpful: int = 0
    ts: float = 0.0


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _summary_key(s: str) -> str:
    """Coarse identity of what a card is ABOUT (for conflict detection)."""
    words = re.findall(r"[\wа-яё-]+", _norm(s))
    return " ".join(words[:4])


class CardStore:
    def __init__(self, memory, dedup_high: float = 0.93, dedup_low: float = 0.84):
        self.mem = memory
        self.high = dedup_high
        self.low = dedup_low
        self.db = memory.db  # share the RAG connection (single writer process)
        self.mem._exec("""CREATE TABLE IF NOT EXISTS cards(
            id INTEGER PRIMARY KEY, version INTEGER DEFAULT 1, type TEXT,
            summary TEXT, body TEXT, examples TEXT, phrases TEXT, meta TEXT,
            item_id INTEGER, usages INTEGER DEFAULT 0, helpful INTEGER DEFAULT 0,
            not_helpful INTEGER DEFAULT 0, created REAL, updated REAL)""")
        # NOTE: "drop" is a reserved SQL keyword -> quote the column names.
        self.mem._exec("""CREATE TABLE IF NOT EXISTS card_conflicts(
            "keep" INTEGER, "drop" INTEGER, note TEXT,
            resolved INTEGER DEFAULT 0, ts REAL)""")
        self.mem._commit()

    # ------------------------------------------------------------------ write
    def add(self, type: str, summary: str, body: str,
            examples: list[str] | None = None, phrases: list[str] | None = None,
            meta: dict | None = None, value: float = 1.0) -> Card:
        examples = list(examples or [])
        phrases = list(phrases or [])
        payload = self._payload(summary, body, phrases)
        best_id, best_sim = self._nearest(payload)

        if best_id is not None and best_sim >= self.high:
            return self._merge(best_id, examples, phrases, value)

        if best_id is not None and self.low <= best_sim < self.high \
                and _summary_key(summary) != _summary_key(self.get(best_id).summary):
            other = self.get(best_id)
            # newer source wins provisionally, old one flagged for review
            cur = self.mem._exec(
                'INSERT INTO card_conflicts("keep","drop",note,ts) VALUES(?,?,?,?)',
                (None, other.id, f"similar {best_sim:.2f} to #{other.id}", time.time()))
            self.mem._commit()

        item_id = self.mem.add(COLLECTION, payload, type, meta, value=value)
        now = time.time()
        cur = self.mem._exec(
            "INSERT INTO cards(version,type,summary,body,examples,phrases,meta,"
            "item_id,usages,helpful,not_helpful,created,updated)"
            " VALUES(1,?,?,?,?,?,?,?,0,0,0,?,?)",
            (type, summary, body, json.dumps(examples, ensure_ascii=False),
             json.dumps(phrases, ensure_ascii=False),
             json.dumps(meta or {}, ensure_ascii=False), item_id, now, now))
        cid = cur.lastrowid
        self.mem._commit()
        card = self.get(cid)
        if card is not None and best_id is not None and self.low <= best_sim < self.high:
            card.meta = dict(card.meta or {})
            card.meta["conflict_with"] = best_id
            self.mem._exec("UPDATE cards SET meta=? WHERE id=?",
                            (json.dumps(card.meta, ensure_ascii=False), cid))
            self.mem._commit()
        return card

    def _merge(self, card_id: int, examples: list[str], phrases: list[str],
               value_delta: float) -> Card:
        c = self.get(card_id)
        ex = list(dict.fromkeys(c.examples + examples))[:8]
        ph = list(dict.fromkeys(c.phrases + phrases))[:12]
        self.mem._exec(
            "UPDATE cards SET version=version+1, examples=?, phrases=?, updated=?"
            " WHERE id=?",
            (json.dumps(ex, ensure_ascii=False), json.dumps(ph, ensure_ascii=False),
             time.time(), card_id))
        self.mem._commit()
        if c.item_id:
            self.mem.reinforce(c.item_id, delta=max(0.1, value_delta - 1.0))
        log.debug("card #%d merged (v%d)", card_id, c.version + 1)
        return self.get(card_id)

    # ------------------------------------------------------------------- read
    def get(self, card_id: int) -> Card | None:
        row = self.mem._exec(
            "SELECT id,version,type,summary,body,examples,phrases,meta,item_id,"
            "usages,helpful,not_helpful,updated FROM cards WHERE id=?",
            (card_id,)).fetchone()
        if not row:
            return None
        value = 1.0
        if row[8]:
            r = self.mem._exec("SELECT value FROM items WHERE id=?",
                                (row[8],)).fetchone()
            value = r[0] if r else 0.0
        return Card(id=row[0], version=row[1], type=row[2], summary=row[3],
                    body=row[4], examples=json.loads(row[5]),
                    phrases=json.loads(row[6]), meta=json.loads(row[7]),
                    item_id=row[8],
                    value=value, usages=row[9], helpful=row[10],
                    not_helpful=row[11], ts=row[12])

    def list_cards(self, limit: int = 50, type: str | None = None) -> list[Card]:
        q = ("SELECT id FROM cards" + (" WHERE type=?" if type else "") +
             " ORDER BY updated DESC LIMIT ?")
        rows = self.mem._exec(q, ((type, limit) if type else (limit,))).fetchall()
        return [c for c in (self.get(r[0]) for r in rows) if c]

    def search(self, query: str, top_k: int = 4,
               types: tuple[str, ...] | None = None,
               project: str | None = None) -> list[Card]:
        hits = self.mem.search(query, (COLLECTION,), top_k=top_k * 3)
        out: list[Card] = []
        for h in hits:
            c = self._by_item(h.id)
            if c is None:
                continue
            if types and c.type not in types:
                continue
            if project and (c.meta or {}).get("project") not in ("", None, project):
                continue
            out.append(c)
            if len(out) >= top_k:
                break
        return out

    def _by_item(self, item_id: int) -> Card | None:
        row = self.mem._exec("SELECT id FROM cards WHERE item_id=?",
                              (item_id,)).fetchone()
        return self.get(row[0]) if row else None

    # ------------------------------------------------------------- usage stats
    def record_usage(self, card_id: int) -> None:
        self.mem._exec("UPDATE cards SET usages=usages+1 WHERE id=?", (card_id,))
        self.mem._commit()

    def rate(self, card_id: int, helpful: bool) -> None:
        col = "helpful" if helpful else "not_helpful"
        self.mem._exec(f"UPDATE cards SET {col}={col}+1 WHERE id=?", (card_id,))
        self.mem._commit()
        c = self.get(card_id)
        if c and c.item_id:
            self.mem.reinforce(c.item_id, delta=0.2 if helpful else -0.3)

    # ---------------------------------------------------------------- conflicts
    def pending_conflicts(self) -> list[tuple[Card, int]]:
        rows = self.mem._exec(
            'SELECT rowid, "drop" FROM card_conflicts WHERE resolved=0').fetchall()
        out = []
        for rid, drop in rows:
            c = self.get(drop)
            if c:
                out.append((c, drop))
        return out

    def resolve_conflict(self, keep_id: int, drop_id: int) -> None:
        """Human decision: 'keep' survives unchanged, 'drop' is tombstoned."""
        self.mem._exec(
            'UPDATE card_conflicts SET resolved=1, "keep"=?, '
            "note=note||' [resolved]' "
            'WHERE "keep" IS NULL AND "drop"=?', (keep_id, drop_id))
        self.mem._commit()
        if keep_id != drop_id:
            self.remove(drop_id)

    # ------------------------------------------------------------------ delete
    def remove(self, card_id: int) -> None:
        c = self.get(card_id)
        if c is None:
            return
        self.mem._exec("DELETE FROM cards WHERE id=?", (card_id,))
        self.mem._commit()
        if c.item_id:
            self.mem._exec("DELETE FROM items WHERE id=?", (c.item_id,))
            self.mem._commit()
            self.mem.invalidate_cache()

    # ---------------------------------------------------------------- nightly
    def quality_report(self) -> dict:
        total = self.mem._exec("SELECT COUNT(*) FROM cards").fetchone()[0]
        conflicts = self.mem._exec(
            "SELECT COUNT(*) FROM card_conflicts WHERE resolved=0").fetchone()[0]
        stale = self.mem._exec(
            "SELECT COUNT(*) FROM cards WHERE usages=0 AND updated < ?",
            (time.time() - 7 * 86400,)).fetchone()[0]
        return {"cards": total, "open_conflicts": conflicts, "stale_7d": stale}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _payload(summary: str, body: str, phrases: list[str]) -> str:
        head = summary if not phrases else summary + "\n" + " | ".join(phrases)
        return head + "\n" + body

    def _nearest(self, payload: str) -> tuple[int | None, float]:
        """Cosine of payload against every existing card item (top-1 only)."""
        for h in self.mem.search(payload, (COLLECTION,), top_k=1):
            return h.id, h.score
        return None, -1.0
