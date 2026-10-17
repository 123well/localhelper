"""External memory + RAG. All factual knowledge lives HERE, never in weights.

sqlite store with three logical collections:
  * docs     : documentation snippets (man pages, Arch wiki exports, API refs)
  * episodes : action history, outcomes, interventions & user reactions
  * prefs    : learned user preferences / "intentional" patterns

Vectors: frozen lightweight embedder (hash+ngram projection by default — a
deterministic stand-in you can swap for e.g. bge-small ONNX without touching
callers). Stored as float32 blobs; cosine search via numpy (sqlite-vec when
available). Forgetting curve: low-value noisy items decay and are pruned at
night (biological principle: forget noise, reinforce useful patterns).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("kde-agent.memory")


# --------------------------------------------------------------------------
# Frozen embedder (swap-in point for a real ONNX text encoder)
# --------------------------------------------------------------------------
class HashEmbedder:
    """Deterministic bag-of-ngrams -> fixed dim projection. Frozen, tiny,
    CPU-cheap; good enough to prove the RAG pipeline. Replace `embed()` with
    an ONNX Runtime call for production quality retrieval."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        text = unicodedata.normalize("NFKC", text.lower())
        v = np.zeros(self.dim, dtype=np.float32)
        toks = re.findall(r"[\wà-ÿ]+[^\w\s]?|[^\w\s]", text)
        for i in range(len(toks)):
            for n in (1, 2):
                if i + n <= len(toks):
                    g = " ".join(toks[i:i + n])
                    h = hashlib.blake2b(g.encode(), digest_size=8).digest()
                    idx = int.from_bytes(h[:4], "little") % self.dim
                    sign = 1.0 if h[4] & 1 else -1.0
                    v[idx] += sign
        norm = np.linalg.norm(v)
        return v / norm if norm else v


def _blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


@dataclass
class Hit:
    id: int
    collection: str
    text: str
    meta: dict
    score: float
    value: float


class Memory:
    def __init__(self, db_path: Path, dim: int = 256):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path))
        self.db.execute("""CREATE TABLE IF NOT EXISTS items(
            id INTEGER PRIMARY KEY, collection TEXT, kind TEXT,
            text TEXT, meta TEXT, vec BLOB, value REAL DEFAULT 1.0,
            hits INTEGER DEFAULT 0, ts REAL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_coll ON items(collection)")
        self.embedder = HashEmbedder(dim)
        self._cache: list[Hit] | None = None

    # ---- write ---------------------------------------------------------
    def add(self, collection: str, text: str, kind: str = "",
            meta: dict | None = None, value: float = 1.0) -> int:
        cur = self.db.execute(
            "INSERT INTO items(collection,kind,text,meta,vec,value,ts) VALUES(?,?,?,?,?,?,?)",
            (collection, kind, text, json.dumps(meta or {}, ensure_ascii=False),
             _blob(self.embedder.embed(text)), value, time.time()))
        self._cache = None
        self.db.commit()
        return cur.lastrowid

    def episode(self, kind: str, text: str, meta: dict | None = None,
                value: float = 1.0) -> int:
        return self.add("episodes", text, kind, meta, value)

    def preference(self, pattern: str, meta: dict, value: float = 2.0) -> int:
        return self.add("prefs", pattern, "pattern", meta, value=value)

    # ---- read -----------------------------------------------------------
    def _all(self) -> list[Hit]:
        if self._cache is None:
            rows = self.db.execute(
                "SELECT id,collection,text,meta,vec,value,hits FROM items").fetchall()
            self._cache = [Hit(r[0], r[1], r[2], json.loads(r[3]), 0.0, r[5]) for r in rows]
            self._vecs = np.stack([_vec(r[4]) for r in rows]) if rows else np.zeros((0, self.embedder.dim), np.float32)
        return self._cache

    def search(self, query: str,
               collections: tuple[str, ...] = ("docs", "episodes", "prefs", "cards"),
               top_k: int = 5, min_value: float = 0.0) -> list[Hit]:
        hits = self._all()
        if not hits:
            return []
        q = self.embedder.embed(query)
        sims = self._vecs @ q  # normalized vectors -> cosine
        order = np.argsort(-sims)
        out = []
        for i in order:
            h = hits[int(i)]
            if h.collection in collections and h.value >= min_value:
                h.score = float(sims[int(i)])
                out.append(h)
                if len(out) >= top_k:
                    break
        self.db.executemany("UPDATE items SET hits=hits+1 WHERE id=?",
                            [(h.id,) for h in out])
        self.db.commit()
        return out

    def recent_episodes(self, limit: int = 30) -> list[Hit]:
        rows = self.db.execute(
            "SELECT id,meta,text,value FROM episodes_view ORDER BY ts DESC LIMIT ?",
            (limit,)).fetchall() if False else self.db.execute(
            "SELECT id,meta,text,value,ts FROM items WHERE collection='episodes' "
            "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [Hit(r[0], "episodes", r[2], json.loads(r[1]), 1.0, r[3]) for r in rows]

    # ---- maintenance (nightly consolidation) ----------------------------
    def reinforce(self, hit_id: int, delta: float = 0.2) -> None:
        self.db.execute("UPDATE items SET value=value+? WHERE id=?", (delta, hit_id))
        self.db.commit()
        self._cache = None

    def decay_and_prune(self, floor: float = 0.05) -> int:
        """Ebbinghaus-style forgetting: unused/noisy items sink below floor."""
        now = time.time()
        self.db.execute(
            "UPDATE items SET value = value * EXP(-(? - ts)/7776000.0) WHERE collection!='prefs'",
            (now,))
        cur = self.db.execute("DELETE FROM items WHERE value < ? AND collection='episodes'",
                              (floor,))
        self.db.commit()
        self._cache = None
        return cur.rowcount

    def export_day(self) -> list[dict]:
        """Episodes of the last day. High-value items (errors, initiative
        outcomes) stay eligible for consolidation for 3 days — a single session
        may not fill the buffer."""
        rows = self.db.execute(
            "SELECT id,kind,text,meta,value FROM items WHERE collection='episodes' "
            "AND (ts > ? OR value >= 1.2) ORDER BY ts", (time.time() - 86400,)).fetchall()
        return [{"id": r[0], "kind": r[1], "text": r[2],
                 "meta": json.loads(r[3]), "value": r[4]} for r in rows]

    def close(self) -> None:
        self.db.close()
