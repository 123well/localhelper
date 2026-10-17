"""Background study pipeline: source queue -> segmentation -> cards.

Design goals (spec): priority queue of sources, energy gates, privacy
masking before anything touches the DB, example sandboxing, self-check of
freshly built cards, and a per-project profile tag so retrieval can filter.

Energy gate: heavy indexing only when allowed — plugged-in OR idle OR night
batch window; on battery with active user it defers (queue persists in sqlite
so restarts don't lose work). Sources carry priority: user-requested topics >
project files > docs dirs > opportunistic web exports.
"""
from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from agent.knowledge.cards import CardStore
from agent.knowledge.segmentation import segment_document

log = logging.getLogger("kde-agent.study")


# --------------------------------------------------------------------------
# Privacy masking: secrets NEVER enter memory (spec requirement)
# --------------------------------------------------------------------------
_SECRET_PATTERNS = [
    re.compile(r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)\b\s*[:=]\s*\S+"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),                       # GitHub PAT (classic)
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),               # GitHub fine-grained
    re.compile(r"\bx-[a-f0-9]{32,}\b"),                             # AWS secret-ish hex
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                            # AWS access key id
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),              # PEM marker
    re.compile(r"\b[0-9]{13,16}\b"),                                # card-like numbers
]


def mask_secrets(text: str) -> tuple[str, int]:
    """Return (masked_text, n_masks). Applied to EVERY segment before add()."""
    n = 0

    def sub(rx: re.Pattern, s: str) -> str:
        nonlocal n
        out, last = [], 0
        for m in rx.finditer(s):
            out.append(s[last:m.start()])
            out.append("[REDACTED]")
            last = m.end()
            n += 1
        out.append(s[last:])
        return "".join(out)

    for rx in _SECRET_PATTERNS:
        text = sub(rx, text)
    return text, n


# --------------------------------------------------------------------------
# Sandbox for doc examples (never run untrusted snippets unsandboxed)
# --------------------------------------------------------------------------
_SANDBOX_FORBIDDEN = re.compile(
    r"(rm\s+-rf|mkfs|\bdd\b|:\(\)|>/dev/sd|chmod\s+-R|curl[^|]*\|\s*(ba)?sh|"
    r"wget[^|]*\|\s*(ba)?sh|sudo|shutdown|reboot|systemctl)")


@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    error: str = ""
    skipped: str = ""


def run_in_sandbox(code: str, lang: str = "python", timeout: float = 5.0,
                   workdir: Path | None = None) -> SandboxResult:
    """Stage-1 sandbox: forbidden-pattern veto + rlimits via `ulimit` + tmp cwd
    + no network tools on PATH assumption. Production swap-in point: systemd-run
    --scope -p PrivateNetwork=yes -p MemoryMax=256M (documented, not required)."""
    if _SANDBOX_FORBIDDEN.search(code):
        return SandboxResult(False, skipped="vetoed by sandbox blacklist")
    wd = Path(workdir or Path("/tmp")).resolve()
    wd.mkdir(parents=True, exist_ok=True)
    try:
        if lang == "python":
            cmd = ["bash", "-c",
                   f"cd {shlex.quote(str(wd))} && ulimit -t 5 -v 500000 -f 20000 -u 128 && "
                   f"timeout {int(timeout)} python -I -S -c {shlex.quote(code)}"]
        elif lang == "bash":
            cmd = ["bash", "-c",
                   f"cd {shlex.quote(str(wd))} && ulimit -t 5 -v 500000 -f 20000 -u 128 && "
                   f"timeout {int(timeout)} bash -c {shlex.quote(code)}"]
        else:
            return SandboxResult(False, skipped=f"unsupported lang {lang}")
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 2)
        if p.returncode == 0:
            return SandboxResult(True, stdout=p.stdout[:2000])
        return SandboxResult(False, stdout=p.stdout[:1000],
                             error=(p.stderr or "")[:1000])
    except (subprocess.TimeoutExpired, OSError) as e:
        return SandboxResult(False, error=str(e)[:200])


# --------------------------------------------------------------------------
# Source queue (persistent, priority-sorted)
# --------------------------------------------------------------------------
@dataclass
class Source:
    path: str
    priority: int          # lower number = studied first
    project: str = ""      # profile tag for retrieval filtering
    status: str = "queued" # queued | done | failed
    error: str = ""


class StudyQueue:
    """sqlite-persisted so deferred work survives restarts.

    check_same_thread=False + RLock: the daemon's idle study loop and the
    control plane (`ctl study add <path>`) touch this DB from different
    threads; serialize every statement through _exec/_commit."""

    def __init__(self, db_path: Path):
        import sqlite3 as _sql
        import threading
        self.db = _sql.connect(str(db_path), check_same_thread=False)
        self.lock = threading.RLock()
        self._exec("""CREATE TABLE IF NOT EXISTS study_queue(
            path TEXT PRIMARY KEY, priority INTEGER, project TEXT,
            status TEXT DEFAULT 'queued', error TEXT DEFAULT '', ts REAL)""")
        self._commit()

    def _exec(self, sql, params=()):
        with self.lock:
            return self.db.execute(sql, params)

    def _commit(self):
        with self.lock:
            self.db.commit()

    def enqueue(self, src: Source) -> None:
        self._exec(
            "INSERT OR REPLACE INTO study_queue(path,priority,project,status,error,ts)"
            " VALUES(?,?,?,?,?,?)",
            (src.path, src.priority, src.project, src.status, src.error, time.time()))
        self._commit()

    def next(self) -> Source | None:
        row = self._exec(
            "SELECT path,priority,project FROM study_queue WHERE status='queued'"
            " ORDER BY priority ASC, ts ASC LIMIT 1").fetchone()
        if not row:
            return None
        return Source(path=row[0], priority=row[1], project=row[2])

    def mark(self, path: str, status: str, error: str = "") -> None:
        self._exec("UPDATE study_queue SET status=?,error=? WHERE path=?",
                   (status, error, path))
        self._commit()

    def stats(self) -> dict:
        rows = self._exec("SELECT status,COUNT(*) FROM study_queue GROUP BY status").fetchall()
        return dict(rows)

    def close(self) -> None:
        self.db.close()


# --------------------------------------------------------------------------
# Energy gate
# --------------------------------------------------------------------------
class EnergyGate:
    """Battery/idle/load aware. Inputs are cheap polls done elsewhere; this is
    pure policy. On Arch/KDE real signals come from UPower D-Bus + KIdleTime;
    here they are injectable callables for testability."""

    def __init__(self, on_battery=lambda: False, idle_sec=lambda: 0.0,
                 load_ratio=lambda: 0.0, night_batch_hours=(2, 6)):
        self.on_battery = on_battery
        self.idle_sec = idle_sec
        self.load_ratio = load_ratio
        self.night_batch_hours = night_batch_hours

    def allow_heavy_work(self) -> tuple[bool, str]:
        hour = time.localtime().tm_hour
        night = self.night_batch_hours[0] <= hour < self.night_batch_hours[1]
        if self.load_ratio() > 0.8 and not night:
            return False, "system busy"
        if self.on_battery():
            if self.idle_sec() >= 120:
                return True, "battery but idle"
            return False, "battery + active user"
        if self.idle_sec() >= 30 or night:
            return True, "plugged and idle/night"
        return False, "plugged but user active"


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
class StudyPipeline:
    def __init__(self, cards: CardStore, queue: StudyQueue, gate: EnergyGate,
                 max_segments_per_source: int = 40):
        self.cards = cards
        self.queue = queue
        self.gate = gate
        self.max_segments = max_segments_per_source
        self.last_report: dict = {}

    def ingest_dir(self, root: str, project: str = "", priority: int = 5,
                   patterns=("*/*.md", "*.md", "*.rst", "*.py", "*.txt")) -> int:
        rp = Path(root).expanduser()
        if not rp.exists():
            log.warning("ingest: %s missing", rp)
            return 0
        n = 0
        for pat in patterns:
            for p in sorted(rp.rglob(pat)):
                if p.is_file() and ".git" not in p.parts:
                    self.queue.enqueue(Source(path=str(p), priority=priority, project=project))
                    n += 1
        return n

    def process_one(self, src: Source) -> dict:
        report = {"path": src.path, "segments": 0, "cards": 0, "masked": 0,
                  "sandbox_checked": 0, "conflicts": 0, "skipped": ""}
        try:
            content = Path(src.path).read_text(errors="replace")
        except OSError as e:
            self.queue.mark(src.path, "failed", str(e))
            report["skipped"] = f"read error: {e}"
            return report
        segs = segment_document(src.path, content)[: self.max_segments]
        report["segments"] = len(segs)
        for seg in segs:
            masked, nm = mask_secrets(seg.text)
            report["masked"] += nm
            meta = {"source": seg.source, "lines": [seg.start_line, seg.end_line],
                    "kind": seg.kind, "tags": seg.tags, "project": src.project}
            examples = _extract_examples(masked)
            sandbox_err = ""
            for ex in examples[:1]:
                res = run_in_sandbox(ex, lang=_guess_lang(seg.source))
                report["sandbox_checked"] += 1
                if not res.ok and not res.skipped:
                    sandbox_err = res.error[:120]
                    break
            if sandbox_err:
                meta["example_status"] = f"sandbox_fail: {sandbox_err}"
            existing_conflict = None
            # CardStore.add() calls the parameter `phrases` — passing
            # `search_phrases=` crashed every study batch (TypeError).
            card = self.cards.add(type=_guess_type(seg, src), summary=seg.title,
                                  body=masked, examples=examples,
                                  phrases=seg.search_phrases(), meta=meta)
            if card.meta.get("conflict_with"):
                report["conflicts"] += 1
            report["cards"] += 1
        self.queue.mark(src.path, "done")
        self.last_report = report
        return report

    def run_pending(self, budget_sec: float = 20.0) -> list[dict]:
        """Respect the energy gate; drain queue within wall-clock budget."""
        allowed, why = self.gate.allow_heavy_work()
        if not allowed:
            log.info("study deferred: %s", why)
            return []
        reports, t0 = [], time.time()
        while time.time() - t0 < budget_sec:
            src = self.queue.next()
            if src is None:
                break
            reports.append(self.process_one(src))
        self.last_report = {"ran": len(reports), "reason": why}
        return reports


def _guess_lang(path: str) -> str:
    return "python" if path.endswith(".py") else "bash"


def _guess_type(seg, src: Source) -> str:
    low = (seg.title + " " + seg.text[:200]).lower()
    if seg.kind == "symbol":
        return "api"
    if any(k in low for k in ("команда", "command", "$ ", "pacman", "systemctl")):
        return "command"
    if "warn" in low or "опасно" in low or "danger" in low:
        return "safety_rule"
    return "concept"


_FENCE = re.compile(r"```(?:[\w+-]*)?\n(.*?)```", re.S)
_PROMPT_LINE = re.compile(r"^[$%]\s+(.+)$", re.M)


def _extract_examples(text: str) -> list[str]:
    ex = [m.strip() for m in _FENCE.findall(text) if m.strip()]
    ex += [m.strip() for m in _PROMPT_LINE.findall(text)][:4]
    return ex[:6]
