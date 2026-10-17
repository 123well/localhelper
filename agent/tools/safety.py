"""Risk gating for every tool call. Blacklist blocks, prefixes confirm,
undo journal for destructive ops. No elevated privileges ever (no sudo -S)."""
from __future__ import annotations

import fnmatch
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RiskDecision:
    level: str          # safe | needs_confirm | blocked
    reason: str = ""
    safer: str | None = None


DANGEROUS_PATTERNS = [r"\brm\s+(-[a-zA-Z]*\s+)*(/|/home\b|~)(\s|$)", r">\s*/etc/",
                      r"\bchmod\s+-R\b", r"\bchown\s+-R\b", r"\bgit\s+push\s+.*--force",
                      r"\bgit\s+reset\s+--hard\b", r"\bkill(all)?\b", r"\bshutdown\b",
                      r"\breboot\b"]


class SafetyGate:
    HOME = Path.home()

    def __init__(self, cfg):
        self.cfg = cfg  # SafetyConfig

    # ---- operating mode (spec: safe / confirm_all / read_only) ------------
    @property
    def mode(self) -> str:
        return getattr(self.cfg, "mode", "normal")

    def set_mode(self, mode: str) -> RiskDecision | None:
        if mode not in ("normal", "safe", "confirm_all", "read_only"):
            return RiskDecision("blocked", f"неизвестный режим: {mode}")
        self.cfg.mode = mode
        return None

    def assess_tool_call(self, tool: str, mutating: bool) -> RiskDecision:
        """Mode policy applied BEFORE any per-tool risk assessment.
        Read-only tools (search/read/list/screen_state) stay available even in
        safe mode — the agent must still answer and propose, just not act."""
        m = self.mode
        if m == "safe" and mutating:
            return RiskDecision("blocked",
                                "безопасный режим: выполнение/запись запрещены, только анализ и предложения")
        if m == "read_only" and mutating:
            return RiskDecision("blocked", "режим «только чтение»: изменения запрещены")
        if m == "confirm_all":
            return RiskDecision("needs_confirm", "режим подтверждения всех действий")
        return RiskDecision("safe")

    # ---- command risk ----------------------------------------------------
    _SHELL_META = re.compile(r"[|;&<>`$*]|\$\(")

    def assess_command(self, cmd: str) -> RiskDecision:
        c = cmd.strip()
        low = c.lower()
        if any(bl in low for bl in self.cfg.blacklist):
            return RiskDecision("blocked", "команда в чёрном списке")
        if any(re.search(p, low) for p in DANGEROUS_PATTERNS):
            safer = None
            m = re.match(r"^rm\s+(-[rf]*\s+)*(\S.*)$", c)
            if m and shutil.which("trash-put"):
                safer = f"trash-put {m.group(2)}"
            return RiskDecision("needs_confirm", "деструктивная операция", safer)
        if any(low.startswith(p.strip()) or p in low for p in self.cfg.require_confirm_prefixes):
            return RiskDecision("needs_confirm", f"префикс «{next((p for p in self.cfg.require_confirm_prefixes if p.strip() in low), '')}» требует подтверждения")
        # Whitelist is a shortcut ONLY for plain single commands. A composite
        # shell line that merely STARTS with a whitelisted word ("ls; rm -rf ~",
        # "echo x > /home/user/.ssh/rc") must never inherit the exemption —
        # otherwise prefix matching silently legalizes arbitrary payloads run
        # through shell=True. (Previously whitelist returned "safe" before any
        # such check and its result was indistinguishable from the default.)
        if self._SHELL_META.search(c):
            return RiskDecision("needs_confirm",
                                "shell-мета-символы (; | & > < $ *) требуют подтверждения")
        if any(fnmatch.fnmatch(low, w) or (low.split() and low.split()[0] == w)
               for w in self.cfg.whitelist):
            return RiskDecision("safe")
        return RiskDecision("safe")

    # ---- file sandbox ------------------------------------------------------
    def assess_path(self, path: str, write: bool = False) -> RiskDecision:
        p = Path(os.path.expanduser(path)).resolve()
        protected = [Path("/etc"), Path("/boot"), Path("/usr"), Path("/var"), Path("/sys"), Path("/proc")]
        if write:
            if any(p == pr or pr in p.parents for pr in protected):
                return RiskDecision("blocked", "системный путь только через явное подтверждение администратора")
            if not (p == self.HOME or self.HOME in p.parents or p == Path.cwd()
                    or Path.cwd() in p.parents):
                return RiskDecision("needs_confirm", f"запись вне домашней директории: {p}")
        return RiskDecision("safe")

    # ---- undo journal --------------------------------------------------------
    def snapshot_for(self, cmd: str) -> dict | None:
        """Before executing rm/mv/chmod-like ops, record enough to undo."""
        if not self.cfg.undo_enabled:
            return None
        try:
            parts = shlex.split(cmd)
        except ValueError:
            return None
        if not parts:
            return None
        op = os.path.basename(parts[0])
        targets = [a for a in parts[1:] if not a.startswith("-")]
        if op == "rm" and targets:
            files = []
            for t in targets:
                tp = Path(os.path.expanduser(t))
                if tp.is_file():
                    files.append(str(tp))
                elif tp.is_dir():
                    files.extend(str(x) for x in tp.rglob("*") if x.is_file())
            if len(files) > self.cfg.max_files_in_one_op:
                return {"op": "skip_undo", "reason": "too many files"}
            return {"op": "backup", "files": files}
        if op == "mv" and len(targets) >= 2:
            return {"op": "move", "src": targets[0], "dst": targets[-1]}
        if op in ("chmod", "chown") and targets:
            return {"op": "stat", "files": targets}
        return None

    def perform_backup(self, snap: dict) -> Path | None:
        """Copy targeted files into data_dir/undo/<ts>/ so undo is possible."""
        if snap.get("op") != "backup":
            return None
        ts = int(__import__("time").time())
        base = Path(os.environ.get("XDG_DATA_HOME", self.HOME / ".local/share"))
        dest = base / "kde-agent" / "undo" / str(ts)
        dest.mkdir(parents=True, exist_ok=True)
        for f in snap["files"]:
            src = Path(f)
            if src.exists():
                rel = src.relative_to(src.anchor) if src.is_absolute() else src
                tgt = dest / str(rel).lstrip("/")
                tgt.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(src, tgt)
                except OSError:
                    pass
        return dest

    def undo(self, snap: dict, backup_dir: Path | None) -> str:
        op = snap.get("op")
        try:
            if op == "backup" and backup_dir and backup_dir.exists():
                n = 0
                for f in backup_dir.rglob("*"):
                    if f.is_file():
                        orig = Path("/") / f.relative_to(backup_dir)
                        orig.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, orig)
                        n += 1
                return f"восстановлено файлов: {n}"
            if op == "move":
                shutil.move(snap["dst"], snap["src"])
                return f"возврат {snap['dst']} -> {snap['src']}"
        except Exception as e:  # noqa: BLE001
            return f"undo не удался: {e}"
        return "нечего отменять"
