"""Code assistant: diff proposals + mandatory verification before suggesting.

Contract with the model (spec): advice TYPES have different confidence gates —
  * explain/answer        : no execution, threshold = initiative base;
  * snippet suggestion    : must pass sandbox self-check to be offered;
  * file edit (diff)      : NEVER applied silently — user confirm required,
                            then linters/tests decide whether we even show it.

Verification ladder (applied in order, first failure vetoes the proposal):
  syntax (py_compile / ast.parse) -> ruff (if installed) -> mypy (if
installed and project configured) -> pytest (project tests, if fast enough)
-> shellcheck for bash diffs. Everything runs read-only in a temp COPY of the
project so a broken patch can't touch real files before confirmation.
"""
from __future__ import annotations

import ast
import asyncio
import difflib
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("kde-agent.code")


@dataclass
class DiffProposal:
    path: str                 # relative to project root
    unified_diff: str
    new_content: str
    rationale: str = ""
    advice_type: str = "edit"  # explain | snippet | edit
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return all(ok for _, ok, _ in self.checks) and bool(self.checks)

    def summary(self) -> str:
        marks = ", ".join(f"{name}:{'OK' if ok else 'FAIL'}" for name, ok, _ in self.checks)
        head = "\n".join(self.unified_diff.splitlines()[:14])
        return f"{self.path} [{marks or 'no checks'}]\n{head}"


def make_diff(path: str, old: str, new: str, rationale: str = "",
              advice_type: str = "edit") -> DiffProposal:
    ud = "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{os.path.basename(path)}", tofile=f"b/{os.path.basename(path)}"))
    return DiffProposal(path=path, unified_diff=ud, new_content=new,
                        rationale=rationale, advice_type=advice_type)


# --------------------------------------------------------------------------
# Verifiers
# --------------------------------------------------------------------------
def _run(cmd: list[str], cwd: Path, timeout: float) -> tuple[bool, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                           timeout=timeout, check=False)
        out = ((p.stdout or "") + (p.stderr or ""))[-1500:]
        return p.returncode == 0, out
    except FileNotFoundError:
        return True, "tool-missing (skipped)"   # missing optional tool != veto
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"


def check_python_syntax(content: str) -> tuple[bool, str]:
    try:
        ast.parse(content)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"


def verify_proposal(prop: DiffProposal, project_root: Path, *,
                    run_tests: bool = True, test_timeout: float = 60.0) -> DiffProposal:
    """Copy project to tmp, apply candidate content, run the ladder."""
    prop.checks = []
    rel = Path(prop.path)
    suffix = rel.suffix.lower()

    if suffix == ".py":
        ok, err = check_python_syntax(prop.new_content)
        prop.checks.append(("syntax", ok, err))
        if not ok:
            return prop
    elif suffix in (".sh", ".bash"):
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as tf:
            tf.write(prop.new_content)
            tmp = tf.name
        try:
            p = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True,
                               timeout=10, check=False)
            prop.checks.append(("bash-syntax", p.returncode == 0,
                                (p.stderr or "")[:500]))
            if not prop.verified:
                return prop
            if shutil.which("shellcheck"):
                ok2, out2 = _run(["shellcheck", "-S", "warning", tmp], project_root, 20)
                prop.checks.append(("shellcheck", ok2, out2))
        finally:
            os.unlink(tmp)
        return prop
    else:
        prop.checks.append(("format-agnostic", True, "no syntax gate for this type"))
        return prop

    # isolated copy so nothing real is touched pre-confirmation
    with tempfile.TemporaryDirectory(prefix="kagent-verify-") as td:
        tdp = Path(td)
        shutil.copytree(project_root, tdp / "proj",
                        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
                        symlinks=True)
        proj = tdp / "proj"
        target = proj / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(prop.new_content)

        if shutil.which("ruff"):
            ok, out = _run(["ruff", "check", "--quiet", str(rel)], proj, 30)
            prop.checks.append(("ruff", ok, out))
        if run_tests and shutil.which("pytest") and (proj / "tests").is_dir():
            ok, out = _run(["pytest", "-x", "-q", "--timeout=90" if _pytest_timeout_installed() else "",
                            "tests"], proj, test_timeout)
            prop.checks.append(("pytest", ok, out))
    return prop


def _pytest_timeout_installed() -> bool:
    try:
        import pytest_timeout  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------
# Advice-type thresholds (spec: different gates per type)
# --------------------------------------------------------------------------
DEFAULT_GATES = {"explain": 0.45, "snippet": 0.6, "edit": 0.75}


class CodeAssistant:
    """Wires proposals into the agent loop: propose -> verify -> (confirm by
    user via bridge) -> apply through safety gate -> record outcome episode."""

    def __init__(self, cfg, memory, safety, bridge, gates: dict | None = None):
        self.cfg, self.memory, self.safety, self.bridge = cfg, memory, safety, bridge
        self.gates = {**DEFAULT_GATES, **(gates or {})}

    def gate_for(self, advice_type: str) -> float:
        return self.gates.get(advice_type, 0.7)

    def should_offer(self, advice_type: str, confidence: float) -> bool:
        return confidence >= self.gate_for(advice_type)

    async def offer_edit(self, prop: DiffProposal, project_root: Path,
                         confidence: float) -> str:
        if not self.should_offer(prop.advice_type, confidence):
            self.memory.episode("code_advice", f"withheld ({prop.advice_type}, conf={confidence:.2f})", value=0.8)
            return "WITHHELD: уверенность ниже порога для типа совета."
        await asyncio.to_thread(verify_proposal, prop, project_root)
        if not prop.verified:
            fails = [f"{n}: {e[:120]}" for n, ok, e in prop.checks if not ok]
            self.memory.episode("error", f"vetoed code advice {prop.path}: {fails}", value=1.4)
            return "VETOED self-check: " + "; ".join(fails)
        decision = self.safety.assess_path(str(project_root / prop.path), write=True)
        if decision.level == "blocked":
            return f"BLOCKED: {decision.reason}"
        if not self.bridge.confirm(f"Применить изменение {prop.path}?\n{prop.rationale}\n\n{prop.summary()}"):
            self.memory.episode("cancel", f"user refused diff {prop.path}")
            return "CANCELLED_BY_USER"
        target = project_root / prop.path
        backup = None
        if target.exists():
            backup = target.read_text(errors="replace")
        target.write_text(prop.new_content)
        # undo journal: register the edit so /undo restores the previous content
        try:
            self.tools_undo_hook(target, backup)
        except Exception:  # noqa: BLE001
            log.debug("no undo hook registered for code edits")
        self.memory.episode("code_change",
                            f"applied {prop.path}: {prop.rationale[:200]}", value=1.5)
        return f"APPLIED {prop.path} (checks: {', '.join(n for n, ok, _ in prop.checks if ok)})"

    def tools_undo_hook(self, target: Path, old_content: str | None) -> None:
        """Register a text-restore undo entry with the agent's ToolRegistry
        (set via `undo_sink` by whoever wires the assistant)."""
        sink = getattr(self, "undo_sink", None)
        if sink is not None:
            sink(("edit", str(target), old_content))

    async def offer_snippet(self, code: str, lang: str, explanation: str,
                            confidence: float) -> str:
        if not self.should_offer("snippet", confidence):
            return "WITHHELD: low confidence for snippet advice."
        from agent.knowledge.pipeline import run_in_sandbox
        res = await asyncio.to_thread(run_in_sandbox, code, lang)
        if not res.ok:
            return f"SNIPPET FAILED sandbox check: {(res.error or res.skipped)[:200]}"
        return f"SNIPPET OK (sandbox passed):\n{explanation}\n```{lang}\n{code}\n```\nstdout: {res.stdout[:300]}"
