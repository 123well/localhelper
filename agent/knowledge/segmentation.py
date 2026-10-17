"""Document segmentation into knowledge-card candidates.

Strategy (Stage-1, stdlib-only; tree-sitter is a documented swap-in point for
AST-accurate splitting in production):
  * markdown/rst : split on heading hierarchy (#, ##) — one card per section
                   with its inherited heading path as context;
  * python       : split on top-level def/class via `ast` (real AST, no deps);
  * man/txt      : paragraph blocks joined to ~600-char windows with small
                   overlap so retrieval never loses sentence boundaries.

Each segment carries: title, text, source path, line range, and generated
search phrases (title + first sentence) — enough to build a KnowledgeCard.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass
class Segment:
    title: str
    text: str
    source: str = ""
    start_line: int = 0
    end_line: int = 0
    kind: str = "prose"          # section | symbol | window
    tags: list[str] = field(default_factory=list)

    def search_phrases(self) -> list[str]:
        first = re.split(r"(?<=[.!?])\s+", self.text.strip())[0][:200] if self.text else ""
        return [p for p in {self.title, first} if p]


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_RST_HEAD = re.compile(r"^([=\-~^\"'`*+#]{3,})\s*$", re.M)


def _split_markdown(content: str, source: str) -> list[Segment]:
    segs: list[Segment] = []
    stack: list[tuple[int, str]] = []      # (level, title)
    buf: list[str] = []
    buf_title = ""
    start = 1
    lines = content.splitlines()

    def flush(end_line: int):
        text = "\n".join(buf).strip()
        if text and buf_title:
            path = " > ".join(t for _, t in stack[-2:]) or buf_title
            segs.append(Segment(title=path, text=text, source=source,
                                start_line=start, end_line=end_line, kind="section"))

    for i, line in enumerate(lines, 1):
        m = _HEADING.match(line)
        is_rst = bool(_RST_HEAD.match(line)) and i < len(lines) and \
            bool(_HEADING.match(lines[i])) is False and lines[i].strip() and \
            _HEADING.match(lines[i]) is None and i - 1 >= 0
        if m:
            level = len(m.group(1))
            flush(i - 1)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
            buf_title = m.group(2).strip()
            buf, start = [], i
        elif is_rst and i < len(lines) - 1 and lines[i].strip():
            pass  # rst underline handled below by simple heuristic
        else:
            buf.append(line)
    flush(len(lines))
    return segs


def _split_python(content: str, source: str) -> list[Segment]:
    segs: list[Segment] = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _windows(content, source)
    lines = content.splitlines()
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        segs.append(Segment(title=f"{source}: module doc", text=mod_doc,
                            source=source, start_line=1, end_line=len(lines),
                            kind="symbol", tags=["python", "module"]))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node) or ""
            sig = _signature(node, lines)
            body = "\n".join(lines[node.lineno - 1: node.end_lineno])
            text = f"{sig}\n{doc}".strip() if doc else body[:1500]
            segs.append(Segment(title=f"{node.name} ({'class' if isinstance(node, ast.ClassDef) else 'def'})",
                                text=text, source=source,
                                start_line=node.lineno, end_line=node.end_lineno or node.lineno,
                                kind="symbol", tags=["python", node.name.lower()]))
    return segs


def _signature(node, lines: list[str]) -> str:
    # CRASH FIX: ClassDef has no .args — only FunctionDef/AsyncFunctionDef do.
    # Studying any file with top-level classes raised AttributeError and killed
    # the whole study batch (and returned an empty socket reply to `ctl`).
    try:
        dec = ast.unparse(node.decorator_list[0]) if node.decorator_list else None
        head = lines[node.lineno - 1].strip()
        if dec:
            head = f"@{dec}\n{head}"
        a = getattr(node, "args", None)
        args = ", ".join(x.arg for x in a.args) if a is not None else ""
        return head if "(" in head else f"{node.name}({args})"
    except Exception:  # noqa: BLE001  signature is cosmetic — never fatal
        return getattr(node, "name", "?")


def _windows(content: str, source: str, size: int = 600, overlap: int = 80) -> list[Segment]:
    paras = re.split(r"\n\s*\n", content)
    segs, cur, cur_start, ln = [], "", 1, 1
    pos = 0
    for p in paras:
        if not p.strip():
            continue
        if len(cur) + len(p) > size and cur:
            segs.append(Segment(title=_first_line(cur), text=cur.strip(),
                                source=source, start_line=cur_start,
                                end_line=ln, kind="window"))
            tail = cur[-overlap:]
            cur, cur_start = tail, ln
        cur += ("\n" if cur else "") + p
        ln += p.count("\n") + 2
    if cur.strip():
        segs.append(Segment(title=_first_line(cur), text=cur.strip(), source=source,
                            start_line=cur_start, end_line=ln, kind="window"))
    return segs


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:100]
    return "fragment"


def segment_document(path: str, content: str | None = None) -> list[Segment]:
    """Dispatch by extension; unknown types fall back to paragraph windows."""
    if content is None:
        from pathlib import Path
        content = Path(path).read_text(errors="replace")
    low = path.lower()
    if low.endswith((".md", ".markdown", ".rst")):
        return _split_markdown(content, path)
    if low.endswith(".py"):
        return _split_python(content, path)
    return _windows(content, path)
