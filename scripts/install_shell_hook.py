"""Install the shell preexec hook so every typed command reaches the agent.

Writes a small snippet into ~/.bashrc / ~/.zshrc (idempotent, marked block)
that POSTs {"cmd": "..."} as one JSON line to $XDG_RUNTIME_DIR/kde-agent.sock
before each command executes. Zero polling: push-only, ~0 energy when idle.

Usage:  python scripts/install_shell_hook.sh is bash; here it's python:
        python scripts/install_shell_hook.py [--zsh|--bash] [--remove]
"""
from __future__ import annotations

import sys
from pathlib import Path

MARK = "# >>> kde-agent shell hook >>>"
END = "# <<< kde-agent shell hook <<<"

BASH_BLOCK = f"""{MARK}
_kde_agent_preexec() {{
  printf '{{"cmd": "%s"}}\\n' "$(printf '%s' "$1" | sed 's/"/\\\\"/g')" \\
    | timeout 1 socat - UNIX-CONNECT:"$XDG_RUNTIME_DIR/kde-agent.sock" 2>/dev/null || true
}}
if command -v socat >/dev/null && [ -S "$XDG_RUNTIME_DIR/kde-agent.sock" ]; then
  trap '_kde_agent_preexec "$BASH_COMMAND"' DEBUG
fi
{END}
"""

ZSH_BLOCK = f"""{MARK}
kde_agent_preexec() {{
  local esc=${{(1)//\\"/\\\\\\"}}
  printf '{{"cmd": "%s"}}\\n' "$esc" |
    timeout 1 socat - UNIX-CONNECT:"$XDG_RUNTIME_DIR/kde-agent.sock" 2>/dev/null || true
}}
if command -v socat >/dev/null && [ -S "$XDG_RUNTIME_DIR/kde-agent.sock" ]; then
  autoload -Uz add-zsh-hook && add-zsh-hook preexec kde_agent_preexec
fi
{END}
"""


def strip_block(text: str) -> str:
    out, skip = [], False
    for ln in text.splitlines(keepends=True):
        if MARK in ln:
            skip = True
        if not skip:
            out.append(ln)
        if END in ln:
            skip = False
    return "".join(out)


def main(argv: list[str]) -> int:
    remove = "--remove" in argv
    shell = "zsh" if "--zsh" in argv else "bash"
    rc = Path.home() / f".{shell}rc"
    text = rc.read_text() if rc.exists() else ""
    cleaned = strip_block(text)
    if remove:
        rc.write_text(cleaned)
        print(f"hook removed from {rc}")
        return 0
    block = ZSH_BLOCK if shell == "zsh" else BASH_BLOCK
    rc.write_text(cleaned + "\n" + block)
    print(f"hook installed into {rc} (requires socat: pacman -S socat)")
    print("open a new terminal or `source` the rc file")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
