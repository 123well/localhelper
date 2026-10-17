"""Event monitor (Component 4): event-driven perception, zero screenshot polling.

Sources (all async, all cheap):
  * stdin / KRunner runner — explicit user requests
  * shell hook             — ~/.bashrc/.zshrc preexec sends commands to our socket
                             ("user typed a command" event, no polling)
  * kdotool getactivewindow change — announced by KDE when focus changes via
                             org.kde.KWin /Activities; prototype uses an
                             internal notify from tools + optional watcher
  * idle                   — ScreenSaver GetIdleSeconds checked ONLY on timer
                             that fires once per minute (near-zero cost)
  * app started/closed     — KWin windowNew/windowRemoved signals (dbus) or manual
Energy rule: when nothing happens the process sleeps in epoll; background draw
is dominated by the NPU idling at <1W, not by us.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

log = logging.getLogger("kde-agent.events")


@dataclass
class Event:
    kind: str          # user_request | user_command | command_result | error |
                       # window_changed | idle | action_cancelled | help_requested
    payload: dict
    ts: float = field(default_factory=time.time)


Handler = Callable[[Event], Awaitable[None] | None]


class EventBus:
    def __init__(self):
        self.handlers: dict[str, list[Handler]] = {}
        self.queue: asyncio.Queue[Event] = asyncio.Queue()

    def subscribe(self, kind: str, h: Handler) -> None:
        self.handlers.setdefault(kind, []).append(h)

    def emit(self, kind: str, **payload) -> None:
        self.queue.put_nowait(Event(kind, payload))
        log.debug("event %s", kind)

    async def run(self) -> None:
        while True:
            ev = await self.queue.get()
            for h in self.handlers.get(ev.kind, []) + self.handlers.get("*", []):
                try:
                    r = h(ev)
                    if asyncio.iscoroutine(r):
                        await r
                except Exception as e:  # noqa: BLE001
                    log.exception("handler error for %s: %s", ev.kind, e)


class StdinSource:
    """Terminal interface: lines become user_request events."""

    def __init__(self, bus: EventBus):
        self.bus = bus

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        try:
            protocol = asyncio.StreamReaderProtocol(reader)
            await loop.connect_read_pipe(lambda: protocol, __import__("sys").stdin)
        except (OSError, ValueError):        # not a tty (tests)
            return
        while True:
            line = (await reader.readline()).decode(errors="replace").strip()
            if not line:
                continue
            self.bus.emit("user_request", text=line)


class ShellSocketSource:
    """Listens on $XDG_RUNTIME_DIR/kde-agent.sock for shell-preexec events:
    one JSON line {'cmd': ...} per executed command (hook installed by
    scripts/install_shell_hook.sh). Pure push, no polling."""

    def __init__(self, bus: EventBus):
        self.bus = bus

    async def run(self) -> None:
        import os
        from pathlib import Path
        sock = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "kde-agent.sock"
        sock.unlink(missing_ok=True)
        try:
            server = await asyncio.start_unix_server(self._cb, path=str(sock))
        except OSError as e:
            log.debug("shell socket unavailable: %s", e)
            return
        os.chmod(sock, 0o600)
        async with server:
            await server.serve_forever()

    async def _cb(self, reader: asyncio.StreamReader, writer) -> None:
        try:
            data = await asyncio.wait_for(reader.readline(), 2)
            obj = json.loads(data.decode())
            if "cmd" in obj:
                self.bus.emit("user_command", **obj)
        except Exception:  # noqa: BLE001
            pass
        finally:
            writer.close()


class IdleWatcher:
    """One cheap check per minute; emits 'idle' crossing configured threshold."""

    def __init__(self, bus: EventBus, bridge, threshold_sec: int):
        self.bus, self.bridge, self.thr = bus, bridge, threshold_sec
        self.fired_at = 0.0

    async def run(self) -> None:
        while True:
            await asyncio.sleep(60)          # 1 wake/min: negligible energy
            idle = self.bridge.idle_seconds()
            if idle >= self.thr and time.time() - self.fired_at > 1800:
                self.fired_at = time.time()
                self.bus.emit("idle", idle_sec=idle)
