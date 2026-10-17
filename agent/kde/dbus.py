"""KDE Plasma 6 integration via D-Bus (pure-Python dbus-next; qdbus fallback).

* org.freedesktop.Notifications      — notifications with action callbacks
* KDialog / zenzen Qt6 dialogs       — confirmations (blocking helper)
* kdotool (AUR)                      — GUI automation: click/type/window info
* org.kde.KWin / org.kde.plasmashell — active window, widget updates
* org.kde.kwalletd5/6                — secrets (never plaintext on disk)
* idle time                          — org.freedesktop.ScreenSaver.GetIdleSeconds
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("kde-agent.kde")


def _has_display() -> bool:
    return bool(shutil.which("kdotool")) and bool(
        __import__("os").environ.get("DISPLAY") or __import__("os").environ.get("WAYLAND_DISPLAY"))


class KDEBridge:
    """Facade over KDE services. Every method degrades gracefully to a log
    line when the desktop is absent (headless dev boxes / CI)."""

    def __init__(self):
        self.bus = None
        self._notif = None
        # `available` is read by the output-channel router and diagnostics;
        # it must exist BEFORE connect() is ever called (regression: it was
        # never assigned anywhere, so Channels._notify raised AttributeError
        # on every notification attempt in headless mode).
        self.available = False
        self.reply_hooks: dict[str, Callable[[str], None]] = {}
        self._id_to_key: dict[int, str] = {}     # notif id -> reply_key
        self._awaiting_key: str | None = None    # last sent keyed notification

    # ---------- connection ------------------------------------------------
    async def connect(self) -> bool:
        try:
            from dbus_next.aio import MessageBus
            self.bus = await MessageBus().connect()
            introspec = await self.bus.get_introspection(
                "/org/freedesktop/Notifications", "org.freedesktop.Notifications")
            self._notif = self.bus.get_object(
                "org.freedesktop.Notifications",
                "/org/freedesktop/Notifications").get_interface(introspec)
            # notification button presses (Принять/Отклонить) come back as
            # org.freedesktop.Notifications.ActionInvoked(id, action_key).
            # Without this the whole feedback loop of initiative lived only in
            # the terminal (/feedback) — spec requires closing it via desktop.
            try:
                self._notif.on_action_invoked(self._on_action_invoked)
            except AttributeError:
                log.debug("ActionInvoked signal unavailable on this server")
            self.available = True
            log.info("connected to plasmashell notifications")
            return True
        except Exception as e:  # noqa: BLE001
            self.available = False
            log.debug("D-Bus unavailable: %s (headless mode)", e)
            return False

    # ---------- notifications ----------------------------------------------
    def notify(self, title: str, body: str, actions: list[str] | None = None,
               reply_key: str | None = None, urgency: str = "low",
               timeout_ms: int = 12000) -> None:
        if reply_key and actions:
            self.reply_hooks[reply_key] = lambda act: log.info(
                "notification answer %s -> %s", reply_key, act)
        if self._notif is None:
            log.info("[notify:%s] %s — %s %s", urgency, title, body, actions or "")
            return
        self._awaiting_key = reply_key
        try:
            asyncio.get_running_loop().create_task(self._notify_async(title, body, actions, urgency, timeout_ms))
        except RuntimeError:
            self._notify_sync(title, body)

    async def _notify_async(self, title, body, actions, urgency, timeout_ms) -> None:
        # dbus-next renders snake_case members: call_notify -> Notify,
        # on_action_invoked -> ActionInvoked (verified against dbus_next README
        # codegen rules; guarded by try/except so a wrong guess degrades to
        # "no click feedback", never to a crash).
        flat = [k for a in (actions or []) for k in ("", a)]  # id,key,label triples: empty ids ok
        try:
            reply = await self._notif.call_notify(
                "KDE Agent", 0, "", title, body, flat, {"urgency": _u(urgency)}, timeout_ms)
            log.info("notification id=%s", reply)
            nid = reply[0] if isinstance(reply, (list, tuple)) and reply else None
            if getattr(self, "_awaiting_key", None):
                self._id_to_key[nid] = self._awaiting_key
                self._awaiting_key = None
        except Exception as e:  # noqa: BLE001
            log.debug("notify failed: %s", e)

    # ---------- notification click -> reaction pipeline ---------------------
    def _on_action_invoked(self, notification_id, action_key) -> None:
        key = self._id_to_key.pop(notification_id, None)
        if not key:
            return
        hook = self.reply_hooks.get(key)
        if hook:
            try:
                hook(str(action_key))
            except Exception:  # noqa: BLE001
                log.exception("reply hook failed for %s", key)

    def _notify_sync(self, title: str, body: str) -> None:
        if shutil.which("kdialog"):
            subprocess.run(["kdialog", "--title", title, "--passivepopup", body, "8"],
                           check=False)

    # ---------- confirmation dialogs ---------------------------------------
    def confirm(self, question: str, dangerous: bool = True,
                timeout_sec: float | None = None) -> bool:
        """Blocking yes/no. Headless (no display, no stdin tty) => deny by default.

        Deny-by-default is required for the always-on daemon: a background
        service has no terminal, and blocking on input() would freeze the
        event loop forever. Non-interactive callers must pass an explicit
        confirmation channel or run in interactive mode.
        """
        interactive_stdin = False
        try:
            interactive_stdin = __import__("sys").stdin.isatty()
        except Exception:  # noqa: BLE001
            interactive_stdin = False
        if shutil.which("kdialog") and _has_display():
            cmd = ["kdialog", "--title", "KDE Agent"]
            cmd += ["--warning" if dangerous else "--info"]
            cmd += ["--yesno", question]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   check=False, timeout=timeout_sec)
                return r.returncode == 0
            except subprocess.TimeoutExpired:
                return False
        if not interactive_stdin:
            log.warning("confirm denied (non-interactive): %s", question[:80])
            return False
        try:
            ans = input(f"[AGENT CONFIRM] {question} [y/N]: ").strip().lower()
        except (EOFError, OSError):
            return False
        return ans in ("y", "да", "yes")

    def ask(self, question: str) -> str:
        if shutil.which("kdialog") and _has_display():
            r = subprocess.run(["kdialog", "--title", "KDE Agent", "--inputbox", question, ""],
                               capture_output=True, text=True, check=False)
            return r.stdout.strip()
        return input(f"[AGENT ASK] {question}: ").strip()

    # ---------- GUI automation via kdotool (Wayland/X11) --------------------
    def kdotool(self, *args: str) -> subprocess.CompletedProcess:
        if not shutil.which("kdotool"):
            log.warning("kdotool missing (install AUR kdotool); op skipped: %s", args)
            return subprocess.CompletedProcess(list(args), returncode=127, stdout="", stderr="no kdotool")
        return subprocess.run(["kdotool", *args], capture_output=True, text=True, check=False)

    def active_window(self) -> dict:
        p = self.kdotool("getactivewindow")
        wid = p.stdout.strip()
        out = {"window_id": wid, "title": "", "class": ""}
        if wid:
            out["title"] = self.kdotool("getwindowname", wid).stdout.strip()
            out["class"] = self.kdotool("getwindowclass", wid).stdout.strip()
        return out

    def screen_state(self) -> dict:
        """On-demand snapshot (the model must request it explicitly — never polled)."""
        w = self.active_window()
        p = self.kdotool("getdisplaygeometry")
        return {"active_window": w, "geometry": p.stdout.strip()}

    def click(self, x: int, y: int) -> None:
        self.kdotool("click", f"{x}", f"{y}")

    def type_text(self, text: str) -> None:
        self.kdotool("type", "--clearmodifiers", text)

    def open_app(self, app: str) -> None:
        ok = shutil.which(app)
        if ok:
            subprocess.Popen([ok], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif shutil.which("kioclient5") or shutil.which("kioclient"):
            launcher = shutil.which("kioclient") or shutil.which("kioclient5")
            subprocess.run([launcher, "exec", app], check=False)

    def idle_seconds(self) -> float:
        """Best-effort idle seconds. NOTE (explicit prototype/NPU split): the
        production path is org.freedesktop.ScreenSaver.GetIdleSeconds /
        KIdleTime over D-Bus (async); this sync fallback uses xprintidle
        (X11 only) and returns -1 when NO idle source exists, so callers can
        distinguish 'unknown' from 'active now' (0). Returning 0 previously
        made IdleWatcher believe the user is always active, so idle-based
        initiatives could never fire."""
        if shutil.which("xprintidle"):
            try:
                r = subprocess.run(["xprintidle"], capture_output=True, text=True,
                                   timeout=2, check=False)
                if r.returncode == 0 and r.stdout.strip().isdigit():
                    return int(r.stdout.strip()) / 1000.0
            except (OSError, subprocess.TimeoutExpired):
                pass
        return -1.0

    # ---------- power (energy gate input; cheap sysfs poll, no D-Bus needed) --
    def on_battery(self) -> bool:
        """UPower-free fallback: read /sys/class/power_supply. True only when a
        battery exists AND is discharging. Headless/dev machines -> False."""
        try:
            for ps in Path("/sys/class/power_supply").glob("BAT*"):
                status = (ps / "status").read_text().strip().lower()
                if status == "discharging":
                    return True
        except OSError:
            pass
        return False

    # ---------- KDE Wallet ---------------------------------------------------
    def wallet_store(self, key: str, value: str, folder: str = "kde-agent") -> bool:
        if not shutil.which("qdbus6") and not shutil.which("qdbus"):
            return False
        qdbus = shutil.which("qdbus6") or shutil.which("qdbus")
        try:
            wid = subprocess.run([qdbus, "org.kde.kwalletd6", "/modules/kwalletd6",
                                  "open", folder, "kde-agent"], capture_output=True,
                                 text=True, check=True).stdout.strip()
            subprocess.run([qdbus, "org.kde.kwalletd6", "/modules/kwalletd6",
                            "writePassword", wid, folder, key, value, "kde-agent"], check=True)
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("kwallet store failed: %s", e)
            return False


def _u(urgency: str) -> int:
    return {"low": 0, "normal": 1, "critical": 2}[urgency]
