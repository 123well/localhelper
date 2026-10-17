"""Window & workspace perception (initiative layer for windows/apps).

Spec: the agent must notice which windows exist, which are stale, which
belong to a finished project, and learn co-occurrence patterns — WITHOUT
polling the screen. Design decisions (verified against reality, not assumed):

* Event source priority:
    1. KWin D-Bus signals org.kde.KWin /KWin windowActivated/windowCreated/
       windowDestroyed. NOTE: Plasma 6 migrated to the virtual-desktop API;
       these legacy signal names exist on X11 sessions but are NOT guaranteed
       on Wayland. We subscribe only if the interface actually exposes the
       member (introspection-guarded), so a wrong assumption degrades to
       source #2 instead of crashing.
    2. kdotool snapshot (AUR) — cheap one-shot `getactivewindow` +
       `getwindows`. Taken ONLY when another event fires (user command,
       idle tick, focus hint) or at a very slow heartbeat (default 5 min);
       never per-second polling.
    3. Nothing available -> WindowState stays empty, detectors stay silent.
       (This is the headless/CI case.)

* The agent NEVER closes windows by itself: close_window lives in the tool
  registry behind the safety gate + user confirmation; here we only produce
  *suggestions* (Interventions) through the initiative engine.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field

log = logging.getLogger("kde-agent.windows")


@dataclass
class WindowInfo:
    wid: str
    title: str = ""
    cls: str = ""            # WM_CLASS / app id
    active: bool = False
    first_seen: float = field(default_factory=time.time)
    last_focus: float = field(default_factory=time.time)
    focus_count: int = 0

    def key(self) -> str:
        """Stable identity across snapshots: class+normalized title."""
        t = re.sub(r"[\W_]+", " ", self.title.lower()).strip()[:48]
        return f"{self.cls}|{t}"


class WindowState:
    """In-memory model of open windows + learned profiles (JSON-persisted).

    Profiles (spec: «строить профили задач и рабочие сценарии»):
      cooccur[clsA][clsB]  — how often apps were focused within one session
      context[project_kw]  — classes focused while such a title was active
      ignored[key]         — suggestions about this window the user rejected
    """

    def __init__(self, data_dir=None):
        self.windows: dict[str, WindowInfo] = {}     # wid -> info
        self.by_key: dict[str, WindowInfo] = {}      # stable key -> info
        self.cooccur: dict[str, dict[str, int]] = {}
        self.ctx_apps: dict[str, dict[str, int]] = {}
        self.recent_focus: list[str] = []            # last ~8 classes
        self.ignored: dict[str, int] = {}            # stable-key -> reject count
        self.last_snapshot_ts: float = 0.0
        self._data_dir = data_dir
        if data_dir:
            self._load()

    # ---------- persistence ---------------------------------------------
    def _path(self):
        from pathlib import Path
        return Path(self._data_dir) / "window_profiles.json"

    def _load(self) -> None:
        try:
            p = self._path()
            if p.exists():
                d = json.loads(p.read_text())
                self.cooccur = d.get("cooccur", {})
                self.ctx_apps = d.get("ctx_apps", {})
                self.ignored = d.get("ignored", {})
        except Exception as e:  # noqa: BLE001  corrupt profile must not kill boot
            log.warning("window profiles unreadable: %s", e)

    def save(self) -> None:
        if not self._data_dir:
            return
        try:
            self._path().write_text(json.dumps(
                {"cooccur": self.cooccur, "ctx_apps": self.ctx_apps,
                 "ignored": self.ignored}, ensure_ascii=False))
        except OSError as e:
            log.warning("window profiles save failed: %s", e)

    # ---------- ingestion -------------------------------------------------
    def apply_snapshot(self, wins: list[dict]) -> list[str]:
        """Merge a normalized snapshot [{'wid','title','class','active'}].
        Returns change descriptors: ['new:<key>', 'closed:<key>', 'focus'].
        Closed-window bookkeeping keeps last-known info for ignore lookups."""
        now = time.time()
        changes: list[str] = []
        seen_keys = set()
        prev_active = next((w.wid for w in self.windows.values() if w.active), None)
        for raw in wins:
            wi = WindowInfo(wid=str(raw.get("wid", "")),
                            title=str(raw.get("title", ""))[:120],
                            cls=str(raw.get("class", "")) or "unknown",
                            active=bool(raw.get("active")))
            seen_keys.add(wi.key())
            old = self.by_key.get(wi.key())
            if old is None:
                changes.append(f"new:{wi.key()}")
                wi.first_seen = now
            else:
                wi.first_seen = old.first_seen
                wi.last_focus = old.last_focus
                wi.focus_count = old.focus_count
            wi.wid = wid_str = str(raw.get("wid", ""))
            self.windows[wid_str] = wi
            self.by_key[wi.key()] = wi
            if wi.active:
                self.on_focus(wi)
        for k in list(self.by_key):
            if k not in seen_keys:
                changes.append(f"closed:{k}")
                w = self.by_key.pop(k)
                self.windows.pop(w.wid, None)
        if not any(c == "focus" for c in changes) and \
                any(w.active for w in self.windows.values()) and \
                prev_active != next((w.wid for w in self.windows.values() if w.active), prev_active):
            changes.append("focus")
        self.last_snapshot_ts = now
        return changes

    def mark_closed_by_user(self, key: str) -> None:
        """User feedback that this window WAS unwanted -> reinforce cleanup
        detector (spec: явные оценки запоминаются)."""
        self.ignored[key] = max(0, self.ignored.get(key, 0) - 1)   # negative ignore = wanted-gone
        self.save()

    def note_ignore(self, key: str) -> None:
        self.ignored[key] = self.ignored.get(key, 0) + 1
        self.save()

    # ---------- learning --------------------------------------------------
    def on_focus(self, wi: WindowInfo) -> None:
        now = time.time()
        # rate-limit: one focus event per window per second (debounces KWin
        # signal storms / synthetic rapid events; real switching is <1/sec)
        if now - getattr(wi, "_last_event_ts", 0.0) < 1.0:
            return
        wi._last_event_ts = now
        wi.last_focus = now
        wi.focus_count += 1
        # co-occurrence: count each *transition* (prev class -> this class),
        # not every window still present in the recent buffer — otherwise a
        # single repeated focus inflates all older entries and ranking breaks.
        if self.recent_focus and self.recent_focus[-1] != wi.cls:
            prev = self.recent_focus[-1]
            self.cooccur.setdefault(prev, {})[wi.cls] = \
                self.cooccur.setdefault(prev, {}).get(wi.cls, 0) + 1
        self.recent_focus.append(wi.cls)
        del self.recent_focus[:-8]
        for kw in _project_keywords(wi.title):
            d = self.ctx_apps.setdefault(kw, {})
            d[wi.cls] = d.get(wi.cls, 0) + 1

    def suggest_for_context(self, keyword: str) -> list[tuple[str, int]]:
        """Apps associated with a project keyword, ranked by focus events."""
        d = self.ctx_apps.get(keyword.lower(), {})
        return sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:3]

    def companions(self, cls: str) -> list[tuple[str, int]]:
        d = self.cooccur.get(cls, {})
        return sorted(d.items(), key=lambda kv: -kv[1])[:3]


def _project_keywords(title: str) -> list[str]:
    """Extract project-ish tokens from a window title.

    'main.py — myproj'      -> ['myproj']
    '~/dev/myproj — main.py'-> ['dev', 'myproj']  ('dev' is generic but kept;
                                                    suggest_for_context ranks
                                                    by accumulated counts, so
                                                    the specific kw wins)
    Plain titles give nothing useful. Stoplist drops filenames (contain '.')
    and common noise words.
    """
    kws = []
    for tok in re.split(r"[\s/~—|:,.\-]+", title):   # em-dash is a separator
        t = tok.lower().strip()
        if not t or len(t) < 3 or len(t) > 30:
            continue
        if "." in t or t.isdigit():                  # file.ext / numbers
            continue
        if t in {"the", "and", "for", "with", "konsole", "dolphin", "firefox",
                "plasma", "agent", "file", "txt", "pdf", "home"}:
            continue
        kws.append(t)
    return kws


# ---------------------------------------------------------------------------
class KWinWatcher:
    """D-Bus signal source #1. Graceful: no dbus-next / no KWin members -> inactive."""

    def __init__(self, bus, bridge):
        self.bus, self.bridge = bus, bridge

    async def run(self) -> None:
        try:
            from dbus_next.aio import MessageBus
            from dbus_next.service import ServiceInterface, Method  # noqa: F401
        except ImportError:
            log.info("KWin watcher inactive (no dbus-next)")
            return
        if self.bridge.bus is None:
            log.info("KWin watcher inactive (no session bus)")
            return
        try:
            import dbus_next
            bus = await MessageBus(bus_type=dbus_next.BusType.SESSION).connect()
            intro = await bus.introspect("org.kde.KWin", "/KWin")
            obj = bus.get_object("org.kde.KWin", "/KWin")
            iface = obj.get_interface("org.kde.KWin")
            names = {m.name for m in intro.nodes["/KWin"].interfaces[0].signals} \
                if intro.nodes else set()
            wired = []
            for sig, kind in (("windowActivated", "activated"),
                              ("windowCreated", "created"),
                              ("windowDestroyed", "destroyed")):
                if sig in names:
                    getattr(iface, f"on_{_lc(sig)}").connect(
                        lambda *_a, k=kind: self.bus.emit("window_changed", source=k))
                    wired.append(sig)
            if not wired:
                log.info("KWin exposes no legacy window signals (Wayland?) — "
                         "snapshot-on-event mode")
                return
            log.info("KWin watcher active: %s", ",".join(wired))
            while True:
                await asyncio.sleep(3600)
        except Exception as e:  # noqa: BLE001
            log.info("KWin watcher inactive: %s", e)


def _lc(name: str) -> str:
    return name[0].lower() + name[1:]


# ---------------------------------------------------------------------------
class WindowProber:
    """Source #2: on-demand kdotool snapshot, rate-limited (event-driven,
    plus a slow heartbeat handled by the caller)."""

    MIN_INTERVAL = 20.0

    def __init__(self, bridge, state: WindowState, bus):
        self.bridge, self.state, self.bus = bridge, state, bus
        self._last = 0.0

    def available(self) -> bool:
        import shutil
        return bool(shutil.which("kdotool"))

    async def probe(self, force: bool = False) -> list[str]:
        if not self.available():
            return []
        if not force and time.time() - self._last < self.MIN_INTERVAL:
            return []
        self._last = time.time()
        snaps = await asyncio.to_thread(self._snapshot_sync)
        if snaps is None:
            return []
        changes = self.state.apply_snapshot(snaps)
        if changes:
            self.bus.emit("windows_updated", changes=len(changes))
        return changes

    def _snapshot_sync(self):
        import shutil
        import subprocess
        kt = shutil.which("kdotool")
        if not kt:
            return None
        try:
            act = subprocess.run([kt, "getactivewindow"], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
            lst = subprocess.run([kt, "getwindows"], capture_output=True,
                                 text=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired) as e:
            log.debug("kdotool snapshot failed: %s", e)
            return None
        wins = []
        for line in lst.splitlines():
            m = re.match(r"\s*(0x[0-9a-fA-F]+|\d+)\b(.*)", line)
            if not m:
                continue
            wid = m.group(1)
            wins.append({"wid": wid, "title": m.group(2).strip(),
                         "class": "", "active": wid == act})
        # enrich with classes/titles via dedicated calls only for a bounded set
        for w in wins[:12]:
            try:
                c = subprocess.run([kt, "getwindowclass", w["wid"]],
                                   capture_output=True, text=True, timeout=3)
                if c.returncode == 0:
                    w["class"] = c.stdout.strip() or w["class"]
                n = subprocess.run([kt, "getwindowname", w["wid"]],
                                   capture_output=True, text=True, timeout=3)
                if n.returncode == 0 and n.stdout.strip():
                    w["title"] = n.stdout.strip() or w["title"]
            except (OSError, subprocess.TimeoutExpired):
                pass
        return wins


# ---------------------------------------------------------------------------
class WindowInitiative:
    """Detectors turning WindowState into candidate Interventions.

    Types: stale_windows | many_windows | project_open | window_wanted.
    Every suggestion respects HabitModel thresholds/cooldowns/budget via the
    shared InitiativeEngine pipeline (we return Intervention objects only).
    """

    STALE_SEC = 1800          # not focused for 30 min
    MANY_MIN = 4              # >=4 windows of one class

    def __init__(self, cfg, state: WindowState, memory=None):
        self.cfg = cfg                      # InitiativeConfig (has .enabled)
        self.state = state
        self.memory = memory

    def detect(self, active_hint: str = "") -> list:
        from agent.core.initiative import Intervention
        if not getattr(self.cfg, "enabled", True):
            return []
        out = []
        now = time.time()
        by_cls: dict[str, list[WindowInfo]] = {}
        for w in self.state.windows.values():
            by_cls.setdefault(w.cls, []).append(w)
        # 1. too many windows of one app, most untouched
        for cls, ws in by_cls.items():
            if len(ws) >= self.MANY_MIN:
                stale = [w for w in ws if now - w.last_focus > self.STALE_SEC]
                if len(stale) >= 2:
                    keys = "|".join(sorted(w.key() for w in stale))
                    if self.state.ignored.get(keys, 0) >= 2:
                        continue    # user twice said these are fine
                    out.append(Intervention(
                        "many_windows",
                        f"Открыто {len(ws)} окон «{cls}», {len(stale)} из них "
                        f"не использовались >{int(self.STALE_SEC // 60)} мин. "
                        f"Закрыть ненужные или сгруппировать?",
                        0.6, {"keys": keys, "cls": cls, "stale": len(stale)},
                        {"action": "group_or_close", "cls": cls}))
        # 2. globally stale single windows after a long absence of use
        stale_all = [w for w in self.state.windows.values()
                     if not w.active and now - w.last_focus > 2 * self.STALE_SEC
                     and now - w.first_seen > 2 * self.STALE_SEC]
        if len(stale_all) >= 3:
            keys = "|".join(sorted(w.key() for w in stale_all)[:6])
            if self.state.ignored.get(keys, 0) < 2:
                out.append(Intervention(
                    "stale_windows",
                    f"{len(stale_all)} окон не использовались долго — похоже на "
                    f"остатки завершённых задач. Навести порядок?",
                    0.55, {"keys": keys}, {"action": "cleanup"}))
        # 3. context suggestion: user started working on project X -> offer
        #    the apps they usually have open for it (learned co-occurrence)
        if active_hint:
            for kw in _project_keywords(active_hint):
                sugg = self.state.suggest_for_context(kw)
                open_classes = set(by_cls)
                missing = [(c, n) for c, n in sugg if c not in open_classes and n >= 3]
                if missing:
                    names = ", ".join(c for c, _n in missing[:2])
                    out.append(Intervention(
                        "project_open",
                        f"Вы в контексте «{kw}». Обычно здесь открыты {names}. "
                        f"Открыть их?",
                        0.6, {"kw": kw, "apps": [c for c, _ in missing]},
                        {"action": "open_apps", "apps": [c for c, _ in missing[:2]]}))
        return out
