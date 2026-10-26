#!/usr/bin/env python3
"""Fullscreen Claude quota display for a 640x480 screen (Raspberry Pi kiosk).

Big horizontal bars + clock. Data is fetched in a background thread so the
clock stays smooth and a network hiccup never freezes the screen.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import threading
import time
import urllib.error

# pygame imports numpy (if installed) for surfarray, which we never use: ~10 MB
sys.modules.setdefault("numpy", None)

import pygame  # noqa: E402

import quota_api  # noqa: E402

# ---- layout / theme -------------------------------------------------------
WIDTH, HEIGHT = 640, 480
PAD = 28
POLL_INTERVAL = 5 * 60      # seconds between usage fetches (quotas move slowly)
MAX_BACKOFF = 30 * 60       # cap for exponential backoff after errors
NET_RETRY = 5               # first retry when the network isn't up (e.g. at boot)
MAX_RETRY_AFTER = 60 * 60   # cap on a server Retry-After hint
TAP_COOLDOWN = 30           # seconds after any fetch before a tap can trigger another
STALE_AFTER = 20 * 60       # seconds before cached data is flagged stale

BG = (16, 18, 24)
FG = (235, 237, 243)
MUTED = (170, 176, 190)     # light enough to read small text on a 3.5in panel
TRACK = (38, 42, 52)
ACCENT = (210, 120, 70)     # Claude-ish terracotta for the title

# utilization thresholds -> bar colour
GREEN = (60, 190, 120)
YELLOW = (220, 200, 70)
ORANGE = (230, 150, 60)
RED = (225, 80, 80)


def bar_color(pct: float) -> tuple[int, int, int]:
    if pct >= 95:
        return RED
    if pct >= 80:
        return ORANGE
    if pct >= 50:
        return YELLOW
    return GREEN


# ---- shared state between fetch thread and render loop --------------------
class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.usage: dict | None = None
        self.last_success: float = 0.0       # wall clock, for display
        self.last_success_mono: float = 0.0  # monotonic, for age/staleness
        self.error: str | None = None
        self.refresh_not_before: float = 0.0  # monotonic; taps are ignored until then

    def update_ok(self, usage: dict) -> None:
        with self.lock:
            self.usage = usage
            self.last_success = time.time()
            self.last_success_mono = time.monotonic()
            self.error = None

    def update_err(self, msg: str) -> None:
        with self.lock:
            self.error = msg

    def snapshot(self):
        with self.lock:
            return self.usage, self.last_success, self.last_success_mono, self.error

    def hold_refresh(self, seconds: float) -> None:
        with self.lock:
            self.refresh_not_before = time.monotonic() + seconds

    def refresh_allowed(self) -> bool:
        with self.lock:
            return time.monotonic() >= self.refresh_not_before


def _retry_after_seconds(exc: urllib.error.HTTPError) -> float | None:
    """Parse a Retry-After header (delta-seconds form) if the server sent one."""
    value = exc.headers.get("Retry-After") if exc.headers else None
    if value and value.strip().isdigit():
        # a bogus huge value must not park the poller for days
        return min(float(value.strip()), MAX_RETRY_AFTER)
    return None


def fetch_loop(state: State, stop: threading.Event, wake: threading.Event) -> None:
    """Poll the usage endpoint, backing off exponentially on errors so we never
    hammer the API — and honouring Retry-After when we're rate-limited (429).
    Setting `wake` (a tap) fetches early; State.refresh_allowed() gates that."""
    backoff = POLL_INTERVAL
    net_backoff = NET_RETRY
    while not stop.is_set():
        # no tap-triggered fetch within TAP_COOLDOWN of this one, however it started
        state.hold_refresh(TAP_COOLDOWN)
        try:
            state.update_ok(quota_api.fetch_usage())
            backoff = POLL_INTERVAL
            net_backoff = NET_RETRY
            wait = POLL_INTERVAL
        except urllib.error.HTTPError as exc:
            state.update_err(str(exc))
            backoff = min(backoff * 2, MAX_BACKOFF)
            # On a rate-limit, prefer the server's own Retry-After hint.
            wait = backoff
            if exc.code == 429:
                wait = max(_retry_after_seconds(exc) or 0, backoff)
                # taps must not jump a rate limit
                state.hold_refresh(wait)
        except urllib.error.URLError as exc:
            # couldn't connect (wifi/dns not up yet), so a quick retry costs the api nothing
            state.update_err(str(exc))
            wait = net_backoff
            net_backoff = min(net_backoff * 2, POLL_INTERVAL)
        except Exception as exc:  # noqa: BLE001 - keep the display alive
            state.update_err(str(exc))
            backoff = min(backoff * 2, MAX_BACKOFF)
            wait = backoff
        # drop a wake that landed mid-fetch, then sleep until the next poll or a tap
        wake.clear()
        wake.wait(wait)


# ---- formatting helpers ---------------------------------------------------
def parse_reset(iso: str | None) -> dt.datetime | None:
    if not iso:
        return None
    try:
        return dt.datetime.fromisoformat(iso).astimezone()
    except ValueError:
        return None


def fmt_reset_short(when: dt.datetime | None) -> str:
    if when is None:
        return ""
    now = dt.datetime.now().astimezone()
    if when.date() == now.date():
        return f"resets {when:%H:%M}"
    if (when.date() - now.date()).days == 1:
        return f"resets tomorrow {when:%H:%M}"
    # Build the day without %-d, which is glibc-only.
    return f"resets {when:%a %b} {when.day}"


def model_limits(usage: dict) -> list[tuple[str, float, str | None]]:
    """Per-model weekly caps (e.g. Fable) from the `limits` list, as
    (label, percent, resets_at) rows."""
    rows = []
    for lim in usage.get("limits") or []:
        if lim.get("kind") != "weekly_scoped":
            continue
        model = ((lim.get("scope") or {}).get("model") or {}).get("display_name")
        if model:
            rows.append((model.upper(), float(lim.get("percent") or 0), lim.get("resets_at")))
    return rows


# ---- drawing --------------------------------------------------------------
class Display:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.f_title = self._font(40, bold=True)
        self.f_clock = self._font(40, bold=True)
        self.f_label = self._font(26, bold=True)
        self.f_pct = self._font(30, bold=True)
        self.f_small = self._font(24)
        self.f_tiny = self._font(20)

    @staticmethod
    def _font(size: int, bold: bool = False) -> pygame.font.Font:
        return pygame.font.SysFont("dejavusans", size, bold=bold)

    def _text(self, surf_font, text, color, x, y, right=False, center=False):
        img = surf_font.render(text, True, color)
        rect = img.get_rect()
        if right:
            rect.topright = (x, y)
        elif center:
            rect.midtop = (x, y)
        else:
            rect.topleft = (x, y)
        self.screen.blit(img, rect)
        return rect

    def _bar(self, x, y, w, h, pct):
        pct = max(0.0, min(100.0, pct))
        radius = h // 2
        pygame.draw.rect(self.screen, TRACK, (x, y, w, h), border_radius=radius)
        fill_w = int(w * pct / 100)
        if fill_w >= h:
            pygame.draw.rect(self.screen, bar_color(pct), (x, y, fill_w, h),
                             border_radius=radius)
        elif fill_w > 0:
            pygame.draw.rect(self.screen, bar_color(pct), (x, y, fill_w, h),
                             border_radius=min(radius, fill_w // 2 or 1))

    def quota_block(self, y, label, pct, reset_text):
        self._text(self.f_label, label, FG, PAD, y)
        self._text(self.f_pct, f"{pct:.0f}%", bar_color(pct), WIDTH - PAD, y - 2,
                   right=True)
        self._bar(PAD, y + 34, WIDTH - 2 * PAD, 24, pct)
        if reset_text:
            self._text(self.f_small, reset_text, MUTED, PAD, y + 62)

    @staticmethod
    def signature(state: State) -> tuple:
        """A cheap fingerprint of everything on screen, so the main loop can
        skip the redraw+flip when nothing has changed (keeps CPU near idle)."""
        usage, last_success, last_success_mono, error = state.snapshot()
        now = dt.datetime.now().astimezone()
        stale = bool(last_success_mono) and (
            time.monotonic() - last_success_mono > STALE_AFTER
        )
        return (now.strftime("%H:%M:%S"), last_success, error, usage is None, stale)

    def render(self, state: State) -> None:
        usage, last_success, last_success_mono, error = state.snapshot()
        self.screen.fill(BG)

        # header
        now = dt.datetime.now().astimezone()
        self._text(self.f_title, "CLAUDE", ACCENT, PAD, PAD - 6)
        self._text(self.f_clock, now.strftime("%H:%M:%S"), FG, WIDTH - PAD, PAD - 6,
                   right=True)
        pygame.draw.line(self.screen, TRACK, (PAD, PAD + 48),
                         (WIDTH - PAD, PAD + 48), 2)

        if usage is None:
            msg = (error or "connecting...")[:42]
            self._text(self.f_label, msg, MUTED, WIDTH // 2, HEIGHT // 2 - 20,
                       center=True)
            return

        five = usage.get("five_hour") or {}
        seven = usage.get("seven_day") or {}
        extra = usage.get("extra_usage") or {}

        rows = [
            ("5-HOUR", float(five.get("utilization") or 0), five.get("resets_at")),
            ("7-DAY", float(seven.get("utilization") or 0), seven.get("resets_at")),
            *model_limits(usage),
        ]
        # close up the rows when the credits row has to fit too
        y, pitch = (90, 98) if extra.get("is_enabled") else (92, 110)
        for label, pct, resets in rows:
            self.quota_block(y, label, pct, fmt_reset_short(parse_reset(resets)))
            y += pitch

        # credits
        if extra.get("is_enabled"):
            # amounts arrive in minor units (cents), with the scale alongside
            scale = 10 ** int(extra.get("decimal_places") or 0)
            used = float(extra.get("used_credits") or 0) / scale
            limit = float(extra.get("monthly_limit") or 0) / scale
            cur = extra.get("currency", "USD")
            sym = "$" if cur == "USD" else f"{cur} "
            self._text(self.f_label, "CREDITS", FG, PAD, y)
            self._text(self.f_pct, f"{sym}{used:.0f} / {sym}{limit:.0f}", FG,
                       WIDTH - PAD, y - 2, right=True)
            pct = (used / limit * 100) if limit else 0
            self._bar(PAD, y + 34, WIDTH - 2 * PAD, 16, pct)

        # status footer
        if error and last_success:
            footer = f"stale - {error[:40]}"
            color = ORANGE
        elif last_success:
            age = time.monotonic() - last_success_mono
            stale = age > STALE_AFTER
            footer = f"updated {dt.datetime.fromtimestamp(last_success):%H:%M}"
            if stale:
                footer += "  (stale)"
            color = ORANGE if stale else MUTED
        else:
            footer = ""
            color = MUTED
        if footer:
            self._text(self.f_tiny, footer, color, WIDTH // 2, HEIGHT - 32,
                       center=True)


def _want_windowed(argv: list[str]) -> bool:
    """Fullscreen kiosk by default; windowed on macOS (dev) or when asked.

    `--windowed`/`--fullscreen` win, then $QUOTA_WINDOWED, then the platform
    default — so the Pi stays fullscreen while a Mac comes up in a window.
    """
    if "--windowed" in argv:
        return True
    if "--fullscreen" in argv:
        return False
    env = os.environ.get("QUOTA_WINDOWED")
    if env is not None:
        return env.strip().lower() not in ("", "0", "false", "no")
    return sys.platform == "darwin"


def main() -> None:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    windowed = _want_windowed(sys.argv[1:])
    # sdl2 picks x11 first, which under labwc means running an extra xwayland server
    if os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("SDL_VIDEODRIVER", "wayland,x11")
    # not pygame.init(): that also opens the audio device, which wakes ~80x/s
    pygame.display.init()
    pygame.font.init()
    # Hide the cursor in the kiosk; keep it in a window so you can move/close it.
    pygame.mouse.set_visible(windowed)
    # No SCALED: the window already matches the native 640x480, so scaling
    # would just burn CPU on a per-frame blit.
    flags = 0 if windowed else pygame.FULLSCREEN
    screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
    pygame.display.set_caption("Claude Quota")

    state = State()
    stop = threading.Event()
    wake = threading.Event()
    thread = threading.Thread(target=fetch_loop, args=(state, stop, wake), daemon=True)
    thread.start()

    display = Display(screen)
    last_sig = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (
                pygame.K_ESCAPE, pygame.K_q,
            ):
                running = False
            elif event.type == pygame.FINGERDOWN or (
                event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
            ):
                # tap to refresh; a touch arrives as both events, the cooldown dedupes
                if state.refresh_allowed():
                    wake.set()
        # Only repaint when something visible actually changed (the clock ticks
        # once a second), so the loop is effectively idle the rest of the time.
        sig = display.signature(state)
        if sig != last_sig:
            display.render(state)
            pygame.display.flip()
            last_sig = sig
        # wake just past each second; pygame.event.wait() would poll every 1ms
        time.sleep(1.005 - time.time() % 1)

    stop.set()
    wake.set()
    pygame.quit()


if __name__ == "__main__":
    main()
