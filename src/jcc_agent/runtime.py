"""Bound-window observation loop. No input is dispatched without --execute."""
from __future__ import annotations

import ctypes
import time
from copy import deepcopy
from pathlib import Path

from .context import UIContextTracker
from .control import ActionController, RecordingBackend, Win32Backend
from .perception import FrameAnalyzer, OCRReader
from .session import AgentSession
from .storage import EventLog


def list_windows() -> list[dict]:
    """Read-only listing for the operator to choose a game window explicitly."""
    import sys
    from ctypes import wintypes

    if sys.platform != "win32":
        raise RuntimeError("Window enumeration requires Windows")
    api = ctypes.WinDLL("user32", use_last_error=True)
    api.GetForegroundWindow.restype = wintypes.HWND
    api.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    api.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    api.IsWindowVisible.argtypes = [wintypes.HWND]
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    api.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    foreground = api.GetForegroundWindow()
    windows = []

    @callback_type
    def callback(hwnd, _):
        if api.IsWindowVisible(hwnd):
            length = api.GetWindowTextLengthW(hwnd)
            if length:
                buffer = ctypes.create_unicode_buffer(length + 1)
                api.GetWindowTextW(hwnd, buffer, length + 1)
                windows.append({"hwnd": hex(hwnd), "title": buffer.value, "foreground": hwnd == foreground})
        return True

    if not api.EnumWindows(callback, 0):
        raise RuntimeError("Cannot enumerate windows")
    return windows


class WindowCapture:
    def __init__(self, hwnd: int):
        import sys
        from ctypes import wintypes
        if sys.platform != "win32":
            raise RuntimeError("Window capture requires Windows")
        if not hwnd:
            raise ValueError("An explicit game window handle is required")
        self.hwnd = hwnd
        self.api = ctypes.WinDLL("user32", use_last_error=True)
        self.api.GetForegroundWindow.restype = wintypes.HWND
        self.api.IsWindow.argtypes = [wintypes.HWND]
        self.api.IsIconic.argtypes = [wintypes.HWND]
        self.api.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.api.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        try:
            self.api.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except AttributeError:
            self.api.SetProcessDPIAware()

    def viewport(self) -> list[int]:
        from ctypes import wintypes
        if not self.api.IsWindow(self.hwnd) or self.api.IsIconic(self.hwnd):
            raise RuntimeError("Game window missing or minimized")
        if self.api.GetForegroundWindow() != self.hwnd:
            raise RuntimeError("Game window lost focus; no input will be sent")
        rect, origin = wintypes.RECT(), wintypes.POINT(0, 0)
        if not self.api.GetClientRect(self.hwnd, ctypes.byref(rect)):
            raise RuntimeError("Cannot read game window client bounds")
        if not self.api.ClientToScreen(self.hwnd, ctypes.byref(origin)):
            raise RuntimeError("Cannot map client bounds to screen")
        return [origin.x, origin.y, rect.right - rect.left, rect.bottom - rect.top]

    def frame(self):
        import cv2
        import numpy as np
        from PIL import ImageGrab
        viewport = self.viewport()
        x, y, width, height = viewport
        image = ImageGrab.grab(bbox=(x, y, x + width, y + height), all_screens=True)
        # Recheck the window after capture; a mixed frame must never lead to input.
        if self.viewport() != viewport:
            raise RuntimeError("Game window moved during capture")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), viewport

    def stop_pressed(self) -> bool:
        return bool(self.api.GetAsyncKeyState(0x1B) & 0x8000)  # Escape

    def input_tick(self) -> int:
        from ctypes import wintypes

        class LastInputInfo(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        info = LastInputInfo()
        info.cbSize = ctypes.sizeof(info)
        if not self.api.GetLastInputInfo(ctypes.byref(info)):
            raise RuntimeError("Cannot verify exclusive input ownership")
        return int(info.dwTime)


def run_live(*, hwnd, layout, rules, compositions, output, execute=False,
             duration=60.0, interval=1.5, stop_file="STOP", choice_preferences=None,
             initial_state=None, scout=False):
    from .scouting import ScoutingScheduler
    from .tracking import SelfTracker

    if execute and not layout.get("calibrated"):
        raise ValueError("Live execution requires a reviewed calibrated layout")
    if initial_state is not None:
        # A reviewed inventory is a baseline for THIS run, not another run's
        # frame clock. Otherwise exported version/timestamps block tracking.
        initial_state = deepcopy(initial_state)
        if hasattr(initial_state, "to_dict"):
            initial_state = initial_state.to_dict()
        initial_state["version"] = 0
        initial_state["observed_at"] = 0.0
        for group in ("board", "bench"):
            for slot in initial_state.get(group, {}).values():
                slot["observed_at"] = 0.0
        if execute and initial_state.get("ordinary_merges") is not True:
            raise ValueError("Review the initial inventory and explicitly set ordinary_merges=true before execution")
    capture = WindowCapture(hwnd)
    analyzer = FrameAnalyzer(OCRReader(**layout.get("ocr_options", {})), rules, layout)
    session = AgentSession(rules, compositions, choice_preferences=choice_preferences)
    tracker = SelfTracker(initial_state, rules)
    ui_context = UIContextTracker(initial_state)
    if execute and not tracker.export().get("ownership_complete"):
        raise ValueError("Execution requires a complete reviewed initial inventory; unknown positions cannot be assumed empty")
    scout_config = layout.get("scouting", {})
    scheduler = None
    if scout:
        scheduler = ScoutingScheduler(
            scout_config.get("players", []), scout_config.get("slots"),
            budget_seconds=float(scout_config.get("budget_seconds", 8)),
            player_points=scout_config.get("player_points"),
            layout=layout,
        )
    bound_layout = deepcopy(layout)
    controller = ActionController(Win32Backend(hwnd) if execute else RecordingBackend(),
                                  bound_layout, rules, dry_run=not execute, live_enabled=execute,
                                  min_action_interval=interval)
    log = EventLog(Path(output) / "events.jsonl")
    start, version = time.monotonic(), 0
    input_tick = capture.input_tick() if execute else None
    log.append("started", mode="live" if execute else "observe", hwnd=hwnd, duration=duration)
    try:
        while time.monotonic() - start < duration:
            iteration = time.monotonic()
            if Path(stop_file).exists() or capture.stop_pressed():
                log.append("stopped", reason="stop_file_or_escape")
                break
            if execute and capture.input_tick() != input_tick:
                raise RuntimeError("External keyboard/mouse input detected; restart after reviewing the current inventory")
            frame, viewport = capture.frame()
            captured_at = time.monotonic()
            reference = layout.get("reference_size")
            if reference and list(viewport[2:]) != list(reference):
                raise RuntimeError(f"Window size {viewport[2:]} differs from calibration {reference}")
            bound_layout["viewport"] = viewport
            version += 1
            observation = analyzer.analyze(frame, timestamp=captured_at - start, version=version)
            if execute and capture.input_tick() != input_tick:
                raise RuntimeError("External input during OCR invalidated the observation")
            inference_age = time.monotonic() - captured_at
            if observation.seconds_left is not None:
                observation.seconds_left = max(0, observation.seconds_left - inference_age)
            observation = ui_context.fuse(observation)
            refreshing = controller.pending is not None and controller.pending.action.kind == "refresh"
            raw_refresh_shop = deepcopy(observation.shop) if refreshing else None
            if not refreshing:
                # Bought-slot evidence belongs to the pre-refresh generation.
                # It may help confirm another purchase, never a new shop.
                observation = tracker.fuse_shop(observation)
            pending_before = deepcopy(controller.pending.before) if controller.pending else None
            result = controller.observe(observation)
            if result:
                log.append("confirmation", result=result)
                if scheduler:
                    scheduler.on_result(result, observation)
                if result.status == "confirmed" and pending_before and result.action.kind in {"buy", "refresh", "level", "sell", "move"}:
                    applied = tracker.apply_confirmed(result, pending_before, observation)
                    if result.action.kind in {"buy", "sell", "move"} and not applied:
                        tracker.mark_uncertain("confirmed_action_could_not_be_applied")
                        controller.paused = True
                        controller.pause_reason = "confirmed_action_could_not_be_applied"
                    if result.action.kind in {"buy", "refresh", "level"}:
                        session.account_confirmed_spend(pending_before, observation)
            if scheduler:
                snapshots = scheduler.export_snapshots()
                self_snapshot = snapshots.pop("self", None)
                if self_snapshot:
                    for key, slot in self_snapshot.slots.items():
                        (observation.bench if key.startswith("bench_") else observation.board)[key] = slot
                observation.opponents.update(snapshots)
            observation = tracker.fuse(observation)
            if refreshing and not (result and result.status == "confirmed" and result.action.kind == "refresh"):
                # fuse() also reconciles inventory and may reapply old shop
                # empties. Keep pending/failed refresh output raw as well, so
                # neither reporting nor planning claims unknown cards are empty.
                observation.shop = raw_refresh_shop
            state, pool, plan, action = session.decide(observation)
            log.append("observation", state=state, pool=pool, plan=plan, action=action,
                       frame_age_seconds=inference_age)
            if controller.paused:
                log.append("stopped", reason=controller.pause_reason)
                break
            if inference_age > float(layout.get("max_frame_age_seconds", 2.0)):
                log.append("stale_frame", age=inference_age, action_suppressed=True)
                continue
            if scheduler and state.phase == "preparation" and state.round:
                scheduler.start_round(state.round)
            if scheduler and scheduler.active:
                action = scheduler.next_action(state)
            if action and not controller.pending:
                # A long OCR call must not use old coordinates after focus changes.
                if capture.viewport() != viewport:
                    raise RuntimeError("Window changed before action dispatch")
                if execute and capture.input_tick() != input_tick:
                    raise RuntimeError("External input before dispatch invalidated the observation")
                if execute:
                    ui_context.before_dispatch(action)
                dispatched = controller.submit(action, state)
                if execute:
                    input_tick = capture.input_tick()
                log.append("dispatch", result=dispatched)
                if scheduler:
                    scheduler.on_result(dispatched, state)
            remaining = interval - (time.monotonic() - iteration)
            if remaining > 0:
                time.sleep(min(remaining, 2.0))
    except KeyboardInterrupt:
        log.append("stopped", reason="keyboard_interrupt")
    except Exception as exc:
        log.append("stopped", reason=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        log.append("finished", elapsed=time.monotonic() - start, observations=version)
    return {"observations": version, "log": str(log.path), "execute": execute}
