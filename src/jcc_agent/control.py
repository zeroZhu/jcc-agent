"""Serialized, observation-confirmed input. Importing this module never clicks."""
from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .models import Action, ActionResult, GameState, Ruleset
from .pool import ownership_summary_error


class InputBackend(Protocol):
    def click(self, x: int, y: int) -> None: ...


@dataclass
class RecordingBackend:
    """Fake backend; records coordinates, with no operating-system input."""
    events: list[tuple[int, int]] = field(default_factory=list)

    def click(self, x: int, y: int) -> None:
        self.events.append((x, y))


class Win32Backend:
    """Standard Windows input, guarded against focus changes at every click.

    The caller must supply the exact calibrated game window handle. Nothing is
    loaded or sent until click(); this class never activates another window.
    """
    def __init__(self, hwnd: int):
        if not hwnd:
            raise ValueError("A calibrated foreground window handle is required")
        self.hwnd = int(hwnd)

    def click(self, x: int, y: int) -> None:
        import ctypes
        import sys
        from ctypes import wintypes
        if sys.platform != "win32":
            raise RuntimeError("Win32 input is available only on Windows")
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetForegroundWindow.restype = wintypes.HWND
        if user32.GetForegroundWindow() != self.hwnd:
            raise RuntimeError("The calibrated game window is not foreground")
        rect = wintypes.RECT()
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        if not user32.GetWindowRect(self.hwnd, ctypes.byref(rect)):
            raise RuntimeError("Cannot read calibrated window bounds")
        if not (rect.left <= x < rect.right and rect.top <= y < rect.bottom):
            raise RuntimeError("Click falls outside the calibrated game window")

        class MouseInput(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                        ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

        class KeyboardInput(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                        ("dwExtraInfo", ctypes.c_size_t)]

        class HardwareInput(ctypes.Structure):
            _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                        ("wParamH", wintypes.WORD)]

        class InputUnion(ctypes.Union):
            _fields_ = [("mi", MouseInput), ("ki", KeyboardInput), ("hi", HardwareInput)]

        class Input(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("data", InputUnion)]

        user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
        user32.SendInput.restype = wintypes.UINT
        if not user32.SetCursorPos(x, y):
            raise RuntimeError("Could not position pointer")
        if user32.GetForegroundWindow() != self.hwnd:
            raise RuntimeError("Foreground changed before click")
        events = (Input * 2)()
        events[0].data.mi.dwFlags = 0x0002  # MOUSEEVENTF_LEFTDOWN
        events[1].data.mi.dwFlags = 0x0004  # MOUSEEVENTF_LEFTUP
        if user32.SendInput(2, events, ctypes.sizeof(Input)) != 2:
            # Release is safe recovery, not a repeated click.
            release = Input()
            release.data.mi.dwFlags = 0x0004
            user32.SendInput(1, ctypes.byref(release), ctypes.sizeof(Input))
            raise RuntimeError("Input was not completely dispatched; resynchronize")


@dataclass
class PendingAction:
    action: Action
    before: GameState
    dispatched_at: float
    blank_evidence_count: int = 0
    blank_evidence_version: int = -1
    blank_evidence_at: float = -math.inf


class ActionController:
    """One owner of game inputs. Timeout pauses instead of retrying a click.

    Clock values passed as ``now`` must be from one monotonic clock. Observation
    timestamps may be replay timestamps, but must advance with frame versions.
    Dry-run evaluates all guards and returns ``simulated`` without calling the
    backend or claiming an action was confirmed.
    """
    def __init__(self, backend: InputBackend, layout: dict[str, Any], ruleset: Ruleset,
                 *, dry_run: bool = True, live_enabled: bool = False,
                 min_action_interval: float = 1.5, confirmation_timeout: float = 4.0,
                 min_confidence: float = 0.85, min_seconds_left: float = 2.5):
        if min_action_interval < 0 or confirmation_timeout <= 0:
            raise ValueError("Invalid timing configuration")
        self.backend, self.layout, self.ruleset = backend, layout, ruleset
        self.dry_run, self.live_enabled = dry_run, live_enabled
        self.min_action_interval = min_action_interval
        self.confirmation_timeout = confirmation_timeout
        self.min_confidence = min_confidence
        self.min_seconds_left = min_seconds_left
        self.pending: PendingAction | None = None
        self.paused = False
        self.pause_reason = ""
        self.last_dispatched_at = -math.inf
        self.last_economy_at = -math.inf
        self.last_version = -1
        self.history: list[ActionResult] = []
        self._resync_after_version = -1

    def _result(self, action: Action, status: str, reason: str, state: GameState) -> ActionResult:
        result = ActionResult(action, status, reason, state.observed_at)
        self.history.append(result)
        return result

    def _field_ok(self, state: GameState, key: str) -> bool:
        return state.field_confidence.get(key, 0.0) >= self.min_confidence

    def _shop_ok(self, state: GameState) -> bool:
        return len(state.shop) == 5 and len({s.slot for s in state.shop}) == 5 and all(
            s.confidence >= self.min_confidence and s.status in {"occupied", "empty"}
            and (s.status == "empty" or s.hero_id in self.ruleset.heroes)
            for s in state.shop)

    def _point(self, action: Action, state: GameState | None = None) -> tuple[int, int]:
        name = {"refresh": "refresh", "level": "level", "return_home": "self",
                "close_detail": "close_detail", "open_shop": "shop_toggle"}.get(action.kind, action.target)
        if action.kind == "choose" and action.metadata.get("phase", state.phase if state else None) == "encounter":
            name = f"encounter_{action.target}"
        value = self.layout.get("points", {}).get(name)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Missing calibrated point: {name}")
        nx, ny = value
        if not all(isinstance(v, (float, int)) and math.isfinite(v) and 0 <= v <= 1 for v in value):
            raise ValueError(f"Point is not normalized: {name}")
        viewport = self.layout.get("viewport")
        if viewport is None:
            size = self.layout.get("reference_size", [])
            viewport = [0, 0, *size]
        if len(viewport) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in viewport):
            raise ValueError("Invalid viewport")
        x, y, width, height = viewport
        if width <= 1 or height <= 1:
            raise ValueError("Invalid viewport size")
        return round(x + nx * (width - 1)), round(y + ny * (height - 1))

    def _cost(self, action: Action, state: GameState) -> int:
        if action.kind == "refresh":
            return self.ruleset.refresh_cost
        if action.kind == "level":
            return self.ruleset.xp_cost
        if action.kind == "buy":
            slot = next((s for s in state.shop if s.slot == action.target), None)
            if slot and isinstance(slot.cost, int) and slot.cost > 0:
                return slot.cost
        return 0

    def _guard(self, action: Action, state: GameState, now: float) -> str | None:
        if action.state_version != state.version or state.version < self.last_version:
            return "stale_state_version"
        if self.paused:
            return "paused_requires_resynchronization"
        if self.pending:
            return "another_action_pending"
        if action.kind not in {"buy", "refresh", "level", "scout", "close_detail", "return_home", "choose", "open_shop", "wait"}:
            return "unsupported_action"
        if action.kind == "wait":
            return None
        if not self.dry_run and (not self.live_enabled or self.layout.get("calibrated") is not True):
            return "live_requires_explicit_enable_and_calibration"
        if isinstance(self.backend, Win32Backend) and not self.dry_run and "viewport" not in self.layout:
            return "live_requires_bound_window_viewport"
        allowed_phases = {"encounter", "augment"} if action.kind == "choose" else {"preparation", "combat", "pve"}
        if action.kind in {"buy", "refresh", "level", "open_shop", "close_detail", "return_home"}:
            # Coarse gameplay is sufficient for economic UI operations, never
            # evidence that board models are standing in their deployment cells.
            allowed_phases.add("gameplay")
        if state.phase not in allowed_phases or not self._field_ok(state, "phase"):
            return "phase_unknown_or_not_actionable"
        if action.kind != "choose" and (state.seconds_left is None or not self._field_ok(state, "seconds_left") or state.seconds_left < self.min_seconds_left):
            return "insufficient_or_unknown_remaining_time"
        if now - self.last_dispatched_at < 0.15:
            return "input_cooldown"
        if action.kind == "choose":
            if action.metadata.get("phase", state.phase) != state.phase:
                return "choice_phase_changed"
            if state.viewed_player != "self" or not self._field_ok(state, "viewed_player"):
                return "own_board_not_confirmed"
            choice = next((c for c in state.choices if action.target == f"choice_{c.index}"), None)
            if choice is None or choice.confidence < max(0.9, self.min_confidence) or choice.text != action.metadata.get("text"):
                return "choice_unknown_or_changed"
        if action.kind == "open_shop":
            if state.viewed_player != "self" or not self._field_ok(state, "viewed_player"):
                return "own_board_not_confirmed"
            if state.detail_open is not False or not self._field_ok(state, "detail_open"):
                return "detail_panel_not_confirmed_closed"
            if self._shop_ok(state):
                return "shop_already_open"
            if state.shop_visible is not False or not self._field_ok(state, "shop_visible"):
                return "shop_not_confirmed_closed"
        if action.kind in {"buy", "refresh", "level"}:
            if state.owned_copies is not None or state.ownership_complete:
                ownership_error = ownership_summary_error(state, self.ruleset)
                if ownership_error:
                    return f"ownership_summary_invalid:{ownership_error}"
            if now - self.last_economy_at < self.min_action_interval:
                return "economy_cooldown"
            if state.viewed_player != "self" or not self._field_ok(state, "viewed_player"):
                return "own_board_not_confirmed"
            if not isinstance(state.gold, int) or state.gold < 0 or not self._field_ok(state, "gold"):
                return "gold_unknown_or_low_confidence"
            if state.detail_open is not False:
                return "detail_panel_not_confirmed_closed"
            if action.kind in {"buy", "refresh"} and not self._shop_ok(state):
                return "shop_unknown_or_low_confidence"
            if action.kind == "buy":
                if state.round is None or not self._field_ok(state, "round"):
                    return "round_unknown_or_low_confidence"
                slot = next((s for s in state.shop if s.slot == action.target), None)
                if slot is None or slot.status != "occupied" or slot.hero_id != action.hero_id:
                    return "target_card_changed"
                if not isinstance(slot.cost, int) or slot.cost <= 0:
                    return "visible_purchase_price_unknown"
                empty_bench = any(s.status == "empty" and s.confidence >= self.min_confidence for s in state.bench.values())
                merge_copies = sum(s.hero_id == slot.hero_id and s.star == 1 and s.status == "occupied"
                                   and s.confidence >= self.min_confidence
                                   for s in [*state.board.values(), *state.bench.values()])
                if state.ownership_complete:
                    empty_bench = empty_bench or (state.bench_free is not None and state.bench_free > 0)
                    stars = state.owned_star_counts.get(slot.hero_id, [])
                    if len(stars) == 3:
                        merge_copies = max(merge_copies, stars[0])
                if not empty_bench and merge_copies < 2:
                    return "bench_capacity_unknown_or_full"
            cost = self._cost(action, state)
            try:
                floor = max(0, int(action.metadata.get("min_gold", 0)))
                reserve = max(0, int(action.metadata.get("reserve_gold", 0)))
                remaining = action.metadata.get("max_spend_remaining")
                if remaining is not None and cost > float(remaining):
                    return "spend_budget_exceeded"
            except (TypeError, ValueError):
                return "invalid_budget"
            if cost <= 0 or state.gold - cost < floor + reserve:
                return "insufficient_gold_or_reserve"
        if action.kind == "scout" and action.metadata.get("stage") == "inspect_slot":
            if state.viewed_player != action.metadata.get("player_id"):
                return "wrong_scouted_player"
            if not self._field_ok(state, "viewed_player") or state.detail_open is not False:
                return "detail_must_be_confirmed_closed_first"
            scout_layout = self.layout.get("scouting", {})
            if action.target.startswith("board_"):
                phases = scout_layout.get("static_board_phases", ["preparation"])
            elif action.target.startswith("bench_"):
                phases = scout_layout.get("bench_phases", ["preparation"])
            else:
                return "unknown_inspection_position"
            if state.phase not in phases:
                return "inspection_requires_static_position"
        if action.kind in {"scout", "return_home"} and action.metadata.get("stage") != "inspect_slot":
            scout_layout = self.layout.get("scouting", {})
            phases = scout_layout.get("switch_phases", ["preparation", "combat", "pve"])
            if action.kind == "return_home":
                phases = scout_layout.get("return_home_phases", [*phases, "gameplay"])
            if state.phase not in phases:
                return "player_switch_not_allowed_in_phase"
        return None

    def submit(self, action: Action, state: GameState, now: float | None = None) -> ActionResult:
        now = time.monotonic() if now is None else now
        reason = self._guard(action, state, now)
        if reason:
            return self._result(action, "rejected", reason, state)
        self.last_version = max(self.last_version, state.version)
        if action.kind == "wait":
            return self._result(action, "confirmed", "wait_without_input", state)
        try:
            point = self._point(action, state)
        except (ValueError, TypeError) as exc:
            return self._result(action, "rejected", str(exc), state)
        if self.dry_run:
            return self._result(action, "simulated", f"would_click:{point[0]},{point[1]}", state)
        before = deepcopy(state)
        try:
            self.backend.click(*point)
        except Exception as exc:  # noqa: BLE001 - any backend failure must pause, never retry input
            self.paused, self.pause_reason = True, f"input_failure:{exc}"
            self._resync_after_version = state.version
            return self._result(action, "failed", self.pause_reason, state)
        self.last_dispatched_at = now
        if action.kind in {"buy", "refresh", "level"}:
            self.last_economy_at = now
        self.pending = PendingAction(deepcopy(action), before, now)
        return self._result(action, "pending", "awaiting_new_observation", state)

    def _held(self, state: GameState, hero: str | None) -> int:
        return sum(3 ** (s.star - 1) for s in [*state.board.values(), *state.bench.values()]
                   if s.hero_id == hero and s.status == "occupied" and s.star in {1, 2, 3}
                   and s.confidence >= self.min_confidence)

    def _blank_purchase_evidence(self, pending: PendingAction, state: GameState) -> bool:
        """Two fresh contextual observations distinguish a bought slot from OCR loss."""
        before, action = pending.before, pending.action
        if before.round is None or before.round != state.round:
            pending.blank_evidence_count = 0
            return False
        old_slots = {s.slot: s for s in before.shop}
        new_slots = {s.slot: s for s in state.shop}
        if len(state.shop) != 5 or set(old_slots) != set(new_slots) or len(new_slots) != 5:
            pending.blank_evidence_count = 0
            return False
        slot = new_slots[action.target]
        if slot.status != "unknown" or slot.source != "ocr_blank" or slot.observed_at <= before.observed_at:
            pending.blank_evidence_count = 0
            return False
        confidence = min(state.field_confidence.get("gold", 0), state.field_confidence.get("viewed_player", 0))
        for key, old in old_slots.items():
            if key == action.target:
                confidence = min(confidence, old.confidence)
                continue
            new = new_slots[key]
            if new.observed_at <= before.observed_at or old.confidence < self.min_confidence:
                pending.blank_evidence_count = 0
                return False
            previously_bought = (old.status == "empty" and old.source in {"confirmed_purchase", "confirmed_purchase_empty"}
                                  and new.status == "unknown" and new.source == "ocr_blank")
            unchanged = (new.confidence >= self.min_confidence and
                         (old.status, old.hero_id, old.cost) == (new.status, new.hero_id, new.cost))
            if not (previously_bought or unchanged):
                pending.blank_evidence_count = 0
                return False
            confidence = min(confidence, old.confidence if previously_bought else new.confidence)
        if state.version <= pending.blank_evidence_version or state.observed_at <= pending.blank_evidence_at:
            return False
        pending.blank_evidence_version = state.version
        pending.blank_evidence_at = state.observed_at
        pending.blank_evidence_count += 1
        if pending.blank_evidence_count < 2:
            return False
        state.shop = [replace(s, status="empty", hero_id=None, star=None, cost=None,
                              confidence=confidence, source="confirmed_purchase")
                      if s.slot == action.target else s for s in state.shop]
        return True

    def _confirmed(self, pending: PendingAction, state: GameState) -> bool:
        action, before = pending.action, pending.before
        if state.version <= before.version or state.observed_at <= before.observed_at:
            return False
        if action.kind in {"buy", "refresh", "level"}:
            if state.viewed_player != "self" or not self._field_ok(state, "viewed_player") or not self._field_ok(state, "gold"):
                pending.blank_evidence_count = 0
                return False
            if state.gold != before.gold - self._cost(action, before):
                pending.blank_evidence_count = 0
                return False
            if action.kind == "refresh":
                # The same five cards can legally recur. Gold debit + a complete
                # new-frame shop is stronger evidence than changed hero names.
                return self._shop_ok(state) and all(s.observed_at > before.observed_at for s in state.shop)
            if action.kind == "buy":
                slot = next((s for s in state.shop if s.slot == action.target), None)
                slot_changed = slot is not None and slot.observed_at > before.observed_at and slot.confidence >= self.min_confidence and (
                    slot.status == "empty" or (slot.status == "occupied" and slot.hero_id != action.hero_id))
                held_changed = self._held(state, action.hero_id) > self._held(before, action.hero_id)
                return slot_changed or held_changed or self._blank_purchase_evidence(pending, state)
            return (self._field_ok(state, "level") and self._field_ok(before, "level")
                    and state.level is not None and before.level is not None and state.level > before.level) or (
                self._field_ok(state, "xp") and self._field_ok(before, "xp")
                and state.xp is not None and before.xp is not None and state.xp > before.xp)
        if action.kind == "return_home":
            return state.viewed_player == "self" and self._field_ok(state, "viewed_player")
        if action.kind == "open_shop":
            return (state.viewed_player == "self" and self._field_ok(state, "viewed_player")
                    and self._shop_ok(state) and all(s.observed_at > before.observed_at for s in state.shop))
        if action.kind == "choose":
            return (state.phase not in {"unknown", "transition", before.phase}
                    and self._field_ok(state, "phase") and state.viewed_player == "self"
                    and self._field_ok(state, "viewed_player"))
        if action.kind == "close_detail":
            return state.detail_open is False and self._field_ok(state, "detail_open")
        if action.kind == "scout":
            player_ok = state.viewed_player == action.metadata.get("player_id", action.target) and self._field_ok(state, "viewed_player")
            if action.metadata.get("stage") == "inspect_slot":
                phases_key = "static_board_phases" if action.target.startswith("board_") else "bench_phases"
                phases = self.layout.get("scouting", {}).get(phases_key, ["preparation"])
                if state.phase not in phases or not self._field_ok(state, "phase"):
                    return False  # a phase transition can invalidate the clicked grid position
                return bool(player_ok and state.detail_open is True and state.detail
                            and state.detail.observed_at > before.observed_at
                            and state.detail.confidence >= self.min_confidence
                            and state.detail.status in {"occupied", "empty"}
                            and (state.detail.status == "empty" or state.detail.hero_id in self.ruleset.heroes))
            return player_ok
        return False

    def observe(self, state: GameState, now: float | None = None) -> ActionResult | None:
        now = time.monotonic() if now is None else now
        if state.version < self.last_version:
            return None
        self.last_version = max(self.last_version, state.version)
        if self.pending is None:
            return None
        pending = self.pending
        if self._confirmed(pending, state):
            self.pending = None
            return self._result(pending.action, "confirmed", "confirmed_by_new_observation", state)
        if now - pending.dispatched_at >= self.confirmation_timeout:
            self.pending = None
            action = pending.action
            if (action.kind == "scout" and action.metadata.get("stage") == "inspect_slot"
                    and state.version > pending.before.version and state.observed_at > pending.before.observed_at
                    and state.viewed_player == action.metadata.get("player_id")
                    and self._field_ok(state, "viewed_player")
                    and state.detail_open is False and self._field_ok(state, "detail_open")):
                # A non-economic inspection can safely advance past a position
                # that yielded no detail. This is unknown, never an empty cell,
                # and the same click is not retried.
                return self._result(action, "unresolved", "inspection_no_detail_position_unknown", state)
            self.paused, self.pause_reason = True, "confirmation_timeout_requires_resynchronization"
            self._resync_after_version = state.version
            return self._result(pending.action, "failed", self.pause_reason, state)
        return None

    def resynchronize(self, state: GameState) -> bool:
        """Explicit recovery from a fresh complete own-board observation."""
        if (self.pending is not None or state.version <= self._resync_after_version
                or state.version < self.last_version or state.viewed_player != "self"
                or state.detail_open is not False or not self._shop_ok(state)
                or state.gold is None or not self._field_ok(state, "gold")
                or not self._field_ok(state, "viewed_player") or not self._field_ok(state, "detail_open")):
            return False
        self.last_version = state.version
        self.paused, self.pause_reason = False, ""
        return True
