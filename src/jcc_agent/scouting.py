"""Budgeted round-robin scouting through the shared action controller."""
from __future__ import annotations

import time
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import replace

from .control import ActionController
from .models import Action, ActionResult, GameState, PlayerSnapshot, SlotObservation


class ScoutingScheduler:
    """Resume the cursor across rounds; every inspection closes old details first.

    Opponent IDs must also be calibrated point names (normally opponent_0..6),
    unless ``player_points`` supplies a mapping. Snapshots cover board and bench
    positions, never inferred hidden holdings. Unread slots stay unknown.

    Call start_round once per round, then tick with fresh frames and the *same*
    ActionController used by the buying loop. While active, give scouting sole
    ownership of scheduling; the controller still serializes pending inputs.
    """
    def __init__(self, opponents: Iterable[str], slot_ids: Iterable[str] | None = None,
                 *, budget_seconds: float = 8.0, min_slot_budget: float = 1.0,
                 home_reserve: float = 2.5, player_points: dict[str, str] | None = None,
                 layout: dict | None = None):
        self.opponents = list(opponents)
        self.slot_ids = list(slot_ids) if slot_ids is not None else [
            *[f"board_{i}" for i in range(28)], *[f"bench_{i}" for i in range(9)]]
        if not self.opponents or not self.slot_ids:
            raise ValueError("At least one opponent and slot are required")
        if len(set(self.opponents)) != len(self.opponents) or len(set(self.slot_ids)) != len(self.slot_ids):
            raise ValueError("Duplicate players or positions would double count")
        if budget_seconds <= 0 or min_slot_budget <= 0 or home_reserve < 0:
            raise ValueError("Invalid scouting budget")
        self.budget_seconds, self.min_slot_budget, self.home_reserve = budget_seconds, min_slot_budget, home_reserve
        self.player_points = player_points or {}
        self.layout = layout or {}
        self.snapshots: dict[str, PlayerSnapshot] = {}
        self.player_index = 0
        self.slot_index = 0
        self.stage = "select_player"
        self.active = False
        self.paused = False
        self.round_id: str | None = None
        self.deadline = 0.0
        self._inflight: Action | None = None
        self._cycle_started = False

    @property
    def current_player(self) -> str:
        return self.opponents[self.player_index]

    def start_round(self, round_id: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if round_id == self.round_id:
            return  # callers cannot renew the time budget each frame
        self.round_id = round_id
        self.deadline = now + self.budget_seconds
        self.active = not self.paused
        if self._inflight is None:
            self.stage = "select_player"

    def _action(self, state: GameState, kind: str, target: str, stage: str) -> Action:
        return Action(kind, target, state_version=state.version,
                      reason="budgeted_scouting",
                      metadata={"scouting": True, "stage": stage,
                                "player_id": self.current_player,
                                "slot": self.slot_ids[self.slot_index]})

    def next_action(self, state: GameState, now: float | None = None) -> Action | None:
        now = time.monotonic() if now is None else now
        if not self.active or self.paused or self._inflight is not None:
            return None
        if state.phase not in {"preparation", "combat", "pve", "gameplay"} or state.field_confidence.get("phase", 0) < 0.85:
            return None  # unknown UI cannot authorize even a return-home click
        scouting = self.layout.get("scouting", {})
        switch_phases = scouting.get("switch_phases", ["preparation", "combat", "pve"])
        readable_positions = any(
            state.phase in scouting.get("static_board_phases" if slot.startswith("board_") else "bench_phases", ["preparation"])
            for slot in self.slot_ids)
        if state.phase == "gameplay" or state.phase not in switch_phases or not readable_positions:
            self.stage = "return_home"
        budget_left = self.deadline - now
        phase_left = state.seconds_left if state.seconds_left is not None else 0
        if min(budget_left, phase_left) < self.min_slot_budget + self.home_reserve or self.stage == "return_home":
            self.stage = "return_home"
            if state.viewed_player == "self" and state.field_confidence.get("viewed_player", 0) >= 0.85:
                self.active = False
                self.stage = "select_player"
                return None
            return self._action(state, "return_home", "self", "return_home")
        if state.viewed_player != self.current_player and self.stage != "select_player":
            self.stage = "select_player"
        if self.stage in {"close_detail", "inspect_slot"}:
            scouting = self.layout.get("scouting", {})
            checked = 0
            while checked < len(self.slot_ids):
                slot = self.slot_ids[self.slot_index]
                phases = scouting.get("static_board_phases" if slot.startswith("board_") else "bench_phases", ["preparation"])
                if state.phase in phases:
                    break
                self._advance_slot()
                checked += 1
                if self.stage == "select_player":
                    break
            if checked == len(self.slot_ids):
                self.stage = "return_home"
                return self._action(state, "return_home", "self", "return_home")
        if self.stage == "select_player":
            return self._action(state, "scout", self.player_points.get(self.current_player, self.current_player), "select_player")
        if self.stage == "close_detail":
            return self._action(state, "close_detail", "close_detail", "close_detail")
        if self.stage == "inspect_slot":
            return self._action(state, "scout", self.slot_ids[self.slot_index], "inspect_slot")
        return None

    def _begin_cycle(self, observed_at: float) -> None:
        if not self._cycle_started:
            # Replace the previous cycle, rather than summing snapshots. Never
            # label untouched positions empty or carry an old card as current.
            self.snapshots[self.current_player] = PlayerSnapshot(
                self.current_player,
                {slot: SlotObservation(slot, observed_at=observed_at) for slot in self.slot_ids},
                observed_at=observed_at)
            self._cycle_started = True

    def _advance_slot(self) -> None:
        self.slot_index += 1
        if self.slot_index == len(self.slot_ids):
            snapshot = self.snapshots[self.current_player]
            snapshot.complete = all(s.status in {"occupied", "empty"} for s in snapshot.slots.values())
            self.slot_index = 0
            self.player_index = (self.player_index + 1) % len(self.opponents)
            self.stage = "select_player"
            self._cycle_started = False
        else:
            self.stage = "close_detail"

    def on_result(self, result: ActionResult, state: GameState) -> None:
        action = result.action
        if not action.metadata.get("scouting"):
            return
        stage = action.metadata.get("stage")
        if result.status == "pending":
            self._inflight = action
            return
        if result.status in {"simulated", "rejected"}:
            return  # dry-run is not evidence that input occurred
        self._inflight = None
        if result.status == "unresolved" and stage == "inspect_slot":
            self._begin_cycle(state.observed_at)
            snapshot = self.snapshots[self.current_player]
            snapshot.slots[self.slot_ids[self.slot_index]] = SlotObservation(
                self.slot_ids[self.slot_index], observed_at=state.observed_at,
                source="inspection_unresolved")
            snapshot.observed_at = state.observed_at
            self._advance_slot()
            return
        if result.status == "failed":
            if stage == "inspect_slot":
                self._begin_cycle(state.observed_at)
                snapshot = self.snapshots[self.current_player]
                snapshot.slots[self.slot_ids[self.slot_index]] = SlotObservation(
                    self.slot_ids[self.slot_index], observed_at=state.observed_at,
                    source="inspection_failed")
                snapshot.observed_at = state.observed_at
                self._advance_slot()
            self.paused, self.active = True, False
            return
        if result.status != "confirmed":
            return
        if stage == "select_player":
            self._begin_cycle(state.observed_at)
            self.stage = "close_detail"
        elif stage == "close_detail":
            self.stage = "inspect_slot"
        elif stage == "inspect_slot":
            self._begin_cycle(state.observed_at)
            snapshot = self.snapshots[self.current_player]
            detail = state.detail
            if detail is None:
                detail = SlotObservation(self.slot_ids[self.slot_index], observed_at=state.observed_at)
            snapshot.slots[self.slot_ids[self.slot_index]] = replace(
                detail, slot=self.slot_ids[self.slot_index], observed_at=state.observed_at)
            snapshot.observed_at = state.observed_at
            self._advance_slot()
        elif stage == "return_home":
            self.active = False
            self.stage = "select_player"

    def tick(self, state: GameState, controller: ActionController,
             now: float | None = None) -> ActionResult | None:
        now = time.monotonic() if now is None else now
        observed = controller.observe(state, now)
        if observed is not None:
            self.on_result(observed, state)
            return observed
        if controller.pending is not None or controller.paused:
            return None
        action = self.next_action(state, now)
        if action is None:
            return None
        result = controller.submit(action, state, now)
        self.on_result(result, state)
        return result

    def resume(self, state: GameState, controller: ActionController) -> bool:
        """Only explicit recovery resumes after a failed inspection."""
        if controller.paused or controller.pending is not None or state.viewed_player != "self":
            return False
        self.paused = False
        self.active = False  # a new round receives a fresh bounded budget
        self._inflight = None
        self.stage = "select_player"
        return True

    def export_snapshots(self) -> dict[str, PlayerSnapshot]:
        return deepcopy(self.snapshots)
