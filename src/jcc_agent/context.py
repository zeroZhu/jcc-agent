"""Provenance-preserving UI context under exclusive, supervised input ownership."""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from .models import Action, GameState


class UIContextTracker:
    """Carry an explicitly reviewed view through OCR gaps, never assume self.

    Runtime must stop on focus loss, external mouse/keyboard input, reconnects
    and window changes. This class cannot detect those events. Call
    before_dispatch immediately before any potentially navigating input, even
    if the backend later fails. Only fresh positive OCR can recover afterward.
    """
    INVALID_PHASES = frozenset({"ended", "menu", "lobby", "login", "disconnected", "reconnect"})
    NAVIGATION_KINDS = frozenset({"scout", "return_home", "navigate", "spectate", "reconnect", "new_match"})

    def __init__(self, initial: dict[str, Any] | None = None, *, min_confidence: float = 0.85):
        self.min_confidence = min_confidence
        self.valid = False
        self.player = "unknown"
        self.confidence = 0.0
        self.source_observed_at = 0.0
        self.origin = "uninitialized"
        self.invalidation_reason = "no_reviewed_context"
        self._last_frame_version = -1
        self._last_frame_at = -math.inf
        initial = initial or {}
        if initial.get("ui_context_reviewed") is True and initial.get("viewed_player") == "self":
            timestamp = initial.get("observed_at", 0.0)
            if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("Initial UI context needs a finite session-relative timestamp")
            self.valid = True
            self.player = "self"
            self.confidence = 1.0  # explicit operator assertion, not OCR accuracy
            self.source_observed_at = float(timestamp)
            self.origin = "reviewed_initial_context"
            self.invalidation_reason = ""

    def invalidate(self, reason: str) -> None:
        self.valid = False
        self.player = "unknown"
        self.confidence = 0.0
        self.invalidation_reason = str(reason)

    def before_dispatch(self, action: Action) -> bool:
        """Invalidate before an attempted navigation; failure never restores it."""
        if action.kind in self.NAVIGATION_KINDS or action.metadata.get("changes_view") is True:
            self.invalidate(f"navigation_attempt:{action.kind}")
            return True
        return False

    def fuse(self, observation: GameState) -> GameState:
        state = deepcopy(observation)
        new_frame = state.version > self._last_frame_version and state.observed_at > self._last_frame_at
        self._last_frame_version = max(self._last_frame_version, state.version)
        self._last_frame_at = max(self._last_frame_at, state.observed_at)
        if state.phase in self.INVALID_PHASES:
            self.invalidate(f"ui_phase:{state.phase}")
            return state
        if any(w.startswith(("viewed_player_conflict", "identity_conflict", "unexpected_navigation")) for w in state.warnings):
            self.invalidate("explicit_identity_conflict")
            state.viewed_player = "unknown"
            state.field_confidence["viewed_player"] = 0.0
            state.field_sources["viewed_player"] = "invalidated_identity_conflict"
            return state
        source = state.field_sources.get("viewed_player", "ocr")
        source_time = state.field_observed_at.get("viewed_player", state.observed_at)
        confidence = state.field_confidence.get("viewed_player", 0.0)
        fresh_ocr = (new_frame and source.startswith("ocr") and source_time == state.observed_at
                     and isinstance(confidence, (float, int)) and math.isfinite(confidence)
                     and confidence >= self.min_confidence)
        if state.viewed_player != "unknown" and fresh_ocr:
            if self.valid and state.viewed_player != self.player:
                # Current raw OCR remains visible to downstream guards. A
                # contradictory frame does not silently authorize its reuse.
                self.invalidate("fresh_ocr_conflicts_with_session_context")
                return state
            self.valid, self.player, self.confidence = True, state.viewed_player, confidence
            self.source_observed_at = source_time
            self.origin = "fresh_ocr"
            self.invalidation_reason = ""
            state.field_sources["viewed_player"] = source
            state.field_observed_at["viewed_player"] = source_time
            return state
        if not source.startswith("ocr") and state.viewed_player != "unknown":
            # A previously fused state must not bootstrap itself as fresh OCR,
            # particularly after a navigation invalidated its original context.
            state.viewed_player = "unknown"
            state.field_confidence["viewed_player"] = 0.0
        if state.viewed_player == "unknown" and self.valid:
            state.viewed_player = self.player
            state.field_confidence["viewed_player"] = self.confidence
            state.field_sources["viewed_player"] = "reviewed_session_context"
            state.field_observed_at["viewed_player"] = self.source_observed_at
            warning = "viewed_player_from_session_context_not_fresh_ocr"
            if warning not in state.warnings:
                state.warnings.append(warning)
        return state
