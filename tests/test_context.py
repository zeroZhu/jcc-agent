import pytest

from jcc_agent.context import UIContextTracker
from jcc_agent.models import Action, GameState


def raw(version, player="unknown", confidence=0.0, phase="gameplay"):
    state = GameState(version=version, observed_at=float(version), phase=phase, viewed_player=player)
    state.field_confidence["viewed_player"] = confidence
    if player != "unknown":
        state.field_sources["viewed_player"] = "ocr"
        state.field_observed_at["viewed_player"] = float(version)
    return state


def reviewed():
    return {"ui_context_reviewed": True, "viewed_player": "self", "observed_at": 0}


@pytest.mark.parametrize("initial", [None, {}, {"viewed_player": "self"},
                                    {"ui_context_reviewed": "true", "viewed_player": "self"},
                                    {"ui_context_reviewed": True, "viewed_player": "opponent_0"}])
def test_unknown_does_not_default_to_self_without_explicit_review(initial):
    context = UIContextTracker(initial)
    assert context.fuse(raw(1)).viewed_player == "unknown"
    assert not context.valid


def test_reviewed_context_preserves_original_timestamp_and_non_ocr_source():
    context = UIContextTracker(reviewed())
    for version in (1, 2, 20):
        frame = raw(version)
        fused = context.fuse(frame)
        assert fused.viewed_player == "self"
        assert fused.field_sources["viewed_player"] == "reviewed_session_context"
        assert fused.field_observed_at["viewed_player"] == 0
        assert fused.field_confidence["viewed_player"] == 1
        assert frame.viewed_player == "unknown"


def test_fresh_ocr_updates_source_time_but_context_never_refreshes_itself():
    context = UIContextTracker(reviewed())
    fresh = context.fuse(raw(2, "self", 0.97))
    assert fresh.field_sources["viewed_player"] == "ocr"
    assert context.source_observed_at == 2
    carried = context.fuse(raw(3))
    carried.version, carried.observed_at = 4, 4
    assert context.fuse(carried).field_observed_at["viewed_player"] == 2
    assert context.source_observed_at == 2


@pytest.mark.parametrize("action", [Action("scout", "opponent_0"), Action("return_home"),
                                    Action("choose", metadata={"changes_view": True})])
def test_navigation_invalidates_before_input_and_failed_input_never_restores(action):
    context = UIContextTracker(reviewed())
    old = context.fuse(raw(1))
    assert context.before_dispatch(action)
    assert not context.valid
    assert context.fuse(raw(2)).viewed_player == "unknown"
    old.version, old.observed_at = 3, 3
    assert context.fuse(old).viewed_player == "unknown"
    assert not context.valid
    restored = context.fuse(raw(4, "opponent_0", 0.99))
    assert restored.viewed_player == "opponent_0" and context.valid


def test_buying_does_not_invalidate_the_selected_player():
    context = UIContextTracker(reviewed())
    assert not context.before_dispatch(Action("buy", "shop_0"))
    assert context.fuse(raw(1)).viewed_player == "self"


def test_ocr_conflict_wins_in_current_frame_and_invalidates_cached_context():
    context = UIContextTracker(reviewed())
    conflicting = context.fuse(raw(1, "opponent_0", 0.99))
    assert conflicting.viewed_player == "opponent_0"
    assert conflicting.field_sources["viewed_player"] == "ocr"
    assert not context.valid
    assert context.fuse(raw(2)).viewed_player == "unknown"
    assert context.fuse(raw(3, "opponent_0", 0.99)).viewed_player == "opponent_0"
    assert context.valid


@pytest.mark.parametrize("phase", ["ended", "menu", "lobby", "login", "disconnected"])
def test_ended_or_menu_invalidates_context(phase):
    context = UIContextTracker(reviewed())
    assert context.fuse(raw(1, phase=phase)).viewed_player == "unknown"
    assert not context.valid
    assert context.fuse(raw(2)).viewed_player == "unknown"


def test_low_confidence_or_old_ocr_cannot_recover_invalid_context():
    context = UIContextTracker(reviewed())
    context.fuse(raw(2))
    context.invalidate("focus_lost")
    context.fuse(raw(3, "self", 0.4))
    assert not context.valid
    context.fuse(raw(1, "self", 0.99))
    assert not context.valid
    context.fuse(raw(4, "self", 0.99))
    assert context.valid


def test_runtime_can_invalidate_on_external_input_and_explicit_identity_conflicts():
    context = UIContextTracker(reviewed())
    context.invalidate("external_input")
    assert context.fuse(raw(1)).viewed_player == "unknown"
    context.fuse(raw(2, "self", 0.99))
    conflict = raw(3)
    conflict.warnings.append("identity_conflict: multiple player labels")
    assert context.fuse(conflict).viewed_player == "unknown"
    assert not context.valid
