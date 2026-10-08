from copy import deepcopy

import pytest

from jcc_agent.control import ActionController, RecordingBackend
from jcc_agent.models import Action, Choice, GameState, Hero, Ruleset, SlotObservation


def rules():
    return Ruleset("test", "test", {"a": Hero("a", "甲", 2), "b": Hero("b", "乙", 1)}, {1: 20, 2: 20}, {})


def state(version=1, gold=20):
    return GameState(
        version=version, observed_at=float(version), phase="preparation", round="2-1", gold=gold, level=6,
        seconds_left=20, viewed_player="self", detail_open=False,
        field_confidence={k: 0.99 for k in ["gold", "phase", "round", "seconds_left", "viewed_player", "detail_open"]},
        shop=[SlotObservation(f"shop_{i}", "occupied", "a", 1, 2, 0.99, float(version)) for i in range(5)],
        bench={"bench_0": SlotObservation("bench_0", "empty", confidence=0.99, observed_at=float(version))})


def controller(**kwargs):
    backend = RecordingBackend()
    layout = {"calibrated": True, "reference_size": [1000, 600], "viewport": [100, 50, 1000, 600],
              "points": {"shop_0": [0.2, 0.8], "refresh": [0.1, 0.8], "level": [0.1, 0.7],
                         "self": [0.9, 0.5], "opponent_0": [0.9, 0.2], "board_0": [0.5, 0.5],
                         "bench_0": [0.5, 0.8], "shop_toggle": [0.1, 0.9],
                         "close_detail": [0.7, 0.4], "choice_0": [0.3, 0.5], "encounter_choice_0": [0.1, 0.3]}}
    options = {"dry_run": False, "live_enabled": True}
    options.update(kwargs)
    return ActionController(backend, layout, rules(), **options), backend


def test_dry_run_never_calls_backend_or_creates_pending():
    control, backend = controller(dry_run=True)
    result = control.submit(Action("refresh", state_version=1), state(), now=0)
    assert result.status == "simulated"
    assert control.pending is None
    assert backend.events == []


@pytest.mark.parametrize("change, reason", [
    ({"viewed_player": "opponent_0"}, "own_board_not_confirmed"),
    ({"phase": "augment"}, "phase_unknown_or_not_actionable"),
    ({"gold": None}, "gold_unknown_or_low_confidence"),
    ({"seconds_left": 1}, "insufficient_or_unknown_remaining_time"),
    ({"detail_open": None}, "detail_panel_not_confirmed_closed"),
])
def test_unsafe_economy_observations_never_click(change, reason):
    control, backend = controller()
    observed = state()
    for key, value in change.items():
        setattr(observed, key, value)
    assert control.submit(Action("refresh", state_version=1), observed, now=0).reason == reason
    assert backend.events == []


def test_stale_actions_and_frames_are_rejected():
    control, backend = controller()
    assert control.submit(Action("refresh", state_version=0), state(), 0).reason == "stale_state_version"
    control.observe(state(3), 0)
    assert control.submit(Action("refresh", state_version=1), state(), 0).reason == "stale_state_version"
    assert backend.events == []


def test_low_confidence_shop_and_gold_block_spending():
    control, backend = controller()
    observed = state()
    observed.shop[3].confidence = 0.5
    assert control.submit(Action("refresh", state_version=1), observed, 0).reason == "shop_unknown_or_low_confidence"
    observed = state()
    observed.field_confidence["gold"] = 0.5
    assert control.submit(Action("refresh", state_version=1), observed, 0).reason == "gold_unknown_or_low_confidence"
    assert backend.events == []


def test_refresh_same_cards_needs_gold_debit_and_new_shop_observation():
    control, backend = controller()
    action = Action("refresh", state_version=1)
    assert control.submit(action, state(), 0).status == "pending"
    unchanged = state(2, 20)
    assert control.observe(unchanged, 0.5) is None
    stale_shop = state(3, 18)
    for slot in stale_shop.shop:
        slot.observed_at = 1
    assert control.observe(stale_shop, 0.6) is None
    assert control.observe(state(4, 18), 0.7).status == "confirmed"
    assert len(backend.events) == 1
    assert control.submit(Action("refresh", state_version=4), state(4, 18), 1).reason == "economy_cooldown"
    assert control.submit(Action("refresh", state_version=4), state(4, 18), 1.5).status == "pending"


def test_changed_names_alone_do_not_confirm_refresh_and_timeout_never_reclicks():
    control, backend = controller()
    control.submit(Action("refresh", state_version=1), state(), 0)
    changed = state(2)
    changed.shop[0].hero_id = "b"
    assert control.observe(changed, 1) is None
    assert control.submit(Action("refresh", state_version=2), changed, 2).reason == "another_action_pending"
    assert control.observe(changed, 4).status == "failed"
    assert control.paused
    assert control.submit(Action("refresh", state_version=2), changed, 5).reason == "paused_requires_resynchronization"
    assert not control.resynchronize(changed)
    assert control.resynchronize(state(3))
    assert len(backend.events) == 1


def test_buy_needs_gold_and_slot_or_inventory_evidence():
    control, backend = controller()
    action = Action("buy", "shop_0", "a", 1)
    assert control.submit(action, state(), 0).status == "pending"
    debit_only = state(2, 18)
    assert control.observe(debit_only, 1) is None
    purchased = state(3, 18)
    purchased.shop[0] = SlotObservation("shop_0", "empty", confidence=0.99, observed_at=3)
    assert control.observe(purchased, 1.5).status == "confirmed"
    assert backend.events == [(300, 529)]


def test_visible_discounted_price_overrides_pool_tier_and_missing_price_blocks():
    control, backend = controller()
    discounted = state()
    discounted.shop[0].cost = None
    action = Action("buy", "shop_0", "a", 1)
    assert control.submit(action, discounted, 0).reason == "visible_purchase_price_unknown"
    discounted.shop[0].cost = 1  # one-gold shop, hero's pool tier remains two
    assert control.submit(action, discounted, 0).status == "pending"
    bought = state(2, 19)
    bought.shop[0] = SlotObservation("shop_0", "empty", confidence=0.99, observed_at=2)
    assert control.observe(bought, 1).status == "confirmed"
    assert len(backend.events) == 1


def test_full_bench_only_allows_observed_immediate_merge():
    control, backend = controller()
    observed = state()
    observed.bench = {f"bench_{i}": SlotObservation(f"bench_{i}", "occupied", "a", 1, 2, 0.99, 1)
                      for i in range(2)}
    assert control.submit(Action("buy", "shop_0", "a", 1), observed, 0).status == "pending"
    merged = state(2, 18)
    merged.bench = {"bench_0": SlotObservation("bench_0", "occupied", "a", 2, 2, 0.99, 2)}
    assert control.observe(merged, 1).status == "confirmed"
    assert len(backend.events) == 1


def test_budget_and_live_enable_are_enforced():
    control, backend = controller()
    assert control.submit(Action("refresh", state_version=1, metadata={"min_gold": 19}), state(), 0).reason == "insufficient_gold_or_reserve"
    assert control.submit(Action("buy", "shop_0", "a", 1, metadata={"max_spend_remaining": 1}), state(), 0).reason == "spend_budget_exceeded"
    control.live_enabled = False
    assert control.submit(Action("refresh", state_version=1), state(), 0).reason == "live_requires_explicit_enable_and_calibration"
    assert backend.events == []


def test_input_exception_pauses_without_retry():
    control, _ = controller()
    class BrokenBackend:
        def click(self, x, y):
            raise RuntimeError("foreground changed")
    control.backend = BrokenBackend()
    assert control.submit(Action("refresh", state_version=1), state(), 0).status == "failed"
    assert control.paused


def test_choose_uses_exact_visible_text_and_does_not_require_hidden_gold():
    control, backend = controller()
    observed = state()
    observed.phase, observed.gold, observed.seconds_left = "augment", None, None
    observed.choices = [Choice(0, "明确选项", 0.99)]
    action = Action("choose", "choice_0", state_version=1, metadata={"text": "错误选项"})
    assert control.submit(action, observed, 0).reason == "choice_unknown_or_changed"
    action.metadata["text"] = "明确选项"
    assert control.submit(action, observed, 0).status == "pending"
    still_open = deepcopy(observed)
    still_open.version, still_open.observed_at = 2, 2
    assert control.observe(still_open, 1) is None
    assert control.observe(state(3), 1.5).status == "confirmed"
    assert len(backend.events) == 1


def test_detail_unknown_is_not_a_closed_panel():
    control, backend = controller()
    observed = state()
    observed.viewed_player = "opponent_0"
    observed.detail_open = None
    inspect = Action("scout", "board_0", state_version=1,
                     metadata={"stage": "inspect_slot", "player_id": "opponent_0"})
    assert control.submit(inspect, observed, 0).reason == "detail_must_be_confirmed_closed_first"
    close = Action("close_detail", state_version=1)
    assert control.submit(close, observed, 0).status == "pending"
    unreadable = deepcopy(observed)
    unreadable.version, unreadable.observed_at = 2, 2
    assert control.observe(unreadable, 1) is None
    unreadable.detail_open = False
    assert control.observe(unreadable, 2).status == "confirmed"
    assert len(backend.events) == 1


def blank_purchase_frame(version, gold=18, source="ocr_blank"):
    frame = state(version, gold)
    frame.round = "2-1"
    frame.shop[0] = SlotObservation("shop_0", observed_at=float(version), source=source)
    return frame


def test_purchase_blank_requires_two_distinct_new_frames_with_unchanged_other_cards():
    control, _ = controller()
    before = state()
    before.round = "2-1"
    assert control.submit(Action("buy", "shop_0", "a", 1), before, 0).status == "pending"
    first = blank_purchase_frame(2)
    assert control.observe(first, 0.5) is None
    assert control.observe(first, 0.7) is None  # the same frame cannot count twice
    assert first.shop[0].status == "unknown"
    second = blank_purchase_frame(3)
    assert control.observe(second, 1).status == "confirmed"
    assert second.shop[0].status == "empty"
    assert second.shop[0].source == "confirmed_purchase"


@pytest.mark.parametrize("mode", ["unresolved_text", "no_gold_debit", "whole_shop_hidden", "other_card_changed", "round_changed"])
def test_incomplete_transaction_evidence_does_not_turn_blank_into_empty(mode):
    control, backend = controller()
    before = state()
    before.round = "2-1"
    control.submit(Action("buy", "shop_0", "a", 1), before, 0)
    for version in (2, 3):
        frame = blank_purchase_frame(version)
        if mode == "unresolved_text":
            frame.shop[0].source = "ocr_unresolved"
        elif mode == "no_gold_debit":
            frame.gold = 20
        elif mode == "whole_shop_hidden":
            frame.shop = [SlotObservation(f"shop_{i}", observed_at=version, source="ocr_blank") for i in range(5)]
        elif mode == "other_card_changed":
            frame.shop[1].hero_id = "b"
        else:
            frame.round = "2-2"
        assert control.observe(frame, version / 2) is None
        assert frame.shop[0].status == "unknown"
    assert len(backend.events) == 1


def test_open_shop_confirms_only_a_new_complete_visible_shop():
    control, backend = controller()
    hidden = state()
    hidden.shop = [SlotObservation(f"shop_{i}", source="ocr_blank") for i in range(5)]
    hidden.shop_visible = False
    hidden.field_confidence["shop_visible"] = 1
    assert control.submit(Action("open_shop", state_version=1), hidden, 0).status == "pending"
    still_hidden = deepcopy(hidden)
    still_hidden.version, still_hidden.observed_at = 2, 2
    assert control.observe(still_hidden, 0.5) is None
    assert control.observe(state(3), 1).status == "confirmed"
    assert control.submit(Action("open_shop", state_version=3), state(3), 1.5).reason == "shop_already_open"
    assert len(backend.events) == 1


def test_combat_board_inspection_requires_explicit_static_layout():
    control, backend = controller()
    observed = state()
    observed.phase, observed.viewed_player = "combat", "opponent_0"
    action = Action("scout", "board_0", state_version=1,
                    metadata={"stage": "inspect_slot", "player_id": "opponent_0"})
    assert control.submit(action, observed, 0).reason == "inspection_requires_static_position"
    control.layout["scouting"] = {"bench_phases": ["preparation", "combat"]}
    action.target = "bench_0"
    assert control.submit(action, observed, 0).status == "pending"
    assert len(backend.events) == 1


def test_encounter_uses_its_own_choice_coordinates_and_rejects_changed_phase():
    control, backend = controller()
    observed = state()
    observed.phase = "encounter"
    observed.choices = [Choice(0, "一元店", 0.99)]
    action = Action("choose", "choice_0", state_version=1, metadata={"text": "一元店", "phase": "augment"})
    assert control.submit(action, observed, 0).reason == "choice_phase_changed"
    action.metadata["phase"] = "encounter"
    assert control.submit(action, observed, 0).status == "pending"
    assert backend.events == [(200, 230)]


def test_copy_ledger_can_prove_capacity_without_fabricating_positions():
    control, backend = controller()
    observed = state()
    observed.bench = {}
    observed.owned_copies = {"a": 2}
    observed.owned_star_counts = {"a": [2, 0, 0]}
    observed.ownership_complete, observed.bench_free = True, 0
    assert control.submit(Action("buy", "shop_0", "a", 1), observed, 0).status == "pending"
    assert len(backend.events) == 1
    control, backend = controller()
    observed.owned_copies = {"a": 3}  # conflicting copy/star distribution
    assert control.submit(Action("buy", "shop_0", "a", 1), observed, 0).reason.startswith("ownership_summary_invalid")
    assert backend.events == []


@pytest.mark.parametrize("kind", ["buy", "refresh", "level", "open_shop", "close_detail"])
def test_coarse_gameplay_allows_guarded_economic_ui_actions(kind):
    control, backend = controller()
    observed = state()
    observed.phase = "gameplay"
    if kind == "open_shop":
        observed.shop = [SlotObservation(f"shop_{i}", source="ocr_blank") for i in range(5)]
        observed.shop_visible = False
        observed.field_confidence["shop_visible"] = 1
    action = Action(kind, "shop_0" if kind == "buy" else "", "a" if kind == "buy" else None, 1)
    assert control.submit(action, observed, 0).status == "pending"
    assert len(backend.events) == 1


def test_coarse_gameplay_does_not_authorize_board_inspection():
    control, backend = controller()
    observed = state()
    observed.phase, observed.viewed_player = "gameplay", "opponent_0"
    action = Action("scout", "board_0", state_version=1,
                    metadata={"stage": "inspect_slot", "player_id": "opponent_0"})
    assert control.submit(action, observed, 0).status == "rejected"
    assert backend.events == []


def test_unread_shop_does_not_authorize_blind_toggle():
    control, backend = controller()
    observed = state()
    observed.shop = [SlotObservation(f"shop_{i}", source="ocr_blank") for i in range(5)]
    assert control.submit(Action("open_shop", state_version=1), observed, 0).reason == "shop_not_confirmed_closed"
    assert backend.events == []


@pytest.mark.parametrize("round_id, confidence", [(None, 0.99), ("2-1", 0.4)])
def test_buy_requires_a_trusted_current_round(round_id, confidence):
    control, backend = controller()
    observed = state()
    observed.round = round_id
    observed.field_confidence["round"] = confidence
    assert control.submit(Action("buy", "shop_0", "a", 1), observed, 0).reason == "round_unknown_or_low_confidence"
    assert backend.events == []


def test_gameplay_can_return_home_but_cannot_select_a_new_opponent():
    control, backend = controller()
    observed = state()
    observed.phase, observed.viewed_player = "gameplay", "opponent_0"
    select = Action("scout", "opponent_0", state_version=1, metadata={"stage": "select_player", "player_id": "opponent_0"})
    assert control.submit(select, observed, 0).status == "rejected"
    assert control.submit(Action("return_home", state_version=1), observed, 0).status == "pending"
    assert len(backend.events) == 1


def test_board_inspection_cannot_confirm_after_units_start_moving():
    control, _ = controller()
    before = state()
    before.viewed_player = "opponent_0"
    action = Action("scout", "board_0", state_version=1,
                    metadata={"stage": "inspect_slot", "player_id": "opponent_0"})
    assert control.submit(action, before, 0).status == "pending"
    combat = state(2)
    combat.phase, combat.viewed_player, combat.detail_open = "combat", "opponent_0", True
    combat.detail = SlotObservation("detail", "occupied", "a", 1, confidence=0.99, observed_at=2)
    assert control.observe(combat, 1) is None
    closed = state(3)
    closed.phase, closed.viewed_player = "combat", "opponent_0"
    assert control.observe(closed, 4).status == "unresolved"
    assert not control.paused
