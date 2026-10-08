from copy import deepcopy

from test_control import controller, rules, state

from jcc_agent.models import Action, ActionResult, SlotObservation
from jcc_agent.pool import ownership_summary_error
from jcc_agent.tracking import BENCH_SLOTS, BOARD_SLOTS, SelfTracker


def reviewed_initial():
    return {"ordinary_merges": True,
            "board": {name: {"status": "empty"} for name in BOARD_SLOTS},
            "bench": {name: {"status": "empty"} for name in BENCH_SLOTS}}


def confirmed_buy(tracker, before, after, target="shop_0", hero="a"):
    action = Action("buy", target, hero, before.version)
    return tracker.apply_confirmed(ActionResult(action, "confirmed", "test", after.observed_at), before, after)


def test_missing_initial_positions_remain_unknown_not_empty():
    tracker = SelfTracker({"board": {"board_0": {"status": "occupied", "hero_id": "a", "star": 1}}}, rules())
    fused = tracker.fuse(state())
    assert fused.board["board_0"].hero_id == "a"
    assert fused.board["board_1"].status == "unknown"
    assert fused.bench["bench_1"].status == "unknown"
    assert tracker.holdings() is None
    assert not fused.ownership_complete


def test_reviewed_complete_initial_snapshot_establishes_copy_and_capacity_ledger():
    initial = reviewed_initial()
    initial["board"]["board_0"] = {"status": "occupied", "hero_id": "a", "star": 2}
    tracker = SelfTracker(initial, rules())
    fused = tracker.fuse(state())
    assert tracker.holdings() == {"a": 3}
    assert fused.owned_star_counts == {"a": [0, 1, 0]}
    assert fused.bench_free == 9
    assert ownership_summary_error(fused, rules()) is None


def test_confirmed_purchase_tracks_copies_but_does_not_invent_a_bench_location():
    tracker = SelfTracker(reviewed_initial(), rules())
    before = tracker.fuse(state())
    after = state(2, 18)
    assert confirmed_buy(tracker, before, after)
    fused = tracker.fuse(after)
    assert fused.owned_copies == {"a": 1} and fused.ownership_complete
    assert fused.owned_star_counts == {"a": [1, 0, 0]}
    assert fused.bench_free == 8
    assert fused.bench["bench_1"].status == "unknown"
    assert all(slot.hero_id != "a" for slot in fused.bench.values())
    assert ownership_summary_error(fused, rules()) is None
    assert not confirmed_buy(tracker, before, after)
    assert tracker.holdings() == {"a": 1}


def test_ordinary_merge_updates_stars_without_assuming_survivor_position():
    initial = reviewed_initial()
    for name in ("board_0", "board_1"):
        initial["board"][name] = {"status": "occupied", "hero_id": "a", "star": 1}
    tracker = SelfTracker(initial, rules())
    before = tracker.fuse(state())
    assert confirmed_buy(tracker, before, state(2, 18))
    fused = tracker.fuse(state(2, 18))
    assert fused.owned_copies == {"a": 3}
    assert fused.owned_star_counts == {"a": [0, 1, 0]}
    assert fused.board["board_0"].status == "unknown"
    assert fused.board["board_1"].status == "unknown"
    assert fused.bench_free == 8  # conservative, no guessed merge destination


def test_unknown_merge_mechanism_invalidates_authoritative_totals():
    initial = reviewed_initial()
    initial["ordinary_merges"] = False
    tracker = SelfTracker(initial, rules())
    before = tracker.fuse(state())
    assert confirmed_buy(tracker, before, state(2, 18))
    assert tracker.holdings() is None
    assert "purchase_with_unverified_merge_rules" in tracker.warnings


def test_pending_and_simulated_results_cannot_modify_inventory():
    tracker = SelfTracker(reviewed_initial(), rules())
    for status in ("pending", "simulated", "rejected", "failed"):
        result = ActionResult(Action("buy", "shop_0", "a", 1), status, "test")
        assert not tracker.apply_confirmed(result, state(), state(2, 18))
    assert tracker.holdings() == {}


def test_partial_details_reconcile_known_identity_but_do_not_assume_star_icons():
    tracker = SelfTracker(reviewed_initial(), rules())
    before = tracker.fuse(state())
    confirmed_buy(tracker, before, state(2, 18))
    observation = state(3, 18)
    observation.bench = {"bench_3": SlotObservation("bench_3", "occupied", "a", None,
                                                confidence=0.99, observed_at=3)}
    fused = tracker.fuse(observation)
    assert fused.bench["bench_3"].hero_id == "a"
    assert fused.bench["bench_3"].star is None
    assert fused.owned_copies == {"a": 1} and fused.ownership_complete
    observation = state(4, 18)
    observation.board = {"board_2": SlotObservation("board_2", "occupied", "b", None,
                                                   confidence=0.99, observed_at=4)}
    assert not tracker.fuse(observation).ownership_complete


def test_new_complete_observation_can_correct_ledger_but_historic_fusion_cannot():
    tracker = SelfTracker(reviewed_initial(), rules())
    before = tracker.fuse(state())
    confirmed_buy(tracker, before, state(2, 18))
    stale = deepcopy(before)
    stale.version, stale.observed_at = 3, 3
    assert tracker.fuse(stale).owned_copies == {"a": 1}
    fresh = deepcopy(stale)
    fresh.version, fresh.observed_at = 4, 4
    for slot in [*fresh.board.values(), *fresh.bench.values()]:
        slot.observed_at = 4
    reconciled = tracker.fuse(fresh)
    assert reconciled.owned_copies == {}
    assert reconciled.ownership_complete
    assert "full_observation_corrected_inventory_ledger" in reconciled.warnings


def test_confirmed_move_and_sale_preserve_then_reduce_total_copies():
    initial = reviewed_initial()
    initial["board"]["board_0"] = {"status": "occupied", "hero_id": "a", "star": 2}
    tracker = SelfTracker(initial, rules())
    before = tracker.fuse(state())
    after = state(2)
    move = Action("move", "board_0", "a", 1, metadata={"destination": "bench_1"})
    assert tracker.apply_confirmed(ActionResult(move, "confirmed", "test"), before, after)
    assert tracker.holdings() == {"a": 3}
    assert tracker.bench["bench_1"].star == 2
    assert tracker.bench_free == 8
    sale = Action("sell", "bench_1", "a", 2)
    assert tracker.apply_confirmed(ActionResult(sale, "confirmed", "test"), after, state(3, 23))
    assert tracker.holdings() == {"a": 0}
    assert tracker.bench_free == 9


def test_consumed_shop_slot_requires_transaction_and_same_round_and_visible_shop():
    tracker = SelfTracker(reviewed_initial(), rules())
    before = state()
    before.round = "2-1"
    before = tracker.fuse(before)
    after = state(2, 18)
    after.round = "2-1"
    after.shop[0] = SlotObservation("shop_0", "unknown", observed_at=2, source="ocr_blank")
    assert confirmed_buy(tracker, before, after)
    fused = tracker.fuse(after)
    assert fused.shop[0].status == "empty"
    assert fused.shop[0].source == "confirmed_purchase_empty"
    hidden = deepcopy(after)
    hidden.version, hidden.observed_at = 3, 3
    hidden.shop = [SlotObservation(f"shop_{i}", observed_at=3, source="ocr_blank") for i in range(5)]
    assert all(s.status == "unknown" for s in tracker.fuse(hidden).shop)
    next_round = deepcopy(after)
    next_round.version, next_round.observed_at, next_round.round = 4, 4, "2-2"
    assert tracker.fuse(next_round).shop[0].status == "unknown"


def test_new_card_or_confirmed_refresh_clears_consumption_cache():
    tracker = SelfTracker(reviewed_initial(), rules())
    before, after = state(), state(2, 18)
    before.round = after.round = "2-1"
    before = tracker.fuse(before)
    confirmed_buy(tracker, before, after)
    assert tracker.fuse(after).shop[0].status == "occupied"
    blank = deepcopy(after)
    blank.version, blank.observed_at = 3, 3
    blank.shop[0] = SlotObservation("shop_0", observed_at=3, source="ocr_blank")
    assert tracker.fuse(blank).shop[0].status == "unknown"
    refreshed = state(4, 16)
    refreshed.round = "2-1"
    result = ActionResult(Action("refresh", state_version=3), "confirmed", "test")
    assert tracker.apply_confirmed(result, blank, refreshed)
    assert tracker.export()["consumed_shop"] == {}


def test_export_is_detached_and_preserves_evidence_not_fabricated_positions():
    tracker = SelfTracker(reviewed_initial(), rules())
    before = tracker.fuse(state())
    confirmed_buy(tracker, before, state(2, 18))
    document = tracker.export()
    assert document["owned_copies"] == {"a": 1}
    assert document["bench"]["bench_1"]["status"] == "unknown"
    document["owned_copies"]["a"] = 99
    assert tracker.holdings() == {"a": 1}


def test_two_context_confirmed_purchases_keep_consumed_slots_and_exact_totals():
    tracker = SelfTracker(reviewed_initial(), rules())
    control, backend = controller()
    control.layout["points"]["shop_1"] = [0.4, 0.8]
    observed = state()
    observed.round = "2-1"
    observed = tracker.fuse(observed)
    for purchase_index, initial_version in ((0, 1), (1, 3)):
        action = Action("buy", f"shop_{purchase_index}", "a", initial_version)
        assert control.submit(action, observed, purchase_index * 2).status == "pending"
        before = deepcopy(control.pending.before)
        for frame_index in (1, 2):
            version = initial_version + frame_index
            raw = state(version, 18 - purchase_index * 2)
            raw.round = "2-1"
            for slot_index in range(purchase_index + 1):
                raw.shop[slot_index] = SlotObservation(f"shop_{slot_index}", observed_at=version, source="ocr_blank")
            transaction_frame = tracker.fuse_shop(raw)
            result = control.observe(transaction_frame, purchase_index * 2 + frame_index * 0.5)
            if frame_index == 1:
                assert result is None
            else:
                assert result.status == "confirmed"
                assert tracker.apply_confirmed(result, before, transaction_frame)
                observed = tracker.fuse(transaction_frame)
        assert all(s.status == "empty" for s in observed.shop[:purchase_index + 1])
        assert observed.owned_copies == {"a": purchase_index + 1}
        assert observed.ownership_complete
    assert len(backend.events) == 2


def test_fuse_shop_never_reconciles_new_physical_position_evidence():
    tracker = SelfTracker(reviewed_initial(), rules())
    new_piece = state()
    new_piece.board["board_0"] = SlotObservation("board_0", "occupied", "b", 1,
                                                  confidence=0.99, observed_at=1)
    tracker.fuse_shop(new_piece)
    assert tracker.holdings() == {}
    assert tracker.board["board_0"].status == "empty"


def test_one_unreadable_round_suspends_cache_without_destroying_purchase_evidence():
    tracker = SelfTracker(reviewed_initial(), rules())
    before, after = state(), state(2, 18)
    after.shop[0] = SlotObservation("shop_0", observed_at=2, source="ocr_blank")
    assert confirmed_buy(tracker, before, after)
    missing_round = deepcopy(after)
    missing_round.version, missing_round.observed_at, missing_round.round = 3, 3, None
    assert tracker.fuse_shop(missing_round).shop[0].status == "unknown"
    assert "shop_0" in tracker.export()["consumed_shop"]
    low_confidence_next_round = deepcopy(after)
    low_confidence_next_round.version, low_confidence_next_round.observed_at = 4, 4
    low_confidence_next_round.round = "2-2"
    low_confidence_next_round.field_confidence["round"] = 0.4
    assert tracker.fuse_shop(low_confidence_next_round).shop[0].status == "unknown"
    assert "shop_0" in tracker.export()["consumed_shop"]
    recovered = deepcopy(after)
    recovered.version, recovered.observed_at = 5, 5
    assert tracker.fuse_shop(recovered).shop[0].status == "empty"
    next_round = deepcopy(recovered)
    next_round.version, next_round.observed_at, next_round.round = 6, 6, "2-2"
    assert tracker.fuse_shop(next_round).shop[0].status == "unknown"
    assert tracker.export()["consumed_shop"] == {}


def test_regressed_round_cannot_reuse_cache_or_reset_its_generation():
    tracker = SelfTracker(reviewed_initial(), rules())
    before, after = state(), state(2, 18)
    after.shop[0] = SlotObservation("shop_0", observed_at=2, source="ocr_blank")
    confirmed_buy(tracker, before, after)
    regressed = deepcopy(after)
    regressed.version, regressed.observed_at, regressed.round = 3, 3, "1-4"
    assert tracker.fuse_shop(regressed).shop[0].status == "unknown"
    assert tracker.export()["shop_round"] == "2-1"
    restored = deepcopy(after)
    restored.version, restored.observed_at = 4, 4
    assert tracker.fuse_shop(restored).shop[0].status == "empty"


def test_confirmed_purchases_can_continue_through_two_merges_to_three_star():
    tracker = SelfTracker(reviewed_initial(), rules())
    control, backend = controller()
    free_counts = []
    for purchase_number in range(1, 10):
        version = purchase_number * 2 - 1
        before = state(version, 30)
        before.board, before.bench = {}, {}
        before = tracker.fuse(before)
        action = Action("buy", "shop_0", "a", version)
        assert control.submit(action, before, now=purchase_number * 2).status == "pending"
        after = state(version + 1, 28)
        after.board, after.bench = {}, {}
        after.shop[0] = SlotObservation("shop_0", "empty", confidence=0.99, observed_at=version + 1)
        result = control.observe(after, now=purchase_number * 2 + 0.5)
        assert result.status == "confirmed"
        assert tracker.apply_confirmed(result, before, after)
        fused = tracker.fuse(after)
        assert fused.ownership_complete
        assert fused.owned_copies == {"a": purchase_number}
        free_counts.append(fused.bench_free)
    assert free_counts == [8, 7, 8, 7, 6, 7, 6, 5, 8]
    assert fused.owned_star_counts == {"a": [0, 0, 1]}
    assert all(slot.status == "unknown" for slot in fused.bench.values())
    assert len(backend.events) == 9


def test_board_evidence_used_for_capacity_must_be_recent_and_not_merge_invalidated():
    initial = reviewed_initial()
    initial["board"]["board_0"] = {"status": "occupied", "hero_id": "b", "star": 1}
    recent = SelfTracker(initial, rules(), max_board_age_seconds=5)
    recent.bench_free = 0  # conservatively lost capacity, then recompute from P-B
    recent.bench = {name: SlotObservation(name) for name in BENCH_SLOTS}
    recent_frame = state(2)
    recent_frame.board, recent_frame.bench = {}, {}
    assert recent.fuse(recent_frame).bench_free == 9  # P=1, recent B=1
    stale = SelfTracker(initial, rules(), max_board_age_seconds=5)
    stale.bench_free = 0
    stale.bench = {name: SlotObservation(name) for name in BENCH_SLOTS}
    stale_frame = state(10)
    stale_frame.board, stale_frame.bench = {}, {}
    assert stale.fuse(stale_frame).bench_free == 8  # stale B=0, P=1


def test_duplicate_board_positions_cannot_claim_more_units_than_star_ledger():
    tracker = SelfTracker(reviewed_initial(), rules())
    before = tracker.fuse(state())
    confirmed_buy(tracker, before, state(2, 18))
    # Two uncertain-star locations cannot both hold the one tracked physical unit.
    tracker.board["board_0"] = SlotObservation("board_0", "occupied", "a", None, confidence=0.99, observed_at=3)
    tracker.board["board_1"] = SlotObservation("board_1", "occupied", "a", None, confidence=0.99, observed_at=3)
    observation = state(3, 18)
    observation.board, observation.bench = {}, {}
    assert not tracker.fuse(observation).ownership_complete
