
from test_control import controller, state

from jcc_agent.models import Action, ActionResult, SlotObservation
from jcc_agent.scouting import ScoutingScheduler


def opponent_frame(version, *, opened=False, hero=None):
    observed = state(version)
    observed.viewed_player = "opponent_0"
    observed.detail_open = opened
    if hero:
        observed.detail = SlotObservation("detail", "occupied", hero, 1, 2, 0.99, float(version))
    return observed


def reach_inspection(scheduler, control):
    scheduler.start_round("2-1", 0)
    assert scheduler.tick(state(), control, 0).status == "pending"
    selected = opponent_frame(2, opened=True, hero="b")
    assert scheduler.tick(selected, control, 0.3).status == "confirmed"
    assert scheduler.tick(selected, control, 0.5).action.kind == "close_detail"
    closed = opponent_frame(3)
    assert scheduler.tick(closed, control, 0.8).status == "confirmed"
    result = scheduler.tick(closed, control, 1)
    assert result.status == "pending"
    assert result.action.target == "board_0"


def test_inspection_requires_observed_close_then_fresh_detail_and_stays_serial():
    control, backend = controller()
    scheduler = ScoutingScheduler(["opponent_0"], ["board_0", "bench_0"], budget_seconds=20)
    reach_inspection(scheduler, control)
    inspected = opponent_frame(4, opened=True, hero="a")
    assert scheduler.tick(inspected, control, 1.3).status == "confirmed"
    snapshot = scheduler.snapshots["opponent_0"]
    assert snapshot.slots["board_0"].hero_id == "a"
    assert snapshot.slots["bench_0"].status == "unknown"
    assert not snapshot.complete
    # Another consumer cannot race the pending close with an economic action.
    assert scheduler.tick(inspected, control, 1.6).action.kind == "close_detail"
    result = control.submit(Action("refresh", state_version=4), inspected, 1.8)
    assert result.reason == "another_action_pending"
    assert len(backend.events) == 4


def test_old_detail_that_never_closes_cannot_be_recorded_for_new_position():
    control, backend = controller()
    scheduler = ScoutingScheduler(["opponent_0"], ["board_0"], budget_seconds=20)
    scheduler.start_round("2-1", 0)
    scheduler.tick(state(), control, 0)
    stale = opponent_frame(2, opened=True, hero="a")
    scheduler.tick(stale, control, 0.3)
    scheduler.tick(stale, control, 0.5)
    still_open = opponent_frame(3, opened=True, hero="a")
    assert scheduler.tick(still_open, control, 1) is None
    assert scheduler.tick(still_open, control, 4.6).status == "failed"
    assert control.paused and scheduler.paused
    assert scheduler.snapshots["opponent_0"].slots["board_0"].status == "unknown"
    assert len(backend.events) == 2  # selecting and closing; never inspect-clicked


def test_closed_empty_or_unreadable_inspection_remains_unknown_and_advances_without_retry():
    control, backend = controller()
    scheduler = ScoutingScheduler(["opponent_0"], ["board_0", "bench_0"], budget_seconds=20)
    reach_inspection(scheduler, control)
    empty_click = opponent_frame(4)
    assert scheduler.tick(empty_click, control, 5.1).status == "unresolved"
    assert scheduler.snapshots["opponent_0"].slots["board_0"].status == "unknown"
    assert scheduler.slot_index == 1
    assert not control.paused and not scheduler.paused
    assert scheduler.tick(opponent_frame(5), control, 6).action.kind == "close_detail"
    assert len(backend.events) == 4


def test_inspection_without_view_or_closed_panel_evidence_still_pauses():
    control, backend = controller()
    scheduler = ScoutingScheduler(["opponent_0"], ["board_0", "bench_0"], budget_seconds=20)
    reach_inspection(scheduler, control)
    unknown_view = opponent_frame(4)
    unknown_view.viewed_player = "unknown"
    assert scheduler.tick(unknown_view, control, 5.1).status == "failed"
    assert control.paused and scheduler.paused
    assert len(backend.events) == 3


def test_budget_returns_home_and_resumes_cursor_next_round_without_renewal():
    control, _ = controller()
    scheduler = ScoutingScheduler(["opponent_0"], ["board_0", "bench_0"], budget_seconds=8)
    reach_inspection(scheduler, control)
    inspected = opponent_frame(4, opened=True, hero="a")
    scheduler.tick(inspected, control, 1.3)
    scheduler.start_round("2-1", 5)  # must not replenish the old budget
    assert scheduler.deadline == 8
    result = scheduler.tick(inspected, control, 5)
    assert result.action.kind == "return_home"
    assert scheduler.tick(state(5), control, 5.3).status == "confirmed"
    assert not scheduler.active and scheduler.slot_index == 1
    scheduler.start_round("2-2", 10)
    assert scheduler.active and scheduler.slot_index == 1
    assert scheduler.next_action(state(6), 10).target == "opponent_0"


def test_new_cycle_replaces_previous_holdings_instead_of_accumulating():
    control, _ = controller()
    scheduler = ScoutingScheduler(["opponent_0"], ["board_0"], budget_seconds=20)
    reach_inspection(scheduler, control)
    inspected = opponent_frame(4, opened=True, hero="a")
    scheduler.tick(inspected, control, 1.3)
    assert scheduler.snapshots["opponent_0"].complete
    action = scheduler.next_action(inspected, 1.6)
    scheduler.on_result(ActionResult(action, "confirmed", "fake confirmation", 5), opponent_frame(5))
    refreshed = scheduler.snapshots["opponent_0"]
    assert len(refreshed.slots) == 1
    assert refreshed.slots["board_0"].status == "unknown"
    assert not refreshed.complete
    exported = scheduler.export_snapshots()
    exported["opponent_0"].slots.clear()
    assert len(scheduler.snapshots["opponent_0"].slots) == 1


def test_scouting_dry_run_cannot_advance_or_claim_a_snapshot():
    control, backend = controller(dry_run=True)
    scheduler = ScoutingScheduler(["opponent_0"], ["board_0"])
    scheduler.start_round("2-1", 0)
    assert scheduler.tick(state(), control, 0).status == "simulated"
    assert scheduler.stage == "select_player"
    assert scheduler.snapshots == {} and backend.events == []


def test_combat_skips_moving_board_positions_and_only_inspects_configured_bench():
    control, backend = controller()
    control.layout["scouting"] = {"bench_phases": ["preparation", "combat"]}
    scheduler = ScoutingScheduler(["opponent_0"], ["board_0", "bench_0"],
                                  budget_seconds=20, layout=control.layout)
    scheduler.start_round("2-1", 0)
    initial = state()
    initial.phase = "combat"
    scheduler.tick(initial, control, 0)
    selected = opponent_frame(2)
    selected.phase = "combat"
    scheduler.tick(selected, control, 0.3)
    assert scheduler.tick(selected, control, 0.5).action.kind == "close_detail"
    closed = opponent_frame(3)
    closed.phase = "combat"
    scheduler.tick(closed, control, 0.8)
    result = scheduler.tick(closed, control, 1)
    assert result.action.target == "bench_0"
    assert scheduler.snapshots["opponent_0"].slots["board_0"].status == "unknown"
    assert len(backend.events) == 3


def test_preparation_to_gameplay_returns_home_without_selecting_another_player():
    control, backend = controller()
    scheduler = ScoutingScheduler(["opponent_0"], ["board_0"], budget_seconds=20)
    scheduler.start_round("2-1", 0)
    assert scheduler.tick(state(), control, 0).status == "pending"
    selected = opponent_frame(2)
    assert scheduler.tick(selected, control, 0.3).status == "confirmed"
    changed_phase = opponent_frame(3)
    changed_phase.phase = "gameplay"
    returning = scheduler.tick(changed_phase, control, 0.6)
    assert returning.action.kind == "return_home" and returning.status == "pending"
    unconfirmed_view = state(4)
    unconfirmed_view.phase, unconfirmed_view.viewed_player = "gameplay", "unknown"
    assert scheduler.tick(unconfirmed_view, control, 0.9) is None
    confirmed_home = state(5)
    confirmed_home.phase = "gameplay"
    assert scheduler.tick(confirmed_home, control, 1.2).status == "confirmed"
    assert not scheduler.active
    assert len(backend.events) == 2


def test_unknown_phase_waits_and_does_not_blindly_return_or_select():
    control, backend = controller()
    scheduler = ScoutingScheduler(["opponent_0"], ["board_0"], budget_seconds=20)
    scheduler.start_round("2-1", 0)
    scheduler.tick(state(), control, 0)
    selected = opponent_frame(2)
    scheduler.tick(selected, control, 0.3)
    unknown = opponent_frame(3)
    unknown.phase = "unknown"
    assert scheduler.tick(unknown, control, 0.6) is None
    assert len(backend.events) == 1
    unknown.phase = "gameplay"
    unknown.field_confidence["phase"] = 0.3
    assert scheduler.tick(unknown, control, 0.9) is None
    assert len(backend.events) == 1
