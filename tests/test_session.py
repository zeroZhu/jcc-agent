from jcc_agent.models import GameState, PlayerSnapshot, SlotObservation
from jcc_agent.session import AgentSession, SnapshotBook
from jcc_agent.storage import state_from_dict


def test_partial_snapshot_does_not_erase_other_slots_or_double_count():
    book = SnapshotBook()
    book.update(PlayerSnapshot("p", {
        "board_0": SlotObservation("board_0", "occupied", "a", observed_at=1),
        "bench_0": SlotObservation("bench_0", "occupied", "b", observed_at=1),
    }, 1, True))
    book.update(PlayerSnapshot("p", {
        "board_0": SlotObservation("board_0", "empty", observed_at=2),
    }, 2, False))
    assert len(book.players["p"].slots) == 2
    assert book.players["p"].slots["board_0"].status == "empty"
    assert book.players["p"].slots["bench_0"].hero_id == "b"
    assert not book.players["p"].complete


def test_old_snapshot_cannot_replace_new_information():
    book = SnapshotBook()
    book.update(PlayerSnapshot("p", {}, 5, True, True))
    book.update(PlayerSnapshot("p", {"x": SlotObservation("x", "occupied", "a")}, 4, True))
    assert book.players["p"].eliminated


def test_nested_state_json_roundtrip():
    state = state_from_dict({"version": 1, "shop": [{"slot": "shop_0", "status": "unknown"}],
                             "opponents": {"p": {"player_id": "p", "slots": {}}}})
    assert state_from_dict(state.to_dict()) == state


def test_budget_only_resets_on_trusted_forward_round_change():
    from pathlib import Path

    from jcc_agent.rules import load_rules

    rules = load_rules(Path(__file__).resolve().parents[1] / "configs/inkborn_fables.json")
    session = AgentSession(rules, [])
    session.decide(GameState(version=1, round="2-2", field_confidence={"round": 1}))
    session.spent = 9
    session.decide(GameState(version=2, round="2-3", field_confidence={"round": .1}))
    assert session.spent == 9
    session.decide(GameState(version=3, round="2-1", field_confidence={"round": 1}))
    assert session.spent == 9
    session.decide(GameState(version=4, round="2-3", field_confidence={"round": 1}))
    assert session.spent == 9
    session.decide(GameState(version=5, round="2-3", field_confidence={"round": 1}))
    assert session.spent == 0


def test_round_candidate_cannot_click_then_confirmed_round_resets_before_spending():
    from pathlib import Path

    from jcc_agent.control import ActionController, RecordingBackend
    from jcc_agent.models import Composition
    from jcc_agent.rules import load_rules

    rules = load_rules(Path(__file__).resolve().parents[1] / "configs/inkborn_fables.json")
    comp = Composition("test", "追牌测试", {"senna": 3}, ["senna"], min_gold=0, max_roll_spend=10)
    session = AgentSession(rules, [comp])

    def frame(version, round_id, confidence=1):
        state = GameState(version=version, observed_at=version, round=round_id,
                          phase="combat", gold=40, level=6, seconds_left=20,
                          viewed_player="self", detail_open=False, owned_copies={},
                          ownership_complete=True, bench_free=9, owned_star_counts={})
        state.field_confidence = {key: 1 for key in ("phase", "gold", "level", "seconds_left", "viewed_player", "detail_open")}
        state.field_confidence["round"] = confidence
        state.shop = [SlotObservation(f"shop_{i}", "occupied", "ahri", 1, 1, 1, version) for i in range(5)]
        state.shop[0].hero_id, state.shop[0].cost = "senna", 2
        return state

    session.decide(frame(1, "2-2"))
    session.spent = 5
    backend = RecordingBackend()
    controller = ActionController(backend, {"calibrated": True, "reference_size": [100, 100],
                                            "points": {"shop_0": [.5, .5]}}, rules,
                                  dry_run=False, live_enabled=True)
    candidate, _, _, action = session.decide(frame(2, "2-3"))
    assert action.kind == "wait"
    assert "回合归属" in action.reason
    assert session.spent == 5
    assert controller.submit(action, candidate, now=2).status == "confirmed"
    assert backend.events == []

    stable, _, _, action = session.decide(frame(3, "2-3"))
    assert session.spent == 0
    assert action.kind == "buy"
    assert action.metadata["max_spend_remaining"] == 10
    assert controller.submit(action, stable, now=3).status == "pending"
    assert len(backend.events) == 1
    confirmed = frame(4, "2-3")
    confirmed.gold = 38
    confirmed.shop[0] = SlotObservation("shop_0", "empty", confidence=1, observed_at=4)
    assert controller.observe(confirmed, now=4).status == "confirmed"
    session.account_confirmed_spend(stable, confirmed)
    session.decide(confirmed)
    assert session.spent == 2  # first new-round debit survives subsequent decisions

    for version, round_id, confidence in ((5, None, 1), (6, "2-4", .1), (7, "2-2", 1)):
        _, _, _, action = session.decide(frame(version, round_id, confidence))
        assert action.kind == "wait"
        assert session.spent == 2
