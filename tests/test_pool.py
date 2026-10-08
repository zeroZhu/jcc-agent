import unittest
from pathlib import Path

from jcc_agent.models import GameState, PlayerSnapshot, SlotObservation
from jcc_agent.pool import estimate_pool
from jcc_agent.rules import load_rules

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "inkborn_fables.json"


def empty(prefix, count):
    return {f"{prefix}_{i}": SlotObservation(f"{prefix}_{i}", "empty", confidence=1, observed_at=100)
            for i in range(count)}


def full_state():
    state = GameState(observed_at=100, board=empty("board", 28), bench=empty("bench", 9))
    state.opponents = {str(i): PlayerSnapshot(str(i), {**empty("board", 28), **empty("bench", 9)}, 100, True)
                       for i in range(7)}
    return state


def occupied(slot, hero="ahri", star=1, stamp=100):
    return SlotObservation(slot, "occupied", hero, star, confidence=1, observed_at=stamp)


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.rules = load_rules(CONFIG)

    def test_snapshot_replacement_never_accumulates_repeated_scouts(self):
        state = full_state()
        state.board["board_0"] = occupied("board_0", star=2)
        state.opponents["0"].slots["bench_0"] = occupied("bench_0", star=2)
        first = estimate_pool(state, self.rules, now=100)["ahri"]
        self.assertEqual((6, 16, 16), (first.known_held, first.remaining_low, first.remaining_high))
        self.assertEqual(first, estimate_pool(state, self.rules, now=100)["ahri"])
        state.opponents["0"].slots["bench_0"] = empty("bench", 1)["bench_0"]
        self.assertEqual(3, estimate_pool(state, self.rules, now=100)["ahri"].known_held)

    def test_unknown_star_is_interval_not_one_star_fact(self):
        state = full_state()
        state.opponents["0"].slots["board_0"] = occupied("board_0", star=None)
        value = estimate_pool(state, self.rules, now=100)["ahri"]
        self.assertEqual((0, 13, 21), (value.known_held, value.remaining_low, value.remaining_high))
        self.assertTrue(any(x.startswith("unknown_star") for x in value.warnings))

    def test_unobserved_players_and_partial_snapshots_expand_lower_bound(self):
        state = full_state()
        state.opponents.pop("6")
        state.board["board_0"] = occupied("board_0", star=2)
        value = estimate_pool(state, self.rules, now=100)["ahri"]
        self.assertEqual((0, 19), (value.remaining_low, value.remaining_high))
        self.assertTrue(any(x.startswith("missing_players") for x in value.warnings))
        state = full_state()
        state.opponents["0"].slots = {}
        self.assertEqual(0, estimate_pool(state, self.rules, now=100)["ahri"].remaining_low)

    def test_stale_and_unknown_identity_are_never_absence(self):
        state = full_state()
        state.opponents["0"].slots["board_0"] = occupied("board_0", star=3, stamp=10)
        values = estimate_pool(state, self.rules, now=100)
        self.assertEqual((0, 22), (values["ahri"].remaining_low, values["ahri"].remaining_high))
        self.assertEqual(0, values["senna"].remaining_low)
        self.assertTrue(any(x.startswith("stale_observation") for x in values["ahri"].warnings))

    def test_contradiction_is_reported_not_silently_clipped(self):
        state = full_state()
        for i in range(3):
            state.opponents[str(i)].slots["board_0"] = occupied("board_0", star=3)
        value = estimate_pool(state, self.rules, now=100)["ahri"]
        self.assertEqual(27, value.known_held)
        self.assertEqual(0, value.remaining_high)
        self.assertTrue(any(x.startswith("pool_conflict") for x in value.warnings))
        self.assertTrue(any(x.startswith("unverified_rules") for x in value.warnings))

    def test_eliminated_player_not_subtracted_and_lovers_share_pool(self):
        state = full_state()
        state.opponents["0"].slots["board_0"] = occupied("board_0", "xayah_rakan", 2)
        self.assertEqual(6, estimate_pool(state, self.rules, now=100)["xayah_rakan"].remaining_high)
        state.opponents["0"].eliminated = True
        self.assertEqual(9, estimate_pool(state, self.rules, now=100)["xayah_rakan"].remaining_high)

    def test_reviewed_owned_ledger_replaces_spatial_self_snapshot(self):
        state = full_state()
        state.board, state.bench = {}, {}
        state.owned_copies, state.owned_star_counts = {"senna": 5}, {"senna": [2, 1, 0]}
        state.ownership_complete, state.bench_free = True, 6
        value = estimate_pool(state, self.rules, now=100)["senna"]
        self.assertEqual((5, 15, 15), (value.known_held, value.remaining_low, value.remaining_high))
        state.owned_copies["senna"] = 6
        value = estimate_pool(state, self.rules, now=100)["senna"]
        self.assertEqual((0, 0, 20), (value.known_held, value.remaining_low, value.remaining_high))
        self.assertTrue(any(w.startswith("unknown_ownership") for w in value.warnings))


if __name__ == "__main__":
    unittest.main()
