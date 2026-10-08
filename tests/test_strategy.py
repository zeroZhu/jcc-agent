import unittest
from copy import deepcopy
from pathlib import Path

from jcc_agent.models import Composition, GameState, SlotObservation, StrategyPlan
from jcc_agent.rules import load_rules
from jcc_agent.strategy import choose_plan, next_action

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "inkborn_fables.json"


def observation(slot, hero=None, star=1):
    return SlotObservation(slot, "occupied" if hero else "empty", hero, star if hero else None,
                           cost={"senna": 2, "shen": 2}.get(hero, 1) if hero else None,
                           confidence=1, observed_at=100)


def playable():
    return GameState(version=3, observed_at=100, phase="preparation", gold=50, level=6,
                     hp=80, seconds_left=20, viewed_player="self", detail_open=False,
                     board={f"board_{i}": observation(f"board_{i}") for i in range(28)},
                     bench={f"bench_{i}": observation(f"bench_{i}") for i in range(9)},
                     shop=[observation(f"shop_{i}", "garen") for i in range(5)])


def plan(state, targets=None, **kwargs):
    targets = targets or {"senna": 9}
    values = dict(composition_id="test", targets=targets, buy_priority=list(targets),
                  min_gold=30, max_roll_spend=20, roll_level=6, state_version=state.version)
    values.update(kwargs)
    return StrategyPlan(**values)


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.rules = load_rules(CONFIG)

    def test_buy_priority_then_stop_at_target(self):
        state = playable()
        state.shop[0] = observation("shop_0", "senna")
        state.shop[1] = observation("shop_1", "shen")
        chosen = plan(state, {"shen": 3, "senna": 9})
        self.assertEqual("shen", next_action(state, chosen, self.rules).hero_id)
        state.board["board_0"] = observation("board_0", "shen", 2)
        self.assertEqual("senna", next_action(state, chosen, self.rules).hero_id)
        state.board["board_1"] = observation("board_1", "senna", 3)
        self.assertEqual("wait", next_action(state, chosen, self.rules).kind)

    def test_unknown_own_star_or_missing_slot_never_buys(self):
        state = playable()
        state.shop[0] = observation("shop_0", "senna")
        state.board["board_0"] = observation("board_0", "senna", None)
        self.assertIn("星级未知", next_action(state, plan(state), self.rules).reason)
        state.board.pop("board_0")
        self.assertIn("扫描不完整", next_action(state, plan(state), self.rules).reason)

    def test_budget_counts_all_costs_and_reserves_purchase_money(self):
        state = playable()
        chosen = plan(state)
        self.assertEqual("refresh", next_action(state, chosen, self.rules, spent=16).kind)
        self.assertEqual("wait", next_action(state, chosen, self.rules, spent=17).kind)
        state.shop[0] = observation("shop_0", "senna")
        self.assertEqual("buy", next_action(state, chosen, self.rules, spent=18).kind)
        self.assertEqual("wait", next_action(state, chosen, self.rules, spent=19).kind)
        state.gold = 31
        self.assertEqual("wait", next_action(state, chosen, self.rules).kind)

    def test_full_bench_blocks_reroll_but_allows_confirmed_merge(self):
        state = playable()
        state.bench = {f"bench_{i}": observation(f"bench_{i}", "garen") for i in range(9)}
        self.assertEqual("wait", next_action(state, plan(state), self.rules).kind)
        state.shop[0] = observation("shop_0", "senna")
        state.board["board_0"] = observation("board_0", "senna")
        state.board["board_1"] = observation("board_1", "senna")
        result = next_action(state, plan(state), self.rules)
        self.assertEqual("buy", result.kind)
        self.assertTrue(result.metadata["merge_expected"])

    def test_level_before_rolling_but_buy_natural_shop_first(self):
        state = playable()
        state.level, state.xp, state.xp_to_level = 5, 12, 20
        self.assertEqual("level", next_action(state, plan(state), self.rules).kind)
        state.shop[0] = observation("shop_0", "senna")
        self.assertEqual("buy", next_action(state, plan(state), self.rules).kind)

    def test_phase_version_time_and_unknown_shop_guards(self):
        for alter in (
            lambda s: setattr(s, "phase", "unknown"),
            lambda s: setattr(s, "seconds_left", 2),
            lambda s: setattr(s.shop[0], "status", "unknown"),
            lambda s: setattr(s.shop[0], "cost", None),
            lambda s: setattr(s, "observed_at", 200),
        ):
            state = playable()
            alter(state)
            self.assertEqual("wait", next_action(state, plan(state), self.rules).kind)
        state = playable()
        self.assertEqual("wait", next_action(state, plan(state, state_version=1), self.rules).kind)
        state.viewed_player = "opponent_0"
        self.assertEqual("return_home", next_action(state, plan(state), self.rules).kind)

    def test_economy_actions_continue_in_combat_or_confirmed_gameplay(self):
        for phase in ("combat", "gameplay"):
            state = playable()
            state.phase = phase
            state.shop[0] = observation("shop_0", "senna")
            self.assertEqual("buy", next_action(state, plan(state), self.rules).kind)
            state.shop[0] = observation("shop_0", "garen")
            self.assertEqual("refresh", next_action(state, plan(state), self.rules).kind)
        state.phase = "unknown"
        self.assertEqual("wait", next_action(state, plan(state), self.rules).kind)

    def test_visible_discount_price_overrides_base_tier(self):
        state = playable()
        state.shop[0] = observation("shop_0", "senna")
        state.shop[0].cost = 1
        state.gold = 31
        result = next_action(state, plan(state), self.rules, spent=19)
        self.assertEqual("buy", result.kind)
        self.assertEqual(1, result.metadata["expected_cost"])
        self.assertEqual(2, self.rules.heroes["senna"].cost)

    def test_reviewed_aggregate_inventory_supports_buy_without_empty_cell_ocr(self):
        state = playable()
        state.board, state.bench = {}, {}
        state.owned_copies = {"senna": 2}
        state.owned_star_counts = {"senna": [2, 0, 0]}
        state.ownership_complete, state.bench_free = True, 0
        state.shop[0] = observation("shop_0", "senna")
        result = next_action(state, plan(state, {"senna": 3}), self.rules)
        self.assertEqual("buy", result.kind)
        self.assertTrue(result.metadata["merge_expected"])
        # After a confirmed purchase/merge, count 3 exactly reaches the target.
        state.owned_copies["senna"] = 3
        state.owned_star_counts["senna"] = [0, 1, 0]
        self.assertEqual("wait", next_action(state, plan(state, {"senna": 3}), self.rules).kind)

    def test_inconsistent_or_unreviewed_summary_never_falls_back_to_old_grid(self):
        state = playable()
        state.shop[0] = observation("shop_0", "senna")
        state.owned_copies = {"senna": 2}
        state.owned_star_counts = {"senna": [0, 1, 0]}
        state.ownership_complete, state.bench_free = True, 5
        result = next_action(state, plan(state), self.rules)
        self.assertEqual("wait", result.kind)
        self.assertIn("矛盾", result.reason)
        state.owned_star_counts["senna"] = [2, 0, 0]
        state.ownership_complete = False
        self.assertIn("完整性", next_action(state, plan(state), self.rules).reason)

    def test_aggregate_stars_must_prove_merge_and_budget_metadata_is_explicit(self):
        state = playable()
        state.board, state.bench = {}, {}
        state.owned_copies, state.owned_star_counts = {"senna": 3}, {"senna": [0, 1, 0]}
        state.ownership_complete, state.bench_free = True, 0
        state.shop[0] = observation("shop_0", "senna")
        self.assertEqual("wait", next_action(state, plan(state), self.rules).kind)
        state.bench_free = 1
        result = next_action(state, plan(state), self.rules, spent=4)
        self.assertEqual("buy", result.kind)
        self.assertEqual(30, result.metadata["min_gold"])
        self.assertEqual(16, result.metadata["max_spend_remaining"])
        self.assertEqual(0, result.metadata["reserve_gold"])

    def test_confirmed_purchase_updates_tracked_plan_without_requiring_full_grid_ocr(self):
        from jcc_agent.control import ActionController, RecordingBackend
        from jcc_agent.tracking import SelfTracker

        initial = playable()
        initial.round = "3-1"
        initial.field_confidence = {key: 1 for key in ("phase", "round", "gold", "level", "seconds_left", "viewed_player", "detail_open")}
        initial.board["board_0"] = observation("board_0", "senna")
        initial.board["board_1"] = observation("board_1", "senna")
        initial.shop[0] = observation("shop_0", "senna")
        initial.shop[0].cost = 1
        tracker = SelfTracker({**initial.to_dict(), "ordinary_merges": True}, self.rules)
        before = tracker.fuse(initial)
        action = next_action(before, plan(before, {"senna": 3}), self.rules)
        backend = RecordingBackend()
        controller = ActionController(backend, {"calibrated": True, "reference_size": [100, 100],
                                               "points": {"shop_0": [0.5, 0.5]}}, self.rules,
                                      dry_run=False, live_enabled=True)
        self.assertEqual("pending", controller.submit(action, before, now=1).status)
        after = deepcopy(initial)
        after.version += 1
        after.observed_at += 1
        after.gold -= 1
        after.board, after.bench = {}, {}  # actual OCR has no empty-cell/star text
        after.shop[0] = observation("shop_0")
        after.shop[0].observed_at = after.observed_at
        confirmed = controller.observe(after, now=2)
        self.assertEqual("confirmed", confirmed.status)
        self.assertTrue(tracker.apply_confirmed(confirmed, before, after))
        fused = tracker.fuse(after)
        self.assertEqual(3, fused.owned_copies["senna"])
        self.assertEqual([0, 1, 0], fused.owned_star_counts["senna"])
        following = next_action(fused, plan(fused, {"senna": 3}), self.rules, spent=1)
        self.assertEqual("wait", following.kind)
        self.assertIn("已达成", following.reason)
        self.assertEqual(1, len(backend.events))

    def test_selection_changes_on_strong_evidence_with_hysteresis(self):
        state = playable()
        comps = [Composition("a", "阿狸", {"ahri": 9}, ["ahri"]),
                 Composition("b", "盖伦", {"garen": 9}, ["garen"])]
        previous = StrategyPlan("a", {"ahri": 9}, ["ahri"], 30, 20, 6)
        # A single newly offered card cannot repeatedly flip the plan.
        state.shop = [observation(f"shop_{i}") for i in range(5)]
        state.shop[0] = observation("shop_0", "garen")
        chosen = choose_plan(state, {}, self.rules, comps, previous)
        self.assertEqual("a", chosen.composition_id)
        state.board["board_0"] = observation("board_0", "garen", 3)
        self.assertEqual("b", choose_plan(state, {}, self.rules, comps, previous).composition_id)
        state.hp = 12
        self.assertEqual(0, choose_plan(state, {}, self.rules, comps, previous).min_gold)


if __name__ == "__main__":
    unittest.main()
