"""Latest-snapshot pool estimates, not omniscient card counting.

The intervals describe holdings uncertainty under configured normal-star rules.
Unknown opponent shops and special generators are explicitly excluded. The point
estimate subtracts latest observed holdings; it is a heuristic, not a posterior.
"""
from __future__ import annotations

import time
from collections import defaultdict

from .models import GameState, PoolEstimate, Ruleset, SlotObservation

STALE_AFTER_SECONDS = 60.0
MIN_CONFIDENCE = 0.85
STAR_COPIES = {1: 1, 2: 3, 3: 9}


def ownership_summary_error(state: GameState, rules: Ruleset) -> str | None:
    """Validate an explicitly reviewed/transaction-tracked aggregate ledger.

    Completeness is an upstream provenance assertion, never inferred from an
    empty dict. Star distributions are required to justify merge capacity.
    """
    if state.ownership_complete is not True or state.owned_copies is None:
        return "己方持牌汇总尚未由审阅快照或确认事件建立完整性"
    if not isinstance(state.owned_copies, dict) or not isinstance(state.owned_star_counts, dict):
        return "己方持牌汇总类型无效"
    if (not isinstance(state.bench_free, int) or isinstance(state.bench_free, bool)
            or not 0 <= state.bench_free <= 9):
        return "备战席可用空位下界未确认或超出 0–9"
    if set(state.owned_copies) != set(state.owned_star_counts):
        return "己方副本汇总与星级分布的英雄集合不一致"
    physical_count = 0
    for hid, count in state.owned_copies.items():
        stars = state.owned_star_counts[hid]
        if hid not in rules.heroes or isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return f"己方持牌汇总存在无效英雄或副本数：{hid}"
        if (not isinstance(stars, (list, tuple)) or len(stars) != 3 or
                any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in stars)):
            return f"己方星级分布无效：{hid}"
        if count != stars[0] + 3 * stars[1] + 9 * stars[2]:
            return f"己方副本数与星级分布矛盾：{hid}"
        physical_count += sum(stars)
    if physical_count + state.bench_free > 37:
        return "己方物理棋子和空位数超过普通棋盘及备战席总容量"
    return None


def estimate_pool(state: GameState, rules: Ruleset, now: float | None = None) -> dict[str, PoolEstimate]:
    now = time.time() if now is None else now
    groups: list[tuple[str, list[SlotObservation], bool, float]] = [
        ("self", list(state.board.values()) + list(state.bench.values()),
         len(state.board) == 28 and len(state.bench) == 9, state.observed_at)
    ]
    warnings: list[str] = []
    if state.owned_copies is not None or state.ownership_complete:
        error = ownership_summary_error(state, rules)
        if error is None:
            ledger_slots = [SlotObservation(f"ledger_{hid}_{star}_{i}", "occupied", hid, star,
                                           confidence=1.0, observed_at=state.observed_at, source="owned_ledger")
                            for hid, counts in state.owned_star_counts.items()
                            for star, number in enumerate(counts, 1) for i in range(number)]
            groups[0] = ("self", ledger_slots, True, state.observed_at)
        else:
            # A broken ledger cannot be replaced silently by stale spatial data.
            groups[0] = ("self", [], False, state.observed_at)
            warnings.append(f"unknown_ownership: {error}")
    if len(state.opponents) < 7:
        warnings.append("missing_players: 尚未记录全部 7 名对手，余量下界设为 0")
    incomplete = len(state.opponents) < 7
    for player, snapshot in state.opponents.items():
        if snapshot.eliminated:
            continue
        groups.append((player, list(snapshot.slots.values()),
                       snapshot.complete and len(snapshot.slots) == 37, snapshot.observed_at))
    exact = defaultdict(int)
    fresh_min = defaultdict(int)
    possible_max = defaultdict(int)
    observed_midpoint = defaultdict(float)
    seen_at: dict[str, list[float]] = defaultdict(list)
    per_hero: dict[str, list[str]] = defaultdict(list)
    unknown_any_cost = False
    for player, slots, complete, snapshot_at in groups:
        if not complete:
            incomplete = True
            warnings.append(f"incomplete_snapshot:{player}: 只观察了部分位置")
        # A declared complete snapshot cannot turn unknown cells into empty cells.
        for slot in slots:
            stamp = slot.observed_at or snapshot_at
            age = now - stamp
            stale = age > STALE_AFTER_SECONDS or age < -1
            if slot.status == "empty" and slot.confidence >= MIN_CONFIDENCE and not stale:
                continue
            if slot.status != "occupied" or slot.hero_id not in rules.heroes or slot.confidence < MIN_CONFIDENCE:
                # Slot cost is a visible price and may be discounted. It cannot
                # narrow the pool tier of an unidentified hero.
                unknown_any_cost = True
                warnings.append(f"unknown_slot:{player}/{slot.slot}: 空位/身份/置信度未确认")
                continue
            hid = slot.hero_id
            capacity = rules.pool_by_cost[rules.heroes[hid].cost]
            seen_at[hid].append(stamp)
            count = STAR_COPIES.get(slot.star)
            lower, upper = (count, count) if count is not None else (1, 9)
            if count is not None:
                exact[hid] += count
            else:
                per_hero[hid].append(f"unknown_star:{player}/{slot.slot}: 按普通 1–3 星估计 1–9 张")
            observed_midpoint[hid] += (lower + upper) / 2
            if stale:
                # A stale identified piece could have been sold and replaced by
                # any other hero, not merely changed its own copy count.
                unknown_any_cost = True
                lower, upper = 0, capacity
                per_hero[hid].append(f"stale_observation:{player}/{slot.slot}: 记录过期或时间戳在未来")
            fresh_min[hid] += lower
            possible_max[hid] += upper
    shared = list(dict.fromkeys(warnings))
    shared.append("visible_holdings_only: 未统计隐藏商店；未建模复制、生成及特殊升星机制")
    if rules.provenance != "jcc_verified":
        shared.append(f"unverified_rules:{rules.provenance}: 数值按配置假设，未完成金铲铲目标版本验证")
    estimates: dict[str, PoolEstimate] = {}
    for hid, hero in rules.heroes.items():
        capacity = rules.pool_by_cost[hero.cost]
        local = shared + per_hero[hid]
        low = max(0, capacity - possible_max[hid])
        high = max(0, capacity - fresh_min[hid])
        if incomplete or unknown_any_cost:
            low = 0
        if exact[hid] > capacity or fresh_min[hid] > capacity:
            local.append(f"pool_conflict: 观测持有量 {exact[hid]} / 至少 {fresh_min[hid]} 超过配置容量 {capacity}，请校验版本或记录")
        estimate = min(high, max(low, capacity - observed_midpoint[hid]))
        estimates[hid] = PoolEstimate(
            hero_id=hid, capacity=capacity, known_held=exact[hid],
            remaining_low=low, remaining_high=high, estimated_remaining=estimate,
            oldest_observation=min(seen_at[hid]) if seen_at[hid] else None,
            warnings=list(dict.fromkeys(local)),
        )
    return estimates
