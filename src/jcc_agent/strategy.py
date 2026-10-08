"""Transparent starter policies; no claim of solved or high-rank play.

Selection scores are relative heuristic units, never win probabilities. Actions
are one-step proposals: callers must confirm the result, update state/version,
and carry total spending across purchases, rerolls and XP before calling again.
"""
from __future__ import annotations

from collections import Counter

from .models import Action, Composition, GameState, PoolEstimate, Ruleset, StrategyPlan
from .pool import MIN_CONFIDENCE, STALE_AFTER_SECONDS, STAR_COPIES, ownership_summary_error

SWITCH_MARGIN = 3.0
ACTION_TIME_RESERVE = 3.0


def _held(state: GameState, rules: Ruleset) -> Counter:
    if ownership_summary_error(state, rules) is None:
        return Counter(state.owned_copies)
    counts: Counter = Counter()
    for slot in (*state.board.values(), *state.bench.values()):
        if slot.status == "occupied" and slot.hero_id and slot.confidence >= MIN_CONFIDENCE:
            # Selection may use an identified piece as weak evidence even when its
            # star is unknown; execution below requires exact own inventory.
            counts[slot.hero_id] += STAR_COPIES.get(slot.star, 1)
    return counts


def choose_plan(state: GameState, pool: dict[str, PoolEstimate], rules: Ruleset,
                compositions: list[Composition], previous: StrategyPlan | None = None) -> StrategyPlan:
    if not compositions:
        raise ValueError("至少需要一个候选阵容")
    if len({c.id for c in compositions}) != len(compositions):
        raise ValueError("候选阵容 id 重复")
    held = _held(state, rules)
    shop = Counter(s.hero_id for s in state.shop
                   if s.status == "occupied" and s.confidence >= MIN_CONFIDENCE)
    scores: dict[str, float] = {}
    for comp in compositions:
        if not comp.targets or any(h not in rules.heroes or isinstance(n, bool) or
                                   not isinstance(n, int) or not 1 <= n <= 9
                                   for h, n in comp.targets.items()):
            raise ValueError(f"阵容 {comp.id} 包含未知英雄或无效目标")
        priority = comp.priority or list(comp.targets)
        score = 0.0
        for hid, needed in comp.targets.items():
            weight = 1.8 if hid in priority[:2] else 1.0
            owned = min(held[hid], needed)
            score += weight * owned ** 0.65 * 2
            score += weight * min(shop[hid], max(0, needed - owned)) * 0.7
            estimate = pool.get(hid)
            if estimate is not None and estimate.capacity > 0:
                # Reduce the influence of incomplete, stale or unverified pool
                # estimates instead of treating their point values as facts.
                uncertain = any(w.startswith(("missing_players", "incomplete_snapshot", "unknown_", "stale_", "unverified_rules"))
                                for w in estimate.warnings)
                reliability = 0.25 if uncertain else 1.0
                other_held = max(0, estimate.capacity - estimate.estimated_remaining - held[hid])
                score -= reliability * weight * 4 * other_held / estimate.capacity
        score += 2.0 * len(set(state.items) & set(comp.preferred_items))
        # Being below the roll level is a modest cost, not a prohibition: the
        # executor can level before rolling and still buy useful natural shops.
        if state.level is not None:
            score -= max(0, comp.roll_level - state.level) * 0.6
        scores[comp.id] = round(score, 4)
    chosen = max(compositions, key=lambda comp: scores[comp.id])
    reasons = ["基于已持有副本、当前来牌、装备与同行估计评分；分数不是胜率"]
    by_id = {c.id: c for c in compositions}
    if previous and previous.composition_id in by_id and chosen.id != previous.composition_id:
        old = by_id[previous.composition_id]
        stranded = sum(min(held[h], n) * rules.heroes[h].cost
                       for h, n in old.targets.items() if h not in chosen.targets)
        margin = SWITCH_MARGIN + min(5.0, stranded * 0.15)
        if scores[chosen.id] < scores[old.id] + margin:
            chosen = old
            reasons.append(f"保留原阵容：新方向优势未超过切换门槛 {margin:.2f}")
        else:
            reasons.append(f"切换阵容：评分优势超过含转型成本的门槛 {margin:.2f}")
    minimum = chosen.min_gold
    trusted_hp = state.hp if state.field_confidence.get("hp", 1) >= MIN_CONFIDENCE else None
    if trusted_hp is not None and 0 < trusted_hp <= 15:
        minimum = 0
        reasons.append("血量不高于 15：允许在既定总预算内用尽金币稳血")
    elif trusted_hp is not None and 15 < trusted_hp <= 30:
        minimum = min(minimum, 10)
        reasons.append("血量不高于 30：最低保留金币降至 10")
    if rules.provenance != "jcc_verified":
        reasons.append("规则仍为版本参考；阵容为可解释启发式模板，未验证胜率")
    priority = chosen.priority + [h for h in chosen.targets if h not in chosen.priority]
    return StrategyPlan(chosen.id, dict(chosen.targets), priority, minimum,
                        chosen.max_roll_spend, chosen.roll_level, scores, reasons, state.version)


def next_action(state: GameState, plan: StrategyPlan, rules: Ruleset, spent: int = 0) -> Action | None:
    """Propose one safe action, or a wait with an actionable explanation.

    This policy never sells an unknown piece to make room. A full bench may buy
    only when two confirmed one-star copies guarantee an ordinary merge. Either
    a reviewed/tracked complete ledger or a full own-grid observation is needed;
    missing dict entries never imply empty cells.
    """
    def action(kind: str, reason: str, target: str = "", hero_id: str | None = None, **metadata) -> Action:
        if kind in {"buy", "refresh", "level"}:
            metadata.setdefault("min_gold", plan.min_gold)
            metadata.setdefault("max_spend_remaining", plan.max_roll_spend - spent)
            metadata.setdefault("reserve_gold", 0)
        return Action(kind, target, hero_id, state.version, reason, metadata)

    if state.phase == "ended":
        return None
    # Economy controls remain available during combat. ``gameplay`` means OCR
    # positively established normal gameplay, while its substage is unknown;
    # it does NOT establish a static board or authorize movement/deployment.
    if state.phase not in {"preparation", "pve", "combat", "gameplay"}:
        return action("wait", f"阶段 {state.phase} 不允许本轮买牌/刷新/升级，等待确认正常对局阶段")
    if state.viewed_player != "self":
        return action("return_home", "先返回并确认自己的棋盘")
    if plan.state_version != state.version:
        return action("wait", "计划与当前状态版本不一致，需要重新规划")
    if (any(isinstance(n, bool) or not isinstance(n, int) or n < 0
            for n in (plan.min_gold, plan.max_roll_spend)) or plan.roll_level not in rules.shop_odds):
        return action("wait", "计划金币底线、总预算或目标等级无效")
    if state.detail_open is True:
        return action("close_detail", "关闭棋子详情后重新读取商店")
    if state.detail_open is None:
        return action("wait", "详情面板开关状态未知，先确认商店可操作")
    if not isinstance(spent, int) or isinstance(spent, bool) or spent < 0:
        return action("wait", "累计开支无效；预算必须包含买牌、刷新和升级")
    if state.gold is None or state.gold < 0 or state.level not in rules.shop_odds:
        return action("wait", "金币或等级未知/无效，先更新 OCR 状态")
    # Explicit low-confidence numeric readings cannot authorize spending. Empty
    # field_confidence supports callers with manually verified state fixtures.
    if any(state.field_confidence.get(k, 1) < MIN_CONFIDENCE for k in ("gold", "level")):
        return action("wait", "金币或等级 OCR 置信度不足")
    if state.seconds_left is not None and state.seconds_left <= ACTION_TIME_RESERVE:
        return action("wait", "剩余时间不足，停止产生新操作")
    inventory = [*state.board.values(), *state.bench.values()]
    use_ledger = state.owned_copies is not None or state.ownership_complete
    if use_ledger:
        error = ownership_summary_error(state, rules)
        if error is not None:
            return action("wait", error)
    else:
        if len(state.board) != 28 or len(state.bench) != 9:
            return action("wait", "自身棋盘/备战席扫描不完整，不能将未记录位置当作空位；可载入已审阅持牌快照")
        for slot in inventory:
            if slot.status not in {"empty", "occupied"} or slot.confidence < MIN_CONFIDENCE:
                return action("wait", f"自身位置 {slot.slot} 未确认，先补全持牌")
            if slot.status == "occupied" and (slot.hero_id not in rules.heroes or slot.star not in STAR_COPIES):
                return action("wait", f"自身位置 {slot.slot} 身份或星级未知，不能盲目追买")
            age = state.observed_at - slot.observed_at
            if age > STALE_AFTER_SECONDS or age < -1:
                return action("wait", f"自身位置 {slot.slot} 观察已过期，需要重扫或载入可信持牌账本")
    if len(state.shop) != 5 or len({s.slot for s in state.shop}) != 5:
        return action("wait", "商店未形成五个不同槽位的完整快照")
    for slot in state.shop:
        if slot.status not in {"occupied", "empty"} or slot.confidence < MIN_CONFIDENCE:
            return action("wait", f"商店位置 {slot.slot} 尚未确认，不能购买或跳过")
        if slot.status == "occupied" and slot.hero_id not in rules.heroes:
            return action("wait", f"商店位置 {slot.slot} 英雄不在当前规则中")
        if slot.status == "occupied" and (not isinstance(slot.cost, int) or isinstance(slot.cost, bool) or slot.cost < 1):
            return action("wait", f"商店位置 {slot.slot} 可见售价未知/无效，不能按英雄基础费用推断")
    if not plan.targets or any(h not in rules.heroes or isinstance(n, bool) or not isinstance(n, int) or n < 1 or n > 9
                               for h, n in plan.targets.items()):
        return action("wait", "阵容目标为空或包含不支持的英雄/副本数")
    held = _held(state, rules)
    missing = {hid: n - held[hid] for hid, n in plan.targets.items() if held[hid] < n}
    if not missing:
        return action("wait", "所有目标副本已达成，停止 D 牌")
    budget_left = plan.max_roll_spend - spent
    affordable = min(state.gold - plan.min_gold, budget_left)
    if affordable <= 0:
        return action("wait", "已达最低金币或本轮总支出上限")
    empty_bench = state.bench_free > 0 if use_ledger else any(s.status == "empty" for s in state.bench.values())
    priority = list(dict.fromkeys(plan.buy_priority + list(plan.targets)))
    candidates = [s for s in state.shop if s.status == "occupied" and s.hero_id in missing]
    candidates.sort(key=lambda s: priority.index(s.hero_id))
    blocked_purchase = False
    for slot in candidates:
        # Shop price can differ from pool tier (e.g. the observed 一元店).
        cost = slot.cost
        if cost > affordable:
            blocked_purchase = True
            continue
        one_stars = (state.owned_star_counts.get(slot.hero_id, [0, 0, 0])[0] if use_ledger else
                     sum(s.status == "occupied" and s.hero_id == slot.hero_id and s.star == 1 for s in inventory))
        if empty_bench or one_stars >= 2:
            return action("buy", "购买未完成目标的商店英雄", slot.slot, slot.hero_id,
                          expected_cost=cost, merge_expected=one_stars >= 2)
        blocked_purchase = True
    if blocked_purchase:
        return action("wait", "目标牌尚未处理：预算不足或备战席无法容纳；保留当前商店")
    if state.level < plan.roll_level:
        if state.xp is None or state.xp_to_level is None or not 0 <= state.xp < state.xp_to_level:
            return action("wait", "尚未到计划 D 牌等级且经验进度未知，先读取经验")
        if any(state.field_confidence.get(k, 1) < MIN_CONFIDENCE for k in ("xp", "xp_to_level")):
            return action("wait", "经验 OCR 置信度不足")
        reserve = min(rules.heroes[h].cost for h in missing)
        if rules.xp_cost + reserve > affordable:
            return action("wait", "升级后不足以保留购买目标牌的金币/预算，继续积累经济")
        return action("level", "先升到计划搜牌等级，每次购买经验后重新确认", "xp",
                      expected_cost=rules.xp_cost, expected_xp=rules.xp_per_purchase, reserve_gold=reserve)
    if not empty_bench:
        return action("wait", "备战席已满，不刷新；先由持牌管理释放空间")
    rollable = [h for h in missing if rules.shop_odds[state.level][rules.heroes[h].cost - 1] > 0]
    if not rollable:
        return action("wait", "当前等级无法抽到任何缺失目标，停止无效刷新")
    # Future shop prices cannot be read before rolling. Base tiers provide a
    # conservative reserve for ordinary/discount shops, not a claim of actual
    # future prices; every purchase still requires a new visible price.
    reserve = max([rules.heroes[h].cost for h in rollable] +
                  [s.cost for s in state.shop if s.status == "occupied" and s.hero_id in rollable])
    if rules.refresh_cost + reserve > affordable:
        return action("wait", "刷新后无法预留目标购买费用或会超预算，停止 D 牌")
    return action("refresh", "目标尚未完成，预算包含刷新费及至少一张最高费用目标的购买预留", "refresh",
                  expected_cost=rules.refresh_cost, purchase_reserve=reserve, reserve_gold=reserve)
