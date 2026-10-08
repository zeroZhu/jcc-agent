"""A single session owns strategy, budget and history independently of OCR."""
from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field

from .models import Action, GameState, PlayerSnapshot, StrategyPlan
from .pool import estimate_pool
from .strategy import choose_plan, next_action


@dataclass
class SnapshotBook:
    players: dict[str, PlayerSnapshot] = field(default_factory=dict)

    def update(self, incoming: PlayerSnapshot) -> None:
        previous = self.players.get(incoming.player_id)
        if previous and incoming.observed_at < previous.observed_at:
            return
        if incoming.complete or incoming.eliminated or previous is None:
            self.players[incoming.player_id] = deepcopy(incoming)
            return
        merged = deepcopy(previous)
        for key, slot in incoming.slots.items():
            current = merged.slots.get(key)
            if current is None or slot.observed_at >= current.observed_at:
                # An unknown observation keeps the previous identity as historical,
                # but is explicitly unknown so it cannot become actionable truth.
                merged.slots[key] = deepcopy(slot)
        merged.observed_at = incoming.observed_at
        merged.complete = False
        self.players[incoming.player_id] = merged


class AgentSession:
    def __init__(self, rules, compositions, *, choice_preferences=None):
        self.rules = rules
        self.compositions = compositions
        self.book = SnapshotBook()
        self.plan: StrategyPlan | None = None
        self.spent = 0
        self.last_round: str | None = None
        self._round_candidate: str | None = None
        self._round_candidate_count = 0
        self.last_version = -1
        self.choice_preferences = choice_preferences or {}

    def account_confirmed_spend(self, before: GameState, after: GameState) -> None:
        if before.gold is not None and after.gold is not None:
            self.spent += max(0, before.gold - after.gold)

    def decide(self, observation: GameState):
        state = deepcopy(observation)
        if state.version <= self.last_version:
            return state, {}, self.plan, Action("wait", state_version=state.version, reason="重复或乱序观测")
        self.last_version = state.version
        # OCR corrections and low-confidence readings must not replenish money.
        trusted_round = bool(state.round and re.fullmatch(r"[1-9]-[1-9]", state.round)
                             and state.field_confidence.get("round", 0) >= 0.85)
        if trusted_round:
            current = tuple(map(int, state.round.split("-")))
            previous = tuple(map(int, self.last_round.split("-"))) if self.last_round else None
            if previous is None:
                self.spent = 0
                self.last_round = state.round
            elif current > previous:
                if self._round_candidate == state.round:
                    self._round_candidate_count += 1
                else:
                    self._round_candidate = state.round
                    self._round_candidate_count = 1
                if self._round_candidate_count >= 2:
                    self.spent = 0
                    self.last_round = state.round
                    self._round_candidate = None
                    self._round_candidate_count = 0
            else:
                self._round_candidate = None
                self._round_candidate_count = 0
        else:
            self._round_candidate = None
            self._round_candidate_count = 0
        for snapshot in state.opponents.values():
            self.book.update(snapshot)
        state.opponents = deepcopy(self.book.players)
        pool = estimate_pool(state, self.rules, now=state.observed_at)
        if not self.compositions:
            return state, pool, None, Action("wait", state_version=state.version, reason="未配置候选阵容")
        self.plan = choose_plan(state, pool, self.rules, self.compositions, previous=self.plan)
        if state.phase in {"encounter", "augment"}:
            preferences = self.choice_preferences.get(state.phase, [])
            for preferred in preferences:
                matches = [c for c in state.choices if (c.title or c.text) == preferred and c.confidence >= 0.9]
                if len(matches) == 1:
                    choice = matches[0]
                    return state, pool, self.plan, Action(
                        "choose", f"choice_{choice.index}", state_version=state.version,
                        reason=f"命中配置偏好：{preferred}", metadata={"text": choice.text, "title": choice.title, "phase": state.phase},
                    )
            return state, pool, self.plan, Action(
                "wait", state_version=state.version, reason="选择阶段缺少可信偏好匹配，保留选择待处理",
            )
        action = next_action(state, self.plan, self.rules, spent=self.spent)
        if action is not None and action.kind in {"buy", "refresh", "level"} and (
                not trusted_round or state.round != self.last_round):
            # Spending in the first candidate frame would be confirmed just
            # before the second frame resets the budget, losing that debit.
            # Unknown/regressed rounds also cannot establish spend ownership.
            action = Action("wait", state_version=state.version,
                            reason="回合归属尚未稳定确认，暂停新经济动作以避免预算漏记")
        return state, pool, self.plan, action
