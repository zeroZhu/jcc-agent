"""Shared contracts. Missing information is explicit, never an empty slot."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Hero:
    id: str
    name: str
    cost: int
    traits: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)


@dataclass
class Ruleset:
    id: str
    name: str
    heroes: dict[str, Hero]
    pool_by_cost: dict[int, int]
    shop_odds: dict[int, list[float]]
    refresh_cost: int = 2
    xp_cost: int = 4
    xp_per_purchase: int = 4
    provenance: str = "unverified"
    notes: list[str] = field(default_factory=list)


@dataclass
class SlotObservation:
    slot: str
    status: str = "unknown"  # occupied / empty / unknown
    hero_id: str | None = None
    star: int | None = None
    cost: int | None = None
    confidence: float = 0.0
    observed_at: float = 0.0
    source: str = "ocr"


@dataclass
class PlayerSnapshot:
    player_id: str
    slots: dict[str, SlotObservation] = field(default_factory=dict)
    observed_at: float = 0.0
    complete: bool = False
    eliminated: bool = False


@dataclass
class Choice:
    index: int
    text: str
    confidence: float = 0.0
    title: str = ""


@dataclass
class GameState:
    version: int = 0
    observed_at: float = 0.0
    phase: str = "unknown"  # encounter / transition / pve / augment / preparation / combat / gameplay / ended / unknown
    round: str | None = None
    gold: int | None = None
    level: int | None = None
    hp: int | None = None
    xp: int | None = None
    xp_to_level: int | None = None
    seconds_left: float | None = None
    shop: list[SlotObservation] = field(default_factory=list)
    board: dict[str, SlotObservation] = field(default_factory=dict)
    bench: dict[str, SlotObservation] = field(default_factory=dict)
    opponents: dict[str, PlayerSnapshot] = field(default_factory=dict)
    items: list[str] = field(default_factory=list)
    augments: list[str] = field(default_factory=list)
    choices: list[Choice] = field(default_factory=list)
    viewed_player: str = "unknown"  # self must be positively established for live actions
    detail: SlotObservation | None = None
    detail_open: bool | None = None
    shop_visible: bool | None = None
    field_confidence: dict[str, float] = field(default_factory=dict)
    field_sources: dict[str, str] = field(default_factory=dict)
    field_observed_at: dict[str, float] = field(default_factory=dict)
    owned_copies: dict[str, int] | None = None
    ownership_complete: bool = False
    bench_free: int | None = None
    owned_star_counts: dict[str, list[int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PoolEstimate:
    hero_id: str
    capacity: int
    known_held: int
    remaining_low: int
    remaining_high: int
    estimated_remaining: float
    oldest_observation: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class Composition:
    id: str
    name: str
    targets: dict[str, int]  # desired base copies, not number of physical pieces
    priority: list[str] = field(default_factory=list)
    traits: list[str] = field(default_factory=list)
    preferred_items: list[str] = field(default_factory=list)
    roll_level: int = 6
    min_gold: int = 30
    max_roll_spend: int = 20
    description: str = ""


@dataclass
class StrategyPlan:
    composition_id: str
    targets: dict[str, int]
    buy_priority: list[str]
    min_gold: int
    max_roll_spend: int
    roll_level: int
    scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    state_version: int = 0


@dataclass
class Action:
    kind: str  # buy / refresh / sell / level / move / scout / choose / return_home / close_detail / wait
    target: str = ""
    hero_id: str | None = None
    state_version: int = 0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    action: Action
    status: str  # confirmed / pending / failed / rejected / simulated / unresolved
    reason: str
    observed_at: float = 0.0
