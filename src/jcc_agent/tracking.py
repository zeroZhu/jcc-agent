"""Own-inventory ledger from explicit snapshots and confirmed actions only.

Copies and physical positions are separate facts. A confirmed purchase can
advance a complete copy ledger while its destination remains unknown. Unknown
OCR output never establishes that a position is empty.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

from .models import ActionResult, GameState, Ruleset, SlotObservation

BOARD_SLOTS = tuple(f"board_{i}" for i in range(28))
BENCH_SLOTS = tuple(f"bench_{i}" for i in range(9))
STAR_COPIES = {1: 1, 2: 3, 3: 9}


class SelfTracker:
    """Fuse tracked state without manufacturing occupied/empty grid locations.

    ``initial`` is an explicitly reviewed GameState or a JSON-compatible mapping
    with board/bench slot maps. Omitted positions become unknown. JSON entries
    may omit confidence (manual observations default to 1), but must explicitly
    state occupied/empty and give hero_id/star for occupied entries. The optional
    ordinary_merges flag must be explicitly enabled before inferring star
    distributions after purchases. No special summoning/copying is assumed.
    """
    def __init__(self, initial: dict[str, Any] | GameState | None, rules: Ruleset,
                 *, ordinary_merges: bool | None = None, min_confidence: float = 0.85,
                 max_board_age_seconds: float = 60.0):
        self.rules = rules
        self.min_confidence = min_confidence
        if max_board_age_seconds <= 0:
            raise ValueError("Board evidence lifetime must be positive")
        self.max_board_age_seconds = max_board_age_seconds
        document = initial.to_dict() if isinstance(initial, GameState) else deepcopy(initial or {})
        self.ordinary_merges = bool(document.get("ordinary_merges", False)) if ordinary_merges is None else ordinary_merges
        self.board = self._read_slots(document.get("board", {}), BOARD_SLOTS, initial is not None and not isinstance(initial, GameState))
        self.bench = self._read_slots(document.get("bench", {}), BENCH_SLOTS, initial is not None and not isinstance(initial, GameState))
        self.owned_copies: dict[str, int] = {}
        self.owned_star_counts: dict[str, list[int]] = {}
        self.ownership_complete = False
        self.bench_free: int | None = None
        self.observed_at = float(document.get("observed_at", 0))
        self.last_version = int(document.get("version", 0))
        self.warnings: list[str] = []
        self._events: set[tuple[Any, ...]] = set()
        self._consumed_shop: dict[str, SlotObservation] = {}
        self._shop_round: str | None = None
        self._summarize_positions()

    def _read_slots(self, entries: dict, names: tuple[str, ...], manual: bool) -> dict[str, SlotObservation]:
        if not isinstance(entries, dict) or any(key not in names for key in entries):
            raise ValueError("Initial inventory contains an invalid position")
        result = {name: SlotObservation(name, source="unobserved") for name in names}
        for name, raw in entries.items():
            if isinstance(raw, SlotObservation):
                slot = deepcopy(raw)
            elif isinstance(raw, dict):
                values = dict(raw)
                values["slot"] = name
                if manual:
                    values.setdefault("confidence", 1.0)
                    values.setdefault("source", "manual_initial")
                slot = SlotObservation(**values)
            else:
                raise ValueError("Initial inventory entries must be slot objects")
            if slot.slot != name or slot.status not in {"unknown", "occupied", "empty"}:
                raise ValueError("Invalid initial inventory observation")
            if slot.status == "occupied" and (slot.hero_id not in self.rules.heroes or slot.star not in STAR_COPIES):
                # An identified piece with an unreadable star is still usable as
                # partial evidence, but never makes the total ledger complete.
                if slot.hero_id not in self.rules.heroes:
                    raise ValueError(f"Unknown initial hero: {slot.hero_id}")
            if slot.status == "empty" and (slot.hero_id is not None or slot.star is not None):
                raise ValueError("An empty position cannot contain a hero or star")
            result[name] = slot
        return result

    def _known(self, slot: SlotObservation) -> bool:
        return (slot.confidence >= self.min_confidence and
                (slot.status == "empty" or (slot.status == "occupied" and
                 slot.hero_id in self.rules.heroes and slot.star in STAR_COPIES)))

    def _summarize_positions(self) -> None:
        counts: Counter = Counter()
        stars: dict[str, list[int]] = {}
        slots = [*self.board.values(), *self.bench.values()]
        self.ownership_complete = all(self._known(s) for s in slots)
        for slot in slots:
            if slot.status == "occupied" and self._known(slot):
                counts[slot.hero_id] += STAR_COPIES[slot.star]
                stars.setdefault(slot.hero_id, [0, 0, 0])[slot.star - 1] += 1
        self.owned_copies, self.owned_star_counts = dict(counts), stars
        self.bench_free = sum(s.status == "empty" and self._known(s) for s in self.bench.values())

    def holdings(self) -> dict[str, int] | None:
        return deepcopy(self.owned_copies) if self.ownership_complete else None

    def mark_uncertain(self, reason: str, *, positions: bool = True) -> None:
        """Call for unobserved manual actions, special grants or unknown merges."""
        self.ownership_complete = False
        self.owned_star_counts = {}
        if positions:
            self.board = {k: SlotObservation(k, observed_at=self.observed_at, source="uncertain_event") for k in BOARD_SLOTS}
            self.bench = {k: SlotObservation(k, observed_at=self.observed_at, source="uncertain_event") for k in BENCH_SLOTS}
            self.bench_free = 0
        if reason not in self.warnings:
            self.warnings.append(reason)

    def _invalidate_purchase_positions(self, hero: str, observed_at: float) -> None:
        # A purchase may consume any matching unit during automatic merging and
        # land in any open bench position. Preserve only unaffected board units.
        for name, slot in list(self.board.items()):
            if slot.hero_id == hero:
                self.board[name] = SlotObservation(name, observed_at=observed_at, source="purchase_location_unknown")
        for name, slot in list(self.bench.items()):
            if slot.status != "occupied" or slot.hero_id == hero:
                self.bench[name] = SlotObservation(name, observed_at=observed_at, source="purchase_location_unknown")

    def _increment_star_distribution(self, hero: str) -> None:
        counts = self.owned_star_counts.setdefault(hero, [0, 0, 0])
        counts[0] += 1
        for index in (0, 1):
            merges, counts[index] = divmod(counts[index], 3)
            counts[index + 1] += merges

    def _refresh_capacity_bound(self) -> None:
        """At most P-B tracked physical pieces can occupy the nine bench cells."""
        if not self.ownership_complete:
            return
        physical_by_hero = {hero: sum(stars) for hero, stars in self.owned_star_counts.items()}
        on_board: Counter = Counter()
        for slot in self.board.values():
            age = self.observed_at - slot.observed_at
            if (slot.status == "occupied" and slot.hero_id in self.rules.heroes
                    and slot.confidence >= self.min_confidence
                    and 0 <= age <= self.max_board_age_seconds):
                on_board[slot.hero_id] += 1
        if any(number > physical_by_hero.get(hero, 0) for hero, number in on_board.items()):
            self.mark_uncertain("board_physical_units_exceed_tracked_total", positions=False)
            return
        total_physical = sum(physical_by_hero.values())
        board_physical = sum(on_board.values())
        guaranteed_free = max(0, 9 - (total_physical - board_physical))
        self.bench_free = max(self.bench_free or 0, guaranteed_free)

    def apply_confirmed(self, result: ActionResult, before: GameState, after: GameState) -> bool:
        """Apply a controller-confirmed mutation once; never apply pending/dry-run."""
        action = result.action
        if result.status != "confirmed" or action.kind not in {"buy", "sell", "move", "refresh"}:
            return False
        if (action.state_version != before.version or after.version <= before.version
                or after.observed_at <= before.observed_at or after.version < self.last_version):
            return False
        event = (action.kind, action.target, action.hero_id, before.version)
        if event in self._events:
            return False
        if action.kind == "refresh":
            self._consumed_shop.clear()
            self._shop_round = after.round if self._trusted_round(after) else None
            self._events.add(event)
            return True
        hero = action.hero_id
        if action.kind == "buy":
            slot = next((s for s in before.shop if s.slot == action.target), None)
            if slot is None or slot.hero_id != hero or hero not in self.rules.heroes:
                return False
            self.owned_copies[hero] = self.owned_copies.get(hero, 0) + 1
            if self.ordinary_merges and self.ownership_complete:
                self._increment_star_distribution(hero)
            else:
                self.mark_uncertain("purchase_with_unverified_merge_rules", positions=False)
            # No assertion about where the merged piece survives. This lower
            # bound deliberately gives no credit for potentially freed slots.
            if self.bench_free is not None:
                self.bench_free = max(0, self.bench_free - 1)
            self._invalidate_purchase_positions(hero, after.observed_at)
            # Missing round OCR after a confirmed purchase is not a shop reset.
            # Keep its generation under the trusted pre-action round until a
            # future positive observation can either reuse or invalidate it.
            same_generation = (self._trusted_round(before) and
                               (not self._trusted_round(after) or after.round == before.round))
            if same_generation:
                if self._shop_round != before.round:
                    self._consumed_shop.clear()
                self._shop_round = before.round
                self._consumed_shop[action.target] = SlotObservation(
                    action.target, "empty", confidence=min(slot.confidence, after.field_confidence.get("gold", 0)),
                    observed_at=after.observed_at, source="confirmed_purchase_empty")
        elif action.kind == "sell":
            position = self.board.get(action.target) or self.bench.get(action.target)
            if position is None or not self._known(position) or position.status != "occupied":
                self.mark_uncertain("sold_piece_was_not_identified")
            else:
                hero = position.hero_id
                copies = STAR_COPIES[position.star]
                if self.owned_copies.get(hero, 0) < copies:
                    self.mark_uncertain("sale_conflicts_with_copy_ledger")
                else:
                    self.owned_copies[hero] -= copies
                    stars = self.owned_star_counts.get(hero)
                    if stars and stars[position.star - 1] > 0:
                        stars[position.star - 1] -= 1
                    mapping = self.board if action.target in self.board else self.bench
                    mapping[action.target] = SlotObservation(action.target, "empty", confidence=1.0,
                                                            observed_at=after.observed_at, source="confirmed_sale")
                    if action.target in self.bench and self.bench_free is not None:
                        self.bench_free = min(9, self.bench_free + 1)
        else:
            destination = action.metadata.get("destination")
            source_map = self.board if action.target in self.board else self.bench
            destination_map = self.board if destination in self.board else self.bench
            source, target = source_map.get(action.target), destination_map.get(destination)
            if source is None or source.status != "occupied" or not self._known(source):
                self.mark_uncertain("move_source_not_identified")
            elif target is None or not self._known(target):
                # Total copies survive a move, but an unverified swap/landing
                # cannot establish locations or create bench capacity.
                source_map[action.target] = SlotObservation(action.target, observed_at=after.observed_at,
                                                            source="move_location_unknown")
                if target is not None:
                    destination_map[destination] = SlotObservation(destination, observed_at=after.observed_at,
                                                                  source="move_location_unknown")
                self.bench_free = 0
            elif target.status == "empty":
                source_map[action.target] = SlotObservation(action.target, "empty", confidence=1,
                                                            observed_at=after.observed_at, source="confirmed_move")
                destination_map[destination] = replace(source, slot=destination, observed_at=after.observed_at,
                                                       source="confirmed_move")
                if self.bench_free is not None:
                    self.bench_free += int(action.target in self.bench) - int(destination in self.bench)
                    self.bench_free = min(9, max(0, self.bench_free))
            else:
                # Occupied-target movement is only a swap when explicitly
                # confirmed as such; otherwise it could have failed or merged.
                if action.metadata.get("swap_confirmed") is True:
                    source_map[action.target] = replace(target, slot=action.target, observed_at=after.observed_at)
                    destination_map[destination] = replace(source, slot=destination, observed_at=after.observed_at)
                else:
                    self.mark_uncertain("occupied_destination_move_not_resolved")
        self._events.add(event)
        self.observed_at = after.observed_at
        self.last_version = after.version
        self._refresh_capacity_bound()
        return True

    def _reconcile(self, observation: GameState) -> None:
        if observation.viewed_player != "self" or observation.version < self.last_version:
            return
        incoming = {**observation.board, **observation.bench}
        full = all(name in incoming and self._known(incoming[name])
                   and incoming[name].observed_at >= observation.observed_at
                   for name in (*BOARD_SLOTS, *BENCH_SLOTS))
        if full:
            previous = dict(self.owned_copies)
            previously_complete = self.ownership_complete
            self.board = {name: deepcopy(incoming[name]) for name in BOARD_SLOTS}
            self.bench = {name: deepcopy(incoming[name]) for name in BENCH_SLOTS}
            self._summarize_positions()
            if previously_complete and previous != self.owned_copies:
                self.warnings.append("full_observation_corrected_inventory_ledger")
            self.observed_at = observation.observed_at
            self.last_version = observation.version
            return
        for name, slot in incoming.items():
            mapping = self.board if name in self.board else self.bench
            previous = mapping.get(name)
            if previous is None or slot.observed_at < previous.observed_at or slot.confidence < self.min_confidence:
                continue
            if slot.status == "unknown":
                continue  # no text is not evidence of absence
            if slot.status == "occupied" and slot.hero_id not in self.rules.heroes:
                self.mark_uncertain("observation_contains_unknown_hero", positions=False)
                continue
            if self._known(previous):
                identity_changed = (previous.status, previous.hero_id) != (slot.status, slot.hero_id)
                star_changed = slot.star in STAR_COPIES and previous.star != slot.star
                if identity_changed or star_changed:
                    self.mark_uncertain("partial_observation_conflicts_with_tracked_position", positions=False)
            mapping[name] = deepcopy(slot)
        if self.ownership_complete:
            visible = Counter()
            for slot in [*self.board.values(), *self.bench.values()]:
                if slot.status == "occupied" and slot.confidence >= self.min_confidence and slot.hero_id in self.rules.heroes:
                    visible[slot.hero_id] += STAR_COPIES.get(slot.star, 1)
            if any(count > self.owned_copies.get(hero, 0) for hero, count in visible.items()):
                self.mark_uncertain("observed_copies_exceed_tracked_total", positions=False)
        explicit_empty = sum(slot.status == "empty" and self._known(slot) for slot in self.bench.values())
        if self.bench_free is not None and self.ownership_complete:
            self.bench_free = max(self.bench_free, explicit_empty)
        elif not self.ownership_complete:
            self.bench_free = explicit_empty
        self.observed_at = max(self.observed_at, observation.observed_at)
        self.last_version = max(self.last_version, observation.version)
        self._refresh_capacity_bound()

    def fuse(self, observation: GameState) -> GameState:
        """Return a copy with tracked self inventory and optional exact totals.

        Fusion never rewrites the incoming OCR/shop timestamp or version.
        Historic tracked positions keep their original timestamps.
        """
        self._reconcile(observation)
        state = deepcopy(observation)
        state.board, state.bench = deepcopy(self.board), deepcopy(self.bench)
        state.owned_copies = deepcopy(self.owned_copies)
        state.ownership_complete = self.ownership_complete
        state.owned_star_counts = deepcopy(self.owned_star_counts)
        state.bench_free = self.bench_free
        state = self.fuse_shop(state)
        state.warnings.extend(w for w in self.warnings if w not in state.warnings)
        return state

    def fuse_shop(self, observation: GameState) -> GameState:
        """Only reconcile transaction-derived shop empties; do not touch holdings.

        Suitable before controller.observe. It never creates a purchase, updates
        copy counts or treats tracked physical locations as fresh OCR evidence.
        """
        state = deepcopy(observation)
        new_card = any(slot.status == "occupied" and slot.confidence >= self.min_confidence
                       and slot.slot in self._consumed_shop for slot in state.shop)
        if new_card:
            # Positive card evidence supersedes cached consumption independently
            # of whether the round label was readable in this frame.
            self._consumed_shop.clear()
        if not self._trusted_round(state):
            state.warnings.append("shop_round_unverified_cache_suspended")
            return state
        if self._shop_round is None:
            self._shop_round = state.round
        elif state.round != self._shop_round:
            old_key, new_key = self._round_key(self._shop_round), self._round_key(state.round)
            if old_key is None or new_key is None or new_key <= old_key:
                state.warnings.append("shop_round_conflict_cache_suspended")
                return state
            self._consumed_shop.clear()
            self._shop_round = state.round
        # Without any positive shop evidence, a hidden shop and an empty shop
        # are indistinguishable. Do not turn the entire hidden UI into empties.
        visible_shop = any(slot.confidence >= self.min_confidence and slot.status == "occupied" for slot in state.shop)
        if visible_shop and state.viewed_player == "self":
            state.shop = [replace(self._consumed_shop[slot.slot], observed_at=slot.observed_at)
                          if slot.slot in self._consumed_shop and slot.status == "unknown" and slot.source == "ocr_blank"
                          else slot for slot in state.shop]
        return state

    def _trusted_round(self, state: GameState) -> bool:
        return state.round is not None and state.field_confidence.get("round", 0) >= self.min_confidence

    @staticmethod
    def _round_key(round_id: str) -> tuple[int, int] | None:
        try:
            stage, number = round_id.split("-", 1)
            return int(stage), int(number)
        except (ValueError, AttributeError):
            return None

    def export(self) -> dict[str, Any]:
        """Saveable evidence; review against the live match before reusing it."""
        return {"schema_version": 1, "ordinary_merges": self.ordinary_merges,
                "version": self.last_version, "observed_at": self.observed_at,
                "board": {k: asdict(v) for k, v in self.board.items()},
                "bench": {k: asdict(v) for k, v in self.bench.items()},
                "owned_copies": deepcopy(self.owned_copies),
                "owned_star_counts": deepcopy(self.owned_star_counts),
                "ownership_complete": self.ownership_complete, "bench_free": self.bench_free,
                "consumed_shop": {k: asdict(v) for k, v in self._consumed_shop.items()},
                "shop_round": self._shop_round,
                "warnings": list(self.warnings)}
