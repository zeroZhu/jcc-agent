"""Deterministic, explicitly synthetic end-to-end transaction demonstration."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .control import ActionController, RecordingBackend
from .models import Composition, GameState, SlotObservation
from .session import AgentSession
from .storage import EventLog, write_json
from .tracking import SelfTracker


def run_demo(rules, output):
    output = Path(output)
    if (output / "events.jsonl").exists():
        raise ValueError("Demo output exists; choose another --out")
    log = EventLog(output / "events.jsonl")
    initial = GameState(version=1, observed_at=1, phase="preparation", round="3-1",
                        gold=40, level=6, seconds_left=30, viewed_player="self", detail_open=False)
    initial.field_confidence = {k: 1.0 for k in ("phase", "round", "gold", "level", "seconds_left", "viewed_player", "detail_open")}
    initial.board = {f"board_{i}": SlotObservation(f"board_{i}", "empty", confidence=1) for i in range(28)}
    initial.bench = {f"bench_{i}": SlotObservation(f"bench_{i}", "empty", confidence=1) for i in range(9)}
    initial.bench["bench_0"] = SlotObservation("bench_0", "occupied", "senna", 1, confidence=1)
    initial.bench["bench_1"] = SlotObservation("bench_1", "occupied", "senna", 1, confidence=1)
    initial.shop = [SlotObservation(f"shop_{i}", "occupied", "ahri", 1, 1, 1, 1) for i in range(5)]
    initial.shop[0].hero_id = "senna"
    tracker = SelfTracker({**initial.to_dict(), "ordinary_merges": True}, rules)
    session = AgentSession(rules, [Composition("demo", "合成二星示例", {"senna": 3}, ["senna"], roll_level=6)])
    backend = RecordingBackend()
    controller = ActionController(backend, {"reference_size": [960, 432], "calibrated": True,
                                            "points": {"shop_0": [.25, .25]}}, rules,
                                  dry_run=False, live_enabled=True)
    log.append("demo_started", synthetic=True, actual_game_inputs=0)
    before, pool, plan, action = session.decide(tracker.fuse(initial))
    log.append("decision", state=before, plan=plan, action=action)
    if action is None or action.kind != "buy":
        raise RuntimeError("Demo expected a target purchase")
    result = controller.submit(action, before, now=1)
    log.append("dispatch", result=result)
    if result.status != "pending":
        raise RuntimeError(f"Demo purchase rejected: {result.reason}")
    confirmed = None
    after = None
    for version in (2, 3):
        after = deepcopy(initial)
        after.version, after.observed_at, after.gold = version, float(version), 39
        after.board, after.bench = {}, {}
        for slot in after.shop:
            slot.observed_at = float(version)
        after.shop[0] = SlotObservation("shop_0", observed_at=float(version), source="ocr_blank")
        after = tracker.fuse_shop(after)
        result = controller.observe(after, now=float(version))
        log.append("observation", state=after, result=result)
        if result:
            confirmed = result
    if confirmed is None or confirmed.status != "confirmed":
        raise RuntimeError("Demo transaction did not confirm")
    if not tracker.apply_confirmed(confirmed, before, after):
        raise RuntimeError("Demo inventory event not applied")
    session.account_confirmed_spend(before, after)
    final, pool, plan, action = session.decide(tracker.fuse(after))
    log.append("final_decision", state=final, plan=plan, action=action)
    summary = {"synthetic": True, "actual_game_inputs": 0, "recorded_clicks": len(backend.events),
               "gold_spent": session.spent, "owned_copies": final.owned_copies,
               "owned_star_counts": final.owned_star_counts, "final_action": action.kind,
               "final_reason": action.reason,
               "passed": final.owned_copies.get("senna") == 3 and action.kind == "wait" and len(backend.events) == 1}
    write_json(output / "summary.json", summary)
    if not summary["passed"]:
        raise RuntimeError("Demo did not meet its invariants")
    return summary
