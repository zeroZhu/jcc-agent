"""Exercise the real orchestrator without Windows capture, Paddle, or input."""
import json
from copy import deepcopy
from pathlib import Path

from jcc_agent import runtime
from jcc_agent.control import RecordingBackend
from jcc_agent.models import Composition, GameState, SlotObservation
from jcc_agent.rules import load_rules


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def setup_runtime(monkeypatch, inference_seconds):
    clock = Clock()
    backend = RecordingBackend()
    monkeypatch.setattr(runtime.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(runtime.time, "sleep", clock.sleep)
    monkeypatch.setattr(runtime, "Win32Backend", lambda hwnd: backend)
    monkeypatch.setattr(runtime, "OCRReader", lambda: None)

    class Capture:
        def __init__(self, hwnd):
            pass

        def frame(self):
            return None, self.viewport()

        def viewport(self):
            return [0, 0, 960, 432]

        def stop_pressed(self):
            return False

        def input_tick(self):
            return 0

    class Analyzer:
        def __init__(self, *args):
            pass

        def analyze(self, frame, timestamp, version):
            clock.now += inference_seconds
            state = GameState(version=version, observed_at=timestamp, round="3-1",
                              phase="preparation", viewed_player="self", detail_open=False,
                              gold=40 if version == 1 else 39, level=6, seconds_left=30)
            state.field_confidence = {k: 1 for k in ("round", "phase", "viewed_player", "detail_open", "gold", "level", "seconds_left")}
            state.shop = [SlotObservation(f"shop_{i}", "occupied", "ahri", 1, 1, 1, timestamp) for i in range(5)]
            if version == 1:
                state.shop[0].hero_id = "senna"
            else:
                state.shop[0] = SlotObservation("shop_0", observed_at=timestamp, source="ocr_blank")
            return state

    monkeypatch.setattr(runtime, "WindowCapture", Capture)
    monkeypatch.setattr(runtime, "FrameAnalyzer", Analyzer)
    return backend


def initial_inventory():
    state = GameState(version=100, observed_at=999)
    state.board = {f"board_{i}": SlotObservation(f"board_{i}", "empty", confidence=1, observed_at=999) for i in range(28)}
    state.bench = {f"bench_{i}": SlotObservation(f"bench_{i}", "empty", confidence=1, observed_at=999) for i in range(9)}
    for index in (0, 1):
        state.bench[f"bench_{index}"] = SlotObservation(f"bench_{index}", "occupied", "senna", 1,
                                                        confidence=1, observed_at=999)
    return {**state.to_dict(), "ordinary_merges": True}


def run(monkeypatch, tmp_path, age):
    backend = setup_runtime(monkeypatch, age)
    rules = load_rules(Path(__file__).resolve().parents[1] / "configs/inkborn_fables.json")
    initial = initial_inventory()
    untouched = deepcopy(initial)
    result = runtime.run_live(
        hwnd=42, layout={"reference_size": [960, 432], "calibrated": True,
                         "points": {"shop_0": [.25, .25]}, "max_frame_age_seconds": 2},
        rules=rules, compositions=[Composition("test", "目标二星", {"senna": 3}, ["senna"], roll_level=6)],
        output=tmp_path, execute=True, duration=4.6, interval=1.5,
        stop_file=tmp_path / "STOP", initial_state=initial,
    )
    assert initial == untouched
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    return backend, result, events


def test_saved_inventory_clock_rebased_and_two_blank_frames_confirm_once(monkeypatch, tmp_path):
    backend, _, events = run(monkeypatch, tmp_path, .01)
    assert len(backend.events) == 1
    confirmations = [e["result"] for e in events if e["event"] == "confirmation"]
    assert confirmations[0]["status"] == "confirmed"
    final = [e for e in events if e["event"] == "observation"][-1]
    assert final["state"]["owned_copies"]["senna"] == 3
    assert final["action"]["kind"] == "wait"
    assert final["state"]["ownership_complete"]


def test_slow_ocr_never_clicks_using_an_expired_frame(monkeypatch, tmp_path):
    backend, _, events = run(monkeypatch, tmp_path, 3)
    assert backend.events == []
    assert any(e["event"] == "stale_frame" for e in events)


def test_external_input_during_ocr_stops_without_a_click(monkeypatch, tmp_path):
    import pytest

    backend = setup_runtime(monkeypatch, .01)
    # The first reading establishes ownership; the next is the frame-start
    # check. A changed tick after inference means the frame can no longer act.
    ticks = iter([100, 100, 101])
    monkeypatch.setattr(runtime.WindowCapture, "input_tick", lambda self: next(ticks))
    rules = load_rules(Path(__file__).resolve().parents[1] / "configs/inkborn_fables.json")
    with pytest.raises(RuntimeError, match="External input during OCR"):
        runtime.run_live(hwnd=42, layout={"calibrated": True}, rules=rules, compositions=[],
                         output=tmp_path, execute=True, duration=10,
                         stop_file=tmp_path / "STOP", initial_state=initial_inventory())
    assert backend.events == []


def test_pending_refresh_does_not_reuse_bought_slot_cache_as_new_shop_evidence(monkeypatch, tmp_path):
    backend = setup_runtime(monkeypatch, .01)

    class Analyzer:
        def __init__(self, *args):
            pass

        def analyze(self, frame, timestamp, version):
            state = GameState(version=version, observed_at=timestamp, round="3-1",
                              phase="combat", viewed_player="self", detail_open=False,
                              gold=40 if version == 1 else (39 if version <= 3 else 37),
                              level=6, seconds_left=30)
            state.field_confidence = {key: 1 for key in ("round", "phase", "viewed_player", "detail_open", "gold", "level", "seconds_left")}
            state.shop = [SlotObservation(f"shop_{i}", "occupied", "ahri" if version <= 3 else "garen",
                                          1, 1, 1, timestamp) for i in range(5)]
            if version == 1 or version >= 5:
                state.shop[0].hero_id = "senna"
            else:
                # Versions 2/3 confirm a purchase. Version 4 is a different
                # shop after paying 2 to refresh; slot 0's NEW card was missed.
                state.shop[0] = SlotObservation("shop_0", observed_at=timestamp, source="ocr_blank")
            return state

    monkeypatch.setattr(runtime, "FrameAnalyzer", Analyzer)
    rules = load_rules(Path(__file__).resolve().parents[1] / "configs/inkborn_fables.json")
    runtime.run_live(
        hwnd=42, layout={"reference_size": [960, 432], "calibrated": True,
                         "points": {"shop_0": [.25, .25], "refresh": [.8, .8]}},
        rules=rules, compositions=[Composition("test", "继续追三", {"senna": 9}, ["senna"], roll_level=6)],
        output=tmp_path, execute=True, duration=6.1, interval=1.5,
        stop_file=tmp_path / "STOP", initial_state=initial_inventory(),
    )
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    fourth_frame = next(e for e in events if e["event"] == "observation" and e["state"]["version"] == 4)
    assert fourth_frame["state"]["shop"][0]["status"] == "unknown"
    refresh_confirmation = [e["result"] for e in events if e["event"] == "confirmation"
                            and e["result"]["action"]["kind"] == "refresh"]
    assert len(refresh_confirmation) == 1
    assert refresh_confirmation[0]["status"] == "confirmed"
    assert refresh_confirmation[0]["observed_at"] == 6.0  # version 5 has all five fresh cards
    fourth_frame_inputs = [e for e in events if e["event"] == "dispatch"
                           and e["result"]["action"]["state_version"] == 4]
    assert fourth_frame_inputs == []
    assert len(backend.events) == 3  # buy, refresh, then buy only after complete new shop
