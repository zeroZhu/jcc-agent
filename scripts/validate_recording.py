"""Reproducible real-Paddle benchmark against manually inspected video frames.

The annotations are consumed by evaluation only, never by FrameAnalyzer.
Run from the repository root after installing the OCR extras.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jcc_agent.perception import FrameAnalyzer, OCRReader  # noqa: E402
from jcc_agent.rules import load_rules  # noqa: E402
from jcc_agent.video import iter_video_frames, probe_video  # noqa: E402


def percentile95(values):
    return sorted(values)[max(0, math.ceil(len(values) * .95) - 1)] if values else None


def quality(rows, truth):
    # Use one observation per unique source frame, not repeated copies as extra
    # accuracy samples. Repetitions only add latency observations.
    latest = {round(row["timestamp"]): row for row in rows}
    names_total = names_correct = groups = groups_correct = 0
    prices_total = prices_read = prices_correct = details_total = details_correct = 0
    phase_total = phase_correct = shop_false_positives = detail_false_positives = 0
    close_total = close_read = close_correct = 0
    for expected in truth:
        row = latest.get(expected["timestamp"])
        if row is None:
            continue
        state = row["state"]
        if expected.get("shop_visible"):
            groups += 1
            actual = [slot["hero_id"] for slot in state["shop"]]
            groups_correct += actual == expected["shop"]
            for index, hero in enumerate(expected["shop"]):
                if hero is None:
                    continue
                names_total += 1
                names_correct += actual[index] == hero
                price = expected["prices"][index]
                prices_total += 1
                read = state["shop"][index]["cost"]
                prices_read += read is not None
                prices_correct += read == price
        else:
            shop_false_positives += sum(slot["hero_id"] is not None for slot in state["shop"])
        detail = state["detail"]["hero_id"] if state["detail"] else None
        if expected.get("detail"):
            details_total += 1
            details_correct += detail == expected["detail"]
        else:
            detail_false_positives += detail is not None
        close_total += 1
        close_read += state["detail_open"] is not None
        close_correct += state["detail_open"] == bool(expected.get("detail"))
        if expected.get("phase"):
            phase_total += 1
            phase_correct += state["phase"] == expected["phase"]
    return {"unique_frames": len(latest), "shop_names_correct": names_correct, "shop_names_total": names_total,
            "shop_groups_correct": groups_correct, "shop_groups_total": groups,
            "shop_false_positive_names_when_hidden": shop_false_positives,
            "prices_correct": prices_correct, "prices_read": prices_read, "prices_total": prices_total,
            "price_coverage": prices_read / prices_total if prices_total else None,
            "price_accuracy_when_read": prices_correct / prices_read if prices_read else None,
            "detail_names_correct": details_correct, "detail_names_total": details_total,
            "detail_false_positive_names": detail_false_positives,
            "detail_visibility_read": close_read, "detail_visibility_correct": close_correct,
            "detail_visibility_total": close_total,
            "phase_exact_matches": phase_correct, "phase_labeled_frames": phase_total,
            "phase_metric_note": "gameplay confirms only a usable match scene, not preparation/combat; unknown is not a correct exact match"}


def main():
    # Windows console code pages may otherwise corrupt Chinese paths and labels;
    # JSON artifacts already use explicit UTF-8 below.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="示例视频.mp4")
    parser.add_argument("--rules", default="configs/inkborn_fables.json")
    parser.add_argument("--layout", default="configs/layout.example.json")
    parser.add_argument("--truth", default="validation/example_video_ground_truth.json")
    parser.add_argument("--out", default="artifacts/perception/final_validation")
    parser.add_argument("--modes", default="recognition")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--no-mkldnn", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "observations.jsonl").exists():
        raise ValueError("Output already contains observations; select a fresh --out directory")
    rules = load_rules(args.rules)
    layout = json.loads(Path(args.layout).read_text(encoding="utf-8"))
    truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))["frames"]
    frames = list(iter_video_frames(args.video, [row["timestamp"] for row in truth]))
    rows, metrics = [], {}
    # Save exact input configuration: benchmark results must remain interpretable
    # after someone recalibrates the working layout.
    (out / "layout_used.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "truth_used.json").write_text(Path(args.truth).read_text(encoding="utf-8"), encoding="utf-8")
    for mode in args.modes.split(","):
        reader = OCRReader(enable_mkldnn=not args.no_mkldnn, cpu_threads=4)
        analyzer = FrameAnalyzer(reader, rules, {**layout, "ocr_mode": mode})
        mode_rows = []
        for repetition in range(args.repeats):
            for index, (timestamp, frame) in enumerate(frames):
                start = time.perf_counter()
                state = analyzer.analyze(frame, timestamp, len(rows) + 1)
                elapsed = time.perf_counter() - start
                row = {"mode": mode, "timestamp": timestamp, "repetition": repetition,
                       "cold_call": repetition == 0 and index == 0, "elapsed_seconds": elapsed,
                       "state": state.to_dict(),
                       "regions": {key: [asdict(token) for token in value]
                                   for key, value in analyzer.last_regions.items()}}
                rows.append(row)
                mode_rows.append(row)
                with (out / "observations.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"{mode} repeat={repetition} t={timestamp:.2f}s OCR={elapsed:.3f}s "
                      f"phase={state.phase} shop={[(slot.hero_id, slot.cost) for slot in state.shop]} "
                      f"detail={state.detail.hero_id if state.detail else None}", flush=True)
        warm = [row["elapsed_seconds"] for row in mode_rows if not row["cold_call"]]
        metrics[mode] = {**quality(mode_rows, truth), "first_call_seconds": mode_rows[0]["elapsed_seconds"],
                         "warm_samples": len(warm), "warm_p50_seconds": statistics.median(warm) if warm else None,
                         "warm_p95_seconds": percentile95(warm), "warm_max_seconds": max(warm) if warm else None}
    with Path(args.video).open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    versions = {}
    for package in ("paddleocr", "paddlepaddle", "opencv-python", "opencv-contrib-python"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    summary = {"video": asdict(probe_video(args.video)), "sha256": source_hash,
               "models": ["PP-OCRv5_mobile_rec"] +
                         (["PP-OCRv5_mobile_det"] if args.modes != "recognition" else []),
               "mkldnn": not args.no_mkldnn, "cpu_threads": 4,
               "packages": versions,
               "metrics": metrics,
               "scope": "decoded frame to structured GameState; excludes capture, click, animation and action confirmation",
               "actual_game_inputs": 0, "annotations_used_by_detector": False}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
