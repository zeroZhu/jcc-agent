"""Command-line entry points. Importing commands does not load Paddle models."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .storage import EventLog, read_json, state_from_dict, write_json


def _config(args):
    from .rules import load_compositions, load_rules
    return load_rules(args.rules), load_compositions(args.rules)


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def doctor(args):
    versions = {}
    for package in ("jcc-agent", "numpy", "opencv-contrib-python", "Pillow", "paddleocr", "paddlepaddle", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {"python": sys.version, "packages": versions, "ocr_installed":
              bool(versions["paddleocr"] and versions["paddlepaddle"])}
    if Path(args.rules).exists():
        rules, comps = _config(args)
        result["rules"] = {"id": rules.id, "heroes": len(rules.heroes),
                           "compositions": len(comps), "provenance": rules.provenance}
    if args.video:
        from .video import probe_video
        result["video"] = asdict(probe_video(args.video))
    _print(result)


def extract(args):
    from .video import extract_frames
    rows = extract_frames(args.video, args.out, [float(v) for v in args.timestamps.split(",")])
    _print({"frames": len(rows), "manifest": str(Path(args.out) / "frames.json")})


def analyze(args):
    import cv2

    from .perception import FrameAnalyzer, OCRReader
    from .reporting import render_report
    from .session import AgentSession
    from .video import annotated_phase, iter_video_frames, probe_video
    rules, comps = _config(args)
    layout = read_json(args.layout)
    info = probe_video(args.video)
    if args.timestamps:
        timestamps = [float(v) for v in args.timestamps.split(",")]
    else:
        end = min(args.end if args.end is not None else info.duration, info.duration)
        if args.every <= 0 or args.start < 0 or args.start >= end:
            raise ValueError("Invalid sampling range")
        timestamps = []
        t = args.start
        while t < end:
            timestamps.append(t)
            t += args.every
    output = Path(args.out)
    if (output / "observations.jsonl").exists():
        raise ValueError("Output already contains a run; use a new --out directory")
    (output / "frames").mkdir(parents=True, exist_ok=True)
    analyzer = FrameAnalyzer(OCRReader(**layout.get("ocr_options", {})), rules, layout)
    session = AgentSession(rules, comps, choice_preferences=layout.get("choice_preferences"))
    log = EventLog(output / "observations.jsonl")
    rows = []
    for version, (timestamp, frame) in enumerate(iter_video_frames(args.video, timestamps), start=1):
        start = time.perf_counter()
        state = analyzer.analyze(frame, timestamp=timestamp, version=version)
        elapsed = (time.perf_counter() - start) * 1000
        state, pool, plan, action = session.decide(state)
        relative = f"frames/{version:04d}_{timestamp:.2f}.jpg"
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise RuntimeError("Could not save frame")
        encoded.tofile(str(output / relative))
        row = {"timestamp": timestamp, "reference_phase": annotated_phase(timestamp),
               "latency_ms": elapsed, "state": state.to_dict(),
               "plan": asdict(plan) if plan else None, "action": asdict(action) if action else None,
               "pool": {key: asdict(value) for key, value in pool.items()}, "image": relative,
               "ocr_regions": {k: [asdict(t) for t in v] for k, v in analyzer.last_regions.items()}}
        rows.append(row)
        log.append("observation", **row)
        print(f"{version}/{len(timestamps)} {timestamp:.2f}s phase={state.phase} "
              f"shop={sum(s.status == 'occupied' for s in state.shop)}/5 "
              f"ocr={elapsed:.0f}ms", flush=True)
    comparable = [r for r in rows if r["reference_phase"] != "later_gameplay"]
    matches = sum(r["state"]["phase"] == r["reference_phase"] for r in comparable)
    latencies = [r["latency_ms"] for r in rows]
    with Path(args.video).open("rb") as video_stream:
        video_hash = hashlib.file_digest(video_stream, "sha256").hexdigest()
    summary = {"video": asdict(info), "source_sha256": video_hash,
               "frames": len(rows), "reference_source": "user_timeline_not_detector_input",
               "phase_matches": matches, "phase_evaluable_frames": len(comparable),
               "phase_accuracy": matches / len(comparable) if comparable else None,
               "latency_ms_median": statistics.median(latencies) if latencies else None,
               "latency_ms_max": max(latencies) if latencies else None,
               "cold_start_included": True, "rules_provenance": rules.provenance,
               "actual_game_inputs": 0,
               "limitations": ["后续游戏阶段未给出细分真值，不计入阶段准确率",
                               "观测来自录像，不能证明拟执行动作在游戏中成功",
                               "首次耗时包含模型加载/下载，连续D牌须单独端到端验证"]}
    write_json(output / "summary.json", summary)
    report = render_report(output, summary, rows)
    _print({"summary": summary, "report": str(report.resolve())})


def replay(args):
    from .session import AgentSession
    rules, comps = _config(args)
    session = AgentSession(rules, comps)
    log = EventLog(Path(args.out) / "decisions.jsonl")
    count = 0
    for line in Path(args.observations).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        state = state_from_dict(row.get("state", row))
        state, pool, plan, action = session.decide(state)
        log.append("decision", state_version=state.version, plan=plan, action=action, pool=pool)
        count += 1
    _print({"observations": count, "log": str(log.path.resolve()), "actual_game_inputs": 0})


def live(args):
    from .runtime import run_live
    rules, comps = _config(args)
    layout = read_json(args.layout)
    if args.duration <= 0 or args.interval < 1.0:
        raise ValueError("Duration must be positive and action interval at least 1 second")
    _print(run_live(hwnd=args.hwnd, layout=layout, rules=rules, compositions=comps,
                    output=args.out, execute=args.execute, duration=args.duration,
                    interval=args.interval, stop_file=args.stop_file,
                    choice_preferences=layout.get("choice_preferences"),
                    initial_state=read_json(args.self_state) if args.self_state else None,
                    scout=args.scout))


def calibrate(args):
    from .calibration import build_calibration_page
    _print({"calibration_page": str(build_calibration_page(args.image, read_json(args.layout), args.out).resolve())})


def windows(args):
    from .runtime import list_windows
    _print(list_windows())


def demo(args):
    from .demo import run_demo
    rules, _ = _config(args)
    _print(run_demo(rules, args.out))


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="金铲铲 OCR Agent：录像验证、快照估计和确认式操作")
    parser.add_argument("--rules", default="configs/inkborn_fables.json")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor", help="检查依赖、规则来源和录像")
    p.add_argument("--video")
    p.set_defaults(func=doctor)
    p = sub.add_parser("windows", help="只读列出Windows窗口及句柄，供显式选择")
    p.set_defaults(func=windows)
    p = sub.add_parser("demo", help="合成状态演示交易确认与达标停止，绝不操作游戏")
    p.add_argument("--out", default="artifacts/demo")
    p.set_defaults(func=demo)
    p = sub.add_parser("extract", help="按时间提取真实录像帧")
    p.add_argument("--video", required=True)
    p.add_argument("--timestamps", default="0,10,24,31,60,132,133,140,149,260,270")
    p.add_argument("--out", default="artifacts/frames")
    p.set_defaults(func=extract)
    p = sub.add_parser("analyze", help="运行PaddleOCR、决策与独立录像验证报告")
    p.add_argument("--video", required=True)
    p.add_argument("--layout", default="configs/layout.example.json")
    p.add_argument("--out", default="artifacts/analysis")
    p.add_argument("--timestamps")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--end", type=float)
    p.add_argument("--every", type=float, default=15)
    p.set_defaults(func=analyze)
    p = sub.add_parser("replay", help="从结构化观测重新计算策略，不操作游戏")
    p.add_argument("--observations", required=True)
    p.add_argument("--out", default="artifacts/replay")
    p.set_defaults(func=replay)
    p = sub.add_parser("calibrate", help="生成本地坐标/文字区域校准页面")
    p.add_argument("--image", required=True)
    p.add_argument("--layout", default="configs/layout.example.json")
    p.add_argument("--out", default="artifacts/calibration.html")
    p.set_defaults(func=calibrate)
    p = sub.add_parser("live", help="绑定Windows窗口观察；--execute显式启用操作")
    p.add_argument("--hwnd", type=lambda v: int(v, 0), required=True)
    p.add_argument("--layout", default="configs/local.layout.json")
    p.add_argument("--out", default="artifacts/live")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--interval", type=float, default=1.5)
    p.add_argument("--stop-file", default="STOP")
    p.add_argument("--self-state", help="经核对的己方初始GameState JSON；缺省保留未知")
    p.add_argument("--scout", action="store_true", help="启用布局中配置的回合间侦察")
    p.set_defaults(func=live)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
