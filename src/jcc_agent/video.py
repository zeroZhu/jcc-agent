"""Video sampling utilities. User annotations are ground truth, never predictions."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


# The user supplied these intervals for 示例视频. They may be compared with OCR
# predictions by an evaluator; neither perception nor control imports this table.
EXAMPLE_ANNOTATIONS = (
    (0.0, 24.0, "encounter"),
    (24.0, 31.0, "transition"),
    (31.0, 133.0, "pve"),
    (133.0, 149.0, "augment"),
    (149.0, None, "later_gameplay"),
)


def _open(path: str | Path):
    import cv2

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path.resolve()))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {path}")
    return cv2, capture


def probe_video(path: str | Path) -> VideoInfo:
    cv2, cap = _open(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count <= 0:
            raise ValueError("Video has no usable frame rate/frame count")
        return VideoInfo(str(Path(path).resolve()), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps, count, count / fps)
    finally:
        cap.release()


def annotated_phase(timestamp: float, annotations=EXAMPLE_ANNOTATIONS) -> str:
    """Return a reference annotation, not a detected game phase."""
    for start, end, label in annotations:
        if start <= timestamp and (end is None or timestamp < end):
            return label
    return "unannotated"


def iter_video_frames(path: str | Path, timestamps: Iterable[float]) -> Iterator[tuple[float, object]]:
    """Seek to requested seconds and yield the decoded frame's actual timestamp."""
    cv2, cap = _open(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count <= 0:
            raise ValueError("Video has no usable frame rate/frame count")
        for requested in timestamps:
            if requested < 0 or requested >= count / fps:
                raise ValueError(f"Timestamp outside video: {requested}")
            frame_index = min(int(round(requested * fps)), count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode frame at {requested}s")
            actual_index = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            yield actual_index / fps, frame
    finally:
        cap.release()


def extract_frames(path: str | Path, output_dir: str | Path,
                   timestamps: Iterable[float]) -> list[dict]:
    import cv2

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    info = probe_video(path)
    manifest = []
    for ordinal, (actual, frame) in enumerate(iter_video_frames(path, timestamps)):
        filename = f"frame_{ordinal:03d}_{actual:08.3f}.png"
        target = output / filename
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise ValueError(f"Cannot encode frame {actual}")
        # imwrite historically fails for Windows Unicode filenames; tofile does not.
        encoded.tofile(str(target))
        manifest.append({"timestamp": actual, "path": str(target.resolve()),
                         "reference_phase": annotated_phase(actual),
                         "reference_source": "user_annotation_not_prediction"})
    (output / "frames.json").write_text(json.dumps({"video": asdict(info), "frames": manifest},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
