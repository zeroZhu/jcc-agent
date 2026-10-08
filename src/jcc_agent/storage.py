"""Portable JSON observations and append-only logs; no pickle or executable input."""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .models import Choice, GameState, PlayerSnapshot, SlotObservation


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def json_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_value), encoding="utf-8")


class EventLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, **data: Any) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, **data}, ensure_ascii=False, default=json_value) + "\n")


def state_from_dict(data: dict) -> GameState:
    """Rebuild nested contracts, rejecting unknown top-level fields."""
    data = dict(data)
    data["shop"] = [SlotObservation(**v) for v in data.get("shop", [])]
    for name in ("board", "bench"):
        data[name] = {k: SlotObservation(**v) for k, v in data.get(name, {}).items()}
    data["opponents"] = {
        key: PlayerSnapshot(
            **{**value, "slots": {k: SlotObservation(**v) for k, v in value.get("slots", {}).items()}}
        )
        for key, value in data.get("opponents", {}).items()
    }
    data["choices"] = [Choice(**v) for v in data.get("choices", [])]
    if data.get("detail") is not None:
        data["detail"] = SlotObservation(**data["detail"])
    return GameState(**data)
