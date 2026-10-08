"""PaddleOCR-only perception: all semantic decisions use recognized text.

No hero portrait classifier, visual language model, or video-timestamp oracle is
used. An unreadable region remains unknown; blank pixels never prove emptiness.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from .models import Choice, GameState, Ruleset, SlotObservation


@dataclass(frozen=True)
class OCRToken:
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in input-image pixels


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).strip()


class OCRReader:
    """Lazy PaddleOCR 3.x adapter; tests can inject a predict-compatible engine."""

    def __init__(self, engine: Any = None, lang: str = "ch", recognition_engine: Any = None, **engine_options: Any):
        self._engine = engine
        self._recognition_engine = recognition_engine
        self.lang = lang
        self.engine_options = engine_options

    @property
    def engine(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError("Install the OCR optional dependencies before reading images") from exc
            options = {"use_doc_orientation_classify": False,
                       "use_doc_unwarping": False, "use_textline_orientation": False,
                       "text_detection_model_name": "PP-OCRv5_mobile_det",
                       "text_recognition_model_name": "PP-OCRv5_mobile_rec",
                       "enable_mkldnn": True, "cpu_threads": 4}
            if self.lang != "ch":
                options["lang"] = self.lang
                options.pop("text_detection_model_name")
                options.pop("text_recognition_model_name")
            options.update(self.engine_options)
            self._engine = PaddleOCR(**options)
        return self._engine

    def read(self, image) -> list[OCRToken]:
        result = self.engine.predict(image)
        if isinstance(result, dict):
            result = [result]
        tokens: list[OCRToken] = []
        for page in result:
            if isinstance(page, OCRToken):
                tokens.append(page)
                continue
            if not isinstance(page, dict):
                page = getattr(page, "json", page)
                if callable(page):
                    page = page()
                if isinstance(page, str):
                    page = json.loads(page)
            if not isinstance(page, dict):
                raise TypeError("Expected PaddleOCR 3.x result with rec_texts/rec_scores/rec_polys")
            page = page.get("res", page)
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            polygons = page.get("rec_polys")
            if polygons is None:
                polygons = page.get("dt_polys")
            if polygons is None or len(texts) != len(scores) or len(texts) != len(polygons):
                raise ValueError("PaddleOCR returned unaligned text, score and polygon arrays")
            for text, score, polygon in zip(texts, scores, polygons):
                points = polygon.tolist() if hasattr(polygon, "tolist") else polygon
                if len(points) == 4 and all(isinstance(p, (int, float)) for p in points):
                    box = tuple(float(p) for p in points)
                else:
                    box = (min(float(p[0]) for p in points), min(float(p[1]) for p in points),
                           max(float(p[0]) for p in points), max(float(p[1]) for p in points))
                tokens.append(OCRToken(str(text), float(score), box))
        return tokens

    def recognize_regions(self, image, regions: dict) -> dict[str, list[OCRToken]]:
        """Batch known single text-line crops without a text detection pass.

        This is only suitable for deliberately calibrated single-line ROIs.
        A recognition hypothesis does not prove that a card or panel exists.
        """
        if self._recognition_engine is None:
            from paddleocr import TextRecognition
            options = {"model_name": "PP-OCRv5_mobile_rec", "enable_mkldnn": True, "cpu_threads": 4}
            for key in ("enable_mkldnn", "cpu_threads", "device"):
                if key in self.engine_options:
                    options[key] = self.engine_options[key]
            self._recognition_engine = TextRecognition(**options)
        height, width = image.shape[:2]
        patches, rectangles, keys = [], [], []
        for key, region in regions.items():
            x1, y1, x2, y2 = validate_region(region)
            rect = (int(x1 * width), int(y1 * height), max(int(x1 * width) + 1, int(x2 * width)),
                    max(int(y1 * height) + 1, int(y2 * height)))
            patches.append(image[rect[1]:rect[3], rect[0]:rect[2]])
            rectangles.append(rect)
            keys.append(key)
        if not patches:
            return {}
        predictions = list(self._recognition_engine.predict(patches, batch_size=16))
        if len(predictions) != len(keys):
            raise ValueError("TextRecognition result count does not match ROI count")
        result = {}
        for key, rect, prediction in zip(keys, rectangles, predictions):
            if not isinstance(prediction, dict):
                prediction = prediction.json
                if callable(prediction):
                    prediction = prediction()
                if isinstance(prediction, str):
                    prediction = json.loads(prediction)
            prediction = prediction.get("res", prediction)
            value, score = str(prediction.get("rec_text", "")), float(prediction.get("rec_score", 0))
            result[key] = [OCRToken(value, score, rect)] if value else []
        return result


def validate_region(region: Iterable[float]) -> tuple[float, float, float, float]:
    values = tuple(float(v) for v in region)
    if len(values) != 4 or not (0 <= values[0] < values[2] <= 1 and 0 <= values[1] < values[3] <= 1):
        raise ValueError(f"Invalid normalized ROI: {values}")
    return values


def _within(token: OCRToken, rect: tuple[float, float, float, float]) -> bool:
    x = (token.bbox[0] + token.bbox[2]) / 2
    y = (token.bbox[1] + token.bbox[3]) / 2
    return rect[0] <= x < rect[2] and rect[1] <= y < rect[3]


def read_regions(reader, image, regions: dict, *, mode: str = "full_frame", scale: float = 2.0):
    """Return tokens grouped by normalized ROI. Mosaic uses one OCR invocation.

    Mosaic packing and resizing are preprocessing only, not hero image analysis.
    Grouped token boxes are relative to the analyzed image (full frame or mosaic).
    """
    height, width = image.shape[:2]
    rectangles = {key: tuple(v * (width if i % 2 == 0 else height)
                              for i, v in enumerate(validate_region(value)))
                  for key, value in regions.items()}
    if mode == "recognition":
        return reader.recognize_regions(image, regions)
    if mode == "full_frame":
        tokens = reader.read(image)
        return {key: [token for token in tokens if _within(token, rect)]
                for key, rect in rectangles.items()}
    if mode != "mosaic":
        raise ValueError(f"Unknown OCR mode: {mode}")
    import cv2
    import numpy as np

    if not 0 < scale <= 4:
        raise ValueError("ROI scale must be between 0 and 4")
    patches = []
    for key, (x1, y1, x2, y2) in rectangles.items():
        patch = image[int(y1):max(int(y1) + 1, int(y2)), int(x1):max(int(x1) + 1, int(x2))]
        patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        patches.append((key, patch))
    if not patches:
        return {}
    # Two columns keep the detector input aspect ratio moderate. Empty margins
    # separate crops so nearby names cannot be merged across region boundaries.
    cell_width = max(p.shape[1] for _, p in patches) + 32
    rows = []
    for offset in range(0, len(patches), 2):
        rows.append(max(p.shape[0] for _, p in patches[offset:offset + 2]) + 32)
    mosaic = np.zeros((sum(rows), cell_width * 2, 3), dtype=image.dtype)
    mapped = {}
    y = 0
    for index, (key, patch) in enumerate(patches):
        if index and index % 2 == 0:
            y += rows[index // 2 - 1]
        x = (index % 2) * cell_width + 16
        py = y + 16
        ph, pw = patch.shape[:2]
        mosaic[py:py + ph, x:x + pw] = patch
        mapped[key] = (x, py, x + pw, py + ph)
    tokens = reader.read(mosaic)
    return {key: [token for token in tokens if _within(token, rect)] for key, rect in mapped.items()}


class FrameAnalyzer:
    def __init__(self, reader, rules: Ruleset, layout: dict):
        self.reader, self.rules, self.layout = reader, rules, layout
        self.threshold = float(layout.get("min_confidence", 0.80))
        self.last_regions: dict[str, list[OCRToken]] = {}
        self._names: dict[str, set[str]] = {}
        for hero in rules.heroes.values():
            for name in [hero.name, *hero.aliases]:
                self._names.setdefault(normalize_text(name), set()).add(hero.id)
        for region in layout.get("regions", {}).values():
            validate_region(region)

    def _tokens(self, key: str) -> list[OCRToken]:
        keys = self.layout.get("field_regions", {}).get(key, [key])
        return [t for name in keys for t in self.last_regions.get(name, []) if t.confidence >= self.threshold]

    def _number(self, key: str, pattern: str, low: int, high: int):
        found = []
        for token in self._tokens(key):
            match = re.fullmatch(pattern, normalize_text(token.text), re.I)
            if match and low <= int(match.group(1)) <= high:
                found.append((int(match.group(1)), token.confidence))
        if len({value for value, _ in found}) == 1:
            return found[0][0], max(score for _, score in found)
        return None, 0.0

    def _hero(self, key: str, timestamp: float) -> SlotObservation:
        tokens = self._tokens(key)
        matched: dict[str, float] = {}
        for token in tokens:
            # Exact name/alias only; speculative fuzzy correction must not buy a
            # different champion. Currency digits and star icons are not names.
            name = normalize_text(token.text).strip("★☆·:")
            for hero_id in self._names.get(name, ()):
                matched[hero_id] = max(matched.get(hero_id, 0), token.confidence)
        if len(matched) == 1:
            hero_id, confidence = next(iter(matched.items()))
            star = None
            stars = set()
            for token in tokens:
                text = normalize_text(token.text)
                match = re.fullmatch(r"(?:星级[:：]?)?([123一二三])星", text)
                if match:
                    stars.add({"一": 1, "二": 2, "三": 3}.get(match[1], int(match[1]) if match[1].isdigit() else 0))
            if len(stars) == 1:
                star = stars.pop()
            return SlotObservation(key, "occupied", hero_id, star, self.rules.heroes[hero_id].cost,
                                   confidence, timestamp)
        empty_markers = self.layout.get("empty_slot_text", {}).get(key, [])
        if not matched and any(normalize_text(t.text) in empty_markers for t in tokens):
            return SlotObservation(key, "empty", confidence=min(t.confidence for t in tokens), observed_at=timestamp)
        source = "ocr_blank" if not self.last_regions.get(key, []) else "ocr_unresolved"
        return SlotObservation(key, observed_at=timestamp, source=source)

    def analyze(self, image, timestamp: float = 0, version: int = 0) -> GameState:
        mode = self.layout.get("ocr_mode", "full_frame")
        regions = self.layout.get("regions", {})
        if mode == "recognition":
            regions = self.layout.get("single_line_regions", {})
            if not regions:
                raise ValueError("recognition mode requires calibrated single_line_regions")
        self.last_regions = read_regions(self.reader, image, regions,
                                         mode=mode,
                                         scale=float(self.layout.get("roi_scale", 2)))
        state = GameState(version=version, observed_at=timestamp)
        for key, pattern, low, high in (("gold", r"(?:金币[:：]?)?(\d{1,3})", 0, 999),
                                        ("level", r"(?:等级[:：]?|Lv\.?|LV)?(\d{1,2})(?:级)?", 1, 11),
                                        ("hp", r"(?:生命[:：]?|血量[:：]?)?(\d{1,3})", 0, 150),
                                        ("timer", r"(\d{1,2})(?:秒|s)?", 0, 99)):
            value, confidence = self._number(key, pattern, low, high)
            attribute = "seconds_left" if key == "timer" else key
            setattr(state, attribute, value)
            state.field_confidence[attribute] = confidence
        rounds = [(normalize_text(t.text).replace("—", "-").replace("–", "-").strip("-"), t.confidence)
                  for t in self._tokens("round")]
        rounds = [(t, c) for t, c in rounds if re.fullmatch(r"[1-9]-[1-9]", t)]
        if len({t for t, _ in rounds}) == 1:
            state.round, state.field_confidence["round"] = max(rounds, key=lambda value: value[1])
        for token in self._tokens("xp"):
            match = re.fullmatch(r"(\d{1,3})/(\d{1,3})", normalize_text(token.text))
            if match and 0 <= int(match[1]) < int(match[2]):
                state.xp, state.xp_to_level = int(match[1]), int(match[2])
                state.field_confidence["xp"] = token.confidence
                state.field_confidence["xp_to_level"] = token.confidence
        state.shop = [self._hero(f"shop_{index}", timestamp) for index in range(5)]
        for index, slot in enumerate(state.shop):
            if slot.status == "occupied":
                # Encounter/modifier prices can differ from the hero's pool tier.
                # Unreadable prices remain None; the executor must not guess.
                price_region = f"shop_price_{index}" if f"shop_price_{index}" in self.last_regions else f"shop_{index}"
                price, confidence = self._number(price_region, r"(?:售价[:：]?)?(\d{1,2})", 0, 99)
                slot.cost = price
                state.field_confidence[f"shop_{index}_price"] = confidence
        for phase, markers in self.layout.get("phase_markers", {}).items():
            phase_region = self.layout.get("phase_regions", {}).get(phase, "phase")
            hits = [t for t in self._tokens(phase_region)
                    if any(normalize_text(marker) in normalize_text(t.text) for marker in markers)]
            if hits:
                state.phase = phase
                state.field_confidence["phase"] = max(t.confidence for t in hits)
                break
        if state.round in self.layout.get("pve_rounds", []) and state.phase in ("unknown", "preparation", "combat"):
            state.phase = "pve"
            state.field_confidence["phase"] = min(state.field_confidence.get("phase", 1), state.field_confidence["round"])
        if (state.phase == "unknown" and state.round is not None and state.seconds_left is not None
                and sum(slot.status == "occupied" for slot in state.shop) >= 3):
            # The shop is usable in both preparation and combat. Do not invent
            # the substage from its visibility, but economic actions may use it.
            state.phase = "gameplay"
            state.field_confidence["phase"] = min(state.field_confidence["round"],
                                                   state.field_confidence["seconds_left"],
                                                   min(slot.confidence for slot in state.shop if slot.hero_id))
            state.warnings.append("phase_substage_unknown")
        for index in range(int(self.layout.get("choice_count", 3))):
            prefix = "encounter_choice" if state.phase == "encounter" else "choice"
            title_tokens = self._tokens(f"{prefix}_title_{index}")
            tokens = self._tokens(f"{prefix}_{index}") or title_tokens
            if tokens and state.phase in ("encounter", "augment"):
                title = "".join(t.text for t in title_tokens)
                state.choices.append(Choice(index, "\n".join(t.text for t in tokens),
                                            min(t.confidence for t in tokens), title=title))
        names = {normalize_text(name) for name in self.layout.get("self_names", [])}
        if self.layout.get("self_player_name"):
            names.add(normalize_text(self.layout["self_player_name"]))
        identity_region = "viewed_player" if "viewed_player" in self.layout.get("regions", {}) else "self_name"
        own = [t for t in self._tokens(identity_region) if normalize_text(t.text) in names]
        if own:
            state.viewed_player = "self"
            state.field_confidence["viewed_player"] = max(t.confidence for t in own)
        else:
            # Opponent identity is configurable; unnamed screens remain unknown.
            player_names = self.layout.get("player_name_map", self.layout.get("player_names", {}))
            matches = [(pid, t.confidence) for pid, name in player_names.items()
                       for t in self._tokens(identity_region) if normalize_text(t.text) == normalize_text(name)]
            if len({pid for pid, _ in matches}) == 1:
                state.viewed_player, state.field_confidence["viewed_player"] = max(matches, key=lambda item: item[1])
        state.detail = self._hero("detail", timestamp)
        open_markers = self.layout.get("detail_open_text", [])
        closed_markers = self.layout.get("detail_closed_text", [])
        open_hits = [t for t in self._tokens("detail") if normalize_text(t.text) in open_markers]
        closed_hits = [t for t in self._tokens("detail_closed") if normalize_text(t.text) in closed_markers]
        refresh_hits = [t for t in self._tokens("refresh_label") if normalize_text(t.text) == "刷新"]
        if state.detail.status == "occupied" or open_hits:
            state.detail_open = True
            state.field_confidence["detail_open"] = max([state.detail.confidence, *[t.confidence for t in open_hits]])
        elif closed_hits:
            state.detail_open = False
            state.field_confidence["detail_open"] = max(t.confidence for t in closed_hits)
            state.detail = None
        else:
            state.detail_open = None
            state.warnings.append("detail_unknown")
        if refresh_hits and state.detail_open is False:
            state.shop_visible = True
            state.field_confidence["shop_visible"] = max(t.confidence for t in refresh_hits)
        if any(slot.status == "unknown" for slot in state.shop):
            state.warnings.append("shop_incomplete")
        if state.viewed_player == "unknown":
            state.warnings.append("viewed_player_unknown")
        for key, confidence in state.field_confidence.items():
            if confidence > 0:
                state.field_sources[key] = "ocr"
                state.field_observed_at[key] = timestamp
        if state.phase == "pve":
            state.field_sources["phase"] = "ocr_round_and_configured_pve_rule"
        elif state.phase == "gameplay":
            state.field_sources["phase"] = "ocr_round_timer_and_shop"
        if state.detail_open is False:
            state.field_sources["detail_open"] = "ocr_uncovered_background_text"
        return state
