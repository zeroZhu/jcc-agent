"""Versioned, deliberately conservative rule loading and OCR name matching."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

from .models import Composition, Hero, Ruleset


def _read(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("规则配置必须是 JSON 对象")
    return data


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} 必须是 >= {minimum} 的整数")
    return value


def _strings(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(f"{label} 必须是非空字符串的列表")
    return value


def normalize_name(text: str) -> str:
    """Ignore layout whitespace and typographic punctuation, never OCR edit distance."""
    return re.sub(r"[\s·•.'’\-]", "", unicodedata.normalize("NFKC", text)).casefold()


def load_rules(path: str | Path) -> Ruleset:
    data = _read(path)
    for key in ("id", "name", "provenance"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"规则缺少 {key}")
    heroes: dict[str, Hero] = {}
    name_owner: dict[str, str] = {}
    records = data.get("heroes")
    if not isinstance(records, list) or not records:
        raise ValueError("heroes 必须是非空列表")
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("英雄缺少 id")
        hero_id = row["id"]
        if hero_id in heroes:
            raise ValueError(f"重复英雄 id: {hero_id}")
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"英雄 {hero_id} 缺少名称")
        cost = _integer(row.get("cost"), f"{hero_id}.cost", 1)
        if cost > 5:
            raise ValueError("当前规则只支持 1 至 5 费普通商店英雄")
        hero = Hero(hero_id, name, cost,
                    _strings(row.get("traits", []), "traits"),
                    _strings(row.get("aliases", []), "aliases"))
        for candidate in (hero.id, hero.name, *hero.aliases):
            key = normalize_name(candidate)
            if not key or (key in name_owner and name_owner[key] != hero_id):
                raise ValueError(f"英雄名或别名歧义: {candidate}")
            name_owner[key] = hero_id
        heroes[hero_id] = hero
    pool_data = data.get("pool_by_cost")
    if not isinstance(pool_data, dict):
        raise ValueError("pool_by_cost 必须是对象")
    pools = {int(k): _integer(v, f"pool_by_cost.{k}", 1)
             for k, v in pool_data.items()}
    if len(pools) != len(pool_data):
        raise ValueError("pool_by_cost 存在等价的重复费用键")
    if set(pools) != {1, 2, 3, 4, 5}:
        raise ValueError("pool_by_cost 必须明确包含 1 至 5 费")
    odds: dict[int, list[float]] = {}
    odds_data = data.get("shop_odds")
    if not isinstance(odds_data, dict):
        raise ValueError("shop_odds 必须是对象")
    for key, values in odds_data.items():
        level = int(key)
        if level in odds:
            raise ValueError(f"重复商店概率等级: {key}")
        if not 1 <= level <= 11 or not isinstance(values, list) or len(values) != 5:
            raise ValueError(f"无效商店概率等级/长度: {key}")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError(f"商店概率须在 [0, 1] 内: {key}")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError(f"商店概率之和必须等于 1: {key}")
        odds[level] = list(values)
    if not odds:
        raise ValueError("shop_odds 不能为空")
    return Ruleset(
        id=data["id"], name=data["name"], heroes=heroes,
        pool_by_cost=pools, shop_odds=odds,
        refresh_cost=_integer(data.get("refresh_cost", 2), "refresh_cost", 1),
        xp_cost=_integer(data.get("xp_cost", 4), "xp_cost", 1),
        xp_per_purchase=_integer(data.get("xp_per_purchase", 4), "xp_per_purchase", 1),
        provenance=data["provenance"], notes=_strings(data.get("notes", []), "notes"),
    )


def load_compositions(path: str | Path) -> list[Composition]:
    data = _read(path)
    rules = load_rules(path)
    rows = data.get("compositions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("compositions 必须是非空列表")
    compositions: list[Composition] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("阵容必须是对象")
        cid, name, targets = row.get("id"), row.get("name"), row.get("targets")
        if not isinstance(cid, str) or not cid or cid in seen or not isinstance(name, str) or not name:
            raise ValueError("阵容 id/name 为空或重复")
        if not isinstance(targets, dict) or not targets:
            raise ValueError(f"阵容 {cid} 缺少 targets")
        for hero, count in targets.items():
            if hero not in rules.heroes or _integer(count, f"{cid}.{hero}", 1) > 9:
                raise ValueError(f"阵容 {cid} 使用未知英雄或目标副本数超过 9: {hero}")
        priority = _strings(row.get("priority", list(targets)), "priority")
        if len(priority) != len(set(priority)) or not set(priority).issubset(targets):
            raise ValueError(f"阵容 {cid} 优先级有重复或非目标英雄")
        level = _integer(row.get("roll_level", 6), "roll_level", 1)
        if level not in rules.shop_odds:
            raise ValueError(f"阵容 {cid} 缺少等级 {level} 的商店概率")
        compositions.append(Composition(
            id=cid, name=name, targets=targets,
            priority=priority + [h for h in targets if h not in priority],
            traits=_strings(row.get("traits", []), "traits"),
            preferred_items=_strings(row.get("preferred_items", []), "preferred_items"),
            roll_level=level,
            min_gold=_integer(row.get("min_gold", 30), "min_gold"),
            max_roll_spend=_integer(row.get("max_roll_spend", 20), "max_roll_spend", 1),
            description=str(row.get("description", "")),
        ))
        seen.add(cid)
    return compositions


def match_hero(text: str, rules: Ruleset) -> str | None:
    """Only a complete normalized name or explicitly reviewed alias may match.

    No substring matching: e.g. '不是阿狸' or a panel naming several heroes is unknown.
    Recheck ambiguity even for Ruleset objects constructed outside the JSON loader.
    """
    key = normalize_name(text)
    matches = {hero.id for hero in rules.heroes.values()
               if key and any(key == normalize_name(x) for x in (hero.id, hero.name, *hero.aliases))}
    return next(iter(matches)) if len(matches) == 1 else None
