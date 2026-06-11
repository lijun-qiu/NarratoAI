#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""《罚罪2》内置人物关系（WebUI 默认填入 JSON；分析仅使用页面填写内容）。"""

from __future__ import annotations

import json
import os
from typing import Any

_CHARACTER_GRAPH_JSON = os.path.join(os.path.dirname(__file__), "fazu2_character_graph.json")

FAZU2_DRAMA_IDS = frozenset({"罚罪2", "罚罪", "fazu2", "FAZU2"})


def is_fazu2_drama(*, drama_id: str = "", drama_title: str = "") -> bool:
    for raw in (drama_id, drama_title):
        text = (raw or "").strip()
        if not text:
            continue
        if text in FAZU2_DRAMA_IDS:
            return True
        if "罚罪" in text:
            return True
    return False


def bundled_fazu2_character_graph_path() -> str:
    if os.path.isfile(_CHARACTER_GRAPH_JSON):
        return _CHARACTER_GRAPH_JSON
    return ""


def bundled_fazu2_relationships_path() -> str:
    """兼容旧名：人物关系已迁至 `fazu2_character_graph.json`。"""
    return bundled_fazu2_character_graph_path()


def load_fazu2_character_graph() -> dict[str, Any]:
    path = bundled_fazu2_character_graph_path()
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def format_character_graph_for_prompt(
    graph: dict[str, Any],
    *,
    max_chars: int = 12000,
) -> str:
    if not graph:
        return ""
    text = json.dumps(graph, ensure_ascii=False, indent=2)
    limit = max(500, int(max_chars or 12000))
    if len(text) <= limit:
        return text
    return text[: limit - 16].rstrip() + "\n…（已截断）"


def is_legacy_fazu2_relationship_markdown(text: str) -> bool:
    """识别旧版 Markdown 人物关系默认文案（需迁移为 JSON）。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if cleaned.startswith("# 《罚罪2》人物关系"):
        return True
    return "fazu2_character_graph.json" in cleaned[:800] and cleaned.startswith("#")


def get_default_character_relationship_text(
    *,
    drama_id: str = "",
    drama_title: str = "",
    max_chars: int = 12000,
) -> str:
    """WebUI 人物关系默认：罚罪2 等内置剧目使用 `fazu2_character_graph.json`。"""
    if not is_fazu2_drama(drama_id=drama_id, drama_title=drama_title):
        return ""
    return format_character_graph_for_prompt(
        load_fazu2_character_graph(),
        max_chars=max_chars,
    )


def get_default_plot_reference_text(
    *,
    drama_id: str = "",
    drama_title: str = "",
    max_chars: int = 12000,
) -> str:
    """剧情参考无内置默认（与人物关系 JSON 分离）。"""
    del drama_id, drama_title, max_chars
    return ""
