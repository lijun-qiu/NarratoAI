#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Parse and repair LLM-generated narration script JSON."""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

NARRATION_ITEM_PATTERN = re.compile(
    r'\{\s*"_id"\s*:\s*(\d+)\s*,\s*"timestamp"\s*:\s*"([^"]+)"\s*,'
    r'\s*"picture"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"narration"\s*:\s*"((?:[^"\\]|\\.)*)"'
    r'(?:\s*,\s*"OST"\s*:\s*(\d+))?\s*\}',
    re.DOTALL,
)


def _normalize_smart_quotes(text: str) -> str:
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _clean_json_text(raw: str) -> str:
    text = _normalize_smart_quotes((raw or "").strip())
    if not text:
        return ""

    code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if code_block:
        text = code_block.group(1).strip()

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx > start_idx:
        text = text[start_idx : end_idx + 1]
    return text.strip()


def _apply_common_json_repairs(text: str) -> str:
    fixed = text.replace("{{", "{").replace("}}", "}")
    fixed = re.sub(r"^\s*#.*$", "", fixed, flags=re.MULTILINE)
    fixed = re.sub(r"^\s*//.*$", "", fixed, flags=re.MULTILINE)
    fixed = re.sub(r",\s*}", "}", fixed)
    fixed = re.sub(r",\s*]", "]", fixed)
    fixed = re.sub(r"'([^']*)':", r'"\1":', fixed)
    fixed = re.sub(r'""([^"]*?)""', r'"\1"', fixed)
    return fixed


def _close_truncated_json(text: str) -> str:
    """Best-effort close truncated JSON objects/arrays."""
    open_curly = text.count("{")
    close_curly = text.count("}")
    open_square = text.count("[")
    close_square = text.count("]")

    suffix = ""
    if open_square > close_square:
        suffix += "]" * (open_square - close_square)
    if open_curly > close_curly:
        suffix += "}" * (open_curly - close_curly)
    return text + suffix


def _load_json_candidates(raw: str) -> list[str]:
    cleaned = _clean_json_text(raw)
    if not cleaned:
        return []

    candidates = [cleaned, _apply_common_json_repairs(cleaned)]
    candidates.append(_close_truncated_json(candidates[-1]))

    seen = set()
    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    return unique_candidates


def parse_llm_json(raw: str) -> dict[str, Any] | None:
    """Parse generic LLM JSON output with several repair strategies."""
    for candidate in _load_json_candidates(raw):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _normalize_script_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    normalized["_id"] = int(normalized.get("_id") or 0)
    normalized["timestamp"] = str(normalized.get("timestamp") or "").strip()
    normalized["picture"] = str(normalized.get("picture") or "").strip()
    normalized["narration"] = str(normalized.get("narration") or "").strip()

    ost = normalized.get("OST", 0)
    try:
        normalized["OST"] = int(ost)
    except (TypeError, ValueError):
        normalized["OST"] = 0

    if re.match(r"^播放原片\d*$", normalized["narration"]) or re.match(
        r"^播放原生[_a-f0-9]*$", normalized["narration"]
    ):
        normalized["OST"] = 1

    return normalized


def _extract_items_by_regex(raw: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for match in NARRATION_ITEM_PATTERN.finditer(raw):
        item = {
            "_id": int(match.group(1)),
            "timestamp": match.group(2),
            "picture": match.group(3).replace('\\"', '"'),
            "narration": match.group(4).replace('\\"', '"'),
            "OST": int(match.group(5) or 0),
        }
        items.append(_normalize_script_item(item))
    return items


def _extract_items_from_parsed(parsed: dict[str, Any]) -> list[dict[str, Any]] | None:
    raw_items = parsed.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        return None

    items: list[dict[str, Any]] = []
    for entry in raw_items:
        if isinstance(entry, dict):
            items.append(_normalize_script_item(entry))
    return items or None


def parse_narration_script_items(raw: str) -> list[dict[str, Any]] | None:
    """
    Parse narration script output into a validated items list.

    Returns None when parsing fails completely.
    """
    if not raw or not str(raw).strip():
        logger.error("解说脚本 JSON 为空")
        return None

    parsed = parse_llm_json(raw)
    if parsed is not None:
        items = _extract_items_from_parsed(parsed)
        if items:
            logger.info(f"解说脚本 JSON 解析成功，共 {len(items)} 个片段")
            return items

    regex_items = _extract_items_by_regex(raw)
    if regex_items:
        logger.warning(
            f"JSON 标准解析失败，已通过正则恢复 {len(regex_items)} 个片段；"
            "建议检查模型输出是否被截断。"
        )
        return regex_items

    logger.error(f"解说脚本 JSON 解析失败，原始内容前 500 字: {str(raw)[:500]}")
    return None


def parse_and_fix_json(json_string: str) -> dict[str, Any] | None:
    """Backward-compatible wrapper used by subtitle analysis steps."""
    parsed = parse_llm_json(json_string)
    if parsed is not None:
        return parsed

    items = _extract_items_by_regex(json_string)
    if items:
        return {"items": items}
    return None
