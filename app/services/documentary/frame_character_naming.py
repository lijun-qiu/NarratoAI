#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""抽帧人名：字幕/硬字幕优先，关系表仅作谐音校正，禁止凭关系表猜人。"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from app.services.documentary.documentary_settings import (
    FRAME_UNKNOWN_CHARACTER_FEMALE,
    FRAME_UNKNOWN_CHARACTER_MALE,
)
from app.services.short_drama_drama_knowledge import (
    PLOT_BLUEPRINT_NAME_ALIAS_GROUPS,
    correct_name_mistakes_in_text,
)

_NAME_IN_TEXT_RE = re.compile(r"[\u4e00-\u9fffA-Za-z·]{2,6}")
_GENDER_SUFFIX_RE = re.compile(r"[\(（][男女不明][\)）]$")


def build_frame_naming_priority_rules(
    *,
    has_drama_knowledge: bool = False,
    has_character_references: bool = False,
    is_carryover_batch: bool = False,
) -> str:
    """抽帧写人名的优先级（关系表/参照图启用时追加）。"""
    lines = [
        "## 人名写入优先级（硬性 · 高于人物关系表/关系图）",
        "1. **本批次硬字幕、SRT 对白摘录、subtitle_entries.text** 中出现的人名/称呼 → 可写入 observation/action/characters；",
        "2. **定妆照面孔匹配**：仅当本批关键帧中**清晰可见**某张脸，且与已上传定妆照一致时，才可写对应规范姓名；",
        "3. **人物关系表/关系图** 仅用于：谐音校正（秦峰→秦枫）、简称归并（老叶→叶天佑）、禁止张冠李戴；"
        "**禁止**因关系表里有某角色，就把该名字写到本批并未出现/未识别的人物上；",
        f"4. 以上皆不满足 → 写「{FRAME_UNKNOWN_CHARACTER_MALE}」「{FRAME_UNKNOWN_CHARACTER_FEMALE}」，禁止凭剧情印象填名。",
    ]
    if has_drama_knowledge:
        lines.append(
            "- 关系表**不是**「本集出场名单」：未在本批字幕/硬字幕/可见面孔中出现的人物，**不得**写入本批 scene_segments。"
        )
    if has_character_references:
        lines.append(
            "- 定妆照**不是**「默认全员在场」：每张脸须在本批画面中可见且与参照图匹配，才可写该姓名。"
        )
    if is_carryover_batch:
        lines.append(
            "- 本批未重复发送参照图：**不得**沿用首批姓名覆盖本批字幕未出现的人；"
            "仅当本批字幕/硬字幕/可见面孔支持时才写人名。"
        )
    return "\n".join(lines)


def _strip_gender_suffix(name: str) -> str:
    return _GENDER_SUFFIX_RE.sub("", (name or "").strip())


def _canonical_for_name(name: str) -> str:
    cleaned = _strip_gender_suffix(name)
    for canonical, aliases in PLOT_BLUEPRINT_NAME_ALIAS_GROUPS:
        if cleaned == canonical or cleaned in aliases:
            return canonical
    return cleaned


def _name_tokens_for_matching(name: str) -> set[str]:
    cleaned = _strip_gender_suffix(name)
    if not cleaned:
        return set()
    tokens = {cleaned}
    for canonical, aliases in PLOT_BLUEPRINT_NAME_ALIAS_GROUPS:
        if cleaned == canonical or cleaned in aliases:
            tokens.add(canonical)
            tokens.update(aliases)
            break
    return {token for token in tokens if len(token) >= 2}


def collect_subtitle_evidence_text(
    segment: dict[str, Any],
    *,
    batch_observations: list[dict[str, Any]] | None = None,
) -> str:
    parts: list[str] = []
    subtitle = str(segment.get("subtitle") or "").strip()
    if subtitle:
        parts.append(subtitle)
    entries = segment.get("subtitle_entries")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict):
                text = str(entry.get("text") or "").strip()
                if text:
                    parts.append(text)
    for observation in batch_observations or []:
        if not isinstance(observation, dict):
            continue
        if observation.get("has_burned_in_subtitle"):
            burned = str(observation.get("burned_in_subtitle") or "").strip()
            if burned:
                parts.append(burned)
    return "\n".join(parts)


def is_character_name_evidence_backed(name: str, evidence_text: str) -> bool:
    """人名是否有本批字幕/硬字幕/谐音依据。"""
    if not name or not evidence_text.strip():
        return False
    for token in _name_tokens_for_matching(name):
        if token in evidence_text:
            return True
    return False


def sanitize_segment_character_names(
    segment: dict[str, Any],
    *,
    evidence_text: str,
    reference_names: set[str] | None = None,
) -> list[str]:
    """
    过滤 characters 中无字幕依据的名字；返回被移除的姓名列表。
    reference_names 仅用于日志，不单独作为写入依据。
    """
    if not isinstance(segment, dict):
        return []

    characters = segment.get("characters")
    if isinstance(characters, str):
        char_list = [part.strip() for part in re.split(r"[、,，/]", characters) if part.strip()]
    elif isinstance(characters, list):
        char_list = [str(name).strip() for name in characters if str(name).strip()]
    else:
        return []

    kept: list[str] = []
    removed: list[str] = []
    for name in char_list:
        corrected = correct_name_mistakes_in_text(name)
        if is_character_name_evidence_backed(corrected, evidence_text):
            kept.append(corrected)
        else:
            removed.append(name)

    if removed:
        logger.debug(
            f"抽帧 characters 移除无字幕依据人名: {removed}"
            + (f"（参照表含 {sorted(reference_names or [])}，仍须本批字幕/硬字幕支持）" if reference_names else "")
        )
    segment["characters"] = kept
    return removed


def apply_subtitle_gated_names_to_artifact(artifact: dict[str, Any]) -> None:
    """整份 artifact：按各 segment 字幕依据过滤 characters。"""
    if not isinstance(artifact, dict):
        return

    ref_names = {
        str(item.get("name") or "").strip()
        for item in (artifact.get("character_references") or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }

    obs_by_batch: dict[int, list[dict[str, Any]]] = {}
    for observation in artifact.get("frame_observations") or []:
        if not isinstance(observation, dict):
            continue
        batch_index = int(observation.get("batch_index", 0))
        obs_by_batch.setdefault(batch_index, []).append(observation)

    total_removed = 0
    for segment in artifact.get("scene_segments") or []:
        if not isinstance(segment, dict):
            continue
        batch_index = int(segment.get("batch_index", 0))
        evidence = collect_subtitle_evidence_text(
            segment,
            batch_observations=obs_by_batch.get(batch_index),
        )
        removed = sanitize_segment_character_names(
            segment,
            evidence_text=evidence,
            reference_names=ref_names,
        )
        total_removed += len(removed)

    if total_removed:
        logger.info(f"抽帧 artifact：已移除 {total_removed} 条无字幕依据的 characters 人名")
