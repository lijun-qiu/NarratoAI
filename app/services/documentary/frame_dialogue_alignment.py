#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""抽帧对白：区分画面人物与说话人；人名靠头像匹配，同一人须身形服装一致才可回溯。"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from app.services.documentary.documentary_settings import (
    FRAME_UNKNOWN_CHARACTER_MALE,
)

_LISTENER_CUES = (
    "听着",
    "静立",
    "聆听",
    "紧闭双眼",
    "闭眼神",
    "一言不发",
    "表情一动不动",
    "目光深沉",
    "极度压抑",
    "眉头紧锁",
    "侧脸",
    "低头思考",
    "目视远方",
)
_SPEAKER_CUES = (
    "开口说话",
    "开口",
    "语带嘲讽",
    "语带",
    "言辞",
    "挥手",
    "伸手指向",
)
_SPEAKER_STRONG = ("说话", "言辞犀利", "对白")
_SUBTITLE_LISTENER_NOTE = "（硬字幕为台词原文，说话者未必是本帧画面人物）"


def build_frame_dialogue_speaker_rules() -> str:
    return """## 对白与说话人（硬性 · 勿把硬字幕归属搞混）

- **画面人物 ≠ 说话人**：硬字幕/burned_in_subtitle 仅复制**屏幕上的台词文字**，**不表示**本帧可见人物一定在说话
- 写 observation 须分开：
  - **谁入画/谁的脸**：对照定妆照/头像，**本帧**面孔匹配成功才写规范姓名
  - **暂称须可追踪**：脸不可辨时用带可见特征的暂称（如「深色夹克便衣男」），勿笼统写「便衣男」了事
  - **同一人回溯**：后帧确认某人后，前序帧仅当**同一身形+同一服装/发型**可判定为同一人时才改规范名；无法确认则保留暂称
  - **禁止**把本批所有便衣/暂称一律改成同一名；也**禁止**凭硬字幕/SRT 猜人
  - **是否在本帧说话**：仅当**本帧可见**其嘴型张开/手势发言时，才写「开口说话/语带…」
  - 若为**反应镜/聆听镜**（静立、听着、闭眼神），写「聆听/静听/压抑反应」，**禁止**写「开口说话」
- **过肩镜头**：前景背影+后景对脸时，硬字幕往往属于**背对镜头者**，勿把对脸聆听者标为说话人
- **人名须有据**：规范姓名来自头像面孔匹配；硬字幕/SRT **不得**用于猜人
- scene_segments 的 action **禁止**把整段对白全算给一人；按逐帧可见说话/聆听分开描述"""


def _resolve_reference_names(artifact: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for item in artifact.get("character_references") or []:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if name:
                names.add(name)
    return names


def _default_gender_for_name(name: str) -> str:
    if name in {"文江燕", "文琴", "赵子怡", "彭含章"}:
        return "女"
    return "男"


def _count_name_mentions_in_texts(name: str, texts: list[str]) -> int:
    count = 0
    for text in texts:
        if not text:
            continue
        if f"{name}(男)" in text or f"{name}(女)" in text:
            count += text.count(f"{name}(男)") + text.count(f"{name}(女)")
        elif re.search(rf"(?<![\u4e00-\u9fff]){re.escape(name)}(?![\u4e00-\u9fff])", text):
            count += 1
    return count


def _demote_sporadic_unsubstantiated_names_in_text(
    text: str,
    *,
    ref_names: set[str],
    dominant_name: str,
    demoted_names: set[str],
) -> str:
    if not text:
        return text
    updated = text
    gender_fallback = FRAME_UNKNOWN_CHARACTER_MALE
    for name in demoted_names:
        if name not in ref_names or name == dominant_name:
            continue
        gender = _default_gender_for_name(name)
        updated = updated.replace(f"{name}({gender})", gender_fallback)
        updated = re.sub(
            rf"(?<![\u4e00-\u9fff]){re.escape(name)}(?![\u4e00-\u9fff])",
            "未名人员",
            updated,
        )
    return updated


def demote_sporadic_unsubstantiated_names(
    frames: list[dict[str, Any]],
    segments: list[dict[str, Any]] | None,
    ref_names: set[str],
) -> set[str]:
    """本批次内零星误认的参照人名（如张冠李戴的刘天也）降级为未名人员。"""
    if not ref_names:
        return set()

    valid_frames = [item for item in frames if isinstance(item, dict)]
    obs_texts = [str(item.get("observation") or "") for item in valid_frames]
    seg_texts: list[str] = []
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        for key in ("observation", "action", "key_visual"):
            value = str(segment.get(key) or "").strip()
            if value:
                seg_texts.append(value)
    all_texts = obs_texts + seg_texts
    if not all_texts:
        return set()

    mention_counts = {
        name: _count_name_mentions_in_texts(name, all_texts)
        for name in ref_names
        if _count_name_mentions_in_texts(name, all_texts) > 0
    }
    if not mention_counts:
        return set()

    dominant_name = max(mention_counts, key=lambda key: mention_counts[key])
    dominant_count = mention_counts[dominant_name]
    demoted: set[str] = set()

    for name, count in mention_counts.items():
        if name == dominant_name:
            continue
        if dominant_count >= 2 and count <= max(1, dominant_count // 2):
            demoted.add(name)

    if not demoted:
        return set()

    for frame in valid_frames:
        obs = str(frame.get("observation") or "")
        new_obs = _demote_sporadic_unsubstantiated_names_in_text(
            obs,
            ref_names=ref_names,
            dominant_name=dominant_name,
            demoted_names=demoted,
        )
        if new_obs != obs:
            frame["observation"] = new_obs

    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        for key in ("observation", "action", "key_visual"):
            value = str(segment.get(key) or "")
            if not value:
                continue
            updated = _demote_sporadic_unsubstantiated_names_in_text(
                value,
                ref_names=ref_names,
                dominant_name=dominant_name,
                demoted_names=demoted,
            )
            if updated != value:
                segment[key] = updated

    return demoted


def fix_listener_speaker_confusion_in_observation(observation: str, *, has_subtitle: bool) -> str:
    """聆听/反应镜误标「开口说话」时纠正。"""
    obs = (observation or "").strip()
    if not obs:
        return obs

    is_listener = any(cue in obs for cue in _LISTENER_CUES)
    is_speaker = any(cue in obs for cue in _SPEAKER_CUES) or (
        any(cue in obs for cue in _SPEAKER_STRONG) and "听着" not in obs
    )

    if is_listener and is_speaker:
        for cue in _SPEAKER_CUES:
            obs = obs.replace(f"{cue}，", "静听，")
            obs = obs.replace(cue, "静听")
        for cue in _SPEAKER_STRONG:
            if cue in obs and "静听" not in obs:
                obs = obs.replace(cue, "静听")

    if has_subtitle and is_listener and _SUBTITLE_LISTENER_NOTE not in obs:
        if "秦枫" in obs and not any(cue in obs for cue in ("说话", "开口", "语带", "挥手")):
            obs = f"{obs}{_SUBTITLE_LISTENER_NOTE}"

    return obs


def apply_dialogue_alignment_to_artifact(artifact: dict[str, Any]) -> None:
    """整份 artifact：降级零星误认人名；纠正说话/聆听混淆（不做便衣统一）。"""
    if not isinstance(artifact, dict):
        return

    ref_names = _resolve_reference_names(artifact)
    listener_fixes = 0

    def _process_batch(frames: list[dict[str, Any]], segments: list[dict[str, Any]]) -> None:
        nonlocal listener_fixes
        demoted = demote_sporadic_unsubstantiated_names(frames, segments, ref_names)
        if demoted:
            logger.info(f"抽帧 batch：已降级零星误认人名 {sorted(demoted)}")

        for frame in frames:
            if not isinstance(frame, dict):
                continue
            obs = str(frame.get("observation") or "")
            has_sub = bool(frame.get("has_burned_in_subtitle")) and bool(
                str(frame.get("burned_in_subtitle") or "").strip()
            )
            new_obs = fix_listener_speaker_confusion_in_observation(obs, has_subtitle=has_sub)
            if new_obs != obs:
                frame["observation"] = new_obs
                listener_fixes += 1

    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        frames = [
            item for item in (batch.get("frame_observations") or []) if isinstance(item, dict)
        ]
        segments = [
            item for item in (batch.get("scene_segments") or []) if isinstance(item, dict)
        ]
        _process_batch(frames, segments)

    top_frames = [
        item for item in (artifact.get("frame_observations") or []) if isinstance(item, dict)
    ]
    top_segments = [
        item for item in (artifact.get("scene_segments") or []) if isinstance(item, dict)
    ]
    if top_frames and not any(
        isinstance(batch, dict) and batch.get("frame_observations")
        for batch in (artifact.get("batches") or [])
    ):
        _process_batch(top_frames, top_segments)

    if listener_fixes:
        logger.info(f"抽帧 artifact：聆听镜修正 {listener_fixes} 处")
