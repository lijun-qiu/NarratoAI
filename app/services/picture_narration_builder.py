#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""原声段旁白（picture）生成：参照字幕上下文与画面信息。"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from app.services.srt_utils import (
    SrtEntry,
    clean_subtitle_dialogue_text,
    extract_entries_in_range,
)
from app.services.video_output_settings import get_video_output_settings

_RULES_FILENAME = "picture_narration_rules.md"
_RULES_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "prompts",
    "video_output",
)

_MOOD_RULES: list[tuple[tuple[str, ...], str]] = [
    (("怒", "吼", "滚", "混蛋", "岂有此理"), "愤怒"),
    (("笑", "呵", "有趣"), "讽刺"),
    (("哭", "泪", "伤心"), "悲伤"),
    (("怕", "恐", "颤抖"), "紧张"),
    (("枪", "杀", "死", "血"), "爆发"),
    (("秘密", "内鬼", "阴谋", "证据"), "悬疑"),
    (("静", "沉", "默"), "压抑"),
]

_SCENE_RULES: list[tuple[tuple[str, ...], str]] = [
    (("电话", "手机", "短信"), "手机响起关键信息"),
    (("门", "推", "闯"), "推门闯入对峙"),
    (("车", "追", "逃"), "追逐场面升级"),
    (("查", "证据", "档案"), "追查线索收紧"),
    (("威胁", "警告", "别查"), "当面警告施压"),
    (("摊牌", "真相", "原来"), "真相浮出水面"),
]


def load_picture_narration_rules(
    overrides: Optional[dict] = None,
) -> str:
    """读取旁白生成规则 Markdown。"""
    path = os.path.join(_RULES_DIR, _RULES_FILENAME)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            template = handle.read().strip()
    except OSError:
        return ""

    cfg = get_video_output_settings(overrides)
    params = {
        "picture_narration_max_chars": str(int(cfg.get("picture_narration_max_chars", 16))),
        "picture_narration_duration": str(float(cfg.get("picture_narration_duration", 5.0))),
    }
    for key, value in params.items():
        template = template.replace(f"${{{key}}}", value)
    return template


def _infer_mood(text: str) -> str:
    for keywords, mood in _MOOD_RULES:
        if any(kw in text for kw in keywords):
            return mood
    if "？" in text or "吗" in text:
        return "悬疑"
    return ""


def _infer_scene_action(text: str) -> str:
    for keywords, action in _SCENE_RULES:
        if any(kw in text for kw in keywords):
            return action
    if len(text) <= 8:
        return "关键对白时刻"
    return "人物对峙交锋"


def _looks_like_raw_dialogue(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if t.startswith("（") and t.endswith("）"):
        return False
    if "：" in t[:12] or ":" in t[:12]:
        return True
    if t.count("，") >= 2 or t.count("。") >= 1:
        return len(t) > 10
    return len(t) > 18


def build_picture_narration_from_subtitle_context(
    srt_entries: List[SrtEntry],
    start_sec: float,
    end_sec: float,
    *,
    max_chars: Optional[int] = None,
    context_ms: int = 3000,
) -> str:
    """
    根据时间轴内字幕及前后上下文，生成原声段旁白 picture 文案。
    不复读对白原文，输出画面/动作/情绪描述。
    """
    if not srt_entries:
        return "关键画面"

    max_len = int(max_chars or get_video_output_settings().get("picture_narration_max_chars", 16))
    start_ms = int(round(start_sec * 1000))
    end_ms = int(round(end_sec * 1000))

    clipped = extract_entries_in_range(srt_entries, start_ms, end_ms)
    context = extract_entries_in_range(
        srt_entries,
        max(0, start_ms - context_ms),
        end_ms + context_ms,
    )

    segment_text = " ".join(
        clean_subtitle_dialogue_text(e.text) for e in clipped if clean_subtitle_dialogue_text(e.text)
    )
    context_text = " ".join(
        clean_subtitle_dialogue_text(e.text)
        for e in context
        if clean_subtitle_dialogue_text(e.text) and e not in clipped
    )
    combined = f"{segment_text} {context_text}".strip()
    if not combined:
        return "剧情过渡"

    mood = _infer_mood(combined)
    action = _infer_scene_action(combined)

    if clipped and not _looks_like_raw_dialogue(segment_text):
        # 字幕本身已是短画面描述，可微调使用
        hint = segment_text
    else:
        hint = f"（{mood}）{action}" if mood else action

    hint = re.sub(r"\s+", "", hint)
    if len(hint) > max_len:
        hint = hint[: max_len - 1].rstrip("，。！？") + "…"
    return hint or "关键画面"
