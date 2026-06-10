#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""剧情参考：用户输入的背景说明，辅助抽帧/整片视频分析理解上下文。"""

from __future__ import annotations

DEFAULT_PLOT_REFERENCE_MAX_CHARS = 4000


def normalize_plot_reference(text: str, *, max_chars: int = DEFAULT_PLOT_REFERENCE_MAX_CHARS) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    limit = max(200, int(max_chars or DEFAULT_PLOT_REFERENCE_MAX_CHARS))
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 12].rstrip() + "\n…（剧情参考已截断）"


def build_plot_reference_prompt_section(
    plot_reference: str,
    *,
    max_chars: int = DEFAULT_PLOT_REFERENCE_MAX_CHARS,
) -> str:
    """返回注入视觉模型 prompt 的剧情参考块；空输入返回空字符串。"""
    body = normalize_plot_reference(plot_reference, max_chars=max_chars)
    if not body:
        return ""
    return "\n".join(
        [
            "## 剧情参考（用户提供 · 理解辅助）",
            "以下为分析前提供的剧情/背景说明，用于帮助理解画面、人物关系与对白语境：",
            "- **不得**据此编造画面中未出现的人物、地点、事件或台词",
            "- 画面描述仍以**可见内容**为准；与参考冲突时以画面为准",
            "- 可参考其梳理人物身份、前情、本集主线，辅助填写 characters / involved_characters",
            "",
            body,
        ]
    )
