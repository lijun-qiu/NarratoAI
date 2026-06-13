#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""人物关系文本：用户输入，辅助整片视频分析与剧情解剖。"""

from __future__ import annotations

DEFAULT_CHARACTER_RELATIONSHIP_MAX_CHARS = 12000


def normalize_character_relationship(
    text: str,
    *,
    max_chars: int = DEFAULT_CHARACTER_RELATIONSHIP_MAX_CHARS,
) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    limit = max(500, int(max_chars or DEFAULT_CHARACTER_RELATIONSHIP_MAX_CHARS))
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 12].rstrip() + "\n…（人物关系已截断）"


def build_character_relationship_prompt_section(
    character_relationship: str,
    *,
    max_chars: int = DEFAULT_CHARACTER_RELATIONSHIP_MAX_CHARS,
) -> str:
    body = normalize_character_relationship(character_relationship, max_chars=max_chars)
    if not body:
        return ""
    return "\n".join(
        [
            "## 人物关系表（JSON · 用户提供 · 识别人名与消歧）",
            "- 结构化人物/阵营/关系边；用于规范 `involved_characters` / `important_dialogues.speaker`",
            "- **不得**凭关系表推断未入画人物；画面仍以可见面孔与定妆照匹配为准",
            "- 多人同屏时可用关系（师徒/兄弟/夫妻等）消歧，**禁止**张冠李戴",
            "",
            "```json",
            body,
            "```",
        ]
    )
