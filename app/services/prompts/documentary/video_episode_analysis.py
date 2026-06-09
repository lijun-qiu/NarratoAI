#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""整片视频单集剧情分析 prompt（直接传视频，非抽帧）。"""

from __future__ import annotations

from app.services.documentary.documentary_settings import (
    FRAME_FACE_MATCH_SIMILARITY_HINT,
    FRAME_UNKNOWN_CHARACTER_FEMALE,
    FRAME_UNKNOWN_CHARACTER_MALE,
)


def _resolve_drama_title(drama_title: str | dict[str, str] | None) -> str:
    if isinstance(drama_title, dict):
        raw = drama_title.get("label") or drama_title.get("id") or "罚罪2"
    else:
        raw = drama_title or "罚罪2"
    return str(raw).strip() or "罚罪2"


def _format_duration_label(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours}小时{minutes}分{secs}秒"
    if minutes > 0:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def build_video_episode_vision_reference_prompt_section(
    *,
    drama_label: str,
    character_references: list[dict[str, str]] | None = None,
    relationship_diagram_attached: bool = False,
    reference_image_count: int = 0,
    character_collage: bool = False,
) -> str:
    """人物头像/关系图参照说明（图片排在视频之前）。"""
    refs = [
        item
        for item in (character_references or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    if reference_image_count <= 0:
        return ""

    work = (drama_label or "本剧").strip()
    lines = [
        f"## 视觉参照图（本请求最前 {reference_image_count} 张 · **不是视频**）",
    ]
    image_index = 1
    if relationship_diagram_attached:
        lines.extend(
            [
                f"**图 #{image_index}**：**{work}** 人物关系图 — 仅用于校正身份/关系，**不可**凭关系图猜在场人物。",
                "",
            ]
        )
        image_index += 1

    if refs:
        if character_collage and len(refs) >= 2:
            numbered = "、".join(
                f"#{idx + 1}{str(item.get('name') or '').strip()}"
                for idx, item in enumerate(refs)
            )
            lines.append(
                f"**图 #{image_index}**：人物定妆照拼图（从左到右依次为 {numbered}），"
                "用于对照视频中可见面孔确认姓名。"
            )
        else:
            lines.append("以下人物定妆照/头像按顺序提供，**用于对照视频中面孔确认姓名**：")
            for item in refs:
                name = str(item.get("name") or "").strip()
                lines.append(f"- **{name}** — 参照图 #{image_index}")
                image_index += 1
        lines.append("")

    lines.extend(
        [
            f"**最后 1 张**为待分析视频片段。",
            build_character_naming_guidance_block(has_references=True),
        ]
    )
    return "\n".join(lines)


def build_character_naming_guidance_block(*, has_references: bool = False) -> str:
    if not has_references:
        return (
            "## 人物命名规则\n"
            "- 无法从画面确认身份时，`involved_characters` / `speaker` 写「剧中未明确交代」。\n"
            "- **禁止**凭字幕称呼（如二师兄、老叶）直接当作规范姓名，除非片内字幕已写明全名。"
        )

    return (
        "## 人物命名规则（须对照上传头像）\n"
        f"- 视频中**脸/侧脸清晰**且与上传定妆照/头像对照匹配（{FRAME_FACE_MATCH_SIMILARITY_HINT}）"
        " → **必须**写规范姓名（如秦枫、胡小跃）。\n"
        f"- 面孔无法匹配或不可辨认时，写「{FRAME_UNKNOWN_CHARACTER_MALE}」「{FRAME_UNKNOWN_CHARACTER_FEMALE}」"
        " 或「剧中未明确交代」，**禁止**便衣男/年轻警员等泛称。\n"
        "- **禁止**仅凭字幕称呼（二师兄、大师兄等）猜规范姓名；须面孔与头像匹配后才能写全名。\n"
        "- `important_dialogues.speaker` 同样遵守上述规则。"
    )


def build_reference_carryover_naming_block(
    *,
    drama_label: str,
    character_references: list[dict[str, str]] | None = None,
    relationship_diagram_attached: bool = False,
) -> str:
    names = [
        str(item.get("name") or "").strip()
        for item in (character_references or [])
        if isinstance(item, dict) and item.get("name")
    ]
    if not names and not relationship_diagram_attached:
        return ""

    work = (drama_label or "本剧").strip()
    lines = [
        "## 视觉参照沿用（本段不再重复发送头像）",
        f"首段已提供 **{work}** 人物定妆照；本段写规范姓名仍须**对照视频可见面孔**与首段头像达到约70%面部相似度。",
    ]
    if relationship_diagram_attached:
        lines.append("- 关系图仅作关系/身份**校正**，不可凭关系图猜人。")
    if names:
        lines.append(
            f"- 定妆照人物（{'、'.join(names)}）须面孔匹配后才可写规范名；"
            "硬字幕称呼**不得**猜人。"
        )
    lines.append(build_character_naming_guidance_block(has_references=True))
    return "\n".join(lines)


def build_analysis_density_guidance(video_duration_seconds: float | None) -> str:
    """台词与其他字段的粒度要求。"""
    duration = max(0.0, float(video_duration_seconds or 0))
    dialogue_min = max(5, int(round(duration / 45))) if duration > 0 else 5
    duration_label = _format_duration_label(duration) if duration > 0 else "未知"
    return (
        f"## 其他字段要求（本视频约 {duration_label}）\n"
        f"- `important_dialogues` **至少 {dialogue_min} 条**；`quote` 必须是视频中实际听到的原话。\n"
        "- 优先收录：质问/反驳、动机表白、案件信息、威胁承诺、伏笔暗示等台词。"
    )


def build_video_episode_analysis_prompt(
    *,
    drama_title: str = "罚罪2",
    video_duration_seconds: float | None = None,
    segment_schedule_block: str = "",
    character_naming_block: str = "",
) -> str:
    title = _resolve_drama_title(drama_title)
    density_guidance = build_analysis_density_guidance(video_duration_seconds)
    naming_block = character_naming_block or build_character_naming_guidance_block(
        has_references=False
    )
    return f"""请扮演一位资深的影视剧编剧兼分析师，你的任务是对我上传的电视剧集（《{title}》单集）进行深入分析，并严格按照下方 JSON 格式输出结果。

请关注以下核心原则：
1. **叙事连贯性**：理解剧情的起承转合，识别因果链条和伏笔。
2. **冲突分析**：识别剧中的核心矛盾、权力博弈和角色间的对抗。
3. **角色动机**：分析主角行为背后的情感和利益动机。
4. **杜绝幻觉**：所有陈述必须基于视频内容，不确定的信息请注明「剧中未明确交代」。

【输出JSON格式模板】
{{
  "overall_summary": "用一段200字以内的话，概括本集的核心剧情、主要冲突和高潮。",
  "key_conflict": "用一句话点名本集最核心的矛盾冲突是什么。",
  "episodic_segments": [
    {{
      "segment_id": 1,
      "title": "为这个片段起一个4-6字的标题，例如「天台对话」",
      "time_range": "00:00:00-00:00:10",
      "key_events": "用一句话描述该片段发生的关键事件。",
      "narration": "第三人称纪录片旁白，20-50字，描述该10秒内画面与对话，可直接用于后期配音。",
      "environment_description": "场景环境：地点、室内外、光线、氛围、可见陈设等，15-40字。",
      "involved_characters": ["角色A", "角色B"]
    }},
    {{
      "segment_id": 2
    }}
  ],
  "important_dialogues": [
    {{
      "speaker": "角色名",
      "timestamp": "时间戳",
      "quote": "引用的关键台词",
      "significance": "解释这句台词为何重要，揭示了什么信息或反映了什么心态"
    }}
  ],
  "cliffhangers_or_foreshadowing": [
    {{
      "description": "描述一个悬念或伏笔",
      "possible_interpretation": "分析它可能为后续剧情埋下的伏笔是什么"
    }}
  ]
}}

【输出要求】
- 严格按照上方 JSON 格式输出，请确保输出是有效的 JSON。
- `episodic_segments` 必须逐条对应下方固定时间窗口，不得少条、不得合并。
- 每条 `episodic_segments` 除 `title` / `key_events` / `involved_characters` 外，还必须填写：
  - `narration`：第三人称纪录片旁白，基于该 10 秒画面与对话撰写，20-50 字，语气客观、可用于配音；
  - `environment_description`：该片段的场景环境（地点、室内外、时间感、光线、氛围、关键布景），15-40 字。
- 若某 10 秒仅为延续上段画面，`narration` 可写「画面延续上段」，`environment_description` 可写「环境延续上段」。
- 若某个字段没有相应内容，请输出空列表 `[]`（`key_conflict` 若无内容则输出空字符串 `""`）。
- 只返回 JSON，不要 markdown 代码块或额外说明。

{segment_schedule_block}

{naming_block}

{density_guidance}"""


def build_video_episode_chunk_prompt(
    *,
    drama_title: str,
    chunk_index: int,
    total_chunks: int,
    offset_seconds: float,
    chunk_duration_seconds: float | None = None,
    segment_schedule_block: str = "",
    character_naming_block: str = "",
) -> str:
    """长片分段分析：要求 time_range / timestamp 使用全片绝对时间。"""
    hours = int(offset_seconds // 3600)
    minutes = int((offset_seconds % 3600) // 60)
    seconds = int(offset_seconds % 60)
    offset_label = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    base = build_video_episode_analysis_prompt(
        drama_title=drama_title,
        video_duration_seconds=chunk_duration_seconds,
        segment_schedule_block=segment_schedule_block,
        character_naming_block=character_naming_block,
    )
    return (
        f"{base}\n\n"
        f"## 分段说明\n"
        f"这是全片第 {chunk_index + 1}/{total_chunks} 段，本段在整片时间轴上的起始时刻为 **{offset_label}**。\n"
        f"`episodic_segments.time_range` 与 `important_dialogues.timestamp` 必须使用**全片绝对时间**（HH:MM:SS），"
        f"不要从 00:00:00 重新计时。\n"
        f"`overall_summary` 仅概括本段剧情（80字内）；`key_conflict` 仅描述本段核心冲突；"
        f"`episodic_segments` 仅填写本段固定窗口，条数与窗口列表一致。"
    )
