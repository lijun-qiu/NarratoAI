#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""整片视频单集剧情分析 prompt（直接传视频，非抽帧）。"""

from __future__ import annotations

from typing import Any

from app.services.documentary.documentary_settings import (
    FRAME_FACE_MATCH_SIMILARITY_HINT,
    FRAME_FACE_MATCH_SIMILARITY_MIN_PERCENT,
    FRAME_UNKNOWN_CHARACTER_FEMALE,
    FRAME_UNKNOWN_CHARACTER_MALE,
)
from app.services.documentary.frame_reference_images import (
    REFERENCE_COLLAGE_MAX_HEADS_PER_SHEET,
    split_character_references_into_collage_sheets,
)
from app.services.documentary.video_episode_constants import (
    SEGMENT_MAX_SECONDS,
    SEGMENT_MIN_SECONDS,
)
from app.services.documentary.video_episode_segment_schedule import segment_policy_summary


def _resolve_drama_title(drama_title: str | dict[str, str] | None) -> str:
    if isinstance(drama_title, dict):
        raw = drama_title.get("label") or drama_title.get("id") or "本片"
    else:
        raw = drama_title or "本片"
    return str(raw).strip() or "本片"


def _format_hms_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
    collage_sheets: list[list[dict[str, str]]] | None = None,
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

    sheets = collage_sheets or []
    if refs:
        if character_collage and sheets:
            total_sheets = len(sheets)
            global_idx = 1
            for sheet_index, sheet in enumerate(sheets, start=1):
                numbered = "、".join(
                    f"#{global_idx + idx}{str(item.get('name') or '').strip()}"
                    for idx, item in enumerate(sheet)
                )
                global_idx += len(sheet)
                if len(sheet) == 1:
                    name = str(sheet[0].get("name") or "").strip()
                    lines.append(f"**图 #{image_index}**：人物定妆照 **{name}**（第 {sheet_index}/{total_sheets} 张参照图）。")
                else:
                    lines.append(
                        f"**图 #{image_index}**：人物定妆照拼图 **第 {sheet_index}/{total_sheets} 张**"
                        f"（每张拼图最多 {REFERENCE_COLLAGE_MAX_HEADS_PER_SHEET} 人；本张从左到右依次为 {numbered}），"
                        "须**逐脸放大对照**后确认姓名。"
                    )
                image_index += 1
        elif character_collage and len(refs) >= 2:
            numbered = "、".join(
                f"#{idx + 1}{str(item.get('name') or '').strip()}"
                for idx, item in enumerate(refs)
            )
            lines.append(
                f"**图 #{image_index}**：人物定妆照拼图（从左到右依次为 {numbered}），"
                "须逐脸对照视频中可见面孔确认姓名。"
            )
        else:
            lines.append("以下人物定妆照/头像按顺序提供，**须逐脸对照**视频中面孔确认姓名：")
            for item in refs:
                name = str(item.get("name") or "").strip()
                lines.append(f"- **{name}** — 参照图 #{image_index}")
                image_index += 1
        lines.append("")

    lines.extend(
        [
            f"**最后 1 张**为待分析视频片段。",
            build_character_naming_guidance_block(has_references=True),
            build_per_grid_face_identification_rules(),
        ]
    )
    return "\n".join(lines)


def build_video_speaker_inference_rules() -> str:
    return """## 说话人归属（硬性 · 仅依据本段视频画面）

- `important_dialogues.speaker` 须为**该台词时间窗内可见**且与定妆照匹配（≥75% 相似）的人物
- 结合画面嘴型/手势：嘴型张开或明显发言姿态 → 候选说话人；静听/反应镜 → **不是**说话人
- 过肩镜头：台词常属**背对镜头者**，勿把对脸聆听者标为 speaker
- 多人同屏时结合用户提供的人物关系消歧；仍须面孔匹配，**禁止**听声猜人
- 无法确认时写「剧中未明确交代」；**禁止**写未入画或未匹配成功的人物规范名"""


def build_frame_timeline_speaker_inference_rules() -> str:
    """兼容旧引用；视频分析已不再注入 frame_timeline。"""
    return build_video_speaker_inference_rules()


def build_character_naming_guidance_block(*, has_references: bool = False) -> str:
    if not has_references:
        return (
            "## 人物命名规则\n"
            "- 无法从画面确认身份时，`involved_characters` / `speaker` 写「剧中未明确交代」。\n"
            "- **禁止**凭字幕称呼直接当作规范姓名，除非片内字幕已写明全名。"
        )

    policy_label = segment_policy_summary()
    return (
        "## 人物命名规则（须对照上传头像 · 严格）\n"
        f"- **每个时间窗独立识脸**（{policy_label}）：每条 `episodic_segments` 须对照该 `time_range` 窗口内**实际可见面孔**与定妆照匹配后再写 `involved_characters`\n"
        f"- 仅当脸/侧脸清晰且与定妆照匹配（{FRAME_FACE_MATCH_SIMILARITY_HINT}）"
        f" → 写规范姓名；**未达 {FRAME_FACE_MATCH_SIMILARITY_MIN_PERCENT}% 相似度禁止写规范名**\n"
        f"- 该窗口内**无清晰人脸 / 无法匹配 / 相似度不足** → 写「{FRAME_UNKNOWN_CHARACTER_MALE}」「{FRAME_UNKNOWN_CHARACTER_FEMALE}」"
        " 或「剧中未明确交代」，**禁止**便衣男/年轻警员/警服男子等泛称，**禁止**凭剧情印象猜名\n"
        "- **禁止**仅凭字幕称呼、对白内容、上下格人物、服装等猜规范姓名\n"
        "- 外貌相近的角色须**逐脸细对**，不可互换\n"
        "- `important_dialogues.speaker` 同样遵守上述规则；说话人须在该台词时间窗内可见且面孔匹配，**禁止**听声猜人"
    )


def build_per_grid_face_identification_rules() -> str:
    policy_label = segment_policy_summary()
    return (
        f"## 逐窗面孔识别（硬性 · {policy_label}）\n"
        "- 每条 `episodic_segments` = 一个预计算 time_range → **必须**基于该窗口画面独立填写 `involved_characters`\n"
        "- 窗口内有多张清晰人脸 → **逐脸**对照定妆照（拼图按标注序号+姓名逐张对照），匹配者全部写入数组\n"
        f"- 仅写**该窗口内确实可见**且相似度 **≥{FRAME_FACE_MATCH_SIMILARITY_MIN_PERCENT}%** 匹配成功的人物；人物离场/未入画/无法确认则不要写入\n"
        "- 禁止把上一窗或下一窗的人物照抄到本窗；**禁止**凭对白、剧情、警服等猜不在画面中的人"
    )


def build_within_chunk_grid_rules() -> str:
    """自适应场景窗：段内独立描述。"""
    policy_label = segment_policy_summary()
    return (
        f"## 场景时间窗填写说明（{policy_label}）\n"
        "- 每条 `episodic_segments` 对应**一个预计算 time_range**，按该窗口内**实际画面**独立描述\n"
        "- **同一段上传视频内**，场景/人物可以随切镜自然变化；**切镜边界格**在 `key_events` 须写「场景切换至…」\n"
        "- `narration` / `environment_description` 写「延续上段」时，仅表示同场景画面未大变，**仍须与当前窗口人物一致**\n"
        "- **禁止**为了「承接」而编造与画面无关的人物或事件"
    )


def build_upload_chunk_boundary_rules() -> str:
    """约 300s 上传分段边界：与上一段末尾衔接。"""
    return (
        "## 上传分段边界衔接（仅本段**开头几窗**需注意）\n"
        "- 长片按约 **5 分钟/300 秒** 切成多段分别分析；**本段前 1–2 个 time_range** 须与上一段末尾画面衔接\n"
        "- 若上一段末尾与当前开头**同一连续场景**（未切镜），人物/环境应一致；若已换场，首窗须写「场景切换至…」\n"
        "- **段内其余时间窗**按各自窗口独立理解，**不要**整段机械沿用上一窗人物\n"
        "- 后处理会在边界处**自动校正**「同场景却误换人且文本未引入新人物」的 involved_characters"
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
        f"首段已提供 **{work}** 人物定妆照；本段写规范姓名仍须**对照视频可见面孔**与头像匹配（{FRAME_FACE_MATCH_SIMILARITY_HINT}）。",
    ]
    if relationship_diagram_attached:
        lines.append("- 关系图仅作关系/身份**校正**，不可凭关系图猜人。")
    if names:
        lines.append(
            f"- 定妆照人物（{'、'.join(names)}）须逐脸对照、{FRAME_FACE_MATCH_SIMILARITY_HINT} 后才可写规范名；"
            "硬字幕/对白**不得**猜人。"
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


def build_previous_chunk_tail_context(
    previous_partial: dict[str, Any] | None,
    *,
    tail_count: int = 3,
) -> str:
    """上一上传分段末尾若干格，供本段开头衔接参考。"""
    if not previous_partial:
        return ""
    segments = previous_partial.get("episodic_segments") or []
    if not segments:
        return ""
    tail = segments[-tail_count:]
    lines = [
        "## 上一上传分段末尾（仅供本段开头衔接，勿整段照搬）",
        "| time_range | 人物 | 关键事件 | 环境 |",
        "|---|---|---|---|",
    ]
    for segment in tail:
        if not isinstance(segment, dict):
            continue
        time_range = str(segment.get("time_range") or "").strip()
        chars = "、".join(segment.get("involved_characters") or []) or "—"
        events = str(segment.get("key_events") or "").strip()[:36]
        env = str(segment.get("environment_description") or "").strip()[:28]
        lines.append(f"| `{time_range}` | {chars} | {events} | {env} |")
    return "\n".join(lines)


def build_episodic_segment_continuity_rules() -> str:
    """兼容旧引用：段内格规则 + 不再做逐格强制承接。"""
    return build_within_chunk_grid_rules()


def build_video_episode_analysis_prompt(
    *,
    drama_title: str = "",
    video_duration_seconds: float | None = None,
    segment_schedule_block: str = "",
    character_naming_block: str = "",
) -> str:
    title = _resolve_drama_title(drama_title)
    policy_label = segment_policy_summary()
    sample_end = _format_hms_timestamp(float(SEGMENT_MAX_SECONDS))
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
      "time_range": "00:00:00-{sample_end}",
      "key_events": "用一句话描述该时间窗内发生的关键事件。",
      "narration": "第三人称纪录片旁白，15-35字，描述该窗口内画面与对话，可直接用于后期配音。",
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
- `episodic_segments` 必须逐条对应下方预计算时间窗口（{policy_label}），不得少条、不得合并。
- 每条 `episodic_segments` 除 `title` / `key_events` / `involved_characters` 外，还必须填写：
  - `narration`：第三人称纪录片旁白，基于该 time_range 画面与对话撰写，15-35 字，语气客观、可用于配音；
  - `environment_description`：该片段的场景环境（地点、室内外、时间感、光线、氛围、关键布景），15-40 字。
- 同场景延续时 `narration` 可写「画面延续上段」，`environment_description` 可写「环境延续上段」。
- **切镜/换场边界窗**在 `key_events` 须写「场景切换至…」并重写 `environment_description`。
- 只返回 JSON，不要 markdown 代码块或额外说明。

{build_within_chunk_grid_rules()}

{build_per_grid_face_identification_rules()}

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
    previous_chunk_partial: dict[str, Any] | None = None,
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
    tail_context = (
        build_previous_chunk_tail_context(previous_chunk_partial)
        if chunk_index > 0
        else ""
    )
    boundary_rules = build_upload_chunk_boundary_rules() if chunk_index > 0 else ""
    extra_blocks = "\n\n".join(
        block for block in (boundary_rules, tail_context) if block
    )
    return (
        f"{base}\n\n"
        f"## 分段说明\n"
        f"这是全片第 {chunk_index + 1}/{total_chunks} 段，本段在整片时间轴上的起始时刻为 **{offset_label}**。\n"
        f"`episodic_segments.time_range` 与 `important_dialogues.timestamp` 必须使用**全片绝对时间**（HH:MM:SS），"
        f"不要从 00:00:00 重新计时。\n"
        f"`overall_summary` 仅概括本段剧情（80字内）；`key_conflict` 仅描述本段核心冲突；"
        f"`episodic_segments` 仅填写本段预计算时间窗，条数与窗口列表一致。"
        + (f"\n\n{extra_blocks}" if extra_blocks else "")
    )
