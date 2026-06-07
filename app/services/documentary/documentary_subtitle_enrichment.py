#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""逐帧解说：字幕分析与抽帧分析结合。"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from loguru import logger

from app.config import config
from app.services.documentary.documentary_settings import (
    compute_ost1_segment_bounds,
    get_documentary_settings,
    is_compact_documentary_settings,
    is_fazu2_compact_settings,
    resolve_append_custom_prompt,
)
from app.services.llm.migration_adapter import _run_async_safely
from app.services.llm.unified_service import UnifiedLLMService
from app.services.srt_utils import SrtEntry, parse_srt
from app.utils import utils

_TRAILING_CLAUSE_PUNCT = re.compile(r"[，。！？、；]+$")
def clean_subtitle_punctuation(text: str) -> str:
    """合并多句字幕后去掉「，；」「。；」等重复标点。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"([，。！？、])；+", "；", cleaned)
    cleaned = re.sub(r"；{2,}", "；", cleaned)
    cleaned = re.sub(r"；+$", "", cleaned).strip()
    return cleaned


def join_subtitle_texts(parts: list[str] | tuple[str, ...]) -> str:
    """多句字幕用分号连接，并去掉各句尾标点避免「，；」叠用。"""
    snippets: list[str] = []
    for part in parts:
        text = str(part or "").strip()
        if not text:
            continue
        snippets.append(_TRAILING_CLAUSE_PUNCT.sub("", text))
    if not snippets:
        return ""
    return clean_subtitle_punctuation("；".join(snippets))


def resolve_segment_subtitle_text(segment: dict[str, Any]) -> str:
    """从 segment 的 subtitle_entries（优先）或 subtitle 得到清洗后的合并字幕。"""
    if not isinstance(segment, dict):
        return ""
    entries = segment.get("subtitle_entries")
    if isinstance(entries, list) and entries:
        merged = join_subtitle_texts(
            str(item.get("text") or "") for item in entries if isinstance(item, dict)
        )
        if merged:
            return merged
    return clean_subtitle_punctuation(str(segment.get("subtitle") or "").strip())


def resolve_segment_time_range(segment: dict[str, Any]) -> str:
    """
    剪辑用时间范围：优先按 subtitle_entries 首尾对位，便于按字幕切分成片。
    无字幕条目时回退 segment.time_range，再回退 timestamp。
    """
    if not isinstance(segment, dict):
        return ""

    entries = segment.get("subtitle_entries")
    if isinstance(entries, list) and entries:
        starts: list[str] = []
        ends: list[str] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            start = str(item.get("start") or "").strip()
            end = str(item.get("end") or "").strip()
            if start:
                starts.append(start)
            if end:
                ends.append(end)
        if starts and ends:
            return f"{starts[0]}-{ends[-1]}"

    explicit = str(segment.get("time_range") or "").strip()
    if explicit:
        return explicit
    return str(segment.get("timestamp") or "").strip()


def _ms_to_hhmmss(ms: int) -> str:
    return utils.seconds_to_time(ms / 1000.0).replace(".", ",")


def _timestamp_to_ms(timestamp: str) -> int:
    text = (timestamp or "").strip()
    try:
        if "," in text:
            time_part, ms_part = text.split(",", 1)
            milliseconds = int(ms_part)
        else:
            time_part = text
            milliseconds = 0
        parts = [int(part) for part in time_part.split(":") if part]
        while len(parts) < 3:
            parts.insert(0, 0)
        hours, minutes, seconds = parts[-3], parts[-2], parts[-1]
        return ((hours * 3600 + minutes * 60 + seconds) * 1000) + milliseconds
    except Exception:
        return 0


def parse_timestamp_range_ms(time_range: str) -> tuple[int, int]:
    text = (time_range or "").strip()
    if "-" not in text:
        ms = _timestamp_to_ms(text)
        return ms, ms
    start_text, end_text = text.split("-", 1)
    start_ms = _timestamp_to_ms(start_text.strip())
    end_ms = _timestamp_to_ms(end_text.strip())
    if end_ms < start_ms:
        start_ms, end_ms = end_ms, start_ms
    return start_ms, end_ms


def truncate_subtitle_content(subtitle_content: str, max_chars: int) -> str:
    text = (subtitle_content or "").strip()
    if not text or len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n…（字幕已截断）"


def subtitle_excerpt_for_time_range(
    subtitle_content: str,
    time_range: str,
    *,
    pad_ms: int = 5000,
    max_lines: int = 8,
) -> str:
    """取某时间段（含前后缓冲）内的字幕对白摘要。"""
    entries = parse_srt(subtitle_content or "")
    if not entries:
        return ""

    start_ms, end_ms = parse_timestamp_range_ms(time_range)
    window_start = max(0, start_ms - pad_ms)
    window_end = end_ms + pad_ms

    lines: list[str] = []
    for entry in entries:
        if entry.end_ms < window_start or entry.start_ms > window_end:
            continue
        text = (entry.text or "").strip().replace("\n", " ")
        if not text:
            continue
        lines.append(f"{_ms_to_hhmmss(entry.start_ms)} {text}")
        if len(lines) >= max_lines:
            break

    if not lines:
        return ""
    excerpt = "；".join(lines)
    return excerpt[:400] + ("…" if len(excerpt) > 400 else "")


def _normalize_subtitle_compare_text(text: str) -> str:
    cleaned = (text or "").strip().replace("\n", " ")
    for ch in "，,。．！？、；：""''…—- ":
        cleaned = cleaned.replace(ch, "")
    return cleaned


def merge_subtitle_text_with_burned_in(srt_text: str, burned_text: str) -> tuple[str, str]:
    """
    字幕文字冲突时以画面硬字幕为准，时间仍用 SRT。
    返回 (最终文本, text_source)；source 为 srt | burned_in_corrected。
    """
    srt = (srt_text or "").strip().replace("\n", " ")
    burned = (burned_text or "").strip().replace("\n", " ")
    if not srt:
        return burned, "burned_in_only" if burned else "srt"
    if not burned:
        return srt, "srt"
    if _normalize_subtitle_compare_text(srt) == _normalize_subtitle_compare_text(burned):
        return srt, "srt"
    return burned, "burned_in_corrected"


def find_srt_entry_at_timestamp(
    entries: list[SrtEntry],
    timestamp: str,
    *,
    pad_ms: int = 800,
) -> SrtEntry | None:
    if not entries:
        return None

    ts_ms = _timestamp_to_ms(timestamp)
    for entry in entries:
        if entry.start_ms <= ts_ms <= entry.end_ms:
            return entry

    nearest: SrtEntry | None = None
    nearest_distance = pad_ms + 1
    for entry in entries:
        distance = min(abs(entry.start_ms - ts_ms), abs(entry.end_ms - ts_ms))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest = entry

    if nearest is None or nearest_distance > pad_ms:
        return None
    return nearest


def subtitle_at_timestamp(
    subtitle_content: str,
    timestamp: str,
    *,
    pad_ms: int = 800,
) -> str:
    """取某一时间点对应的 SRT 对白（优先命中区间，否则取 pad 内最近一条）。"""
    entries = parse_srt(subtitle_content or "")
    entry = find_srt_entry_at_timestamp(entries, timestamp, pad_ms=pad_ms)
    if entry is None:
        return ""
    return (entry.text or "").strip().replace("\n", " ")


def subtitle_entries_for_time_range(
    subtitle_content: str,
    time_range: str,
    *,
    pad_ms: int = 0,
    max_entries: int = 0,
    text_overrides_by_start_ms: dict[int, str] | None = None,
) -> list[dict[str, str]]:
    """取时间范围内的 SRT 条目，含起止时间与对白文本。"""
    entries = parse_srt(subtitle_content or "")
    if not entries:
        return []

    start_ms, end_ms = parse_timestamp_range_ms(time_range)
    window_start = max(0, start_ms - pad_ms)
    window_end = end_ms + pad_ms
    overrides = text_overrides_by_start_ms or {}

    result: list[dict[str, str]] = []
    for entry in entries:
        if entry.end_ms < window_start or entry.start_ms > window_end:
            continue
        srt_text = (entry.text or "").strip().replace("\n", " ")
        if not srt_text and entry.start_ms not in overrides:
            continue
        text, text_source = merge_subtitle_text_with_burned_in(
            srt_text,
            overrides.get(entry.start_ms, ""),
        )
        if not text:
            continue
        payload: dict[str, str] = {
            "start": _ms_to_hhmmss(entry.start_ms),
            "end": _ms_to_hhmmss(entry.end_ms),
            "text": text,
        }
        if text_source == "burned_in_corrected":
            payload["text_source"] = text_source
        result.append(payload)
        if max_entries > 0 and len(result) >= max_entries:
            break
    return result


def build_burned_in_overrides_for_entries(
    observations: list[dict[str, Any]],
    entries: list[SrtEntry],
) -> dict[int, str]:
    """根据逐帧硬字幕，为重叠的 SRT 条目收集文字修正（以画面为准）。"""
    overrides: dict[int, str] = {}
    if not observations or not entries:
        return overrides

    for observation in observations:
        if not isinstance(observation, dict):
            continue
        burned = str(observation.get("burned_in_subtitle") or "").strip()
        if not burned:
            continue
        ts_ms = _timestamp_to_ms(str(observation.get("timestamp") or ""))
        if ts_ms <= 0:
            continue
        for entry in entries:
            if entry.start_ms <= ts_ms <= entry.end_ms:
                srt_text = (entry.text or "").strip().replace("\n", " ")
                if _normalize_subtitle_compare_text(srt_text) != _normalize_subtitle_compare_text(burned):
                    overrides[entry.start_ms] = burned
                break
    return overrides


def apply_subtitle_fields_to_observation(
    observation: dict[str, Any],
    subtitle_content: str,
    *,
    pad_ms: int = 800,
) -> None:
    """为单帧写入 subtitle / subtitle_start / subtitle_end（抽帧阶段调用）。"""
    if not isinstance(observation, dict):
        return

    timestamp = str(observation.get("timestamp") or "").strip()
    burned = str(observation.get("burned_in_subtitle") or "").strip()
    entries = parse_srt(subtitle_content or "")

    entry = find_srt_entry_at_timestamp(entries, timestamp, pad_ms=pad_ms) if timestamp else None
    if entry is not None:
        srt_text = (entry.text or "").strip().replace("\n", " ")
        text, text_source = merge_subtitle_text_with_burned_in(srt_text, burned)
        if text:
            observation["subtitle"] = text
            observation["subtitle_start"] = _ms_to_hhmmss(entry.start_ms)
            observation["subtitle_end"] = _ms_to_hhmmss(entry.end_ms)
            observation["subtitle_text_source"] = text_source
        return

    if burned:
        observation["subtitle"] = burned
        observation["subtitle_start"] = ""
        observation["subtitle_end"] = ""
        observation["subtitle_text_source"] = "burned_in_only"


def apply_subtitle_fields_to_segment(
    segment: dict[str, Any],
    subtitle_content: str,
    *,
    observations: list[dict[str, Any]] | None = None,
    pad_ms: int = 500,
    max_entries: int = 20,
) -> None:
    """为场景段写入 subtitle_entries 与合并 subtitle 文本。"""
    if not isinstance(segment, dict):
        return

    time_range = str(segment.get("timestamp") or segment.get("time_range") or "").strip()
    if not time_range:
        return

    all_entries = parse_srt(subtitle_content or "")
    start_ms, end_ms = parse_timestamp_range_ms(time_range)
    window_start = max(0, start_ms - pad_ms)
    window_end = end_ms + pad_ms
    window_entries = [
        entry
        for entry in all_entries
        if not (entry.end_ms < window_start or entry.start_ms > window_end)
    ]
    overrides = build_burned_in_overrides_for_entries(observations or [], window_entries)
    entries = subtitle_entries_for_time_range(
        subtitle_content,
        time_range,
        pad_ms=pad_ms,
        max_entries=max_entries,
        text_overrides_by_start_ms=overrides,
    )
    if not entries:
        return

    segment["subtitle_entries"] = entries
    segment["subtitle"] = join_subtitle_texts(str(item.get("text") or "") for item in entries)


def attach_subtitles_to_frame_analysis_artifact(
    artifact: dict[str, Any],
    subtitle_content: str,
    *,
    settings: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """抽帧完成后：按 SRT 时间区间补全字幕字段；文字冲突以画面硬字幕为准。"""
    text = (subtitle_content or "").strip()
    if not text or not isinstance(artifact, dict):
        return artifact

    cfg = settings or get_documentary_settings()
    pad_sec = int(cfg.get("subtitle_batch_pad_sec", 5))
    pad_ms = max(0, pad_sec * 1000)
    frame_pad_ms = max(800, int((cfg.get("frame_interval_input") or 1) * 1000))

    observations_by_batch: dict[int, list[dict[str, Any]]] = {}
    for observation in artifact.get("frame_observations") or []:
        if isinstance(observation, dict):
            batch_index = int(observation.get("batch_index", 0))
            observations_by_batch.setdefault(batch_index, []).append(observation)

    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        batch_index = int(batch.get("batch_index", 0))
        batch_observations = list(batch.get("frame_observations") or batch.get("observations") or [])
        if not batch_observations:
            batch_observations = observations_by_batch.get(batch_index, [])

        time_range = str(batch.get("time_range") or "").strip()
        if time_range:
            batch["subtitle_entries"] = subtitle_entries_for_time_range(
                text,
                time_range,
                pad_ms=pad_ms,
                max_entries=30,
                text_overrides_by_start_ms=build_burned_in_overrides_for_entries(
                    batch_observations,
                    parse_srt(text),
                ),
            )
            batch["subtitle_excerpt"] = subtitle_excerpt_for_time_range(text, time_range, pad_ms=pad_ms)
            if batch["subtitle_entries"]:
                batch["subtitle"] = join_subtitle_texts(
                    str(item.get("text") or "") for item in batch["subtitle_entries"]
                )

        for segment in batch.get("scene_segments") or []:
            apply_subtitle_fields_to_segment(
                segment,
                text,
                observations=batch_observations,
                pad_ms=500,
            )
        for observation in batch_observations:
            apply_subtitle_fields_to_observation(observation, text, pad_ms=frame_pad_ms)

    for segment in artifact.get("scene_segments") or []:
        if not isinstance(segment, dict):
            continue
        batch_index = int(segment.get("batch_index", 0))
        apply_subtitle_fields_to_segment(
            segment,
            text,
            observations=observations_by_batch.get(batch_index, []),
            pad_ms=500,
        )

    for observation in artifact.get("frame_observations") or []:
        apply_subtitle_fields_to_observation(observation, text, pad_ms=frame_pad_ms)

    artifact["subtitle_attached"] = True
    return artifact


def build_subtitle_cross_validation_instructions(
    settings: Optional[dict[str, Any]] = None,
) -> str:
    cfg = settings or get_documentary_settings()
    lines = [
        "## 素材优先级（有字幕时必须遵守）",
        "",
        "**生成脚本以字幕为主，抽帧为辅：**",
        "- **字幕（主）**：剧情主线、`narration` 内容、`original_line`、人名、**所有 `timestamp`**",
        "- **抽帧（辅）**：仅用于 `picture` 画面描述，以及截取时间在字幕区间内的**对齐参考**",
        "",
        "## 字幕 × 抽帧 对照规则",
        "",
        "- **对白内容、剧情、人名、台词时间戳** → 以 `<subtitles>` 为准",
        "- **`picture` 画面描述** → 参考抽帧（人物表情、动作、场景氛围）",
        "- **昼夜、天气、光线** → `picture` 以抽帧为准；`narration` 剧情仍以字幕为准",
        "- 两者冲突时：**剧情/台词/时间戳以字幕为准**，画面描述以抽帧为准，勿互相覆盖",
        "- 写 `timestamp` 必须从字幕复制；可参考抽帧在字幕区间内对齐起止，严禁重叠",
    ]
    if is_fazu2_compact_settings(cfg):
        min_ost1, max_ost1 = compute_ost1_segment_bounds(settings=cfg)
        lines.extend(
            [
                "",
                "### 精剪 · 罚罪2 V2",
                "- 生成 JSON 前须完成「字幕×抽帧 对照分析」蓝图，并严格遵循",
                "- 标出本集**最炸裂名场面**（供第 1 段纯原声开头）：金句+时间戳+动作",
                "- 列出正叙剧情情节点（按时间顺序）；多数对白写入 OST=0 解说",
                f"- 从字幕列出 **{min_ost1}–{max_ost1}** 个 OST=1（约 50%）；timestamp 覆盖整句对白，播完再接解说",
                "- 标出收尾情节与道别；人名用胡小跃/秦枫/伟业/罗博等（禁止胡小月/小月）",
            ]
        )
    elif is_compact_documentary_settings(cfg):
        lines.extend(
            [
                "",
                "### 精剪 · 原声对位",
                "- 标注可作 OST=1 的字幕对白 moment 与时间戳",
            ]
        )
    else:
        lines.extend(
            [
                "- 标记 `[高光原声]` 或字幕中有力对白 + 画面张力强的 moment，优先 OST=1",
            ]
        )
    return "\n".join(lines)


def summarize_frame_markdown(frame_markdown: str, max_chars: int) -> str:
    text = (frame_markdown or "").strip()
    if not text:
        return "（无抽帧描述）"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n…（抽帧摘要已截断）"


def analyze_subtitle_with_frames(
    *,
    subtitle_content: str,
    frame_markdown: str,
    video_theme: str = "",
    append_custom_prompt: str = "",
    progress_callback: Optional[Callable[[str], None]] = None,
    documentary_settings: Optional[dict[str, Any]] = None,
    analysis_style: str = "documentary",
) -> str:
    """文本模型：结合字幕与抽帧摘要做对照分析。"""
    if not (subtitle_content or "").strip():
        return ""

    cfg = documentary_settings or get_documentary_settings()
    subtitle_excerpt = truncate_subtitle_content(
        subtitle_content,
        int(cfg.get("subtitle_analysis_max_subtitle_chars", 12000) or 12000),
    )
    frame_summary = summarize_frame_markdown(
        frame_markdown,
        int(cfg.get("subtitle_analysis_max_frame_chars", 8000)),
    )
    theme = (video_theme or "").strip() or "本视频"
    min_chars = max(200, int(cfg.get("subtitle_analysis_min_chars", 500) or 500))
    max_tokens = max(1024, int(cfg.get("subtitle_analysis_max_tokens", 4096) or 4096))

    logger.info(
        f"字幕×抽帧对照分析输入：字幕 {len(subtitle_excerpt)} 字，"
        f"抽帧摘要 {len(frame_summary)} 字，要求输出 ≥{min_chars} 字"
    )
    if len(subtitle_excerpt) < 100:
        logger.warning(
            f"字幕内容过短（{len(subtitle_excerpt)} 字），"
            "请确认已在页面「确认使用」完整 SRT 文件"
        )

    if progress_callback:
        progress_callback("正在分析字幕并对照抽帧画面...")

    compact = is_compact_documentary_settings(cfg)
    append_text = resolve_append_custom_prompt(append_custom_prompt, cfg)
    append_block = ""
    if append_text:
        append_block = f"""## 用户追加要求（最高优先级 · 策划蓝图须优先落实）
{append_text}
若已指定开头高潮/爆燃场面，「开头高潮方案」必须写该场面，不得另选。

"""

    if is_fazu2_compact_settings(cfg):
        min_ost1, max_ost1 = compute_ost1_segment_bounds(settings=cfg)
        prompt = f"""{append_block}你是电视剧「高潮前置型」解说策划。请**以完整字幕为第一依据**，对照抽帧摘要补充画面信息，输出一份**脚本生成蓝图**（800–1200 字，结构化 Markdown，不要 JSON）。

作品/主题：{theme}

必须按以下标题逐项填写（缺一不可）：

## 本集识别
- 作品名；是否**全剧第 1 集**（决定转场句用固定句还是自拟）

## 主要人物表
- 姓名 + 身份/关系 + **性别（据字幕/抽帧）**（如胡小跃-刑警-男、伟业-局长-男）
- 标注各角色性别，供后续 narration/picture 对照画面，避免张冠李戴

## 开头高潮方案（→ JSON 第 1 段 OST=1）
- **默认优先**：胡小跃**楼顶跳楼牺牲**（夜色楼顶纵身跃下），金句优先「天就快亮了。」+ 字幕精确时间戳
- **禁止**用中段台词（如「你跟我说这是狗贩子」）顶替跳楼作第 1 段；此类放正叙 OST=1
- 动作描述（结合抽帧；昼夜/光线须与画面一致）
- **转场句建议**：第2段用「宝子们」+「故事，得从头讲起。」
- 若用户追加要求已指定开头名场面，本节**只写该场面**

## 正叙时间线（→ JSON 主体 OST=0，按片头时间顺序）
- 至少 **8–12 个情节点**，每点含：时间段、人名、事件摘要
- 标注哪些点适合加小钩子（「您猜怎么着？」等）

## OST=1 金句清单（{min_ost1}–{max_ost1} 条，时间戳必须来自字幕）
- 每条：说话人姓名 + 台词原文 + **精确时间戳** + 用途（开头/中段爆点/高潮复现）

## 高潮复现方案（→ JSON 中后段 1 个 OST=1）
- 复用哪句金句、时间戳、加强情绪的方向

## 高潮之后 + 下集钩子（→ JSON 末段）
- 本集高潮之后的关键情节（2–3 点）
- 下集悬念/预告（1–2 句）

## 声画对位注意
- 字幕与画面不一致处、需以字幕为准的台词、需以画面为准的动作
- **环境对位**：逐段标注昼夜/光线/天气（如「天台·白天·日光」）；标出易写错的镜头（画面白天勿写夜风刺骨）
- 正叙开场勿写「第几集」，用「宝子们，一起来看《作品名》。」即可

禁忌：不要警员1/说话人1；不要镜头/导演分析；不要然后/接着；不要编造时间戳；不要臆造与抽帧不符的环境氛围
"""
        system_prompt = (
            "你是影视解说策划，擅长高潮前置型短视频脚本。"
            "输出结构化策划蓝图，时间戳必须来自字幕原文，不要 JSON。"
        )
    elif compact:
        prompt = f"""请以**原始字幕为主**、抽帧画面摘要为辅，输出供精剪脚本使用的对照分析（350–550 字）。

作品/主题：{theme}

请包含剧情主线、建议 OST=1 时刻（带时间戳）、写脚本注意点。
"""
        system_prompt = (
            "你是资深影视解说分析师，输出简洁可执行，不要 JSON。"
        )
    elif (analysis_style or "").strip().lower() == "short_drama":
        prompt = f"""你是短剧解说策划。请以**原始字幕为主**、抽帧画面摘要为辅，输出供写 JSON 脚本使用的对照分析（600–900 字，结构化 Markdown，不要 JSON）。

作品/主题：{theme}

必须包含（缺一不可）：
1. **主要人物表**（姓名、关系、性别须与画面对照）
2. **开头高潮方案**（→ JSON 第 1 段 OST=1，含精确时间戳）
3. **正叙时间线**：至少 **10–15 个情节点**，每点含时间段 + 事件摘要（这是后续切段的骨架，禁止只写 3–5 个大段概括）
4. **建议保留原声 OST=1**：至少 **8–12 条**，每条含台词原文 + 精确时间戳 + 用途
5. **声画对位注意**（昼夜/光线/表情以抽帧为准，台词以字幕为准；各情节点 time_range 以 subtitle_entries 结束时间定界）

重要：本分析用于指导**细粒度切段**，不是让你把整集压成少量大段；后续 JSON items 须覆盖上述时间线上的多个节点。
"""
        system_prompt = (
            "你是短剧解说策划，擅长高潮前置与快节奏细切。"
            "输出结构化策划蓝图，时间戳必须来自字幕原文，不要 JSON。"
        )
        min_chars = max(min_chars, 500)
    else:
        prompt = f"""你是一位拥有 30 年经验的资深解说员，请以**原始字幕为主**、抽帧画面摘要为辅，输出供写脚本使用的对照分析（300–500 字）。

作品/主题：{theme}

请包含：
1. 剧情主线、关键人物与冲突（结合字幕与画面）
2. 字幕对白与画面是否一致或有反差（可点出「声画对位」moment）
3. **建议保留原声 OST=1** 的时刻（标注大致时间戳，来自字幕真实范围）
4. 写解说时的注意点（哪些地方以字幕为准、哪些以画面为准）
"""
        system_prompt = (
            "你是资深影视解说分析师，擅长字幕与画面交叉验证。"
            "输出简洁、可执行，不要 JSON。"
        )

    prompt = prompt + f"""

<video_frame_summary>
{frame_summary}
</video_frame_summary>

<subtitles>
{subtitle_excerpt}
</subtitles>
"""

    text_provider = config.app.get("text_llm_provider", "openai").lower()
    api_key = config.app.get(f"text_{text_provider}_api_key")
    model = config.app.get(f"text_{text_provider}_model_name")
    base_url = config.app.get(f"text_{text_provider}_base_url", "")
    if not api_key or not model:
        logger.warning("未配置文本模型，跳过字幕×画面对照分析")
        return ""

    analysis_temperature = 0.35 if is_fazu2_compact_settings(cfg) else 0.7
    fazu2 = is_fazu2_compact_settings(cfg)
    max_attempts = 3 if fazu2 else 2
    current_prompt = prompt
    analysis = ""

    try:
        for attempt in range(max_attempts):
            result = _run_async_safely(
                UnifiedLLMService.generate_text,
                prompt=current_prompt,
                system_prompt=system_prompt,
                provider=text_provider,
                temperature=analysis_temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                api_base=base_url,
                for_script=True,
            )
            analysis = (result or "").strip()
            if len(analysis) >= min_chars or (not fazu2 and len(analysis) >= 200):
                logger.info(f"字幕×抽帧对照分析完成，约 {len(analysis)} 字")
                return analysis

            preview = analysis[:120].replace("\n", " ")
            logger.warning(
                f"字幕×抽帧对照分析过短（{len(analysis)} 字），"
                f"预览: {preview!r}，重试 {attempt + 1}/{max_attempts}"
            )
            if attempt + 1 >= max_attempts:
                break
            current_prompt = (
                prompt
                + f"\n\n## 重试要求（第 {attempt + 2} 次）\n"
                f"上次输出仅 {len(analysis)} 字，无效。"
                f"必须输出完整策划蓝图，**不少于 {min_chars} 字**，"
                "按上方全部标题逐项填写，禁止只回复一句话或空泛总结。"
            )

        if analysis:
            logger.warning(f"字幕×抽帧对照分析仍未达标，最终仅 {len(analysis)} 字")
        return analysis
    except Exception as exc:
        logger.warning(f"字幕×抽帧对照分析失败，将仅注入原始字幕: {exc}")
        return ""


def build_subtitle_narration_sections(
    *,
    subtitle_content: str,
    subtitle_analysis: str = "",
    settings: Optional[dict[str, Any]] = None,
) -> list[str]:
    """构建写入解说 prompt 的字幕相关区块。"""
    cfg = settings or get_documentary_settings()
    if not cfg.get("enable_subtitle_enrichment", True):
        return []
    if not (subtitle_content or "").strip():
        return []

    sections: list[str] = [
        build_subtitle_cross_validation_instructions(cfg).strip()
    ]

    excerpt = truncate_subtitle_content(
        subtitle_content,
        int(cfg.get("subtitle_max_chars", 15000)),
    )
    sections.append(
        "## 原始字幕（脚本生成第一依据 · timestamp / narration / 台词均以此为准）\n"
        f"<subtitles>\n{excerpt}\n</subtitles>"
    )

    if subtitle_analysis.strip():
        if is_fazu2_compact_settings(cfg):
            sections.append(
                "## 字幕×抽帧 对照分析（策划蓝图 · 剧情与时间戳仍以字幕为准）\n"
                + subtitle_analysis.strip()
            )
        else:
            sections.append(
                "## 字幕×抽帧 对照分析（剧情与时间戳以字幕为准）\n"
                + subtitle_analysis.strip()
            )

    return sections
