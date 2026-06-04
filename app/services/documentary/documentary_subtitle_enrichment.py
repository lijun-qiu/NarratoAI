#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""逐帧解说：字幕分析与抽帧分析结合。"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from loguru import logger

from app.config import config
from app.services.documentary.documentary_settings import (
    get_documentary_settings,
    is_compact_documentary_settings,
    is_fazu2_compact_settings,
)
from app.services.llm.migration_adapter import _run_async_safely
from app.services.llm.unified_service import UnifiedLLMService
from app.services.srt_utils import parse_srt
from app.utils import utils


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


def build_subtitle_cross_validation_instructions(
    settings: Optional[dict[str, Any]] = None,
) -> str:
    cfg = settings or get_documentary_settings()
    lines = [
        "## 字幕 × 抽帧 对照规则（有字幕时必须遵守）",
        "",
        "- **对白内容、台词时间戳** → 以 `<subtitles>` / 字幕分析为准",
        "- **人物表情、动作、场景氛围** → 以 `<video_frame_description>` 抽帧描述为准",
        "- 两者冲突时：台词不要凭画面猜，画面不要凭字幕脑补",
        "- 写 `timestamp` 时优先采用字幕/抽帧已有时间范围，严禁重叠",
    ]
    if is_fazu2_compact_settings(cfg):
        lines.extend(
            [
                "",
                "### 精剪 · 故事讲述型",
                "- 列出本集剧情情节点（按时间顺序）；多数对白写入 OST=0 解说叙述",
                "- 从字幕列出 ≤6 个 OST=1 金句候选：精确时间戳 + 台词原文",
                "- 禁止规划相邻 OST=1；禁止镜头/导演分析；脚本须用**人名**勿用警员1/说话人1",
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
    progress_callback: Optional[Callable[[str], None]] = None,
    documentary_settings: Optional[dict[str, Any]] = None,
) -> str:
    """文本模型：结合字幕与抽帧摘要做对照分析。"""
    if not (subtitle_content or "").strip():
        return ""

    cfg = documentary_settings or get_documentary_settings()
    subtitle_excerpt = truncate_subtitle_content(
        subtitle_content,
        int(cfg.get("subtitle_max_chars", 15000)),
    )
    frame_summary = summarize_frame_markdown(
        frame_markdown,
        int(cfg.get("subtitle_analysis_max_frame_chars", 8000)),
    )
    theme = (video_theme or "").strip() or "本视频"

    if progress_callback:
        progress_callback("正在分析字幕并对照抽帧画面...")

    compact = is_compact_documentary_settings(cfg)
    if is_fazu2_compact_settings(cfg):
        prompt = f"""你是电视剧「故事讲述型」解说策划，请结合**抽帧画面**与**原始字幕**，输出供写脚本用的对照分析（350–550 字）。

作品/主题：{theme}

请包含：
1. **主要人物表**：姓名 + 身份/关系（如胡小月-女警、秦枫-师弟、罗博-金鼎集团）
2. **本集剧情主线**（按时间顺序 5–8 个情节点，口语化，写清**人名**）
3. **OST=1 金句候选（≤6 条）**：台词原文 + **字幕精确时间戳** + 说话人**姓名**
4. **可写入解说的关键对白**（10–15 条）：说话人姓名 + 台词摘要
5. 禁忌：不要警员1/说话人1等编号；不要镜头/导演分析；不要然后/接着
"""
        system_prompt = (
            "你是影视解说策划，擅长故事讲述型短视频脚本。"
            "时间戳必须来自字幕，不要 JSON。"
        )
    elif compact:
        prompt = f"""请结合**抽帧画面摘要**与**原始字幕**，输出供精剪脚本使用的对照分析（350–550 字）。

作品/主题：{theme}

请包含剧情主线、建议 OST=1 时刻（带时间戳）、写脚本注意点。
"""
        system_prompt = (
            "你是资深影视解说分析师，输出简洁可执行，不要 JSON。"
        )
    else:
        prompt = f"""你是一位拥有 30 年经验的资深解说员，请结合**抽帧画面摘要**与**原始字幕**，输出供写脚本使用的对照分析（300–500 字）。

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

    try:
        result = _run_async_safely(
            UnifiedLLMService.generate_text,
            prompt=prompt,
            system_prompt=system_prompt,
            provider=text_provider,
            temperature=0.7,
            api_key=api_key,
            api_base=base_url,
            for_script=True,
        )
        analysis = (result or "").strip()
        if analysis:
            logger.info(f"字幕×抽帧对照分析完成，约 {len(analysis)} 字")
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

    if subtitle_analysis.strip():
        sections.append(f"## 字幕×抽帧 对照分析\n{subtitle_analysis.strip()}")

    excerpt = truncate_subtitle_content(
        subtitle_content,
        int(cfg.get("subtitle_max_chars", 15000)),
    )
    sections.append(f"## 原始字幕\n<subtitles>\n{excerpt}\n</subtitles>")

    return sections
