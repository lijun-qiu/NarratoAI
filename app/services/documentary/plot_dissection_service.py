#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""剧情解剖：对照人物关系/剧情参考，校正视频分析结果中的人名并输出完善脚本。"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from loguru import logger

from app.config.llm_gateway_router import resolve_llm_credentials
from app.services.documentary.character_relationship import (
    build_character_relationship_prompt_section,
)
from app.services.documentary.plot_reference import build_plot_reference_prompt_section
from app.services.documentary.video_episode_analysis import (
    build_video_episode_analysis_markdown,
    load_video_episode_analysis_artifact,
)
from app.services.llm.migration_adapter import _run_async_safely
from app.services.llm.unified_service import UnifiedLLMService

PLOT_DISSECTION_MODEL = "deepseek-v4-pro"
PLOT_DISSECTION_MAX_TOKENS = 16000


def default_plot_dissection_path(video_episode_analysis_path: str) -> str:
    base, _ext = os.path.splitext(video_episode_analysis_path)
    return f"{base}_dissection.json"


def _strip_json_fence(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^```\s*$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def build_plot_dissection_prompt(
    *,
    analysis_markdown: str,
    character_relationship: str = "",
    plot_reference: str = "",
) -> str:
    blocks: list[str] = [
        "你是一位资深影视剧本编辑。请根据下方「视频分析结果」，对照可选的人物关系与剧情参考，"
        "**校正所有主要人名的规范写法**，输出一份**可直接用于后续写脚本**的完善 JSON。",
        "",
        "## 任务要求（硬性）",
        "- 保留原分析的时间结构：`episodic_segments.time_range` 不得改动",
        "- `involved_characters` / `important_dialogues.speaker` 须与人物关系表、定妆照逻辑一致",
        "- 纠正谐音/错字（如胡小月→胡小跃、秦峰→秦枫、罗伯→罗博）",
        "- `important_dialogues.quote` 保持原话，仅校正 speaker",
        "- 不确定的人名写「剧中未明确交代」",
        "- **禁止**编造分析中未出现的重大情节",
        "",
        "## 输出格式",
        "只输出 JSON 对象，包含：",
        "- `overall_summary`（校正人名后的摘要）",
        "- `key_conflict`",
        "- `episodic_segments`（与输入条数、time_range 一致，人名已校正）",
        "- `important_dialogues`（speaker 已校正）",
        "- `cliffhangers_or_foreshadowing`",
        "- `name_corrections`：数组，每项 `{ \"wrong\": \"…\", \"correct\": \"…\", \"reason\": \"…\" }`",
        "",
        "## 视频分析结果（待校正）",
        analysis_markdown.strip(),
    ]
    rel = build_character_relationship_prompt_section(character_relationship)
    if rel.strip():
        blocks.extend(["", rel.strip()])
    plot = build_plot_reference_prompt_section(plot_reference)
    if plot.strip():
        blocks.extend(["", plot.strip()])
    return "\n".join(blocks)


def run_plot_dissection(
    *,
    video_episode_analysis_path: str,
    character_relationship: str = "",
    plot_reference: str = "",
    use_character_relationship: bool = True,
    use_plot_reference: bool = True,
    output_path: str = "",
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    progress = progress_callback or (lambda _m: None)
    if not video_episode_analysis_path or not os.path.isfile(video_episode_analysis_path):
        raise FileNotFoundError(f"视频分析文件不存在: {video_episode_analysis_path}")

    artifact = load_video_episode_analysis_artifact(video_episode_analysis_path)
    analysis_md = build_video_episode_analysis_markdown(artifact)
    if not analysis_md.strip():
        raise ValueError("视频分析内容为空，无法解剖")

    rel_text = (character_relationship or "").strip() if use_character_relationship else ""
    plot_text = (plot_reference or "").strip() if use_plot_reference else ""
    if not rel_text and not plot_text:
        raise ValueError("请至少勾选「人物关系」或「剧情参考」之一，且对应文本框有内容")

    prompt = build_plot_dissection_prompt(
        analysis_markdown=analysis_md,
        character_relationship=rel_text,
        plot_reference=plot_text,
    )
    progress("正在调用 deepseek-v4-pro 进行剧情解剖…")
    api_key, base_url = resolve_llm_credentials(PLOT_DISSECTION_MODEL, role="text")
    raw = _run_async_safely(
        UnifiedLLMService.generate_text,
        prompt=prompt,
        system_prompt=(
            "你是专业剧本编辑，只输出合法 JSON，不附加 markdown 或解释。"
        ),
        provider="openai",
        model=PLOT_DISSECTION_MODEL,
        temperature=0.2,
        max_tokens=PLOT_DISSECTION_MAX_TOKENS,
        api_key=api_key,
        api_base=base_url,
        response_format="json",
    )
    cleaned = _strip_json_fence(raw if isinstance(raw, str) else str(raw))
    if not cleaned:
        raise ValueError("剧情解剖模型返回空响应")
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"剧情解剖 JSON 解析失败: {exc}") from exc

    save_path = (output_path or "").strip() or default_plot_dissection_path(
        video_episode_analysis_path
    )
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    result: dict[str, Any] = {
        "source_analysis_path": video_episode_analysis_path,
        "dissection_model": PLOT_DISSECTION_MODEL,
        "used_character_relationship": bool(rel_text),
        "used_plot_reference": bool(plot_text),
        "corrected": payload,
    }
    with open(save_path, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    logger.info(f"剧情解剖已保存: {save_path}")
    progress(f"剧情解剖完成 · 已保存 {os.path.basename(save_path)}")
    result["output_path"] = save_path
    return result
