#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""整片视频网格快扫 prompt（固定时间格，简化输出）。"""

from __future__ import annotations

from typing import Any


def _resolve_drama_title(drama_title: str | dict[str, str] | None) -> str:
    if isinstance(drama_title, dict):
        raw = drama_title.get("label") or drama_title.get("id") or "本片"
    else:
        raw = drama_title or "本片"
    return str(raw).strip() or "本片"


def build_grid_schedule_prompt_block(time_ranges: list[str], *, grid_interval_seconds: int) -> str:
    if not time_ranges:
        return ""
    lines = [
        "## 时间网格（硬性要求 · 已预计算）",
        (
            f"你必须输出 **恰好 {len(time_ranges)} 条** `grid_segments`，"
            f"每条对应 **{grid_interval_seconds} 秒** 固定时间窗；"
            "`time_range` 必须与下列窗口 **完全一致**（字符级一致，仅用 `-` 连接）："
        ),
    ]
    for index, time_range in enumerate(time_ranges, start=1):
        lines.append(f"{index}. `{time_range}`")
    lines.extend(
        [
            "- 每条填写：`description`（画面/动作/对白，15-30字）、`characters`（可见人物数组，不确定写「剧中未明确交代」）、"
            "`dialogue`（该窗内听到的原话，无则空字符串）。",
            "- **不得跳过任何窗口**；镜头静止可写「画面无明显变化」，禁止用其他时段内容凑数。",
            "- `description` / `dialogue` 必须对应该 `time_range` 内**实际**画面与声音，禁止把片头内容写到后段窗口。",
        ]
    )
    return "\n".join(lines)


def build_grid_batch_time_anchor_block(batch_schedule: list[str]) -> str:
    if not batch_schedule:
        return ""
    start = batch_schedule[0].split("-", 1)[0].strip()
    end = batch_schedule[-1].split("-", 1)[-1].strip()
    return "\n".join(
        [
            "## 本批时间定位（硬性）",
            f"- 本批窗口覆盖全片 **`{start}` – `{end}`**",
            f"- 请先定位到该时段再分析；**只**填写该时段画面与对白",
            "- **禁止**把片头、尾或其他时段的情节/台词写入本批任何 `time_range`",
        ]
    )


def build_previous_grid_batch_tail_context(
    previous_segments: list[dict[str, Any]] | None,
    *,
    tail_count: int = 2,
) -> str:
    if not previous_segments:
        return ""
    tail = [item for item in previous_segments[-tail_count:] if isinstance(item, dict)]
    if not tail:
        return ""
    lines = [
        "## 上一批末尾（仅供时间衔接，勿复述到本批）",
        "| time_range | characters | description |",
        "|---|---|---|",
    ]
    for item in tail:
        time_range = str(item.get("time_range") or "").strip()
        chars = "、".join(item.get("characters") or []) or "—"
        desc = str(item.get("description") or "").strip()[:40]
        lines.append(f"| `{time_range}` | {chars} | {desc} |")
    lines.append("- 本批从上一批结束时间之后继续，**不得**再次输出片头或已写过的台词。")
    return "\n".join(lines)


def build_grid_batch_prompt_addon(
    *,
    batch_index: int,
    batch_count: int,
    batch_size: int,
) -> str:
    if batch_count <= 1:
        return ""
    lines = [
        f"## 本批输出范围（整片视频 · 分批输出 · 第 {batch_index + 1}/{batch_count} 批 · {batch_size} 格）",
        "- 上传的是完整视频；**仅**输出本批时间窗对应的 `grid_segments`",
    ]
    if batch_index > 0:
        lines.append('- 本批 **`overall_summary` / `key_conflict` 填空字符串 `""`**')
    return "\n".join(lines)


def build_whole_grid_analysis_prompt(
    *,
    drama_title: str = "",
    video_duration_seconds: float,
    grid_interval_seconds: int,
    segment_schedule_block: str,
    character_names: list[str] | None = None,  # 无视觉参照图时 fallback 为纯文本人名提示
    require_summary: bool = True,
    plot_reference_section: str = "",
) -> str:
    title = _resolve_drama_title(drama_title)
    names = [str(name).strip() for name in (character_names or []) if str(name).strip()]
    naming_hint = ""
    if names:
        naming_hint = (
            f"\n已知人物（视频可见且能确认时才使用）：{'、'.join(names[:30])}"
            f"{'…' if len(names) > 30 else ''}。"
        )
    summary_block = ""
    if require_summary:
        summary_block = (
            '  "overall_summary": "200字以内概括**本集视频实际内容**",\n'
            '  "key_conflict": "一句话点出**本集视频**最核心的矛盾",\n'
        )
    plot_block = (plot_reference_section or "").strip()
    duration_min = max(1, int(round(video_duration_seconds / 60)))
    return f"""请分析我上传的电视剧视频（《{title}》），按固定时间网格输出简化 JSON。

视频总长约 {duration_min} 分钟；每格 {grid_interval_seconds} 秒。所有 `time_range` 使用全片绝对时间 **HH:MM:SS-HH:MM:SS**。
{naming_hint}
{plot_block}
{segment_schedule_block}

【输出 JSON 模板】
{{
{summary_block}  "grid_segments": [
    {{
      "segment_id": 1,
      "time_range": "00:00:00-00:00:05",
      "description": "画面与关键动作简述",
      "characters": ["角色A"],
      "dialogue": "该时间窗内听到的原话，无则留空"
    }}
  ]
}}

要求：
1. 只依据**该 time_range 内**的视频画面与声音；不确定写「剧中未明确交代」。
2. `overall_summary` / `key_conflict` 必须来自本集视频，不得照搬剧情参考里的设定。
3. 严格 JSON，不要 markdown 代码块或注释。
4. `grid_segments` 条数与时间窗列表一致。
"""
