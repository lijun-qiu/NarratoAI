# -*- coding: UTF-8 -*-

from __future__ import annotations

from typing import Any

import streamlit as st

from app.utils.script_duration_estimate import estimate_script_duration, format_duration_seconds


def _resolve_voice_rate(voice_rate: float | None) -> float:
    if voice_rate is not None and voice_rate > 0:
        return voice_rate
    return float(st.session_state.get("voice_rate", 1.0) or 1.0)


def build_duration_preview_text(
    script_items: list[dict[str, Any]],
    *,
    voice_rate: float | None = None,
) -> str:
    summary = estimate_script_duration(script_items, voice_rate=_resolve_voice_rate(voice_rate))
    ost_counts = summary["ost_counts"]
    narration_label = format_duration_seconds(summary["narration_seconds"])
    original_label = format_duration_seconds(summary["original_seconds"])

    ost_parts = []
    if ost_counts.get(0):
        ost_parts.append(f"解说 {ost_counts[0]}")
    if ost_counts.get(1):
        ost_parts.append(f"原声 {ost_counts[1]}")
    if ost_counts.get(2):
        ost_parts.append(f"混合 {ost_counts[2]}")
    ost_line = " / ".join(ost_parts) if ost_parts else "无片段"

    return (
        f"**预计成片时长：约 {summary['formatted_total']}**（{summary['total_seconds']:.1f} 秒）\n\n"
        f"- 片段数：**{summary['segment_count']}**（{ost_line}）\n"
        f"- 解说段合计约 **{narration_label}** | 原声段合计约 **{original_label}**\n"
        f"- 语速参考：**{summary['voice_rate']:.2f}x**（可在音频设置中调整）\n\n"
        "说明：OST=0 按解说文案估算 TTS 时长；OST=1 按时间戳区间计算；实际成片可能有 ±10% 偏差。"
    )


def render_script_duration_preview(
    script_items: list[dict[str, Any]],
    *,
    voice_rate: float | None = None,
) -> None:
    if not script_items:
        return

    st.markdown("#### 成片时长参考")
    st.info(build_duration_preview_text(script_items, voice_rate=voice_rate))


def store_generated_script(script_items: list[dict[str, Any]]) -> None:
    st.session_state["video_clip_json"] = script_items
