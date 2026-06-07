"""脚本片段 OST 统计与成片时长估算，供 WebUI 展示。"""

from __future__ import annotations

import re
from typing import Any


def summarize_script_ost(items: list[Any] | None) -> dict[str, int | float] | None:
    if not items:
        return None

    ost0 = 0
    ost1 = 0
    ost2 = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        ost = int(item.get("OST", 0))
        if ost == 1:
            ost1 += 1
        elif ost == 2:
            ost2 += 1
        else:
            ost0 += 1

    total = ost0 + ost1 + ost2
    if total <= 0:
        return None

    return {
        "total": total,
        "ost0": ost0,
        "ost1": ost1,
        "ost2": ost2,
        "ost0_pct": round(ost0 * 100 / total, 1),
        "ost1_pct": round(ost1 * 100 / total, 1),
        "ost2_pct": round(ost2 * 100 / total, 1),
    }


def _estimate_narration_duration_sec(text: str) -> float:
    """解说 TTS 时长估算（中文约 0.35 秒/字）。"""
    chars = len(re.sub(r"\s+", "", text or ""))
    return max(3.0, chars * 0.35)


def _segment_playback_duration_sec(item: dict[str, Any]) -> float:
    """按播放顺序估算单段成片时长。"""
    ost = int(item.get("OST", 0))
    timestamp = str(item.get("timestamp") or "").strip()
    narration = str(item.get("narration") or "")

    if ost == 1 and timestamp:
        try:
            from app.services.update_script import calculate_duration

            duration = calculate_duration(timestamp)
            if duration > 0:
                return duration
        except Exception:
            pass

    if ost == 2 and timestamp:
        try:
            from app.services.update_script import calculate_duration

            ts_duration = calculate_duration(timestamp)
            narr_duration = _estimate_narration_duration_sec(narration)
            if ts_duration > 0:
                return max(ts_duration, narr_duration)
        except Exception:
            pass

    return _estimate_narration_duration_sec(narration)


def summarize_script_duration(items: list[Any] | None) -> dict[str, float] | None:
    """按 _id 播放顺序累加各段时长，估算成片总时长。"""
    if not items:
        return None

    ost0_sec = 0.0
    ost1_sec = 0.0
    ost2_sec = 0.0
    counted = 0

    ordered = sorted(
        (item for item in items if isinstance(item, dict)),
        key=lambda item: int(item.get("_id") or 0),
    )
    for item in ordered:
        duration = _segment_playback_duration_sec(item)
        if duration <= 0:
            continue
        counted += 1
        ost = int(item.get("OST", 0))
        if ost == 1:
            ost1_sec += duration
        elif ost == 2:
            ost2_sec += duration
        else:
            ost0_sec += duration

    total_sec = ost0_sec + ost1_sec + ost2_sec
    if counted <= 0 or total_sec <= 0:
        return None

    return {
        "total_sec": round(total_sec, 1),
        "ost0_sec": round(ost0_sec, 1),
        "ost1_sec": round(ost1_sec, 1),
        "ost2_sec": round(ost2_sec, 1),
    }


def format_duration_display(seconds: float) -> str:
    """格式化成「X分Y秒」或「Y.Y秒」。"""
    if seconds <= 0:
        return "0秒"
    minutes = int(seconds // 60)
    remain = seconds % 60
    if minutes > 0:
        if remain >= 0.05:
            return f"{minutes}分{remain:.1f}秒"
        return f"{minutes}分"
    return f"{remain:.1f}秒"


def format_ost_summary_caption(
    summary: dict[str, int | float],
    *,
    min_ost1: int | None = None,
    max_ost1: int | None = None,
) -> str:
    parts = [
        f"总段数 **{summary['total']}**",
        f"解说 OST=0：**{summary['ost0']}** 段（{summary['ost0_pct']}%）",
        f"原声 OST=1：**{summary['ost1']}** 段（{summary['ost1_pct']}%）",
    ]
    if summary["ost2"]:
        parts.append(f"OST=2：**{summary['ost2']}** 段（{summary['ost2_pct']}%）")
    if min_ost1 and max_ost1:
        parts.append(f"要求原声 **{min_ost1}–{max_ost1}** 段")
    elif max_ost1 and max_ost1 > 0:
        parts.append(f"建议原声 ≤{max_ost1} 段")
    return " · ".join(parts)


def render_script_ost_summary(
    items: list[Any] | None,
    *,
    min_ost1: int | None = None,
    max_ost1: int | None = None,
) -> None:
    """在 Streamlit 页面展示 OST=0 / OST=1 段数、占比与预估成片总时长。"""
    import streamlit as st

    summary = summarize_script_ost(items)
    if not summary:
        return

    duration_summary = summarize_script_duration(items)

    st.caption("片段统计")
    column_count = 4 if summary["ost2"] else 3
    if duration_summary:
        column_count += 1
    columns = st.columns(column_count)
    col_index = 0
    columns[col_index].metric("解说 OST=0", f"{summary['ost0']} 段", f"{summary['ost0_pct']}%")
    col_index += 1
    columns[col_index].metric("原声 OST=1", f"{summary['ost1']} 段", f"{summary['ost1_pct']}%")
    col_index += 1
    if summary["ost2"]:
        columns[col_index].metric("OST=2", f"{summary['ost2']} 段", f"{summary['ost2_pct']}%")
        col_index += 1
    columns[col_index].metric("总段数", f"{summary['total']} 段")
    col_index += 1
    if duration_summary:
        columns[col_index].metric(
            "预估成片总时长",
            format_duration_display(duration_summary["total_sec"]),
            help=(
                "按 _id 播放顺序累加：OST=1 取 timestamp 跨度，"
                "OST=0 按解说字数估算 TTS 时长"
            ),
        )

    if duration_summary:
        duration_cols = st.columns(3 if summary["ost2"] else 2)
        duration_cols[0].metric(
            "解说时长",
            format_duration_display(duration_summary["ost0_sec"]),
        )
        duration_cols[1].metric(
            "原声时长",
            format_duration_display(duration_summary["ost1_sec"]),
        )
        if summary["ost2"]:
            duration_cols[2].metric(
                "OST=2 时长",
                format_duration_display(duration_summary["ost2_sec"]),
            )

    if min_ost1 and max_ost1:
        if summary["ost1"] < min_ost1:
            st.warning(
                f"原声 OST=1 仅 {summary['ost1']} 段（{summary['ost1_pct']}%），"
                f"要求 **{min_ost1}–{max_ost1}** 段；金句对白应标 OST=1，narration 填原台词。"
            )
        elif summary["ost1"] > max_ost1:
            st.warning(
                f"原声 OST=1 共 {summary['ost1']} 段（{summary['ost1_pct']}%），"
                f"超过上限 {max_ost1} 段；请合并或改回 OST=0 解说。"
            )
    elif summary["ost1"] <= 0:
        st.warning("当前脚本没有 OST=1 原声段；金句对白应标为 OST=1，narration 填原台词。")
