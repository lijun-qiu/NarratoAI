"""脚本片段 OST 统计，供 WebUI 展示。"""

from __future__ import annotations

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
    """在 Streamlit 页面展示 OST=0 / OST=1 段数与占比。"""
    import streamlit as st

    summary = summarize_script_ost(items)
    if not summary:
        return

    st.caption("片段类型占比")
    columns = st.columns(4 if summary["ost2"] else 3)
    columns[0].metric("解说 OST=0", f"{summary['ost0']} 段", f"{summary['ost0_pct']}%")
    columns[1].metric("原声 OST=1", f"{summary['ost1']} 段", f"{summary['ost1_pct']}%")
    if summary["ost2"]:
        columns[2].metric("OST=2", f"{summary['ost2']} 段", f"{summary['ost2_pct']}%")
        columns[3].metric("总段数", f"{summary['total']} 段")
    else:
        columns[2].metric("总段数", f"{summary['total']} 段")

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
