"""剧情参考输入（抽帧分析 / 整片视频分析共用）。"""

from __future__ import annotations

import streamlit as st

PLOT_REFERENCE_SESSION_KEY = "doc_plot_reference"


def get_plot_reference() -> str:
    return str(st.session_state.get(PLOT_REFERENCE_SESSION_KEY) or "").strip()


def render_plot_reference_input(*, key: str = PLOT_REFERENCE_SESSION_KEY) -> str:
    """渲染剧情参考文本框，返回当前输入内容。"""
    st.text_area(
        "剧情参考",
        height=120,
        key=key,
        placeholder=(
            "可选。填写本集/本片剧情背景，帮助模型理解画面，例如：\n"
            "· 本集主线：主角追查失踪案，与搭档在审讯室对峙\n"
            "· 人物：张三（刑警）、李四（嫌疑人）\n"
            "· 前情：上一集结尾张三在停车场发现线索"
        ),
        help="仅作理解辅助，不会替代画面分析；留空则不注入。",
    )
    return get_plot_reference()
