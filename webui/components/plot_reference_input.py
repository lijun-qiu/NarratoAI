"""剧情参考输入（抽帧 / 蓝图 / 剧情解剖共用；不含默认人物关系）。"""

from __future__ import annotations

import streamlit as st

PLOT_REFERENCE_SESSION_KEY = "doc_plot_reference"


def get_plot_reference() -> str:
    return str(st.session_state.get(PLOT_REFERENCE_SESSION_KEY) or "").strip()


def render_plot_reference_input(*, key: str = PLOT_REFERENCE_SESSION_KEY) -> str:
    st.text_area(
        "剧情参考",
        height=160,
        key=key,
        placeholder=(
            "本集主线、情节背景、名场面说明等（可选）。\n"
            "示例：本集主线为胡小跃坠楼案调查，开篇为楼顶对峙。"
        ),
        help="理解辅助；留空则不注入。人物关系请填上方「人物关系」框。",
    )
    return get_plot_reference()
