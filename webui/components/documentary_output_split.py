"""素材预处理：输出切割份数控件。"""

from __future__ import annotations

import streamlit as st

from app.services.documentary.material_output_split import MAX_SPLIT_PARTS, MIN_SPLIT_PARTS


def render_output_split_control(*, key: str = "doc_output_split_parts") -> int:
    """输出切割份数：1 表示完整单文件，2-10 表示均分另存多份。"""
    if key not in st.session_state:
        st.session_state[key] = 1

    return int(
        st.number_input(
            "输出切割份数",
            min_value=MIN_SPLIT_PARTS,
            max_value=MAX_SPLIT_PARTS,
            step=1,
            key=key,
            help="按素材时长均分为多份文件。1 表示仅输出完整文件；大于 1 时会额外另存 part 文件，完整文件仍保留。",
        )
    )
