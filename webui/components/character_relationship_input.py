"""人物关系输入（整片视频分析 / 剧情解剖共用）。"""

from __future__ import annotations

import streamlit as st

from app.data.drama_knowledge.fazu2_speaker_knowledge import (
    bundled_fazu2_character_graph_path,
    get_default_character_relationship_text,
    is_legacy_fazu2_relationship_markdown,
)
from webui.components.drama_character_input import DRAMA_ID_SESSION_KEY

CHARACTER_RELATIONSHIP_SESSION_KEY = "doc_character_relationship"
_RELATIONSHIP_SYNCED_DRAMA_KEY = "_character_relationship_synced_drama_id"
_RELATIONSHIP_BUILTIN_DEFAULT_KEY = "_character_relationship_builtin_default"


def get_character_relationship() -> str:
    return str(st.session_state.get(CHARACTER_RELATIONSHIP_SESSION_KEY) or "").strip()


def sync_character_relationship_with_drama(drama_id: str | None = None) -> None:
    resolved_id = str(
        drama_id if drama_id is not None else st.session_state.get(DRAMA_ID_SESSION_KEY) or ""
    ).strip()
    default_text = get_default_character_relationship_text(drama_id=resolved_id)
    synced_drama = str(st.session_state.get(_RELATIONSHIP_SYNCED_DRAMA_KEY) or "")
    current = get_character_relationship()
    previous_default = str(st.session_state.get(_RELATIONSHIP_BUILTIN_DEFAULT_KEY) or "")

    if not default_text:
        if synced_drama != resolved_id:
            st.session_state[_RELATIONSHIP_SYNCED_DRAMA_KEY] = resolved_id
        return

    should_apply = False
    if resolved_id != synced_drama:
        if not current or (previous_default and current == previous_default):
            should_apply = True
    elif not current:
        should_apply = True
    elif default_text and is_legacy_fazu2_relationship_markdown(current):
        should_apply = True

    if should_apply:
        st.session_state[CHARACTER_RELATIONSHIP_SESSION_KEY] = default_text
        st.session_state[_RELATIONSHIP_BUILTIN_DEFAULT_KEY] = default_text
    st.session_state[_RELATIONSHIP_SYNCED_DRAMA_KEY] = resolved_id


def render_character_relationship_input(
    *,
    key: str = CHARACTER_RELATIONSHIP_SESSION_KEY,
) -> str:
    sync_character_relationship_with_drama()
    st.text_area(
        "人物关系",
        height=220,
        key=key,
        placeholder=(
            "选择《罚罪2》后会自动填入 fazu2_character_graph.json；也可自行修改。\n"
            "用于视频分析、蓝图生成、剧情解剖的人名与关系校正。"
        ),
        help="默认注入结构化 JSON 人物关系图；留空则不注入。",
    )
    return get_character_relationship()
