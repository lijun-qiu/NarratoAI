#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Generate scripts for Enhanced Mix Narration mode."""

from __future__ import annotations

import streamlit as st

from webui.tools.generate_short_summary import generate_script_short_sunmmary


def generate_script_enhanced(params, subtitle_path: str, video_theme: str, temperature: float):
    """Generate mixed narration script and persist enhanced-mode metadata."""
    if not subtitle_path:
        st.error("请先上传或转写原片字幕，智能混剪解说模式依赖完整 SRT。")
        return

    st.session_state["processing_mode"] = "enhanced"
    st.session_state["source_subtitle_path"] = subtitle_path

    generate_script_short_sunmmary(
        params,
        subtitle_path,
        video_theme,
        temperature,
        require_media_name=True,
        enhanced_mix=True,
    )

    if st.session_state.get("video_clip_json"):
        st.session_state["processing_mode"] = "enhanced"
        st.session_state["source_subtitle_path"] = subtitle_path
        st.info(
            "脚本已按「智能混剪解说·压缩混剪」规则生成："
            "目标成片约为原片 50%；解说文案精简准确；自动混入 BGM。"
        )
