"""逐帧解说/精剪：从默认目录选用字幕与抽帧分析文件。"""

from __future__ import annotations

import streamlit as st

from app.utils import utils
from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService
from webui.components.frame_analysis_settings import (
    render_documentary_frame_analysis_file_picker,
    sync_frame_analysis_with_video,
)
from webui.components.subtitle_transcription_settings import render_documentary_subtitle_file_picker
from webui.utils.documentary_file_picker import consume_pending_reuse_frame_analysis


def render_documentary_material_pickers(
    tr,
    *,
    expanded: bool = False,
    key_prefix: str = "doc_script",
) -> None:
    """字幕目录 + 抽帧分析目录：与「素材预处理」共用，确认后用于脚本生成。"""
    consume_pending_reuse_frame_analysis()
    video_path = (st.session_state.get("video_origin_path") or "").strip()
    if video_path:
        sync_frame_analysis_with_video(video_path)

    subtitle_dir = utils.subtitle_dir()
    analysis_dir = DocumentaryFrameExtractionService.analysis_artifact_dir()

    with st.expander("字幕与抽帧分析（从默认目录选择）", expanded=expanded):
        st.caption(
            f"字幕目录: `{subtitle_dir}` · 抽帧分析目录: `{analysis_dir}`。"
            "也可在「素材预处理」中转录/抽帧；此处可直接选用已有文件。"
        )
        render_documentary_subtitle_file_picker(
            tr,
            path_input_key=f"{key_prefix}_subtitle_path_input",
            pick_key=f"{key_prefix}_subtitle_saved_pick",
            confirm_button_key=f"{key_prefix}_confirm_subtitle_path",
            clear_button_key=f"{key_prefix}_clear_subtitle",
            import_key=f"{key_prefix}_subtitle_uploader",
        )
        st.divider()
        render_documentary_frame_analysis_file_picker(
            tr,
            video_path,
            path_input_key=f"{key_prefix}_frame_analysis_path_input",
            pick_key=f"{key_prefix}_frame_analysis_saved_pick",
            confirm_button_key=f"{key_prefix}_confirm_frame_analysis_path",
            clear_button_key=f"{key_prefix}_clear_frame_analysis",
            import_key=f"{key_prefix}_frame_analysis_uploader",
        )
