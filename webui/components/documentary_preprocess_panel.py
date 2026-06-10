"""逐帧解说素材预处理：字幕转录、抽帧分析、校准字幕（独立功能入口）。"""

from __future__ import annotations

import streamlit as st

from webui.components.documentary_output_split import render_output_split_control
from webui.components.drama_character_input import render_drama_character_input
from webui.components.plot_reference_input import render_plot_reference_input
from webui.components.frame_analysis_settings import render_frame_analysis_panel
from webui.components.video_episode_analysis_settings import render_video_episode_analysis_panel
from webui.components.subtitle_calibration_settings import render_subtitle_calibration_panel
from webui.components.subtitle_transcription_settings import render_subtitle_transcription_panel
from webui.components.voice_calibration_panel import render_voice_calibration_panel
from webui.utils.documentary_file_picker import consume_pending_reuse_frame_analysis


def render_documentary_preprocess_panel(tr, params) -> None:
    """素材预处理：三个独立 Tab，互不嵌套于脚本生成流程。"""
    consume_pending_reuse_frame_analysis()
    st.caption(
        "在此完成字幕转录、抽帧分析与字幕校准；完成后切换到「逐帧精剪」或「逐帧解说」生成脚本。"
        "各步骤可按需单独执行，结果会自动保存供后续复用。"
    )
    render_plot_reference_input()
    render_drama_character_input()
    render_output_split_control(key="doc_output_split_parts")

    tab_transcribe, tab_voice, tab_frame, tab_video, tab_calibrate = st.tabs([
        tr("Subtitle Transcription"),
        "人声音频导出",
        tr("Frame Analysis Tool"),
        "整片视频分析",
        tr("Subtitle Calibration"),
    ])

    with tab_transcribe:
        render_subtitle_transcription_panel(tr, show_output_split=False)

    with tab_voice:
        render_voice_calibration_panel()

    with tab_frame:
        render_frame_analysis_panel(
            tr, params, compact=False, standalone=True, show_output_split=False
        )

    with tab_video:
        render_video_episode_analysis_panel(tr, params)

    with tab_calibrate:
        render_subtitle_calibration_panel(tr, params=params, compact=False)
