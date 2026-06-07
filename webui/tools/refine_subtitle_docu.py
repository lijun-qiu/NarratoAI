# 对照抽帧校正字幕
import os
import time
import traceback

import streamlit as st
from loguru import logger

from app.services.documentary.documentary_settings import get_documentary_compact_settings, get_documentary_settings
from app.services.documentary.subtitle_refinement_service import refine_subtitle_with_frame_analysis
from app.services.subtitle_video_pairing import load_subtitle_content
from webui.tools.subtitle_calibration_session import (
    resolve_calibration_analysis_path,
    resolve_calibration_subtitle_path,
)


def _normalize_progress_value(progress: float | int) -> int:
    try:
        value = float(progress)
    except (TypeError, ValueError):
        return 0
    if 0.0 <= value <= 1.0:
        value *= 100
    return max(0, min(100, int(round(value))))


def refine_subtitle_docu(params, *, compact: bool = False):
    """对照已有抽帧分析 JSON 校正字幕，产出 *_refined.srt。"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress: float, message: str = ""):
        normalized_progress = _normalize_progress_value(progress)
        progress_bar.progress(normalized_progress)
        if message:
            status_text.text(f"📝 {message}")
        else:
            status_text.text(f"📊 进度: {normalized_progress}%")

    try:
        with st.spinner("正在校正字幕..."):
            doc_settings = get_documentary_compact_settings() if compact else get_documentary_settings()
            if "doc_enable_subtitle_enrichment" in st.session_state:
                doc_settings = dict(doc_settings)
                doc_settings["enable_subtitle_enrichment"] = bool(
                    st.session_state.get("doc_enable_subtitle_enrichment")
                )

            subtitle_path = resolve_calibration_subtitle_path()
            if not subtitle_path or not os.path.isfile(subtitle_path):
                st.error("请先选用或导入字幕 SRT，再执行校正")
                return

            analysis_path = resolve_calibration_analysis_path()
            if not analysis_path or not os.path.isfile(analysis_path):
                st.error("未找到抽帧分析 JSON，请先选用或导入分析文件")
                return

            update_progress(10, "正在对照抽帧分析校正字幕...")

            def on_progress(message: str):
                update_progress(50, message)

            output_path = refine_subtitle_with_frame_analysis(
                subtitle_path=subtitle_path,
                analysis_json_path=analysis_path,
                video_theme=st.session_state.get("video_theme", ""),
                documentary_settings=doc_settings,
                progress_callback=on_progress,
            )

            refined_content = load_subtitle_content(output_path)
            st.session_state["subtitle_path"] = output_path
            st.session_state["subtitle_content"] = refined_content
            st.session_state["doc_subtitle_file_processed"] = True
            logger.info(f"字幕校正已保存: {output_path}")

        time.sleep(0.1)
        progress_bar.progress(100)
        status_text.text("🎉 字幕校正完成！")
        st.success(f"✅ 已保存校正字幕: `{os.path.basename(output_path)}`")
        st.caption(f"完整路径: `{output_path}`")
        st.info("生成脚本时将优先使用此校正字幕。")

    except Exception as err:
        st.error(f"❌ 字幕校正失败: {str(err)}")
        logger.exception(f"字幕校正失败\n{traceback.format_exc()}")
    finally:
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
