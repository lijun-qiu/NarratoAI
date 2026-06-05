# 硬字幕 OCR 校准
import os
import time
import traceback

import streamlit as st
from loguru import logger

from app.services.documentary.documentary_settings import get_documentary_compact_settings, get_documentary_settings
from app.services.documentary.frame_analysis_pairing import find_paired_frame_analysis_path
from app.services.documentary.frame_analysis_service import DocumentaryFrameAnalysisService
from app.services.documentary.hard_subtitle_ocr_service import (
    calibrate_subtitle_with_hard_subtitle_ocr,
    get_ocr_refined_subtitle_path,
)
from app.services.subtitle_video_pairing import find_paired_subtitle_path, load_subtitle_content


def _normalize_progress_value(progress: float | int) -> int:
    try:
        value = float(progress)
    except (TypeError, ValueError):
        return 0
    if 0.0 <= value <= 1.0:
        value *= 100
    return max(0, min(100, int(round(value))))


def ocr_calibrate_subtitle_docu(params, *, compact: bool = False):
    """裁剪关键帧底部硬字幕带 OCR，校准 ASR 字幕，产出 *_ocr_refined.srt。"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress: float, message: str = ""):
        normalized_progress = _normalize_progress_value(progress)
        progress_bar.progress(normalized_progress)
        if message:
            status_text.text(f"🔍 {message}")
        else:
            status_text.text(f"📊 进度: {normalized_progress}%")

    try:
        with st.spinner("正在 OCR 硬字幕并校准..."):
            if not params.video_origin_path:
                st.error("请先选择视频文件")
                return

            doc_settings = get_documentary_compact_settings() if compact else get_documentary_settings()

            subtitle_path = (st.session_state.get("subtitle_path") or "").strip()
            if not subtitle_path or not os.path.isfile(subtitle_path):
                subtitle_path = find_paired_subtitle_path(params.video_origin_path) or ""
            if not subtitle_path or not os.path.isfile(subtitle_path):
                st.error("请先转写或上传字幕，再执行 OCR 校准")
                return

            reuse_frame_analysis = bool(st.session_state.get("doc_reuse_frame_analysis", True))
            analysis_path = DocumentaryFrameAnalysisService().resolve_reusable_analysis_path(
                params.video_origin_path,
                explicit_path=(st.session_state.get("frame_analysis_json_path") or "").strip() or None,
                reuse=reuse_frame_analysis,
            )
            if not analysis_path:
                analysis_path = find_paired_frame_analysis_path(params.video_origin_path) or ""
            if not analysis_path or not os.path.isfile(analysis_path):
                st.error("未找到抽帧分析 JSON，请先执行「抽帧并分析」")
                return

            update_progress(10, "正在裁剪关键帧底部并 OCR 硬字幕...")

            def on_progress(message: str):
                update_progress(55, message)

            output_path = calibrate_subtitle_with_hard_subtitle_ocr(
                subtitle_path=subtitle_path,
                analysis_json_path=analysis_path,
                documentary_settings=doc_settings,
                progress_callback=on_progress,
                allow_vision_ocr_fallback=True,
            )

            refined_content = load_subtitle_content(output_path)
            st.session_state["subtitle_path"] = output_path
            st.session_state["subtitle_content"] = refined_content
            st.session_state["doc_subtitle_file_processed"] = True
            logger.info(f"硬字幕 OCR 校准已保存: {output_path}")

        time.sleep(0.1)
        progress_bar.progress(100)
        status_text.text("🎉 硬字幕 OCR 校准完成！")
        st.success(f"✅ 已保存 OCR 校准字幕: `{os.path.basename(output_path)}`")
        st.caption(f"完整路径: `{output_path}`")
        st.info("生成脚本时将优先使用此 OCR 校准字幕。")

    except Exception as err:
        st.error(f"❌ 硬字幕 OCR 校准失败: {str(err)}")
        logger.exception(f"硬字幕 OCR 校准失败\n{traceback.format_exc()}")
    finally:
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
