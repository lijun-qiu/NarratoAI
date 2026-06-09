"""逐帧解说素材：对照抽帧分析校准字幕（独立功能，无需视频文件）。"""

from __future__ import annotations

import json
import os
import time

import streamlit as st

from app.services.documentary.documentary_settings import (
    get_documentary_compact_settings,
    get_documentary_settings,
)
from app.services.documentary.frame_analysis_pairing import is_valid_analysis_artifact
from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService
from webui.components.subtitle_transcription_settings import render_documentary_subtitle_file_picker
from webui.tools.ocr_calibrate_subtitle_docu import ocr_calibrate_subtitle_docu
from webui.tools.refine_subtitle_docu import refine_subtitle_docu
from webui.tools.subtitle_calibration_session import (
    predict_ocr_refined_path,
    predict_refined_path,
    resolve_calibration_analysis_path,
    resolve_calibration_subtitle_path,
)
from webui.utils.documentary_file_picker import (
    apply_frame_analysis_path,
    clear_subtitle_path,
    queue_picker_paths,
    render_saved_file_picker,
)


def _clear_calibration_subtitle_path() -> None:
    clear_subtitle_path()
    st.session_state.pop("doc_calibrate_subtitle_path_input", None)
    st.session_state.pop("doc_calibrate_subtitle_saved_pick", None)


def _clear_calibration_frame_analysis_path() -> None:
    st.session_state["frame_analysis_json_path"] = None
    st.session_state["doc_frame_analysis_file_processed"] = False
    st.session_state["doc_frame_analysis_upload_explicit"] = False
    st.session_state.pop("doc_calibrate_frame_analysis_path_input", None)
    st.session_state.pop("doc_calibrate_frame_analysis_saved_pick", None)


def _import_frame_analysis_file(tr, analysis_file) -> None:
    try:
        payload = json.loads(analysis_file.getvalue().decode("utf-8"))
        from app.services.documentary.frame_analysis_pairing import (
            _LEGACY_ARTIFACT_ERROR,
            is_legacy_analysis_artifact,
        )

        if is_legacy_analysis_artifact(payload):
            st.error(_LEGACY_ARTIFACT_ERROR)
            st.stop()
        if not is_valid_analysis_artifact(payload):
            st.error("无效的抽帧分析 JSON：缺少 scene_segments / batches / frame_observations")
            st.stop()

        safe_filename = os.path.basename(analysis_file.name)
        analysis_dir = DocumentaryFrameExtractionService.analysis_artifact_dir()
        os.makedirs(analysis_dir, exist_ok=True)
        target_path = os.path.join(analysis_dir, safe_filename)
        if os.path.exists(target_path):
            timestamp = time.strftime("%Y%m%d%H%M%S")
            name, ext = os.path.splitext(safe_filename)
            target_path = os.path.join(analysis_dir, f"{name}_{timestamp}{ext}")

        with open(target_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

        apply_frame_analysis_path(target_path)
        queue_picker_paths(
            "doc_calibrate_frame_analysis_path_input",
            "doc_calibrate_frame_analysis_saved_pick",
            target_path,
        )
        st.success(f"抽帧分析已导入: {os.path.basename(target_path)}")
        st.rerun()
    except json.JSONDecodeError:
        st.error("无法解析 JSON 文件，请检查格式")
    except Exception as exc:
        st.error(f"{tr('Upload failed')}: {str(exc)}")


def _render_calibration_frame_analysis_picker(tr) -> None:
    """选用或导入抽帧分析 JSON（校准专用，不依赖视频）。"""
    analysis_dir = DocumentaryFrameExtractionService.analysis_artifact_dir()
    active_path = (st.session_state.get("frame_analysis_json_path") or "").strip()

    render_saved_file_picker(
        label="抽帧分析 JSON",
        directory=analysis_dir,
        glob_pattern="*_frame_analysis*.json",
        path_input_key="doc_calibrate_frame_analysis_path_input",
        pick_key="doc_calibrate_frame_analysis_saved_pick",
        confirm_button_key="doc_calibrate_confirm_frame_analysis",
        clear_button_key="doc_calibrate_clear_frame_analysis",
        active_path=active_path,
        paired_path="",
        on_confirm=apply_frame_analysis_path,
        on_clear=_clear_calibration_frame_analysis_path,
        import_label="导入 JSON 到分析目录",
        import_types=["json"],
        import_key="doc_calibrate_frame_analysis_uploader",
        on_import=lambda uploaded: _import_frame_analysis_file(tr, uploaded),
    )


def render_subtitle_calibration_panel(tr, *, params=None, compact: bool = False) -> None:
    """对照抽帧分析校正 ASR 字幕（产出 *_refined.srt / *_ocr_refined.srt）。"""
    doc_settings = get_documentary_compact_settings() if compact else get_documentary_settings()

    st.caption(
        "仅需 **字幕 SRT** + **抽帧分析 JSON**，无需选择视频。"
        "生成脚本时将优先使用 `*_ocr_refined.srt` / `*_refined.srt`。"
    )

    render_documentary_subtitle_file_picker(
        tr,
        path_input_key="doc_calibrate_subtitle_path_input",
        pick_key="doc_calibrate_subtitle_saved_pick",
        confirm_button_key="doc_calibrate_confirm_subtitle_path",
        clear_button_key="doc_calibrate_clear_subtitle",
        import_key="doc_calibrate_subtitle_uploader",
        on_clear=_clear_calibration_subtitle_path,
    )
    st.divider()
    _render_calibration_frame_analysis_picker(tr)
    st.divider()

    subtitle_path = resolve_calibration_subtitle_path()
    analysis_path = resolve_calibration_analysis_path()

    if subtitle_path:
        st.info(f"将校准字幕: **{os.path.basename(subtitle_path)}**")
        ocr_path = predict_ocr_refined_path(subtitle_path)
        refined_path = predict_refined_path(subtitle_path)
        if os.path.isfile(ocr_path):
            st.success(f"已有 OCR 校准字幕: **{os.path.basename(ocr_path)}**")
        elif os.path.isfile(refined_path):
            st.success(f"已有 LLM 校正字幕: **{os.path.basename(refined_path)}**")
        else:
            st.caption("尚未生成校正字幕")
    else:
        st.warning("请先选用或导入待校准的字幕 SRT")

    if analysis_path:
        st.caption(f"将对照抽帧分析: `{os.path.basename(analysis_path)}`")
    else:
        st.warning("请先选用或导入抽帧分析 JSON")

    can_refine = bool(analysis_path and subtitle_path)

    if doc_settings.get("enable_hard_subtitle_ocr", True):
        st.caption(
            "硬字幕 OCR：优先使用 JSON 内嵌的 burned_in_subtitle；"
            "旧版 JSON 无该字段时会二次调用视觉模型裁剪 OCR。"
        )
        if st.button(
            "硬字幕 OCR 校准",
            key="doc_ocr_calibrate_subtitle_btn",
            use_container_width=True,
            disabled=not can_refine,
        ):
            ocr_calibrate_subtitle_docu(params, compact=compact)

    if doc_settings.get("enable_subtitle_refinement", True):
        if st.button(
            "LLM 校正字幕",
            key="doc_refine_subtitle_btn",
            use_container_width=True,
            disabled=not can_refine,
        ):
            refine_subtitle_docu(params, compact=compact)
