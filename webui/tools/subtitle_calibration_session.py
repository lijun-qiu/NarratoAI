# 字幕校准：session 状态与路径解析（不依赖视频文件）
from __future__ import annotations

import os
from typing import Any

import streamlit as st

from app.services.documentary.frame_analysis_pairing import find_paired_frame_analysis_path
from app.services.documentary.frame_analysis_service import DocumentaryFrameAnalysisService
from app.services.subtitle_video_pairing import find_paired_subtitle_path, load_subtitle_content
from app.utils import utils


def apply_subtitle_calibration_to_session(calibration: dict[str, Any]) -> str:
    """将校准结果写入 session_state，返回最终字幕路径（可能为空）。"""
    final_path = str(calibration.get("subtitle_path") or "").strip()
    if not final_path or not os.path.isfile(final_path):
        return ""

    st.session_state["subtitle_path"] = final_path
    st.session_state["subtitle_content"] = load_subtitle_content(final_path)
    st.session_state["doc_subtitle_file_processed"] = True
    return final_path


def format_subtitle_calibration_summary(calibration: dict[str, Any]) -> str:
    if calibration.get("skipped_reason") == "no_subtitle":
        return "未检测到字幕，已跳过自动校正（请先转写或上传字幕）。"
    if calibration.get("skipped_reason"):
        return ""

    parts: list[str] = []
    ocr_path = calibration.get("ocr_refined_path")
    refined_path = calibration.get("refined_path")
    if ocr_path:
        parts.append(f"OCR 校准 `{os.path.basename(ocr_path)}`")
    if refined_path:
        parts.append(f"LLM 校正 `{os.path.basename(refined_path)}`")
    if not parts:
        return "已尝试自动校正，但未产生新字幕文件（可能无硬字幕或无需修改）。"
    return "已自动完成字幕校正：" + " → ".join(parts)


def _subtitle_stem_for_output(subtitle_path: str) -> str:
    stem = os.path.splitext(os.path.basename(subtitle_path))[0]
    for suffix in ("_ocr_refined", "_refined", "_transcribed"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem or "subtitle"


def predict_ocr_refined_path(subtitle_path: str) -> str:
    stem = _subtitle_stem_for_output(subtitle_path)
    return os.path.join(utils.subtitle_dir(), f"{stem}_ocr_refined.srt")


def predict_refined_path(subtitle_path: str) -> str:
    stem = _subtitle_stem_for_output(subtitle_path)
    return os.path.join(utils.subtitle_dir(), f"{stem}_refined.srt")


def resolve_calibration_subtitle_path() -> str:
    """当前用于校准的字幕路径（session 优先，可选视频配对回退）。"""
    explicit = (st.session_state.get("subtitle_path") or "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit

    input_path = (st.session_state.get("doc_calibrate_subtitle_path_input") or "").strip()
    if input_path and os.path.isfile(input_path):
        return input_path

    input_path = (st.session_state.get("doc_subtitle_path_input") or "").strip()
    if input_path and os.path.isfile(input_path):
        return input_path

    video_path = (st.session_state.get("video_origin_path") or "").strip()
    if video_path:
        return find_paired_subtitle_path(video_path) or ""
    return ""


def resolve_calibration_analysis_path() -> str:
    """当前用于校准的抽帧分析 JSON（session 优先，可选视频配对回退）。"""
    explicit = (st.session_state.get("frame_analysis_json_path") or "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit

    input_path = (st.session_state.get("doc_calibrate_frame_analysis_path_input") or "").strip()
    if input_path and os.path.isfile(input_path):
        return input_path

    input_path = (st.session_state.get("doc_frame_analysis_path_input") or "").strip()
    if input_path and os.path.isfile(input_path):
        return input_path

    video_path = (st.session_state.get("video_origin_path") or "").strip()
    if not video_path:
        return ""

    reuse = bool(st.session_state.get("doc_reuse_frame_analysis", True))
    resolved = DocumentaryFrameAnalysisService().resolve_reusable_analysis_path(
        video_path,
        explicit_path=None,
        reuse=reuse,
    )
    if resolved:
        return resolved
    return find_paired_frame_analysis_path(video_path) or ""
