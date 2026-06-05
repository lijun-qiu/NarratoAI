# 抽帧分析完成后更新 session 中的字幕路径
from __future__ import annotations

import os
from typing import Any

import streamlit as st

from app.services.subtitle_video_pairing import load_subtitle_content


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
