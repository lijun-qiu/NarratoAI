"""逐帧解说 / 精剪：抽帧分析 UI（独立于脚本生成，对标字幕转录）。"""

from __future__ import annotations

import json
import os
import time

import streamlit as st

from app.config import config
from app.services.documentary.documentary_settings import get_documentary_compact_settings, get_documentary_settings
from app.services.documentary.frame_analysis_pairing import (
    default_analysis_path_for_video,
    find_paired_frame_analysis_path,
    is_valid_analysis_artifact,
    load_analysis_artifact,
)
from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService
from app.services.documentary.subtitle_calibration_pipeline import (
    calibrate_subtitles_after_frame_analysis,
    requires_subtitle_before_frame_analysis,
)
from app.services.subtitle_video_pairing import resolve_subtitle_path_for_video
from webui.tools.extract_frame_analysis_docu import extract_frame_analysis_docu
from webui.tools.subtitle_calibration_session import (
    apply_subtitle_calibration_to_session,
    format_subtitle_calibration_summary,
)


def sync_frame_analysis_with_video(video_path: str) -> None:
    """视频切换时自动配对已有抽帧分析 JSON。"""
    video_path = (video_path or "").strip()
    if not video_path:
        return
    if st.session_state.get("_frame_analysis_synced_video_path") == video_path:
        return
    paired = find_paired_frame_analysis_path(video_path)
    if paired:
        st.session_state["frame_analysis_json_path"] = paired
    else:
        st.session_state["frame_analysis_json_path"] = None
    st.session_state["_frame_analysis_synced_video_path"] = video_path
    st.session_state["doc_frame_analysis_file_processed"] = False


def _session_subtitle_path(video_path: str) -> str:
    return resolve_subtitle_path_for_video(
        video_path,
        explicit_path=st.session_state.get("subtitle_path"),
    )


def _render_frame_analysis_status(video_path: str) -> None:
    """在抽帧面板外展示当前视频是否已有可复用的分析 JSON。"""
    if not video_path:
        st.caption("抽帧分析：请先选择视频")
        return

    override_path = (st.session_state.get("frame_analysis_json_path") or "").strip()
    default_path = default_analysis_path_for_video(video_path)
    active_path = ""
    if override_path and os.path.isfile(override_path):
        active_path = override_path
    elif os.path.isfile(default_path):
        active_path = default_path

    if not active_path:
        st.warning("抽帧分析：尚未检测到可用 JSON，请先完成字幕转写/上传，再展开下方「抽帧分析」")
        return

    label = os.path.basename(active_path)
    detail = ""
    try:
        artifact = load_analysis_artifact(active_path)
        batch_count = len(artifact.get("batches") or [])
        frame_count = len(artifact.get("frame_observations") or [])
        detail = f"（{batch_count} 批次 / {frame_count} 帧）"
    except Exception:
        pass

    st.success(f"已检测到与视频同名的抽帧分析: **{label}**{detail}")


def render_frame_analysis_panel(tr, params, *, compact: bool = False) -> None:
    """抽帧分析：复用/上传/独立抽帧按钮。"""
    video_path = (st.session_state.get("video_origin_path") or "").strip()
    if video_path:
        sync_frame_analysis_with_video(video_path)

    doc_settings = get_documentary_compact_settings() if compact else get_documentary_settings()
    require_subtitle = requires_subtitle_before_frame_analysis(doc_settings)
    subtitle_path = _session_subtitle_path(video_path) if video_path else ""
    subtitle_ready = bool(subtitle_path)

    _render_frame_analysis_status(video_path)

    interval_key = "frame_interval_input_compact" if compact else "frame_interval_input_full"
    default_interval = config.frames.get("frame_interval_input", 3)
    if interval_key not in st.session_state:
        st.session_state[interval_key] = default_interval

    with st.expander("抽帧分析（ffmpeg 抽帧 + 视觉模型）", expanded=False):
        if require_subtitle:
            st.caption(
                "须先完成上方字幕转写/上传，再抽帧：同一次视觉调用识读硬字幕，"
                "并自动 OCR + LLM 校正字幕，生成脚本时复用 JSON。"
            )
        else:
            st.caption(
                "抽帧并调用视觉模型写出分析 JSON；生成脚本时将复用该文件。"
            )

        if require_subtitle:
            if subtitle_ready:
                st.success(f"字幕已就绪: **{os.path.basename(subtitle_path)}**，可开始抽帧")
            else:
                st.warning("请先在上方转写或上传字幕，再点击「抽帧并分析」")

        input_cols = st.columns(2)
        with input_cols[0]:
            st.number_input(
                tr("Frame Interval (seconds)"),
                min_value=0.0,
                value=float(st.session_state.get(interval_key, default_interval)),
                help=tr("Frame Interval (seconds) (More keyframes consume more tokens)"),
                key=interval_key,
            )
        with input_cols[1]:
            st.number_input(
                tr("Batch Size"),
                min_value=0,
                value=st.session_state.get("vision_batch_size", config.frames.get("vision_batch_size", 10)),
                help=tr("Batch Size (More keyframes consume more tokens)"),
                key="vision_batch_size",
            )

        st.session_state["frame_interval_input"] = st.session_state.get(
            interval_key,
            default_interval,
        )

        st.checkbox(
            "复用已有抽帧分析（生成脚本时跳过视觉模型）",
            value=bool(st.session_state.get("doc_reuse_frame_analysis", True)),
            key="doc_reuse_frame_analysis",
            help="若存在与当前视频同名的分析文件，或下方已指定 JSON，则生成脚本时不再调用视觉模型",
        )

        default_path = ""
        if video_path:
            default_path = default_analysis_path_for_video(video_path)
            st.caption(f"默认可复用路径: `{default_path}`")

        if "doc_frame_analysis_file_processed" not in st.session_state:
            st.session_state["doc_frame_analysis_file_processed"] = False

        analysis_file = st.file_uploader(
            "上传抽帧分析 JSON",
            type=["json"],
            accept_multiple_files=False,
            key="docu_frame_analysis_uploader",
            help="上传后将保存为与当前视频同名的分析文件；须先有字幕才会自动校准",
        )

        override_path = (st.session_state.get("frame_analysis_json_path") or "").strip()
        if (
            override_path
            and os.path.isfile(override_path)
            and override_path != default_analysis_path_for_video(video_path or "")
        ):
            st.info(f"已指定抽帧分析: {os.path.basename(override_path)}")
            if st.button("清除已上传抽帧分析", key="doc_clear_frame_analysis"):
                st.session_state["frame_analysis_json_path"] = None
                st.session_state["doc_frame_analysis_file_processed"] = False
                st.rerun()

        if analysis_file is not None and not st.session_state.get("doc_frame_analysis_file_processed"):
            if require_subtitle and not subtitle_ready:
                st.error("请先完成上方字幕转写/上传，再上传抽帧分析 JSON（以便自动校准字幕）")
            else:
                try:
                    payload = json.loads(analysis_file.getvalue().decode("utf-8"))
                    if not is_valid_analysis_artifact(payload):
                        st.error("无效的抽帧分析 JSON：缺少 batches 或 frame_observations 字段")
                        st.stop()

                    safe_filename = os.path.basename(analysis_file.name)
                    if video_path:
                        target_path = default_analysis_path_for_video(video_path)
                    else:
                        analysis_dir = DocumentaryFrameExtractionService.analysis_artifact_dir()
                        os.makedirs(analysis_dir, exist_ok=True)
                        target_path = os.path.join(analysis_dir, safe_filename)
                        if os.path.exists(target_path):
                            timestamp = time.strftime("%Y%m%d%H%M%S")
                            name, ext = os.path.splitext(safe_filename)
                            target_path = os.path.join(analysis_dir, f"{name}_{timestamp}{ext}")

                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with open(target_path, "w", encoding="utf-8") as fp:
                        json.dump(payload, fp, ensure_ascii=False, indent=2)

                    st.session_state["frame_analysis_json_path"] = target_path
                    st.session_state["doc_frame_analysis_file_processed"] = True

                    calibration_summary = ""
                    if video_path and subtitle_ready:
                        calibration = calibrate_subtitles_after_frame_analysis(
                            analysis_json_path=target_path,
                            video_path=video_path,
                            subtitle_path=subtitle_path,
                            video_theme=st.session_state.get("video_theme", ""),
                            documentary_settings=doc_settings,
                            allow_vision_ocr_fallback=False,
                        )
                        apply_subtitle_calibration_to_session(calibration)
                        calibration_summary = format_subtitle_calibration_summary(calibration)

                    st.success(f"抽帧分析已保存: {os.path.basename(target_path)}")
                    if calibration_summary:
                        st.info(calibration_summary)
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("无法解析 JSON 文件，请检查格式")
                except Exception as exc:
                    st.error(f"{tr('Upload failed')}: {str(exc)}")

        can_extract = bool(
            video_path
            and os.path.isfile(video_path)
            and (not require_subtitle or subtitle_ready)
        )
        if video_path and os.path.isfile(video_path):
            if st.button(
                "抽帧并分析",
                key="doc_extract_frame_analysis_btn",
                use_container_width=True,
                disabled=not can_extract,
            ):
                extract_frame_analysis_docu(params, compact=compact)
        else:
            st.warning("请先在上方选择或上传视频文件，再进行抽帧分析")
