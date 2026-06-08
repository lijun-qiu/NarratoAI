# 纪录片抽帧分析（独立于脚本生成）
import asyncio
import time
import traceback

import os

import streamlit as st
from loguru import logger

from app.config import config
from app.services.documentary.documentary_settings import (
    get_documentary_compact_settings,
    get_documentary_settings,
)
from app.services.documentary.frame_analysis_pairing import (
    default_analysis_path_for_video,
    find_paired_frame_analysis_path,
)
from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService
from app.services.drama_character_registry import (
    DEFAULT_DRAMA_ID,
    merge_frame_analysis_settings_for_drama,
    resolve_active_relationship_diagram_path,
    resolve_character_references,
)
from app.services.documentary.subtitle_calibration_pipeline import (
    requires_subtitle_before_frame_analysis,
)
from app.services.subtitle_video_pairing import find_paired_subtitle_path, load_subtitle_content, resolve_subtitle_path_for_video
from app.services.srt_utils import clip_entries_to_duration, entries_to_srt, parse_srt


def _normalize_progress_value(progress: float | int) -> int:
    try:
        value = float(progress)
    except (TypeError, ValueError):
        return 0
    if 0.0 <= value <= 1.0:
        value *= 100
    return max(0, min(100, int(round(value))))


def _resolve_subtitle_content(video_path: str) -> str:
    session_content = (st.session_state.get("subtitle_content") or "").strip()
    subtitle_path = (st.session_state.get("subtitle_path") or "").strip()
    if not subtitle_path and video_path:
        subtitle_path = find_paired_subtitle_path(video_path) or ""
    if subtitle_path and os.path.isfile(subtitle_path):
        return load_subtitle_content(subtitle_path).strip() or session_content
    return session_content


def _clip_subtitle_content(content: str, max_duration_seconds: float) -> str:
    cleaned = (content or "").strip()
    if not cleaned or max_duration_seconds <= 0:
        return cleaned
    entries = parse_srt(cleaned)
    if not entries:
        return cleaned
    clipped = clip_entries_to_duration(entries, int(max_duration_seconds * 1000))
    return entries_to_srt(clipped).strip()


def extract_frame_analysis_docu(
    params,
    *,
    compact: bool = False,
    test_mode: bool = False,
    test_duration_seconds: float | None = None,
):
    """抽帧并用视觉模型分析，产出 JSON 供后续脚本生成复用。"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress: float, message: str = ""):
        normalized_progress = _normalize_progress_value(progress)
        progress_bar.progress(normalized_progress)
        if message:
            status_text.text(f"🎞️ {message}")
        else:
            status_text.text(f"📊 进度: {normalized_progress}%")

    analysis_path = ""
    keyframe_count = 0
    part_paths: list[str] = []
    test_duration = float(test_duration_seconds if test_duration_seconds is not None else 5.0)

    try:
        label = "测试抽帧" if test_mode else "抽帧"
        with st.spinner(f"正在{label}并分析..."):
            if not params.video_origin_path:
                st.error("请先选择视频文件")
                return

            vision_llm_provider = (
                st.session_state.get("vision_llm_provider")
                or config.app.get("vision_llm_provider", "openai")
            ).lower()
            vision_api_key = (
                st.session_state.get(f"vision_{vision_llm_provider}_api_key")
                or config.app.get(f"vision_{vision_llm_provider}_api_key")
            )
            vision_model = (
                st.session_state.get(f"vision_{vision_llm_provider}_model_name")
                or config.app.get(f"vision_{vision_llm_provider}_model_name")
            )
            vision_base_url = (
                st.session_state.get(f"vision_{vision_llm_provider}_base_url")
                or config.app.get(f"vision_{vision_llm_provider}_base_url", "")
            )
            if not vision_api_key or not vision_model:
                raise ValueError(
                    f"未配置 {vision_llm_provider} 的视觉模型参数。"
                    f"请在设置中配置 vision_{vision_llm_provider}_api_key 和 vision_{vision_llm_provider}_model_name"
                )

            doc_settings = get_documentary_compact_settings() if compact else get_documentary_settings()
            if "doc_enable_subtitle_enrichment" in st.session_state:
                doc_settings = dict(doc_settings)
                doc_settings["enable_subtitle_enrichment"] = bool(
                    st.session_state.get("doc_enable_subtitle_enrichment")
                )

            drama_id = str(st.session_state.get("doc_frame_drama_id") or DEFAULT_DRAMA_ID).strip()
            enable_knowledge_text = bool(st.session_state.get("doc_frame_enable_drama_knowledge_text"))
            enable_relationship_diagram = bool(st.session_state.get("doc_frame_enable_relationship_diagram"))
            doc_settings = dict(doc_settings)
            doc_settings["frame_reference_token_saver"] = bool(
                st.session_state.get("doc_frame_reference_token_saver", True)
            )
            doc_settings = merge_frame_analysis_settings_for_drama(
                doc_settings,
                drama_id,
                enable_knowledge_text=enable_knowledge_text,
            )
            relationship_diagram_path = resolve_active_relationship_diagram_path(
                drama_id,
                enabled=enable_relationship_diagram,
            )
            selected_names = set(st.session_state.get("doc_frame_selected_character_names") or [])
            character_references = (
                st.session_state.get("doc_frame_character_references")
                or resolve_character_references(drama_id, selected_names=selected_names)
            )
            video_theme = str(st.session_state.get("video_theme") or drama_id).strip()

            subtitle_path = resolve_subtitle_path_for_video(
                params.video_origin_path,
                explicit_path=st.session_state.get("subtitle_path"),
            )
            if requires_subtitle_before_frame_analysis(doc_settings) and not subtitle_path:
                st.error(
                    "请先在「素材预处理 → 字幕转录」中完成转写或上传，再执行抽帧并分析。"
                )
                return

            subtitle_content = ""
            if doc_settings.get("enable_subtitle_enrichment", True):
                subtitle_content = _resolve_subtitle_content(params.video_origin_path)

            test_duration = float(
                test_duration_seconds
                if test_duration_seconds is not None
                else st.session_state.get("doc_frame_test_duration_seconds", test_duration)
            )
            if test_mode:
                subtitle_content = _clip_subtitle_content(subtitle_content, test_duration)

            default_interval = doc_settings.get("frame_interval_input") or config.frames.get(
                "frame_interval_input", 3
            )
            frame_interval_input = st.session_state.get("frame_interval_input") or default_interval
            vision_batch_size = st.session_state.get("vision_batch_size") or config.frames.get(
                "vision_batch_size", 10
            )
            vision_max_concurrency = st.session_state.get("vision_max_concurrency") or config.frames.get(
                "vision_max_concurrency", 2
            )

            service = DocumentaryFrameExtractionService()
            result = asyncio.run(
                service.analyze_video(
                    video_path=params.video_origin_path,
                    video_theme=video_theme,
                    custom_prompt=st.session_state.get("custom_prompt", ""),
                    frame_interval_input=frame_interval_input,
                    vision_batch_size=vision_batch_size,
                    vision_llm_provider=vision_llm_provider,
                    progress_callback=update_progress,
                    vision_api_key=vision_api_key,
                    vision_model_name=vision_model,
                    vision_base_url=vision_base_url,
                    max_concurrency=vision_max_concurrency,
                    documentary_settings=doc_settings,
                    subtitle_content=subtitle_content,
                    drama_id=drama_id,
                    character_references=character_references,
                    relationship_diagram_path=relationship_diagram_path,
                    frame_drama_knowledge_text_enabled=enable_knowledge_text,
                    frame_relationship_diagram_enabled=enable_relationship_diagram,
                    max_duration_seconds=test_duration if test_mode else None,
                    test_mode=test_mode,
                )
            )

            analysis_path = result["analysis_json_path"]
            st.session_state["frame_analysis_json_path"] = analysis_path
            st.session_state["doc_frame_analysis_upload_explicit"] = False
            st.session_state["doc_frame_analysis_file_processed"] = True
            st.session_state["_frame_analysis_synced_video_path"] = params.video_origin_path
            keyframe_count = len(result.get("keyframe_files") or [])
            logger.info(f"{'测试' if test_mode else ''}抽帧分析完成: {analysis_path}，共 {keyframe_count} 帧")

            split_parts = int(st.session_state.get("doc_output_split_parts") or 1)
            if test_mode:
                split_parts = 1
            split_result = DocumentaryFrameExtractionService.save_split_analysis_artifacts(
                analysis_path,
                split_parts,
                artifact=result.get("analysis_artifact"),
            )
            part_paths = split_result.get("part_paths") or []

        time.sleep(0.1)
        progress_bar.progress(100)
        status_text.text("🎉 测试抽帧完成！" if test_mode else "🎉 抽帧分析完成！")
        prefix = "✅ 测试抽帧已保存" if test_mode else "✅ 抽帧分析已保存"
        success_msg = (
            f"{prefix}: `{os.path.basename(analysis_path)}`（{keyframe_count} 帧）"
        )
        if test_mode:
            success_msg += f"\n\n仅分析前 **{test_duration:g}** 秒，不会覆盖完整版 `{os.path.basename(DocumentaryFrameExtractionService.default_analysis_path_for_video(params.video_origin_path))}`。"
        if part_paths:
            success_msg += f"\n\n另存 **{len(part_paths)}** 份切割文件："
            success_msg += "\n".join(f"- `{os.path.basename(path)}`" for path in part_paths)
            success_msg += "\n\n当前仍使用完整 JSON。"
        st.success(success_msg)

    except Exception as err:
        st.error(f"❌ 抽帧分析失败: {str(err)}")
        logger.exception(f"抽帧分析失败\n{traceback.format_exc()}")
    finally:
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
