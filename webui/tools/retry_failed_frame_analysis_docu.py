# 重跑抽帧分析中失败的批次
import asyncio
import os
import time
import traceback

import streamlit as st
from loguru import logger

from app.config import config
from app.services.documentary.documentary_settings import (
    get_documentary_compact_settings,
    get_documentary_settings,
)
from app.services.documentary.frame_analysis_pairing import load_analysis_artifact
from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService
from app.services.drama_character_registry import (
    DEFAULT_DRAMA_ID,
    merge_frame_analysis_settings_for_drama,
    resolve_active_relationship_diagram_path,
    resolve_character_references,
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


def _resolve_subtitle_content(video_path: str) -> str:
    session_content = (st.session_state.get("subtitle_content") or "").strip()
    subtitle_path = (st.session_state.get("subtitle_path") or "").strip()
    if not subtitle_path and video_path:
        subtitle_path = find_paired_subtitle_path(video_path) or ""
    if subtitle_path and os.path.isfile(subtitle_path):
        return load_subtitle_content(subtitle_path).strip() or session_content
    return session_content


def retry_failed_frame_analysis_docu(
    params,
    *,
    compact: bool = False,
    analysis_json_path: str = "",
):
    """仅重跑已有 JSON 中 status=failed 的批次。"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress: float, message: str = ""):
        normalized_progress = _normalize_progress_value(progress)
        progress_bar.progress(normalized_progress)
        if message:
            status_text.text(f"🔁 {message}")
        else:
            status_text.text(f"📊 进度: {normalized_progress}%")

    try:
        target_path = (analysis_json_path or st.session_state.get("frame_analysis_json_path") or "").strip()
        if not target_path or not os.path.isfile(target_path):
            st.error("请先选用或导入抽帧分析 JSON")
            return

        artifact = load_analysis_artifact(target_path)
        failed_count = DocumentaryFrameExtractionService.count_failed_batches(artifact)
        if failed_count <= 0:
            st.info("当前 JSON 没有失败批次，无需重跑")
            return

        video_path = (
            (params.video_origin_path if params else "")
            or str(artifact.get("video_path") or "")
            or (st.session_state.get("doc_material_source_video_path") or "")
        ).strip()

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

        drama_id = str(
            st.session_state.get("doc_frame_drama_id")
            or artifact.get("drama_id")
            or DEFAULT_DRAMA_ID
        ).strip()
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

        subtitle_content = ""
        if doc_settings.get("enable_subtitle_enrichment", True) and video_path:
            subtitle_content = _resolve_subtitle_content(video_path)

        vision_max_concurrency = st.session_state.get("vision_max_concurrency") or config.frames.get(
            "vision_max_concurrency", 2
        )

        with st.spinner(f"正在重跑 {failed_count} 个失败批次..."):
            service = DocumentaryFrameExtractionService()
            result = asyncio.run(
                service.retry_failed_batches(
                    analysis_json_path=target_path,
                    video_path=video_path,
                    video_theme=video_theme,
                    custom_prompt=st.session_state.get("custom_prompt", ""),
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
                )
            )

        st.session_state["frame_analysis_json_path"] = result["analysis_json_path"]
        st.session_state["doc_frame_analysis_file_processed"] = True

        recovered = int(result.get("recovered") or 0)
        still_failed = int(result.get("still_failed") or 0)
        retried = int(result.get("retried") or 0)
        logger.info(
            f"失败批次重跑: {target_path}，重试 {retried}，成功 {recovered}，仍失败 {still_failed}"
        )

        time.sleep(0.1)
        progress_bar.progress(100)
        status_text.text("🎉 失败批次重跑完成！")
        if still_failed:
            artifact_after = load_analysis_artifact(target_path)
            report = DocumentaryFrameExtractionService.format_failed_batches_report(artifact_after)
            logger.warning(report)
            st.warning(
                f"已重跑 {retried} 个批次：成功 {recovered}，仍有 {still_failed} 个失败，可再次点击重试"
            )
            with st.expander("仍失败的批次详情", expanded=True):
                st.text(report)
                for item in DocumentaryFrameExtractionService.list_failed_batch_details(artifact_after):
                    st.markdown(f"**批次 #{item['batch_index']}** · `{item['time_range']}`")
                    st.error(item["error_message"])
                    if item["raw_response_preview"]:
                        st.code(item["raw_response_preview"][:800], language="json")
        else:
            st.success(f"✅ 已补全全部 {recovered} 个失败批次: `{os.path.basename(target_path)}`")

    except Exception as err:
        st.error(f"❌ 重跑失败批次出错: {str(err)}")
        logger.exception(f"重跑失败批次出错\n{traceback.format_exc()}")
    finally:
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
