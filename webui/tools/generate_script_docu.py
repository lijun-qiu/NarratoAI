# 纪录片脚本生成
import asyncio
import json
import time
import traceback

import os

import streamlit as st
from loguru import logger

from app.config import config
from app.services.documentary.frame_analysis_service import DocumentaryFrameAnalysisService
from app.services.documentary.documentary_settings import (
    get_documentary_compact_settings,
    get_documentary_settings,
    compute_ost1_segment_bounds,
)
from app.services.documentary.documentary_material_resolver import (
    load_subtitle_content_for_documentary,
    resolve_frame_analysis_path_for_documentary,
    normalize_material_source_video_path,
)
from webui.utils.script_stats import render_script_ost_summary


def _material_source_video_path() -> str:
    return normalize_material_source_video_path(
        str(st.session_state.get("doc_material_source_video_path") or "")
    )


def _resolve_subtitle_content(video_path: str) -> str:
    session_content = (st.session_state.get("subtitle_content") or "").strip()
    explicit_path = (st.session_state.get("subtitle_path") or "").strip() or None
    if st.session_state.get("doc_subtitle_file_processed") and session_content:
        return session_content
    if st.session_state.get("doc_subtitle_file_processed") and explicit_path:
        return load_subtitle_content_for_documentary(
            video_path,
            material_source_video_path="",
            explicit_path=explicit_path,
            fallback_content=session_content,
        )
    return load_subtitle_content_for_documentary(
        video_path,
        material_source_video_path=_material_source_video_path(),
        explicit_path=explicit_path,
        fallback_content=session_content,
    )


def _normalize_progress_value(progress: float | int) -> int:
    """Normalize mixed progress inputs to Streamlit's 0-100 integer range."""
    try:
        value = float(progress)
    except (TypeError, ValueError):
        return 0

    if 0.0 <= value <= 1.0:
        value *= 100

    return max(0, min(100, int(round(value))))


def generate_script_docu(params, *, compact: bool = False):
    """
    生成纪录片/逐帧解说脚本。
    适合场景: 纪录片、动物搞笑解说、荒野建造等。
    可选上传或转录 SRT，与抽帧分析结合（config: enable_subtitle_enrichment）。

    compact=True 时为逐帧精剪：先字幕×抽帧充分分析，再按高潮前置版规则生成 JSON。
    """
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress: float, message: str = ""):
        normalized_progress = _normalize_progress_value(progress)
        progress_bar.progress(normalized_progress)
        if message:
            status_text.text(f"🎬 {message}")
        else:
            status_text.text(f"📊 进度: {normalized_progress}%")

    try:
        with st.spinner("正在生成脚本..."):
            if not params.video_origin_path:
                st.error("请先选择视频文件")
                return

            vision_llm_provider = (
                st.session_state.get("vision_llm_provider") or config.app.get("vision_llm_provider", "openai")
            ).lower()
            reuse_frame_analysis = bool(st.session_state.get("doc_reuse_frame_analysis", True))
            material_source = _material_source_video_path()
            explicit_analysis = (
                st.session_state.get("frame_analysis_json_path") or ""
            ).strip() or None
            resolved_analysis_path = resolve_frame_analysis_path_for_documentary(
                params.video_origin_path,
                material_source_video_path=material_source,
                explicit_path=explicit_analysis,
                reuse=reuse_frame_analysis,
            )
            if not resolved_analysis_path:
                hint = (
                    "未找到可用的抽帧分析 JSON。请先在「抽帧分析」中点击「抽帧并分析」，"
                    "或上传/复用已有分析文件后再生成脚本。"
                )
                if material_source:
                    hint += (
                        f" 已配置素材来源视频「{os.path.basename(material_source)}」，"
                        "请确认该视频已完成抽帧分析。"
                    )
                raise ValueError(hint)
            update_progress(10, "将复用已有抽帧分析，正在生成脚本...")

            doc_settings = get_documentary_compact_settings() if compact else get_documentary_settings()
            if "doc_enable_subtitle_enrichment" in st.session_state or compact:
                doc_settings = dict(doc_settings)
                if "doc_enable_subtitle_enrichment" in st.session_state:
                    doc_settings["enable_subtitle_enrichment"] = bool(
                        st.session_state.get("doc_enable_subtitle_enrichment")
                    )
                if compact:
                    for key in (
                        "enable_opening_closing_hook",
                        "opening_hook_template",
                        "closing_hook_template",
                        "append_custom_prompt",
                    ):
                        if key in st.session_state:
                            doc_settings[key] = st.session_state[key]
                elif "append_custom_prompt" in st.session_state:
                    doc_settings["append_custom_prompt"] = st.session_state[
                        "append_custom_prompt"
                    ]
            subtitle_content = ""
            if doc_settings.get("enable_subtitle_enrichment", True):
                subtitle_content = _resolve_subtitle_content(params.video_origin_path)
                if subtitle_content:
                    update_progress(12, "已加载字幕，将先分析字幕×抽帧再生成脚本...")
                elif compact and doc_settings.get("require_subtitle_for_script", True):
                    hint = (
                        "逐帧精剪需要字幕文件。请先上传/转录 SRT，"
                        "或完成 OCR 字幕后将 *_ocr_refined.srt 与视频配对。"
                    )
                    if material_source:
                        hint += (
                            f" 可在「抽帧/字幕来源视频」指定有字幕素材"
                            f"（当前：{os.path.basename(material_source)}）。"
                        )
                    raise ValueError(hint)
            default_interval = doc_settings.get("frame_interval_input") or config.frames.get(
                "frame_interval_input", 3
            )
            frame_interval_input = (
                st.session_state.get("frame_interval_input")
                or default_interval
            )
            vision_batch_size = st.session_state.get("vision_batch_size") or config.frames.get("vision_batch_size", 10)
            vision_max_concurrency = st.session_state.get("vision_max_concurrency") or config.frames.get(
                "vision_max_concurrency", 2
            )

            mode_label = "逐帧精剪" if compact else "逐帧解说"
            update_progress(12, f"复用抽帧分析，正在生成{mode_label}脚本...")
            service = DocumentaryFrameAnalysisService()
            script_items = asyncio.run(
                service.generate_documentary_script(
                    video_path=params.video_origin_path,
                    video_theme=st.session_state.get("video_theme", ""),
                    custom_prompt=st.session_state.get("custom_prompt", ""),
                    append_custom_prompt=st.session_state.get("append_custom_prompt", ""),
                    frame_interval_input=frame_interval_input,
                    vision_batch_size=vision_batch_size,
                    vision_llm_provider=vision_llm_provider,
                    progress_callback=update_progress,
                    vision_api_key=None,
                    vision_model_name=None,
                    vision_base_url=None,
                    max_concurrency=vision_max_concurrency,
                    documentary_settings=doc_settings,
                    subtitle_content=subtitle_content,
                    analysis_json_path=resolved_analysis_path,
                    material_source_video_path=material_source,
                    reuse_frame_analysis=reuse_frame_analysis,
                )
            )

            logger.info(f"{mode_label}脚本生成完成，共 {len(script_items)} 个片段")
            st.session_state["documentary_script_mode"] = "auto_compact" if compact else "auto"
            script = json.dumps(script_items, ensure_ascii=False, indent=2)
            if isinstance(script, list):
                st.session_state["video_clip_json"] = script
            elif isinstance(script, str):
                st.session_state["video_clip_json"] = json.loads(script)
            update_progress(100, "脚本生成完成")

        time.sleep(0.1)
        progress_bar.progress(100)
        status_text.text("🎉 脚本生成完成！")
        st.success("✅ 视频脚本生成成功！")
        max_ost1 = None
        min_ost1 = None
        if compact:
            min_ost1, max_ost1 = compute_ost1_segment_bounds(
                len(st.session_state.get("video_clip_json") or script_items),
                doc_settings,
            )
        render_script_ost_summary(
            st.session_state.get("video_clip_json") or script_items,
            min_ost1=min_ost1,
            max_ost1=max_ost1,
        )

    except Exception as err:
        st.error(f"❌ 生成过程中发生错误: {str(err)}")
        logger.exception(f"生成脚本时发生错误\n{traceback.format_exc()}")
    finally:
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
