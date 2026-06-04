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
)
from app.services.subtitle_video_pairing import (
    find_paired_subtitle_path,
    load_subtitle_content,
)


def _resolve_subtitle_content(video_path: str) -> str:
    session_content = (st.session_state.get("subtitle_content") or "").strip()
    subtitle_path = (st.session_state.get("subtitle_path") or "").strip()
    if not subtitle_path and video_path:
        subtitle_path = find_paired_subtitle_path(video_path) or ""
    if subtitle_path and os.path.isfile(subtitle_path):
        return load_subtitle_content(subtitle_path).strip() or session_content
    return session_content


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

    compact=True 时为逐帧精剪：故事讲述型（30–100 字/段，35–45 段，原声≤6）。
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
                    f"未配置 {vision_llm_provider} 的 API Key 或模型名称。"
                    f"请在设置页面配置 vision_{vision_llm_provider}_api_key 和 vision_{vision_llm_provider}_model_name"
                )

            doc_settings = get_documentary_compact_settings() if compact else get_documentary_settings()
            if "doc_enable_subtitle_enrichment" in st.session_state:
                doc_settings = dict(doc_settings)
                doc_settings["enable_subtitle_enrichment"] = bool(
                    st.session_state.get("doc_enable_subtitle_enrichment")
                )
            subtitle_content = ""
            if doc_settings.get("enable_subtitle_enrichment", True):
                subtitle_content = _resolve_subtitle_content(params.video_origin_path)
                if subtitle_content:
                    update_progress(12, "已加载字幕，将与抽帧分析结合...")
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
            update_progress(10, f"正在提取关键帧（{mode_label}）...")
            service = DocumentaryFrameAnalysisService()
            script_items = asyncio.run(
                service.generate_documentary_script(
                    video_path=params.video_origin_path,
                    video_theme=st.session_state.get("video_theme", ""),
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

    except Exception as err:
        st.error(f"❌ 生成过程中发生错误: {str(err)}")
        logger.exception(f"生成脚本时发生错误\n{traceback.format_exc()}")
    finally:
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
