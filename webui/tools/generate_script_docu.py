# 纪录片脚本生成
import asyncio
import time
import traceback

import streamlit as st
from loguru import logger

from app.config import config
from app.services.documentary.frame_analysis_service import DocumentaryFrameAnalysisService
from webui.utils.script_duration_preview import store_generated_script


def _normalize_progress_value(progress: float | int) -> int:
    """Normalize mixed progress inputs to Streamlit's 0-100 integer range."""
    try:
        value = float(progress)
    except (TypeError, ValueError):
        return 0

    if 0.0 <= value <= 1.0:
        value *= 100

    return max(0, min(100, int(round(value))))


def generate_script_docu(params):
    """
    生成纪录片视频脚本。
    要求: 原视频无字幕无配音
    适合场景: 纪录片、动物搞笑解说、荒野建造等
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

            frame_interval_input = st.session_state.get("frame_interval_input") or config.frames.get(
                "frame_interval_input", 3
            )
            vision_batch_size = st.session_state.get("vision_batch_size") or config.frames.get("vision_batch_size", 10)
            vision_max_concurrency = st.session_state.get("vision_max_concurrency") or config.frames.get(
                "vision_max_concurrency", 2
            )

            update_progress(10, "正在提取关键帧...")
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
                )
            )

            logger.info(f"纪录片解说脚本生成完成，共 {len(script_items)} 个片段")
            store_generated_script(script_items)
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
