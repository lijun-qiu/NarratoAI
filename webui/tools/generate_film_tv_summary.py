#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project: NarratoAI
@File   : generate_film_tv_summary.py
@Description: 电影/电视剧解说脚本生成
"""

import os
import json
import time
import traceback
import streamlit as st
from loguru import logger

from app.config import config
from app.services.SDE.short_drama_explanation import analyze_subtitle, generate_narration_script
from app.services.subtitle_text import read_subtitle_text
import app.services.llm  # noqa: F401 — 触发 LLM 提供商注册
from app.services.llm.migration_adapter import SubtitleAnalyzerAdapter
from webui.tools.generate_short_summary import parse_and_fix_json

PROMPT_CATEGORY = "film_tv_narration"


def generate_script_film_tv_summary(params, subtitle_path, video_theme, temperature):
    """
    生成电影/电视剧解说视频脚本
    要求: 提供高质量字幕（可上传 SRT 或通过 Fun-ASR 转写）
    """
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress: float, message: str = ""):
        progress_bar.progress(progress)
        if message:
            status_text.text(f"{progress}% - {message}")
        else:
            status_text.text(f"进度: {progress}%")

    try:
        with st.spinner("正在生成影视解说脚本..."):
            if not params.video_origin_path:
                st.error("请先选择视频文件")
                return

            update_progress(30, "正在解析字幕...")
            if not subtitle_path or not os.path.exists(subtitle_path):
                st.error("字幕文件不存在，请先上传或通过 Fun-ASR 转写字幕")
                return

            text_provider = config.app.get("text_llm_provider", "gemini").lower()
            text_api_key = config.app.get(f"text_{text_provider}_api_key")
            text_model = config.app.get(f"text_{text_provider}_model_name")
            text_base_url = config.app.get(f"text_{text_provider}_base_url")

            subtitle_content = read_subtitle_text(subtitle_path).text
            if not subtitle_content:
                st.error("字幕文件内容为空或无法读取")
                return

            if not (video_theme or "").strip():
                st.warning("建议填写影视作品名称，有助于生成更准确的解说文案")

            analyzer = None
            try:
                logger.info("使用 LLM 服务进行影视字幕分析")
                analyzer = SubtitleAnalyzerAdapter(
                    text_api_key,
                    text_model,
                    text_base_url,
                    text_provider,
                    prompt_category=PROMPT_CATEGORY,
                )
                analysis_result = analyzer.analyze_subtitle(subtitle_content)
            except Exception as e:
                logger.warning(f"使用新 LLM 服务失败，回退到旧实现: {str(e)}")
                analysis_result = analyze_subtitle(
                    subtitle_file_path=subtitle_path,
                    api_key=text_api_key,
                    model=text_model,
                    base_url=text_base_url,
                    save_result=True,
                    temperature=temperature,
                    provider=text_provider,
                    prompt_category=PROMPT_CATEGORY,
                )

            if analysis_result["status"] != "success":
                logger.error(f"分析失败: {analysis_result.get('message', 'unknown')}")
                st.error("剧情分析失败，请检查日志")
                st.stop()

            logger.info("影视字幕分析成功")
            update_progress(60, "正在生成解说文案...")

            narration_result = None
            if analyzer is not None:
                try:
                    narration_result = analyzer.generate_narration_script(
                        short_name=video_theme or "未命名影视作品",
                        plot_analysis=analysis_result["analysis"],
                        subtitle_content=subtitle_content,
                        temperature=temperature,
                    )
                except Exception as e:
                    logger.warning(f"解说文案生成失败，回退到旧实现: {str(e)}")

            if narration_result is None or narration_result.get("status") != "success":
                narration_result = generate_narration_script(
                    short_name=video_theme or "未命名影视作品",
                    plot_analysis=analysis_result["analysis"],
                    subtitle_content=subtitle_content,
                    api_key=text_api_key,
                    model=text_model,
                    base_url=text_base_url,
                    save_result=True,
                    temperature=temperature,
                    provider=text_provider,
                    prompt_category=PROMPT_CATEGORY,
                )

            if narration_result["status"] != "success":
                logger.error(f"解说文案生成失败: {narration_result['message']}")
                st.error("生成脚本失败，请检查日志")
                st.stop()

            narration_dict = parse_and_fix_json(narration_result["narration_script"])
            if narration_dict is None:
                st.error("生成的解说文案格式错误，无法解析为 JSON")
                st.stop()

            if "items" not in narration_dict:
                st.error("生成的解说文案缺少必要的 'items' 字段")
                st.stop()

            script = json.dumps(narration_dict["items"], ensure_ascii=False, indent=2)
            st.session_state["video_clip_json"] = json.loads(script)
            update_progress(90, "整理输出...")

        time.sleep(0.1)
        progress_bar.progress(100)
        status_text.text("脚本生成完成！")
        st.success("影视解说脚本生成成功！")

    except Exception as err:
        st.error(f"生成过程中发生错误: {str(err)}")
        logger.exception(f"生成影视解说脚本时发生错误\n{traceback.format_exc()}")
    finally:
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
