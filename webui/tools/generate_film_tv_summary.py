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
from app.services.SDE.short_drama_explanation import analyze_subtitle, generate_narration_script, research_film_work
from app.services.film_tv_settings import get_film_tv_settings, get_film_tv_script_prompt_params
from app.services.film_tv_script_optimizer import optimize_film_tv_script, validate_film_tv_script_counts
from app.services.subtitle_text import read_subtitle_text
from app.utils.video_processor import VideoProcessor
import app.services.llm  # noqa: F401 — 触发 LLM 提供商注册
from app.services.llm.migration_adapter import SubtitleAnalyzerAdapter
from webui.tools.generate_short_summary import parse_and_fix_json

PROMPT_CATEGORY = "film_tv_narration"


def generate_script_film_tv_summary(params, subtitle_path, video_theme, temperature, film_tv_settings=None):
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

            film_tv_settings = get_film_tv_settings(film_tv_settings)

            source_duration_sec = 0.0
            try:
                source_duration_sec = VideoProcessor(params.video_origin_path).duration
                logger.info(f"原片时长: {source_duration_sec:.1f} 秒")
            except Exception as e:
                logger.warning(f"无法读取原片时长: {e}")

            script_extra_params = get_film_tv_script_prompt_params(
                source_duration_sec, settings=film_tv_settings
            )

            film_name = (video_theme or "").strip()
            if not film_name:
                st.error("请先填写影视作品名称（剧名/片名），AI 需要先调研作品背景再生成脚本")
                return

            work_brief = ""
            analyzer = None
            try:
                logger.info(f"开始调研作品：《{film_name}》")
                update_progress(15, f"专家剪辑师正在调研《{film_name}》...")
                analyzer = SubtitleAnalyzerAdapter(
                    text_api_key,
                    text_model,
                    text_base_url,
                    text_provider,
                    prompt_category=PROMPT_CATEGORY,
                    script_extra_params=script_extra_params,
                )
                brief_result = analyzer.research_work(film_name, temperature=temperature)
            except Exception as e:
                logger.warning(f"作品调研失败，回退到旧实现: {str(e)}")
                brief_result = research_film_work(
                    film_name,
                    api_key=text_api_key,
                    model=text_model,
                    base_url=text_base_url,
                    temperature=temperature,
                    provider=text_provider,
                )

            if brief_result.get("status") == "success":
                work_brief = brief_result["work_brief"]
                logger.info(f"《{film_name}》作品调研完成")
            else:
                logger.warning(f"作品调研未成功: {brief_result.get('message', 'unknown')}，将继续仅依据字幕分析")
                work_brief = "（作品调研未完成，请主要依据字幕内容分析）"

            update_progress(35, "正在分析字幕...")
            try:
                if analyzer is None:
                    analyzer = SubtitleAnalyzerAdapter(
                        text_api_key,
                        text_model,
                        text_base_url,
                        text_provider,
                        prompt_category=PROMPT_CATEGORY,
                        script_extra_params=script_extra_params,
                    )
                logger.info("使用 LLM 服务进行影视字幕分析")
                analysis_result = analyzer.analyze_subtitle(
                    subtitle_content,
                    film_name=film_name,
                    work_brief=work_brief,
                )
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
                    film_name=film_name,
                    work_brief=work_brief,
                )

            if analysis_result["status"] != "success":
                logger.error(f"分析失败: {analysis_result.get('message', 'unknown')}")
                st.error("剧情分析失败，请检查日志")
                st.stop()

            logger.info("影视字幕分析成功")
            update_progress(60, "专家剪辑师正在生成精剪脚本...")

            def _call_generate_narration(plot_analysis: str):
                if analyzer is not None:
                    try:
                        result = analyzer.generate_narration_script(
                            short_name=film_name,
                            plot_analysis=plot_analysis,
                            subtitle_content=subtitle_content,
                            temperature=temperature,
                            work_brief=work_brief,
                        )
                        if result.get("status") == "success":
                            return result
                    except Exception as e:
                        logger.warning(f"解说文案生成失败，回退到旧实现: {str(e)}")
                return generate_narration_script(
                    short_name=film_name,
                    plot_analysis=plot_analysis,
                    subtitle_content=subtitle_content,
                    api_key=text_api_key,
                    model=text_model,
                    base_url=text_base_url,
                    save_result=True,
                    temperature=temperature,
                    provider=text_provider,
                    prompt_category=PROMPT_CATEGORY,
                    script_extra_params=script_extra_params,
                    work_brief=work_brief,
                )

            plot_analysis = analysis_result["analysis"]
            narration_result = None
            narration_dict = None
            validation = None
            max_attempts = 2

            for attempt in range(1, max_attempts + 1):
                if attempt > 1:
                    update_progress(65, f"段数未达标，正在第 {attempt} 次重新生成...")
                    logger.info(f"影视脚本段数未达标，第 {attempt} 次生成")

                narration_result = _call_generate_narration(plot_analysis)
                if narration_result["status"] != "success":
                    break

                narration_dict = parse_and_fix_json(narration_result["narration_script"])
                if narration_dict is None or "items" not in narration_dict:
                    break

                optimized_items = optimize_film_tv_script(
                    narration_dict["items"],
                    subtitle_content=subtitle_content,
                    source_duration_sec=source_duration_sec or None,
                    settings=film_tv_settings,
                )
                validation = validate_film_tv_script_counts(optimized_items, film_tv_settings)
                if validation["ok"]:
                    narration_dict["items"] = optimized_items
                    break

                if attempt < max_attempts:
                    plot_analysis = (
                        f"{analysis_result['analysis']}\n\n"
                        f"【重要修正】上次脚本段数不足：原声 OST=1 仅 {validation['ost1_count']} 段"
                        f"（需≥{validation['ost1_min']}），解说 OST=0 仅 {validation['ost0_count']} 段"
                        f"（需≥{validation['ost0_min']}），总段数 {validation['total']}"
                        f"（需≥{validation['total_min']}）。"
                        f"请重新输出完整 JSON，务必补全至至少 {validation['ost1_min']} 段 OST=1"
                        f" 和 {validation['ost0_min']} 段 OST=0。"
                    )
                else:
                    narration_dict["items"] = optimized_items
                    logger.warning(f"重试后段数仍未达标: {validation['message']}")

            if narration_result is None or narration_result["status"] != "success":
                logger.error(f"解说文案生成失败: {narration_result.get('message', 'unknown') if narration_result else 'unknown'}")
                st.error("生成脚本失败，请检查日志")
                st.stop()

            if narration_dict is None:
                st.error("生成的解说文案格式错误，无法解析为 JSON")
                st.stop()

            if "items" not in narration_dict:
                st.error("生成的解说文案缺少必要的 'items' 字段")
                st.stop()

            if validation and not validation["ok"]:
                st.warning(
                    f"脚本段数未完全达到配置要求：原声 {validation['ost1_count']}/{validation['ost1_min']} 段，"
                    f"解说 {validation['ost0_count']}/{validation['ost0_min']} 段，"
                    f"共 {validation['total']}/{validation['total_min']} 段。"
                    f"已自动重试一次，建议调高 temperature 或再次点击生成。"
                )

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
