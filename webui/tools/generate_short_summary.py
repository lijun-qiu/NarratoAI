#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project: NarratoAI
@File   : 短剧解说脚本生成
@Author : 小林同学
@Date   : 2025/5/10 下午10:26 
'''
import os
import json
import time
import traceback
import streamlit as st
from loguru import logger

from app.config import config
from app.services.SDE.short_drama_explanation import analyze_subtitle, generate_narration_script
from app.services.subtitle_text import read_subtitle_text
from app.utils.script_json_parser import parse_narration_script_items
from app.utils.media_duration import get_video_duration_seconds
from app.utils.enhanced_mix_duration import (
    MIN_OUTPUT_RATIO,
    MAX_OUTPUT_RATIO,
    build_enhanced_mix_duration_plan,
    cap_enhanced_mix_script_playback,
    estimate_script_playback_seconds,
)
from app.utils.utils import format_time
# 导入新的LLM服务模块 - 确保提供商被注册
import app.services.llm  # 这会触发提供商注册
from app.services.llm.migration_adapter import SubtitleAnalyzerAdapter


def generate_script_short_sunmmary(
    params,
    subtitle_path,
    video_theme,
    temperature,
    require_media_name=False,
    enhanced_mix=False,
):
    """
    生成 短剧解说 / 智能混剪解说 视频脚本
    要求: 提供高质量字幕
    适合场景: 短剧、电影/电视剧混剪片段
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
        with st.spinner("正在生成脚本..."):
            if not params.video_origin_path:
                st.error("请先选择视频文件")
                return
            if require_media_name and not (video_theme or "").strip():
                st.error("请先填写电影/电视剧名称，以便 AI 理解片段在整部作品中的内容")
                return
            media_name = (video_theme or "").strip()

            video_duration_sec = get_video_duration_seconds(params.video_origin_path)
            duration_params = None
            video_duration_label = "未知"
            target_duration_label = "未知"
            min_duration_label = "未知"
            prompt_name = (
                "script_generation_enhanced" if enhanced_mix else "script_generation"
            )
            if enhanced_mix:
                if video_duration_sec:
                    plan = build_enhanced_mix_duration_plan(video_duration_sec)
                    duration_params = plan.to_prompt_parameters()
                    video_duration_label = plan.video_duration
                    target_duration_label = plan.target_duration
                    min_duration_label = plan.min_duration
                    st.session_state["enhanced_mix_duration_plan"] = plan.plan_summary
                    logger.info(f"智能混剪动态时长计划：{plan.plan_summary}")
                else:
                    st.warning("无法读取上传视频时长，智能混剪将使用默认片段规则，建议检查视频文件后重试。")

            """
            1. 获取字幕
            """
            update_progress(30, "正在解析字幕...")
            # 判断字幕文件是否存在
            if not os.path.exists(subtitle_path):
                st.error("字幕文件不存在")
                return

            """
            2. 分析字幕总结剧情 - 使用新的LLM服务架构
            """
            text_provider = config.app.get('text_llm_provider', 'gemini').lower()
            text_api_key = config.app.get(f'text_{text_provider}_api_key')
            text_model = config.app.get(f'text_{text_provider}_model_name')
            text_base_url = config.app.get(f'text_{text_provider}_base_url')

            # 读取字幕文件内容（无论使用哪种实现都需要）
            subtitle_content = read_subtitle_text(subtitle_path).text
            if not subtitle_content:
                st.error("字幕文件内容为空或无法读取")
                return

            try:
                # 优先使用新的LLM服务架构
                logger.info("使用新的LLM服务架构进行字幕分析")
                analyzer = SubtitleAnalyzerAdapter(text_api_key, text_model, text_base_url, text_provider)

                analysis_result = analyzer.analyze_subtitle(subtitle_content, drama_name=media_name)

            except Exception as e:
                logger.warning(f"使用新LLM服务失败，回退到旧实现: {str(e)}")
                # 回退到旧的实现
                analysis_result = analyze_subtitle(
                    subtitle_file_path=subtitle_path,
                    drama_name=media_name,
                    api_key=text_api_key,
                    model=text_model,
                    base_url=text_base_url,
                    save_result=True,
                    temperature=temperature,
                    provider=text_provider
                )
            """
            3. 根据剧情生成解说文案
            """
            if analysis_result["status"] == "success":
                logger.info("字幕分析成功！")
                update_progress(60, "正在生成文案...")

                # 根据剧情生成解说文案 - 使用新的LLM服务架构
                try:
                    # 优先使用新的LLM服务架构
                    logger.info("使用新的LLM服务架构生成解说文案")
                    narration_result = analyzer.generate_narration_script(
                        short_name=media_name,
                        plot_analysis=analysis_result["analysis"],
                        subtitle_content=subtitle_content,
                        temperature=temperature,
                        prompt_name=prompt_name,
                        video_duration=video_duration_label,
                        target_duration=target_duration_label,
                        min_duration=min_duration_label,
                        duration_params=duration_params,
                    )
                except Exception as e:
                    logger.warning(f"使用新LLM服务失败，回退到旧实现: {str(e)}")
                    # 回退到旧的实现
                    narration_result = generate_narration_script(
                        short_name=media_name,
                        plot_analysis=analysis_result["analysis"],
                        subtitle_content=subtitle_content,
                        api_key=text_api_key,
                        model=text_model,
                        base_url=text_base_url,
                        save_result=True,
                        temperature=temperature,
                        provider=text_provider,
                        prompt_name=prompt_name,
                        video_duration=video_duration_label,
                        target_duration=target_duration_label,
                        min_duration=min_duration_label,
                        duration_params=duration_params,
                    )

                if narration_result["status"] == "success":
                    logger.info("\n解说文案生成成功！")
                    logger.info(narration_result["narration_script"])
                else:
                    logger.info(f"\n解说文案生成失败: {narration_result['message']}")
                    st.error("生成脚本失败，请检查日志")
                    st.stop()
            else:
                logger.error(f"分析失败: {analysis_result['message']}")
                st.error("生成脚本失败，请检查日志")
                st.stop()

            """
            4. 生成文案
            """
            logger.info("开始准备生成解说文案")

            # 结果转换为JSON字符串
            narration_script = narration_result["narration_script"]

            script_items = parse_narration_script_items(narration_script)
            if not script_items:
                st.error("生成的解说文案格式错误，无法解析为有效脚本。请降低 temperature 后重试，或检查模型输出是否被截断。")
                with st.expander("查看模型原始输出"):
                    st.code(str(narration_script)[:8000])
                logger.error(f"JSON解析失败，原始内容: {str(narration_script)[:1000]}")
                st.stop()

            script = json.dumps(script_items, ensure_ascii=False, indent=2)

            if enhanced_mix and video_duration_sec:
                script_items, cap_note = cap_enhanced_mix_script_playback(
                    script_items, video_duration_sec
                )
                if cap_note:
                    logger.info(f"脚本时长校正：{cap_note}")
                    st.warning(f"已自动校正脚本：{cap_note}")
                    script = json.dumps(script_items, ensure_ascii=False, indent=2)

                estimated = estimate_script_playback_seconds(script_items)
                min_sec = video_duration_sec * MIN_OUTPUT_RATIO
                max_sec = video_duration_sec * MAX_OUTPUT_RATIO
                plan = build_enhanced_mix_duration_plan(video_duration_sec)
                span_msg = (
                    f"预计成片约 {format_time(estimated)}（{len(script_items)} 段）；"
                    f"{plan.plan_summary}"
                )
                logger.info(span_msg)
                if estimated > max_sec * 1.02:
                    st.warning(
                        f"{span_msg}。仍偏长，请重新生成或手动删减片段。"
                    )
                elif estimated < min_sec * 0.85:
                    st.warning(
                        f"{span_msg}。偏短，可适当增加片段后重新生成。"
                    )
                else:
                    st.info(span_msg)

            if script is None:
                st.error("生成脚本失败，请检查日志")
                st.stop()
            logger.success(f"剪辑脚本生成完成")
            if isinstance(script, list):
                st.session_state['video_clip_json'] = script
            elif isinstance(script, str):
                st.session_state['video_clip_json'] = json.loads(script)
            update_progress(90, "整理输出...")

        time.sleep(0.1)
        progress_bar.progress(100)
        status_text.text("脚本生成完成！")
        st.success("视频脚本生成成功！")

    except Exception as err:
        st.error(f"生成过程中发生错误: {str(err)}")
        logger.exception(f"生成脚本时发生错误\n{traceback.format_exc()}")
    finally:
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
