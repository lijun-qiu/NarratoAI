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
from app.services.film_tv_settings import (
    TV_CONTENT_SERIES,
    get_film_tv_settings,
    get_film_tv_script_prompt_params,
)
from app.services.subtitle_text import read_subtitle_text
from app.utils.video_processor import VideoProcessor
import app.services.llm  # noqa: F401 — 触发 LLM 提供商注册
from app.services.llm.migration_adapter import SubtitleAnalyzerAdapter, _run_async_safely
from app.services.llm.unified_service import UnifiedLLMService
from app.services.film_tv_script_optimizer import (
    AUTO_NARRATION_MARKER,
    apply_tv_series_bookends,
    build_film_tv_script_summary,
    fill_auto_narration_placeholders,
    optimize_film_tv_script,
    supplement_film_tv_segment_counts,
    validate_film_tv_script,
    validate_film_tv_script_counts,
)
from webui.tools.generate_short_summary import parse_and_fix_json

PROMPT_CATEGORY = "film_tv_narration"


def _format_llm_error_for_ui(message: str) -> str:
    """将 LLM 接口原始错误转为可操作的界面提示。"""
    msg = (message or "").strip()
    lower = msg.lower()
    if "insufficient_quota" in lower or "quota failed" in lower or "pre-consumed quota" in lower:
        return (
            "**LLM 账户余额不足**（403 insufficient_quota）。\n\n"
            "字幕剧情分析需要一次性提交整集字幕，网关会**预扣约 $0.14** 额度，"
            "而你当前余额不足（日志里 user quota 小于 need quota）。\n\n"
            "**处理办法：**\n"
            "1. 到 `api.4022543.xyz`（或你配置的 text_openai_base_url）充值；\n"
            "2. 在「基础设置」换余额充足的 API Key；\n"
            "3. 换更便宜的文本模型（当前为 config 里 text_openai_model_name）。\n\n"
            f"原始信息：{msg[:500]}"
        )
    if "401" in msg or "authentication" in lower or "invalid api key" in lower:
        return f"**LLM 认证失败**，请检查 config.toml 中 text_openai_api_key。\n\n原始信息：{msg[:500]}"
    if "rate limit" in lower or "429" in msg:
        return f"**LLM 请求过于频繁**（429），请稍后再试。\n\n原始信息：{msg[:500]}"
    return f"剧情分析失败：{msg[:800]}"


def _render_film_tv_script_summary_panel(summary: dict) -> None:
    """在页面上展示成片脚本原声/解说段统计。"""
    if not summary:
        return
    with st.container(border=True):
        st.markdown("**成片脚本组成（原声 / 解说）**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "解说 OST=0",
                f"{summary.get('ost0_count', 0)} 段",
                f"{summary.get('narration_pct', 0):.1f}% · {summary.get('ost0_sec', 0):.0f}s",
            )
        with c2:
            st.metric(
                "原声 OST=1",
                f"{summary.get('ost1_count', 0)} 段",
                f"{summary.get('original_pct', 0):.1f}% · {summary.get('ost1_sec', 0):.0f}s",
            )
        with c3:
            st.metric(
                "成片总时长（估）",
                f"{summary.get('total_sec', 0):.0f}s",
                f"目标 {summary.get('narration_target', 60)}/{summary.get('original_target', 40)}",
            )
        d1, d2 = st.columns(2)
        with d1:
            ost0_ids = summary.get("ost0_ids") or []
            st.caption(
                f"解说段 _id：{', '.join(map(str, ost0_ids)) if ost0_ids else '无'}"
            )
            st.caption(
                f"最长连续解说 {summary.get('max_consecutive_ost0', 0)} 段 "
                f"(上限 {summary.get('max_consecutive_ost0_limit', 3)})"
            )
        with d2:
            ost1_ids = summary.get("ost1_ids") or []
            st.caption(
                f"原声段 _id：{', '.join(map(str, ost1_ids)) if ost1_ids else '无'}"
            )
            st.caption(
                f"最长连续原声 {summary.get('max_consecutive_ost1', 0)} 段 "
                f"(上限 {summary.get('max_consecutive_ost1_limit', 3)})"
            )
        if summary.get("unanchored_ost0_count", 0) > 0:
            st.warning(
                f"仍有 {summary['unanchored_ost0_count']} 段解说时间戳未对齐字幕，建议重新生成。"
            )
        if summary.get("validation_ok"):
            st.success("段数、时长占比与穿插结构均符合当前规则。")
        else:
            st.warning(summary.get("validation_message", "脚本未完全通过校验"))


def _generate_prev_episode_recap(
    *,
    film_name: str,
    episode_number: int,
    work_brief: str,
    plot_analysis: str,
    film_tv_settings: dict,
    text_api_key: str,
    text_model: str,
    text_base_url: str,
    text_provider: str,
    temperature: float,
) -> str:
    """非首集时生成上集剧情回顾（供开场解说拼接）。"""
    if episode_number <= 1:
        return ""
    if not film_tv_settings.get("tv_recap_prev_episode", True):
        return ""

    chars_min = int(film_tv_settings.get("tv_recap_chars_min") or 40)
    chars_max = int(film_tv_settings.get("tv_recap_chars_max") or 80)
    prev_ep = episode_number - 1
    prompt = (
        f"你是影视解说编剧。请为《{film_name}》第 {episode_number} 集解说稿写一段「上集回顾」。\n"
        f"要求：概括第 {prev_ep} 集主要剧情（人物关系、核心冲突、结尾悬念），"
        f"约 {chars_min}-{chars_max} 个汉字，口语化，不要用列表，不要剧透本集。\n"
        f"只输出回顾正文，不要标题、不要 markdown。\n\n"
        f"作品调研：\n{(work_brief or '')[:2500]}\n\n"
        f"本集字幕分析摘要：\n{(plot_analysis or '')[:2500]}"
    )
    try:
        result = _run_async_safely(
            UnifiedLLMService.generate_text,
            prompt=prompt,
            system_prompt="你只输出一段中文解说回顾正文。",
            provider=text_provider,
            temperature=min(1.0, temperature + 0.1),
            api_key=text_api_key,
            api_base=text_base_url,
        )
        recap = (result or "").strip()
        if recap:
            logger.info(f"已生成第 {prev_ep} 集回顾，约 {len(recap)} 字")
        return recap
    except Exception as exc:
        logger.warning(f"上集回顾生成失败: {exc}")
        return ""


def _apply_tv_bookends_if_needed(
    items: list,
    *,
    film_name: str,
    film_tv_settings: dict,
    subtitle_content: str,
    source_duration_sec: float,
    prev_episode_recap: str = "",
) -> list:
    if film_tv_settings.get("content_type") != TV_CONTENT_SERIES:
        return items
    return apply_tv_series_bookends(
        items,
        film_name=film_name,
        settings=film_tv_settings,
        subtitle_content=subtitle_content,
        source_duration_sec=source_duration_sec or None,
        prev_episode_recap=prev_episode_recap,
    )


def _fill_auto_narrations_with_llm(
    items: list,
    *,
    film_name: str,
    plot_analysis: str,
    film_tv_settings: dict,
    text_api_key: str,
    text_model: str,
    text_base_url: str,
    text_provider: str,
    temperature: float,
) -> list:
    pending = [item for item in items if item.get("narration") == AUTO_NARRATION_MARKER]
    if not pending:
        return items

    chars_min = int(film_tv_settings["narration_chars_min"])
    chars_max = int(film_tv_settings["narration_chars_max"])
    brief_items = [
        {
            "_id": item.get("_id"),
            "picture": item.get("picture"),
            "timestamp": item.get("timestamp"),
        }
        for item in pending
    ]
    prompt = (
        f"你是专业影视解说编剧。请为《{film_name}》补写 {len(pending)} 段 OST=0 解说词。\n"
        f"要求：每段 {chars_min}-{chars_max} 个汉字，承上启下，不要写「播放原片」，不要重复对白。\n"
        f"只输出 JSON 数组，每项含 _id（整数）和 narration（字符串）。\n\n"
        f"待补片段：\n{json.dumps(brief_items, ensure_ascii=False, indent=2)}\n\n"
        f"剧情分析摘要：\n{plot_analysis[:4000]}"
    )

    try:
        result = _run_async_safely(
            UnifiedLLMService.generate_text,
            prompt=prompt,
            system_prompt="你只输出合法 JSON 数组，不要 markdown 代码块。",
            provider=text_provider,
            temperature=min(1.0, temperature + 0.15),
            response_format="json",
            api_key=text_api_key,
            api_base=text_base_url,
        )
        parsed = parse_and_fix_json(result)
        if isinstance(parsed, list):
            patches = {int(p["_id"]): p["narration"] for p in parsed if "_id" in p and p.get("narration")}
        elif isinstance(parsed, dict) and "items" in parsed:
            patches = {
                int(p["_id"]): p["narration"]
                for p in parsed["items"]
                if "_id" in p and p.get("narration")
            }
        else:
            patches = {}

        updated = [dict(item) for item in items]
        filled = 0
        for item in updated:
            if item.get("narration") != AUTO_NARRATION_MARKER:
                continue
            narration = patches.get(int(item.get("_id", 0)))
            if narration:
                item["narration"] = str(narration).strip()
                filled += 1
        if filled:
            logger.info(f"LLM 已补写 {filled}/{len(pending)} 段解说文案")
            return updated
    except Exception as exc:
        logger.warning(f"LLM 补写解说失败，使用模板兜底: {exc}")

    return fill_auto_narration_placeholders(items, film_tv_settings)


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

            film_name = (video_theme or "").strip()
            script_extra_params = get_film_tv_script_prompt_params(
                source_duration_sec,
                settings=film_tv_settings,
                film_name=film_name,
            )
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
                err_msg = analysis_result.get("message", "unknown")
                logger.error(f"分析失败: {err_msg}")
                st.error(_format_llm_error_for_ui(str(err_msg)))
                st.stop()

            logger.info("影视字幕分析成功")
            update_progress(60, "专家剪辑师正在生成精剪脚本...")

            def _call_generate_narration(plot_analysis: str, gen_temperature=None):
                use_temp = temperature if gen_temperature is None else gen_temperature
                if analyzer is not None:
                    try:
                        result = analyzer.generate_narration_script(
                            short_name=film_name,
                            plot_analysis=plot_analysis,
                            subtitle_content=subtitle_content,
                            temperature=use_temp,
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
                    temperature=use_temp,
                    provider=text_provider,
                    prompt_category=PROMPT_CATEGORY,
                    script_extra_params=script_extra_params,
                    work_brief=work_brief,
                )

            plot_analysis = analysis_result["analysis"]

            prev_episode_recap = ""
            if film_tv_settings.get("content_type") == TV_CONTENT_SERIES:
                episode_no = max(1, int(film_tv_settings.get("episode_number") or 1))
                if episode_no > 1 and film_tv_settings.get("tv_recap_prev_episode", True):
                    update_progress(58, f"正在生成第 {episode_no - 1} 集回顾...")
                    prev_episode_recap = _generate_prev_episode_recap(
                        film_name=film_name,
                        episode_number=episode_no,
                        work_brief=work_brief,
                        plot_analysis=plot_analysis,
                        film_tv_settings=film_tv_settings,
                        text_api_key=text_api_key,
                        text_model=text_model,
                        text_base_url=text_base_url,
                        text_provider=text_provider,
                        temperature=temperature,
                    )

            narration_result = None
            narration_dict = None
            validation = None
            max_attempts = 3
            supplemented = False
            retry_temperature = temperature

            for attempt in range(1, max_attempts + 1):
                if attempt > 1:
                    update_progress(65, f"段数未达标，正在第 {attempt} 次重新生成...")
                    logger.info(f"影视脚本段数未达标，第 {attempt} 次生成")

                retry_temperature = min(1.2, temperature + 0.1 * (attempt - 1))
                narration_result = _call_generate_narration(plot_analysis, retry_temperature)
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
                validation = validate_film_tv_script(
                    optimized_items, film_tv_settings, subtitle_content
                )
                if validation["ok"]:
                    narration_dict["items"] = optimized_items
                    break

                if attempt < max_attempts:
                    plot_analysis = (
                        f"{analysis_result['analysis']}\n\n"
                        f"【重要修正·第{attempt}次】脚本未达标，当前输出无效。"
                        f"原声 OST=1 {validation['ost1_count']} 段（至少 {validation['ost1_min']}），"
                        f"解说 OST=0 {validation['ost0_count']} 段（至少 {validation['ost0_min']}）。"
                        f"成片时长占比：解说约 {validation.get('narration_pct', 0):.0f}%"
                        f"（目标 {validation.get('narration_target', 60)}%），"
                        f"原声约 {validation.get('original_pct', 0):.0f}%"
                        f"（目标 {validation.get('original_target', 40)}%）。"
                        f"连续解说 {validation.get('max_consecutive_ost0', 0)} 段"
                        f"（上限 {validation.get('max_consecutive_ost0_limit', 3)}），"
                        f"未贴字幕解说 {validation.get('unanchored_ost0_count', 0)} 段。"
                        f"请输出完整 JSON：解说/原声必须穿插，禁止连续超过 3 段同类型；"
                        f"OST=0 的 timestamp 必须来自字幕真实时间点，禁止等差虚构时间轴。"
                    )
                else:
                    narration_dict["items"] = optimized_items
                    logger.warning(f"LLM 重试后段数仍未达标: {validation['message']}")

            if narration_dict and "items" in narration_dict and validation and not validation["ok"]:
                update_progress(72, "段数不足，正在按配置自动补段...")
                supplemented_items, validation = supplement_film_tv_segment_counts(
                    narration_dict["items"],
                    subtitle_content=subtitle_content,
                    source_duration_sec=source_duration_sec or None,
                    settings=film_tv_settings,
                )
                if any(item.get("narration") == AUTO_NARRATION_MARKER for item in supplemented_items):
                    update_progress(78, "正在为补入的解说段撰写文案...")
                    supplemented_items = _fill_auto_narrations_with_llm(
                        supplemented_items,
                        film_name=film_name,
                        plot_analysis=analysis_result["analysis"],
                        film_tv_settings=film_tv_settings,
                        text_api_key=text_api_key,
                        text_model=text_model,
                        text_base_url=text_base_url,
                        text_provider=text_provider,
                        temperature=retry_temperature,
                    )
                optimized_items = optimize_film_tv_script(
                    supplemented_items,
                    subtitle_content=subtitle_content,
                    source_duration_sec=source_duration_sec or None,
                    settings=film_tv_settings,
                )
                validation = validate_film_tv_script(
                    optimized_items, film_tv_settings, subtitle_content
                )
                narration_dict["items"] = optimized_items
                supplemented = True

            if narration_result is None or narration_result["status"] != "success":
                err_msg = (
                    narration_result.get("message", "unknown")
                    if narration_result
                    else "unknown"
                )
                logger.error(f"解说文案生成失败: {err_msg}")
                st.error(_format_llm_error_for_ui(str(err_msg)).replace("剧情分析", "脚本生成"))
                st.stop()

            if narration_dict is None:
                st.error("生成的解说文案格式错误，无法解析为 JSON")
                st.stop()

            if "items" not in narration_dict:
                st.error("生成的解说文案缺少必要的 'items' 字段")
                st.stop()

            narration_dict["items"] = _apply_tv_bookends_if_needed(
                narration_dict["items"],
                film_name=film_name,
                film_tv_settings=film_tv_settings,
                subtitle_content=subtitle_content,
                source_duration_sec=source_duration_sec,
                prev_episode_recap=prev_episode_recap,
            )

            validation = validate_film_tv_script(
                narration_dict["items"], film_tv_settings, subtitle_content
            )
            script_summary = build_film_tv_script_summary(
                narration_dict["items"],
                film_tv_settings,
                subtitle_content,
            )
            st.session_state["film_tv_script_summary"] = script_summary

            if validation and not validation["ok"]:
                st.warning(
                    f"脚本仍未完全达标：原声 {validation['ost1_count']}/{validation['ost1_min']} 段，"
                    f"解说 {validation['ost0_count']}/{validation['ost0_min']} 段；"
                    f"时长占比 解说 {validation.get('narration_pct', 0):.0f}% / "
                    f"原声 {validation.get('original_pct', 0):.0f}% "
                    f"(目标 {validation.get('narration_target', 60)}/{validation.get('original_target', 40)})；"
                    f"连续解说 {validation.get('max_consecutive_ost0', 0)} 段。"
                    f"已自动校正，建议预览后再次生成。"
                )
            elif supplemented and validation and validation["ok"]:
                st.info(
                    f"段数与时长占比已自动校正：原声 {validation['ost1_count']} 段 "
                    f"({validation.get('original_pct', 0):.0f}%)，"
                    f"解说 {validation['ost0_count']} 段 "
                    f"({validation.get('narration_pct', 0):.0f}%)。"
                )

            _render_film_tv_script_summary_panel(script_summary)

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
