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
from app.config.llm_gateway_router import resolve_role_credentials
from app.services.SDE.short_drama_explanation import analyze_subtitle, generate_narration_script
from app.services.subtitle_text import read_subtitle_text
from app.services.documentary.documentary_settings import get_documentary_settings
from app.services.documentary.documentary_material_resolver import (
    normalize_material_source_video_path,
    resolve_video_episode_analysis_path_for_documentary,
)
from app.services.documentary.video_episode_analysis import (
    build_video_episode_analysis_markdown,
    build_video_episode_script_reference_section,
    load_video_episode_analysis_artifact,
    video_episode_summary_usable,
)
from app.services.documentary.documentary_subtitle_enrichment import (
    analyze_subtitle_with_frames,
    truncate_subtitle_content,
)
from app.services.documentary.opening_climax_resolver import (
    apply_opening_climax_chronological_replay,
    apply_opening_climax_fix,
)
from app.services.short_drama_settings import (
    format_ost1_max_segments_rule,
    get_short_drama_script_prompt_params,
    get_short_drama_settings,
    save_short_drama_settings_to_config,
)
from app.services.short_drama_script_optimizer import (
    optimize_short_drama_script_items,
    pick_best_short_drama_script_candidate,
    repair_short_drama_script_timestamps,
    score_short_drama_script_quality,
    validate_short_drama_script_duration,
    validate_short_drama_script_timestamps,
)
from app.utils.video_processor import VideoProcessor
# 导入新的LLM服务模块 - 确保提供商被注册
import app.services.llm  # 这会触发提供商注册
from app.services.llm.migration_adapter import SubtitleAnalyzerAdapter
import re
from webui.tools.plot_blueprint_workflow import get_plot_blueprint, save_plot_blueprint


def parse_and_fix_json(json_string):
    """
    解析并修复JSON字符串

    Args:
        json_string: 待解析的JSON字符串

    Returns:
        dict: 解析后的字典，如果解析失败返回None
    """
    if not json_string or not json_string.strip():
        logger.error("JSON字符串为空")
        return None

    # 清理字符串
    json_string = json_string.strip()

    # 尝试直接解析
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        logger.warning(f"直接JSON解析失败: {e}")

    # 尝试修复双大括号问题（LLM生成的常见问题）
    try:
        # 将双大括号替换为单大括号
        fixed_braces = json_string.replace('{{', '{').replace('}}', '}')
        logger.info("修复双大括号格式")
        return json.loads(fixed_braces)
    except json.JSONDecodeError:
        pass

    # 尝试提取JSON部分
    try:
        # 查找JSON代码块
        json_match = re.search(r'```json\s*(.*?)\s*```', json_string, re.DOTALL)
        if json_match:
            json_content = json_match.group(1).strip()
            logger.info("从代码块中提取JSON内容")
            return json.loads(json_content)
    except json.JSONDecodeError:
        pass

    # 尝试查找大括号包围的内容
    try:
        # 查找第一个 { 到最后一个 } 的内容
        start_idx = json_string.find('{')
        end_idx = json_string.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_content = json_string[start_idx:end_idx+1]
            logger.info("提取大括号包围的JSON内容")
            return json.loads(json_content)
    except json.JSONDecodeError:
        pass

    # 尝试综合修复JSON格式问题
    try:
        fixed_json = json_string

        # 1. 修复双大括号问题
        fixed_json = fixed_json.replace('{{', '{').replace('}}', '}')

        # 2. 提取JSON内容（如果有其他文本包围）
        start_idx = fixed_json.find('{')
        end_idx = fixed_json.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            fixed_json = fixed_json[start_idx:end_idx+1]

        # 3. 移除注释
        fixed_json = re.sub(r'#.*', '', fixed_json)
        fixed_json = re.sub(r'//.*', '', fixed_json)

        # 4. 移除多余的逗号
        fixed_json = re.sub(r',\s*}', '}', fixed_json)
        fixed_json = re.sub(r',\s*]', ']', fixed_json)

        # 5. 修复单引号
        fixed_json = re.sub(r"'([^']*)':", r'"\1":', fixed_json)

        # 6. 修复没有引号的属性名
        fixed_json = re.sub(r'(\w+)(\s*):', r'"\1"\2:', fixed_json)

        # 7. 修复重复的引号
        fixed_json = re.sub(r'""([^"]*?)""', r'"\1"', fixed_json)

        logger.info("尝试综合修复JSON格式问题后解析")
        return json.loads(fixed_json)
    except json.JSONDecodeError as e:
        logger.debug(f"综合修复失败: {e}")
        pass

    # 如果所有方法都失败，尝试创建一个基本的结构
    logger.error(f"所有JSON解析方法都失败，原始内容: {json_string[:200]}...")

    # 尝试从文本中提取关键信息创建基本结构
    try:
        # 这是一个简单的回退方案
        return {
            "items": [
                {
                    "_id": 1,
                    "timestamp": "00:00:00,000-00:00:10,000",
                    "picture": "解析失败，使用默认内容",
                    "narration": json_string[:100] + "..." if len(json_string) > 100 else json_string,
                    "OST": 0
                }
            ]
        }
    except Exception:
        return None


def _material_source_video_path() -> str:
    return normalize_material_source_video_path(
        str(st.session_state.get("doc_material_source_video_path") or "")
    )


def _resolve_short_drama_settings_for_session() -> dict:
    """合并 config 与当前 UI 会话中的短剧解说覆盖项。"""
    overrides: dict = {}
    if "sd_ost1_max_segments" in st.session_state:
        overrides["ost1_max_segments"] = int(st.session_state["sd_ost1_max_segments"])
    return get_short_drama_settings(overrides or None)


def _build_output_duration_hint(*, sd_settings: dict | None = None) -> str:
    cfg = sd_settings or get_short_drama_settings()
    narr_pct = int(cfg.get("narration_percent", 30))
    orig_pct = int(cfg.get("original_audio_percent", 70))
    min_min = int(cfg.get("target_output_minutes_min", 8))
    max_min = int(cfg.get("target_output_minutes_max", 13))
    narr_chars_min = int(cfg.get("narration_chars_min", 20))
    narr_chars_max = int(cfg.get("narration_chars_max", 120))
    ost1_min = int(cfg.get("ost1_duration_min", 8))
    ost1_max = int(cfg.get("ost1_duration_max", 18))
    ost0_min = int(cfg.get("ost0_duration_min", 5))
    max_run = int(cfg.get("max_consecutive_ost1", 4))
    return (
        f"- **成片总时长目标 {min_min}–{max_min} 分钟**（按 `_id` 播放顺序累加各段时长估算）\n"
        f"- **解说为主**：解说 vs 原声成片时长约 **{narr_pct}:{orig_pct}**\n"
        f"- OST=0 解说每段 **{narr_chars_min}–{narr_chars_max} 字**（估算 ≥{ost0_min} 秒）\n"
        f"- **开篇**：播放顺序前 3 段内 **最多 1 段** OST=1（爆燃钩子）；第 2 段起先解说\n"
        f"- **场景原声**：OST=0 解说铺垫 → OST=1 金句（≤{ost1_max} 秒），按蓝图场景取舍\n"
        f"- 原声段数：{format_ost1_max_segments_rule(cfg)}；`picture`/`timestamp`/`original_line` 须同镜\n"
    )


def _log_plot_analysis_block(title: str, content: str) -> None:
    """将剧情构思/分析方案完整输出到控制台。"""
    text = (content or "").strip()
    separator = "=" * 72
    if not text:
        logger.info(f"\n{separator}\n{title}（空）\n{separator}")
        return
    logger.info(f"\n{separator}\n{title}\n{separator}\n{text}\n{separator}")


def _log_short_drama_validation_report(
    *,
    stage: str,
    attempt: int | None,
    dur_validation: dict,
    ts_validation: dict,
    item_count: int = 0,
) -> None:
    """每次校验后将成片时长/时间戳达标情况输出到控制台。"""
    attempt_label = f"第 {attempt} 次" if attempt is not None else "—"
    duration_ok = bool(dur_validation.get("ok"))
    timestamp_ok = bool(ts_validation.get("ok"))
    overall_ok = duration_ok and timestamp_ok
    status = "通过" if overall_ok else "未达标"

    lines = [
        "",
        "-" * 72,
        f"【短剧脚本校验 · {stage} · {attempt_label}】{status}",
        "-" * 72,
        f"段数: {item_count}",
        f"成片时长: {dur_validation.get('total_sec', 0) / 60:.1f} 分钟 "
        f"（解说 {dur_validation.get('narration_pct', 0)}% / "
        f"原声 {dur_validation.get('original_pct', 0)}%）",
        f"时长校验: {'通过' if duration_ok else '未达标'} — "
        f"{dur_validation.get('message') or '无'}",
        f"时间戳校验: {'通过' if timestamp_ok else '未达标'} — "
        f"{ts_validation.get('message') or '无'}",
    ]
    dur_issues = dur_validation.get("issues") or []
    if dur_issues:
        lines.append("时长问题明细:")
        lines.extend(f"  - {issue}" for issue in dur_issues)
    ts_issues = ts_validation.get("issues") or []
    if ts_issues:
        lines.append("时间戳/解说问题明细:")
        lines.extend(f"  - {issue}" for issue in ts_issues)
    lines.append("-" * 72)
    logger.info("\n".join(lines))


def _build_blueprint_execution_note(
    *,
    plot_blueprint: str = "",
    has_video_analysis: bool = False,
) -> str:
    """第二步 JSON 生成：蓝图落实要点（写入 prompt blueprint_execution 块）。"""
    from app.services.short_drama_blueprint_script import (
        build_blueprint_script_execution_note,
    )

    return build_blueprint_script_execution_note(
        plot_blueprint=plot_blueprint,
        has_video_analysis=has_video_analysis,
    )


def _load_video_episode_script_reference(
    video_path: str,
    *,
    explicit_path: str = "",
) -> str:
    """加载供 JSON 脚本生成参考的整片视频分析摘要。"""
    json_path = (explicit_path or "").strip()
    if not json_path:
        json_path = _resolve_video_episode_analysis_for_short_drama(video_path)
    if not json_path:
        return ""
    try:
        artifact = load_video_episode_analysis_artifact(json_path)
        return build_video_episode_script_reference_section(artifact)
    except Exception as exc:
        logger.warning(f"加载整片视频分析供脚本生成失败: {exc}")
        return ""


def _resolve_video_episode_analysis_for_short_drama(video_path: str) -> str:
    """解析短剧解说可用的整片视频分析 JSON 路径。"""
    material_source = _material_source_video_path()
    explicit_path = (
        st.session_state.get("video_episode_analysis_json_path") or ""
    ).strip() or None
    return (
        resolve_video_episode_analysis_path_for_documentary(
            video_path,
            material_source_video_path=material_source,
            explicit_path=explicit_path,
        )
        or ""
    )


def generate_plot_blueprint_short(
    params,
    subtitle_path: str,
    video_theme: str,
    *,
    fingerprint: str = "",
):
    """第一步：生成短剧解说「完美剧情构思方案」并在 session 中保存。"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress: float, message: str = ""):
        progress_bar.progress(int(progress))
        status_text.text(f"📝 {message}" if message else f"进度: {int(progress)}%")

    try:
        with st.spinner("正在生成剧情构思方案..."):
            if not params.video_origin_path:
                st.error("请先选择视频文件")
                return

            enable_video_analysis = bool(
                st.session_state.get("sd_enable_video_episode_analysis", True)
            )
            doc_settings = get_documentary_settings()
            plot_analysis = ""
            source_label = "字幕×整片视频分析×剧情联合分析"
            video_episode_json_path = ""

            if enable_video_analysis:
                video_episode_json_path = _resolve_video_episode_analysis_for_short_drama(
                    params.video_origin_path
                )
                if not video_episode_json_path:
                    st.error(
                        "未找到整片视频分析 JSON。请先在「素材预处理 → 整片视频分析」完成分析；"
                        "或取消勾选「结合整片视频分析」后仅用字幕生成。"
                    )
                    return
                subtitle_content = ""
                if subtitle_path and os.path.exists(subtitle_path):
                    subtitle_content = read_subtitle_text(subtitle_path).text
                source_duration_sec = 0.0
                try:
                    source_duration_sec = float(
                        VideoProcessor(params.video_origin_path).duration or 0.0
                    )
                except Exception:
                    source_duration_sec = 0.0
                update_progress(20, "正在整理整片视频分析结果...")
                video_artifact = load_video_episode_analysis_artifact(video_episode_json_path)
                video_markdown = build_video_episode_analysis_markdown(video_artifact)
                update_progress(
                    40,
                    "正在联合分析人物关系表、字幕与整片视频分析做场景分段..."
                    if (subtitle_content or "").strip()
                    else "正在联合分析人物关系表与整片视频分析做场景分段...",
                )
                plot_analysis = analyze_subtitle_with_frames(
                    subtitle_content=subtitle_content,
                    frame_markdown="",
                    video_theme=video_theme or "本短剧",
                    progress_callback=lambda msg: update_progress(55, msg),
                    documentary_settings=doc_settings,
                    analysis_style="short_drama",
                    frame_json_path=None,
                    append_custom_prompt=str(st.session_state.get("append_custom_prompt") or ""),
                    for_plot_blueprint=True,
                    source_duration_sec=source_duration_sec or None,
                    video_episode_json_path=video_episode_json_path,
                    video_episode_markdown=video_markdown,
                    character_relationship=str(st.session_state.get("doc_character_relationship") or ""),
                )
            else:
                subtitle_content = ""
                if subtitle_path and os.path.exists(subtitle_path):
                    subtitle_content = read_subtitle_text(subtitle_path).text
                if not (subtitle_content or "").strip():
                    st.error("未开启整片视频分析时，构思蓝图需要字幕文件")
                    return
                source_label = "字幕剧情分析"
                text_provider = config.app.get("text_llm_provider", "openai").lower()
                text_model, text_api_key, text_base_url = resolve_role_credentials("text")
                temperature = float(
                    st.session_state.get("temperature")
                    or get_short_drama_settings().get("narration_script_temperature", 0.4)
                )
                update_progress(40, "正在分析字幕剧情...")
                analyzer = SubtitleAnalyzerAdapter(
                    text_api_key,
                    text_model,
                    text_base_url,
                    text_provider,
                )
                try:
                    analysis_result = analyzer.analyze_subtitle(subtitle_content)
                except Exception as exc:
                    logger.warning(f"使用新LLM服务失败，回退到旧实现: {exc}")
                    analysis_result = analyze_subtitle(
                        subtitle_file_path=subtitle_path,
                        api_key=text_api_key,
                        model=text_model,
                        base_url=text_base_url,
                        save_result=True,
                        temperature=temperature,
                        provider=text_provider,
                    )
                if analysis_result.get("status") != "success":
                    st.error(f"剧情分析失败: {analysis_result.get('message', 'unknown')}")
                    return
                plot_analysis = (analysis_result.get("analysis") or "").strip()

            if not (plot_analysis or "").strip():
                st.error("剧情构思方案为空，请检查文本模型 API 与素材")
                return

            save_plot_blueprint(
                content=plot_analysis,
                fingerprint=fingerprint,
                mode="summary",
                meta={
                    "source_label": source_label,
                    "video_episode_analysis_path": video_episode_json_path,
                    "subtitle_path": subtitle_path,
                },
            )
            _log_plot_analysis_block(f"【短剧】{source_label} · 完美剧情构思方案", plot_analysis)
            logger.info(f"剧情构思方案生成完成，约 {len(plot_analysis)} 字")
            st.rerun()

        progress_bar.progress(100)
        status_text.text("🎉 剧情构思方案生成完成！")

    except Exception as err:
        st.error(f"❌ 生成构思方案时发生错误: {str(err)}")
        logger.exception(f"生成构思方案时发生错误\n{traceback.format_exc()}")
    finally:
        time.sleep(1.5)
        progress_bar.empty()
        status_text.empty()


def generate_script_short_sunmmary(
    params,
    subtitle_path,
    video_theme,
    temperature,
    *,
    plot_blueprint: str | None = None,
):
    """
    生成 短剧解说 视频脚本
    要求: 提供高质量短剧字幕；推荐先确认构思蓝图（整片视频分析 + 字幕）
    适合场景: 短剧
    """
    if temperature is None:
        temperature = float(
            get_short_drama_settings().get("narration_script_temperature", 0.4)
        )
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
            """
            1. 获取字幕
            """
            update_progress(30, "正在解析字幕...")
            prebuilt_blueprint = (plot_blueprint or get_plot_blueprint() or "").strip()
            blueprint_only = bool(prebuilt_blueprint)

            if not blueprint_only and not os.path.exists(subtitle_path):
                st.error("字幕文件不存在")
                return

            # 读取字幕文件内容（有构思方案时可选）
            subtitle_content = ""
            if subtitle_path and os.path.exists(subtitle_path):
                subtitle_content = read_subtitle_text(subtitle_path).text
            if not blueprint_only and not subtitle_content:
                st.error("字幕文件内容为空或无法读取")
                return

            """
            2. 分析字幕总结剧情 - 使用新的LLM服务架构
            """
            text_provider = config.app.get("text_llm_provider", "openai").lower()
            text_model, text_api_key, text_base_url = resolve_role_credentials("text")

            doc_settings = get_documentary_settings()
            subtitle_frame_analysis = ""
            video_episode_script_reference = ""
            blueprint_meta = dict(st.session_state.get("plot_blueprint_meta") or {})
            video_episode_json_path = str(
                blueprint_meta.get("video_episode_analysis_path")
                or st.session_state.get("video_episode_analysis_json_path")
                or ""
            ).strip()

            if prebuilt_blueprint:
                update_progress(40, "复用已确认的剧情构思方案，准备生成脚本...")
                plot_analysis = prebuilt_blueprint
                script_subtitle_content = subtitle_content
                video_episode_script_reference = _load_video_episode_script_reference(
                    params.video_origin_path,
                    explicit_path=video_episode_json_path,
                )
                has_video_analysis = video_episode_summary_usable(
                    video_episode_script_reference
                )
                subtitle_frame_analysis = _build_blueprint_execution_note(
                    plot_blueprint=plot_analysis,
                    has_video_analysis=has_video_analysis,
                )
                if has_video_analysis:
                    _log_plot_analysis_block(
                        "【短剧】整片视频分析 · JSON 脚本参考",
                        video_episode_script_reference,
                    )
                if subtitle_content:
                    script_subtitle_content = truncate_subtitle_content(
                        subtitle_content,
                        int(doc_settings.get("subtitle_max_chars", 15000)),
                    )
                analysis_result = {
                    "status": "success",
                    "analysis": plot_analysis,
                    "model": text_model,
                    "temperature": temperature,
                    "source": "confirmed_blueprint",
                }
                _log_plot_analysis_block(
                    "【短剧】复用已确认的完美剧情构思方案",
                    plot_analysis,
                )
            else:
                analysis_result = {
                    "status": "error",
                    "message": "未执行剧情分析",
                }

            source_duration_sec = 0.0
            try:
                source_duration_sec = VideoProcessor(params.video_origin_path).duration
            except Exception as duration_err:
                logger.warning(f"无法读取原片时长: {duration_err}")

            sd_settings = _resolve_short_drama_settings_for_session()
            output_duration_hint = _build_output_duration_hint(sd_settings=sd_settings)
            prompt_extra = {
                "output_duration_hint": output_duration_hint,
                **get_short_drama_script_prompt_params(
                    source_duration_sec=source_duration_sec,
                    settings=sd_settings,
                ),
            }
            if not prebuilt_blueprint:
                script_subtitle_content = subtitle_content

            analyzer = SubtitleAnalyzerAdapter(
                text_api_key,
                text_model,
                text_base_url,
                text_provider,
                script_extra_params=prompt_extra,
            )
            plot_analysis = plot_analysis if prebuilt_blueprint else ""
            script_frame_analysis = subtitle_frame_analysis

            if not prebuilt_blueprint:
                st.error("请先生成并确认「完美剧情构思方案」，再生成 JSON 脚本。")
                return

            if analysis_result["status"] != "success":
                logger.error(f"分析失败: {analysis_result.get('message', 'unknown')}")
                st.error("剧情分析失败，请检查日志")
                st.stop()

            update_progress(60, "正在生成解说脚本...")
            narration_result = None
            narration_dict = None
            max_llm_attempts = 3
            max_repair_passes = 3
            generation_candidates: list[dict] = []
            used_best_effort = False

            for attempt in range(1, max_llm_attempts + 1):
                if attempt > 1:
                    update_progress(
                        65,
                        f"JSON 解析失败，正在第 {attempt} 次重新生成...",
                    )
                plot_for_retry = plot_analysis

                try:
                    logger.info("使用新的LLM服务架构生成解说文案")
                    narration_result = analyzer.generate_narration_script(
                        short_name=video_theme,
                        plot_analysis=plot_for_retry,
                        subtitle_content=script_subtitle_content,
                        temperature=temperature,
                        subtitle_frame_analysis=script_frame_analysis,
                        video_episode_analysis=video_episode_script_reference,
                    )
                except Exception as e:
                    logger.warning(f"使用新LLM服务失败，回退到旧实现: {str(e)}")
                    narration_result = generate_narration_script(
                        short_name=video_theme,
                        plot_analysis=plot_for_retry,
                        subtitle_content=script_subtitle_content,
                        api_key=text_api_key,
                        model=text_model,
                        base_url=text_base_url,
                        save_result=True,
                        temperature=temperature,
                        provider=text_provider,
                        subtitle_frame_analysis=script_frame_analysis,
                        video_episode_analysis=video_episode_script_reference,
                        script_extra_params=prompt_extra,
                    )

                if narration_result["status"] != "success":
                    continue

                narration_dict = parse_and_fix_json(narration_result["narration_script"])
                if narration_dict is None or "items" not in narration_dict:
                    continue

                item_count = len(narration_dict.get("items") or [])
                dur_validation = validate_short_drama_script_duration(
                    narration_dict.get("items") or [],
                    sd_settings,
                )
                ts_validation = validate_short_drama_script_timestamps(
                    narration_dict.get("items") or [],
                    sd_settings,
                )
                quality_score = score_short_drama_script_quality(
                    dur_validation, ts_validation, sd_settings
                )
                generation_candidates.append(
                    {
                        "attempt": attempt,
                        "narration_dict": narration_dict,
                        "dur_validation": dur_validation,
                        "ts_validation": ts_validation,
                        "score": quality_score,
                    }
                )
                _log_short_drama_validation_report(
                    stage="LLM 生成后",
                    attempt=attempt,
                    dur_validation=dur_validation,
                    ts_validation=ts_validation,
                    item_count=item_count,
                )
                logger.info(
                    f"短剧脚本 LLM 输出 第{attempt}次 接近度分数={quality_score:.3f} "
                    f"（时间轴问题将走后处理自动修复，不重写全文）"
                )
                break

            if narration_dict is None and generation_candidates:
                best = pick_best_short_drama_script_candidate(generation_candidates)
                if best:
                    used_best_effort = True
                    narration_dict = best["narration_dict"]
                    logger.warning(
                        f"末次生成未得到有效 JSON，回退到最接近的第 {best['attempt']} 次候选 "
                        f"（分数 {best['score']:.3f}）"
                    )

            if narration_result is None or narration_result.get("status") != "success":
                if generation_candidates and narration_dict is not None:
                    logger.warning(
                        "末次 LLM 调用失败，仍输出已缓存的最接近脚本候选"
                    )
                else:
                    logger.info(
                        f"\n解说文案生成失败: "
                        f"{narration_result.get('message', 'unknown') if narration_result else 'unknown'}"
                    )
                    st.error("生成脚本失败，请检查日志")
                    st.stop()

            """
            4. 生成文案
            """
            logger.info("开始准备生成解说文案")

            if narration_dict is None:
                narration_dict = parse_and_fix_json(narration_result["narration_script"])
            if narration_dict is None:
                st.error("生成的解说文案格式错误，无法解析为JSON")
                if narration_result:
                    logger.error(
                        f"JSON解析失败，原始内容: {narration_result.get('narration_script', '')}"
                    )
                st.stop()

            if 'items' not in narration_dict:
                st.error("生成的解说文案缺少必要的'items'字段")
                logger.error(f"JSON结构错误，缺少items字段: {narration_dict}")
                st.stop()

            frame_analysis_path = (
                str(video_episode_json_path or "").strip()
                or _resolve_video_episode_analysis_for_short_drama(
                    params.video_origin_path
                )
            )

            narration_dict["items"] = apply_opening_climax_fix(
                narration_dict["items"],
                subtitle_content=subtitle_content,
                subtitle_frame_analysis=plot_analysis,
                append_custom_prompt=str(st.session_state.get("append_custom_prompt") or ""),
                frame_analysis_path=frame_analysis_path,
                settings=doc_settings,
                enabled=True,
            )

            sd_settings = _resolve_short_drama_settings_for_session()
            narration_dict["items"] = optimize_short_drama_script_items(
                narration_dict["items"],
                subtitle_content=subtitle_content,
                frame_analysis_path=frame_analysis_path,
                plot_blueprint=plot_analysis,
                settings=sd_settings,
            )

            narration_dict["items"] = apply_opening_climax_fix(
                narration_dict["items"],
                subtitle_content=subtitle_content,
                subtitle_frame_analysis=plot_analysis,
                append_custom_prompt=str(st.session_state.get("append_custom_prompt") or ""),
                frame_analysis_path=frame_analysis_path,
                settings=doc_settings,
                enabled=True,
            )

            replay_settings = {**doc_settings, **sd_settings}
            narration_dict["items"] = apply_opening_climax_chronological_replay(
                narration_dict["items"],
                settings=replay_settings,
                enabled=bool(
                    sd_settings.get("enable_opening_climax_chronological_replay", True)
                ),
            )

            from app.services.short_drama_timestamp_alignment import (
                align_script_items_to_source_material,
            )

            narration_dict["items"] = align_script_items_to_source_material(
                narration_dict["items"],
                subtitle_content=subtitle_content,
                plot_blueprint=plot_analysis,
                settings=sd_settings,
            )

            for repair_pass in range(1, max_repair_passes + 1):
                ts_validation = validate_short_drama_script_timestamps(
                    narration_dict.get("items") or [],
                    sd_settings,
                    subtitle_content=subtitle_content,
                    plot_blueprint=plot_analysis,
                )
                if ts_validation["ok"]:
                    break
                update_progress(
                    72,
                    f"自动修复时间轴（第 {repair_pass}/{max_repair_passes} 次，不改文案）...",
                )
                logger.info(
                    f"时间轴未达标，第 {repair_pass} 次自动扩展/对位："
                    f"{ts_validation.get('message')}"
                )
                narration_dict["items"] = repair_short_drama_script_timestamps(
                    narration_dict["items"],
                    subtitle_content=subtitle_content,
                    frame_analysis_path=frame_analysis_path,
                    plot_blueprint=plot_analysis,
                    settings=sd_settings,
                )
                _log_short_drama_validation_report(
                    stage=f"时间轴修复后 · 第{repair_pass}次",
                    attempt=repair_pass,
                    dur_validation=validate_short_drama_script_duration(
                        narration_dict.get("items") or [],
                        sd_settings,
                    ),
                    ts_validation=validate_short_drama_script_timestamps(
                        narration_dict.get("items") or [],
                        sd_settings,
                        subtitle_content=subtitle_content,
                        plot_blueprint=plot_analysis,
                    ),
                    item_count=len(narration_dict.get("items") or []),
                )

            final_dur_validation = validate_short_drama_script_duration(
                narration_dict.get("items") or [],
                sd_settings,
            )
            final_ts_validation = validate_short_drama_script_timestamps(
                narration_dict.get("items") or [],
                sd_settings,
                subtitle_content=subtitle_content,
                plot_blueprint=plot_analysis,
            )
            _log_short_drama_validation_report(
                stage="后处理完成",
                attempt=None,
                dur_validation=final_dur_validation,
                ts_validation=final_ts_validation,
                item_count=len(narration_dict.get("items") or []),
            )
            if not final_ts_validation["ok"] or not final_dur_validation["ok"]:
                used_best_effort = True
                logger.warning(
                    "后处理后脚本仍未完全达标，仍输出最接近的结果。"
                    f"时长：{final_dur_validation.get('message')}；"
                    f"时间戳：{final_ts_validation.get('message')}"
                )
                st.warning(
                    "脚本后处理后仍有未达标项，已输出当前最接近可用的版本，"
                    "建议人工检查后再生成视频。"
                    f"{final_ts_validation.get('message') or final_dur_validation.get('message')}"
                )
            elif used_best_effort:
                logger.info("已输出最佳努力脚本（生成阶段未完全达标）")

            script = json.dumps(narration_dict['items'], ensure_ascii=False, indent=2)

            if script is None:
                st.error("生成脚本失败，请检查日志")
                st.stop()
            logger.success(f"剪辑脚本生成完成")
            st.session_state["narration_workflow_mode"] = "summary"
            config.app["narration_workflow_mode"] = "summary"
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
