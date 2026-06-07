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
    resolve_frame_analysis_path_for_documentary,
    normalize_material_source_video_path,
)
from app.services.documentary.documentary_subtitle_enrichment import (
    analyze_subtitle_with_frames,
    truncate_subtitle_content,
)
from app.services.documentary.opening_climax_resolver import apply_opening_climax_fix
from app.services.short_drama_settings import (
    compute_short_drama_ost_bounds,
    compute_short_drama_segment_bounds,
    get_short_drama_script_prompt_params,
    get_short_drama_settings,
)
from app.services.short_drama_script_optimizer import (
    optimize_short_drama_script_items,
    validate_short_drama_script_counts,
)
from app.services.generate_narration_script import parse_frame_analysis_to_markdown
from app.utils.video_processor import VideoProcessor
# 导入新的LLM服务模块 - 确保提供商被注册
import app.services.llm  # 这会触发提供商注册
from app.services.llm.migration_adapter import SubtitleAnalyzerAdapter
import re


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


def _prepare_frame_summary(analysis_json_path: str, doc_settings: dict) -> str:
    """将抽帧分析 JSON 整理为剧情/脚本生成用的紧凑摘要。"""
    markdown_output = parse_frame_analysis_to_markdown(analysis_json_path, detail_level="full")
    compact_threshold = int(doc_settings.get("narration_compact_markdown_chars", 120000))
    if len(markdown_output) > compact_threshold:
        compact_markdown = parse_frame_analysis_to_markdown(
            analysis_json_path,
            detail_level="compact",
        )
        logger.info(
            f"抽帧 Markdown 过长（{len(markdown_output)} 字），"
            f"已切换紧凑模式（{len(compact_markdown)} 字）"
        )
        return compact_markdown
    return markdown_output


def _build_segment_count_hint(
    *,
    enabled: bool,
    min_segments: int,
    max_segments: int,
    sd_settings: dict | None = None,
) -> str:
    cfg = sd_settings or get_short_drama_settings()
    narr_pct = int(cfg.get("narration_percent", 30))
    orig_pct = int(cfg.get("original_audio_percent", 70))
    expected = (min_segments + max_segments) // 2
    bounds = compute_short_drama_ost_bounds(expected, cfg)
    ratio_block = (
        f"- **解说 OST=0 约 {narr_pct}%**，**原声 OST=1 约 {orig_pct}%**（按 `_id` 播放顺序统计段数，不是按 timestamp）\n"
        f"- **OST=0 至少 {bounds['ost0_min']} 段**，**OST=1 最多 {bounds['ost1_max']} 段**；"
        f"**同场可连续 {cfg.get('max_consecutive_ost1', 4)} 段 OST=1，整块播完再 OST=0**；"
        f"以成片时长 3:7 为准，OST=0 至少 {bounds['ost0_min']} 段\n"
        f"- 情节点之间须 OST=0 串场；同场可连续 {cfg.get('max_consecutive_ost1', 4)} 段 OST=1 成块\n"
    )
    if not enabled:
        return (
            "按字幕长度自然切段，保持快节奏细切，不要人为压缩段数。\n"
            + ratio_block
        )
    return (
        f"- **items 总数须达到 {min_segments}–{max_segments} 段**（低于 {min_segments} 段视为无效输出）\n"
        f"- 对照分析中的情节点须落实，但**以 OST=0 解说串场为主、OST=1 原声点睛为辅**\n"
        f"- 1/3 成片时长靠**合理切段 + 原声时长**达成，不是靠堆满 OST=1\n"
        + ratio_block
    )


def _resolve_frame_analysis_for_short_drama(video_path: str) -> str:
    """解析短剧解说可用的抽帧分析 JSON 路径。"""
    material_source = _material_source_video_path()
    explicit_analysis = (
        st.session_state.get("frame_analysis_json_path") or ""
    ).strip() or None
    reuse_frame_analysis = bool(st.session_state.get("doc_reuse_frame_analysis", True))
    return resolve_frame_analysis_path_for_documentary(
        video_path,
        material_source_video_path=material_source,
        explicit_path=explicit_analysis,
        reuse=reuse_frame_analysis,
    )


def generate_script_short_sunmmary(params, subtitle_path, video_theme, temperature):
    """
    生成 短剧解说 视频脚本
    要求: 提供高质量短剧字幕；可选结合抽帧分析（参照逐帧精剪工作流）
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
            # 判断字幕文件是否存在
            if not os.path.exists(subtitle_path):
                st.error("字幕文件不存在")
                return

            """
            2. 分析字幕总结剧情 - 使用新的LLM服务架构
            """
            text_provider = config.app.get("text_llm_provider", "openai").lower()
            text_model, text_api_key, text_base_url = resolve_role_credentials("text")

            # 读取字幕文件内容（无论使用哪种实现都需要）
            subtitle_content = read_subtitle_text(subtitle_path).text
            if not subtitle_content:
                st.error("字幕文件内容为空或无法读取")
                return

            enable_frame_analysis = bool(
                st.session_state.get("sd_enable_frame_analysis", True)
            )
            doc_settings = get_documentary_settings()
            frame_summary = ""
            subtitle_frame_analysis = ""
            analysis_json_path = ""

            if enable_frame_analysis:
                analysis_json_path = _resolve_frame_analysis_for_short_drama(
                    params.video_origin_path
                )
                if not analysis_json_path:
                    hint = (
                        "未找到可用的抽帧分析 JSON。请先在「素材预处理」或「抽帧分析」中完成抽帧，"
                        "或上传/选用已有分析文件；也可取消勾选「结合抽帧分析」后仅用字幕生成。"
                    )
                    material_source = _material_source_video_path()
                    if material_source:
                        hint += (
                            f" 已配置素材来源视频「{os.path.basename(material_source)}」，"
                            "请确认该视频已完成抽帧分析。"
                        )
                    st.error(hint)
                    return

                update_progress(32, "正在整理抽帧分析结果...")
                frame_summary = _prepare_frame_summary(analysis_json_path, doc_settings)
                if not (frame_summary or "").strip():
                    st.error("抽帧分析结果为空，请重新执行「抽帧并分析」")
                    return
                logger.info(f"短剧解说复用抽帧分析: {analysis_json_path}")

                update_progress(38, "正在分析字幕并对照抽帧画面...")
                subtitle_frame_analysis = analyze_subtitle_with_frames(
                    subtitle_content=subtitle_content,
                    frame_markdown=frame_summary,
                    video_theme=video_theme or "本短剧",
                    progress_callback=lambda msg: update_progress(40, msg),
                    documentary_settings=doc_settings,
                    analysis_style="short_drama",
                )
                if subtitle_frame_analysis:
                    logger.info(
                        f"字幕×抽帧对照分析完成，约 {len(subtitle_frame_analysis)} 字"
                    )

            source_duration_sec = 0.0
            try:
                source_duration_sec = VideoProcessor(params.video_origin_path).duration
            except Exception as duration_err:
                logger.warning(f"无法读取原片时长: {duration_err}")

            min_segments, max_segments = compute_short_drama_segment_bounds(source_duration_sec)
            sd_settings = get_short_drama_settings()
            segment_count_hint = _build_segment_count_hint(
                enabled=enable_frame_analysis,
                min_segments=min_segments,
                max_segments=max_segments,
                sd_settings=sd_settings,
            )
            prompt_extra = {
                "segment_count_hint": segment_count_hint,
                **get_short_drama_script_prompt_params(
                    source_duration_sec=source_duration_sec,
                    expected_total_segments=(min_segments + max_segments) // 2,
                    settings=sd_settings,
                ),
            }
            script_subtitle_content = subtitle_content
            if enable_frame_analysis and subtitle_frame_analysis:
                script_subtitle_content = truncate_subtitle_content(
                    subtitle_content,
                    int(doc_settings.get("subtitle_max_chars", 15000)),
                )
                logger.info(
                    f"结合抽帧：脚本 prompt 字幕由 {len(subtitle_content)} 字截断为 "
                    f"{len(script_subtitle_content)} 字（时间戳仍以字幕为准）"
                )

            try:
                # 优先使用新的LLM服务架构
                logger.info("使用新的LLM服务架构进行字幕分析")
                analyzer = SubtitleAnalyzerAdapter(
                    text_api_key,
                    text_model,
                    text_base_url,
                    text_provider,
                    script_extra_params=prompt_extra,
                )

                analysis_result = analyzer.analyze_subtitle(
                    subtitle_content,
                    frame_summary=frame_summary,
                )

            except Exception as e:
                logger.warning(f"使用新LLM服务失败，回退到旧实现: {str(e)}")
                # 回退到旧的实现
                analysis_result = analyze_subtitle(
                    subtitle_file_path=subtitle_path,
                    api_key=text_api_key,
                    model=text_model,
                    base_url=text_base_url,
                    save_result=True,
                    temperature=temperature,
                    provider=text_provider,
                    frame_summary=frame_summary,
                )
            """
            3. 根据剧情生成解说文案
            """
            if analysis_result["status"] != "success":
                logger.error(f"分析失败: {analysis_result['message']}")
                st.error("生成脚本失败，请检查日志")
                st.stop()

            logger.info("字幕分析成功！")
            update_progress(60, "正在生成文案...")

            plot_analysis = analysis_result["analysis"]
            narration_result = None
            narration_dict = None
            max_attempts = 3

            for attempt in range(1, max_attempts + 1):
                retry_reasons: list[str] = []
                if attempt > 1 and narration_dict:
                    item_count = len(narration_dict.get("items") or [])
                    if enable_frame_analysis and item_count < min_segments:
                        retry_reasons.append(
                            f"items 仅 {item_count} 段，低于 {min_segments} 段下限"
                        )
                    validation = validate_short_drama_script_counts(
                        narration_dict.get("items") or [],
                        sd_settings,
                    )
                    if not validation["ok"]:
                        retry_reasons.append(validation["message"])
                if attempt > 1 and retry_reasons:
                    update_progress(65, f"比例/段数不足，正在第 {attempt} 次重新生成...")
                    plot_for_retry = (
                        f"{analysis_result['analysis']}\n\n"
                        f"【重要修正·第{attempt}次】上次输出无效："
                        f"{'；'.join(retry_reasons)}。"
                        f"请重新输出完整 JSON：items {min_segments}–{max_segments} 段；"
                        f"OST=0 解说至少 {validation['ost0_min']} 段（约 30%），"
                        f"OST=1 原声最多 {validation['ost1_max']} 段（约 70% 成片时长）；"
                        f"同场可连续 {sd_settings.get('max_consecutive_ost1', 4)} 段原声成块，整块播完再解说；"
                        f"禁止几乎全是 OST=1。"
                    )
                elif attempt > 1:
                    update_progress(65, f"段数不足，正在第 {attempt} 次重新生成...")
                    plot_for_retry = (
                        f"{analysis_result['analysis']}\n\n"
                        f"【重要修正·第{attempt}次】上次 JSON items 仅 {len(narration_dict.get('items', []))} 段，"
                        f"低于要求的 {min_segments} 段下限，输出无效。"
                        f"请重新输出完整 JSON，items 总数须达到 {min_segments}–{max_segments} 段。"
                    )
                else:
                    plot_for_retry = plot_analysis

                try:
                    logger.info("使用新的LLM服务架构生成解说文案")
                    narration_result = analyzer.generate_narration_script(
                        short_name=video_theme,
                        plot_analysis=plot_for_retry,
                        subtitle_content=script_subtitle_content,
                        temperature=min(1.2, temperature + 0.1 * (attempt - 1)),
                        subtitle_frame_analysis=subtitle_frame_analysis,
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
                        temperature=min(1.2, temperature + 0.1 * (attempt - 1)),
                        provider=text_provider,
                        subtitle_frame_analysis=subtitle_frame_analysis,
                        script_extra_params=prompt_extra,
                    )

                if narration_result["status"] != "success":
                    break

                narration_dict = parse_and_fix_json(narration_result["narration_script"])
                if narration_dict is None or "items" not in narration_dict:
                    break

                item_count = len(narration_dict.get("items") or [])
                validation = validate_short_drama_script_counts(
                    narration_dict.get("items") or [],
                    sd_settings,
                )
                segment_ok = item_count >= min_segments
                ratio_ok = validation["ok"]
                if segment_ok and ratio_ok:
                    break
                if attempt >= max_attempts:
                    if enable_frame_analysis and item_count < min_segments:
                        logger.warning(
                            f"短剧解说段数仍不足: {item_count}/{min_segments}，已用尽重试"
                        )
                        st.warning(
                            f"脚本段数偏少（{item_count} 段，建议 ≥{min_segments} 段）。"
                        )
                    if not ratio_ok:
                        logger.warning(f"短剧解说 OST 比例未达标: {validation['message']}")
                        st.warning(
                            f"解说段偏少（OST=0 {validation['ost0_count']} 段，"
                            f"建议 ≥{validation['ost0_min']} 段）。后处理已尝试自动修正。"
                        )
                    break

                logger.info(
                    f"短剧解说未达标（段数 {item_count}/{min_segments}，"
                    f"OST=0 {validation['ost0_count']}/{validation['ost0_min']}），"
                    f"准备第 {attempt + 1} 次生成"
                )

            if narration_result is None or narration_result["status"] != "success":
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
                logger.error(f"JSON解析失败，原始内容: {narration_result['narration_script']}")
                st.stop()

            if 'items' not in narration_dict:
                st.error("生成的解说文案缺少必要的'items'字段")
                logger.error(f"JSON结构错误，缺少items字段: {narration_dict}")
                st.stop()

            narration_dict["items"] = apply_opening_climax_fix(
                narration_dict["items"],
                subtitle_content=subtitle_content,
                subtitle_frame_analysis=subtitle_frame_analysis,
                append_custom_prompt=str(st.session_state.get("append_custom_prompt") or ""),
                frame_analysis_path=analysis_json_path if enable_frame_analysis else "",
                settings=doc_settings,
                enabled=enable_frame_analysis,
            )

            sd_settings = get_short_drama_settings()
            narration_dict["items"] = optimize_short_drama_script_items(
                narration_dict["items"],
                subtitle_content=subtitle_content,
                frame_analysis_path=analysis_json_path if enable_frame_analysis else "",
                settings=sd_settings,
            )

            script = json.dumps(narration_dict['items'], ensure_ascii=False, indent=2)

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
        if enable_frame_analysis and analysis_json_path:
            st.success(
                f"视频脚本生成成功！已结合抽帧分析："
                f"`{os.path.basename(analysis_json_path)}`"
            )
        else:
            st.success("视频脚本生成成功！")

    except Exception as err:
        st.error(f"生成过程中发生错误: {str(err)}")
        logger.exception(f"生成脚本时发生错误\n{traceback.format_exc()}")
    finally:
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
