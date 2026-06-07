#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project: NarratoAI
@File   : 生成介绍文案
@Author : Viccy同学
@Date   : 2025/5/8 上午11:33 
'''

import json
import os
import traceback
import asyncio
from openai import OpenAI
from loguru import logger

# 导入新的LLM服务模块 - 确保提供商被注册
import app.services.llm  # 这会触发提供商注册
from app.services.documentary.documentary_settings import get_narration_script_llm_params
from app.services.llm.migration_adapter import generate_narration as generate_narration_new
# 导入新的提示词管理系统
from app.services.prompts import PromptManager


def parse_frame_analysis_to_markdown(json_file_path, *, detail_level: str = "full"):
    """
    解析视频帧分析JSON文件并转换为Markdown格式
    
    :param json_file_path: JSON文件路径
    :param detail_level: full=含每帧描述；compact=仅批次摘要+首尾帧（长视频省 token）
    :return: Markdown格式的字符串
    """
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        return f"错误: 文件 {json_file_path} 不存在"
    
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        def time_to_milliseconds(time_text):
            time_text = (time_text or "").strip()
            if not time_text:
                return 0
            try:
                if "," in time_text:
                    hhmmss, ms = time_text.split(",", 1)
                    milliseconds = int(ms)
                else:
                    hhmmss = time_text
                    milliseconds = 0

                parts = [int(part) for part in hhmmss.split(":") if part]
                while len(parts) < 3:
                    parts.insert(0, 0)
                hours, minutes, seconds = parts[-3], parts[-2], parts[-1]
                return ((hours * 3600 + minutes * 60 + seconds) * 1000) + milliseconds
            except Exception:
                return 0

        def batch_sort_key(batch):
            time_range = batch.get("time_range", "")
            start = time_range.split("-", 1)[0].strip()
            return time_to_milliseconds(start), batch.get("batch_index", 0)

        compact = (detail_level or "full").lower() == "compact"

        def format_scene_segment(segment: dict, index: int) -> str:
            timestamp = segment.get("timestamp", "")
            scene = segment.get("scene", "")
            characters = segment.get("characters") or []
            if isinstance(characters, list):
                characters_text = "、".join(str(name) for name in characters if str(name).strip())
            else:
                characters_text = str(characters)
            lines = [f"## 场景 {index}", f"- 时间：{timestamp}"]
            if scene:
                lines.append(f"- 场景：{scene}")
            if characters_text:
                lines.append(f"- 人物：{characters_text}")
            observation = str(segment.get("observation") or "").strip()
            if observation:
                lines.append(f"- 观察：{observation}")
            for label, key in (
                ("动作", "action"),
                ("情绪", "emotion"),
                ("关键视觉", "key_visual"),
                ("音效/原声", "audio_cue"),
                ("重要度", "importance"),
                ("字幕", "subtitle"),
            ):
                value = str(segment.get(key) or "").strip()
                if value:
                    lines.append(f"- {label}：{value}")
            entries = segment.get("subtitle_entries")
            if isinstance(entries, list) and entries:
                lines.append("- 字幕明细：")
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    start = str(entry.get("start") or "").strip()
                    end = str(entry.get("end") or "").strip()
                    text = str(entry.get("text") or "").strip()
                    if start and end and text:
                        lines.append(f"  - [{start}-{end}] {text}")
            return "\n".join(lines) + "\n\n"

        def format_sample_frames(observations: list) -> str:
            if not observations:
                return ""
            if len(observations) == 1:
                selected = observations
            else:
                selected = [observations[0], observations[-1]]
            lines = ""
            for frame in selected:
                timestamp = frame.get("timestamp", "")
                observation = frame.get("observation", "")
                subtitle = str(frame.get("subtitle") or "").strip()
                subtitle_start = str(frame.get("subtitle_start") or "").strip()
                subtitle_end = str(frame.get("subtitle_end") or "").strip()
                subtitle_time = ""
                if subtitle_start and subtitle_end:
                    subtitle_time = f"[{subtitle_start}-{subtitle_end}] "
                elif subtitle_start:
                    subtitle_time = f"[{subtitle_start}] "
                if observation and subtitle:
                    lines += f"  - {timestamp}: {observation}｜字幕：{subtitle_time}{subtitle}\n"
                elif observation:
                    lines += f"  - {timestamp}: {observation}\n"
                elif subtitle:
                    lines += f"  - {timestamp}: 字幕：{subtitle}\n"
                else:
                    lines += f"  - {timestamp}: \n"
            return lines

        markdown = ""

        top_level_segments = data.get("scene_segments")
        if isinstance(top_level_segments, list) and top_level_segments:
            for index, segment in enumerate(top_level_segments, 1):
                if isinstance(segment, dict):
                    markdown += format_scene_segment(segment, index)
            return markdown

        # 新结构：按批次保存完整分析产物
        if isinstance(data.get("batches"), list):
            ordered_batches = sorted(data.get("batches", []), key=batch_sort_key)

            for i, batch in enumerate(ordered_batches, 1):
                batch_segments = batch.get("scene_segments") or []
                if batch_segments:
                    for segment in batch_segments:
                        if isinstance(segment, dict):
                            markdown += format_scene_segment(segment, i)
                    continue

                time_range = batch.get("time_range", "")
                summary = (
                    batch.get("overall_activity_summary")
                    or batch.get("summary")
                    or batch.get("fallback_summary")
                    or ""
                )
                observations = batch.get("frame_observations") or batch.get("observations") or []

                markdown += f"## 片段 {i}\n"
                markdown += f"- 时间范围：{time_range}\n"
                markdown += f"- 片段描述：{summary}\n" if summary else "- 片段描述：\n"
                if compact:
                    markdown += "- 详细描述：（已压缩；请以片段描述为主，首尾帧采样如下）\n"
                    markdown += format_sample_frames(observations)
                else:
                    markdown += "- 详细描述：\n"
                    for frame in observations:
                        timestamp = frame.get("timestamp", "")
                        observation = frame.get("observation", "")
                        subtitle = str(frame.get("subtitle") or "").strip()
                        subtitle_start = str(frame.get("subtitle_start") or "").strip()
                        subtitle_end = str(frame.get("subtitle_end") or "").strip()
                        subtitle_time = ""
                        if subtitle_start and subtitle_end:
                            subtitle_time = f"[{subtitle_start}-{subtitle_end}] "
                        elif subtitle_start:
                            subtitle_time = f"[{subtitle_start}] "
                        if observation and subtitle:
                            markdown += f"  - {timestamp}: {observation}｜字幕：{subtitle_time}{subtitle}\n"
                        elif observation:
                            markdown += f"  - {timestamp}: {observation}\n"
                        elif subtitle:
                            markdown += f"  - {timestamp}: 字幕：{subtitle}\n"
                        else:
                            markdown += f"  - {timestamp}: \n"

                markdown += "\n"

            return markdown

        # 兼容旧结构
        summaries = data.get('overall_activity_summaries', [])
        frame_observations = data.get('frame_observations', [])

        batch_frames = {}
        for frame in frame_observations:
            batch_index = frame.get('batch_index')
            if batch_index not in batch_frames:
                batch_frames[batch_index] = []
            batch_frames[batch_index].append(frame)

        for i, summary in enumerate(summaries, 1):
            batch_index = summary.get('batch_index')
            time_range = summary.get('time_range', '')
            batch_summary = summary.get('summary', '')

            markdown += f"## 片段 {i}\n"
            markdown += f"- 时间范围：{time_range}\n"
            markdown += f"- 片段描述：{batch_summary}\n" if batch_summary else f"- 片段描述：\n"
            frames = batch_frames.get(batch_index, [])
            if compact:
                markdown += "- 详细描述：（已压缩；请以片段描述为主，首尾帧采样如下）\n"
                markdown += format_sample_frames(frames)
            else:
                markdown += "- 详细描述：\n"
                for frame in frames:
                    timestamp = frame.get('timestamp', '')
                    observation = frame.get('observation', '')
                    markdown += f"  - {timestamp}: {observation}\n" if observation else f"  - {timestamp}: \n"

            markdown += "\n"

        return markdown
    
    except Exception as e:
        return f"处理JSON文件时出错: {traceback.format_exc()}"


def generate_narration(markdown_content, api_key, base_url, model):
    """
    调用大模型API根据视频帧分析的Markdown内容生成解说文案 - 已重构为使用新的LLM服务架构

    :param markdown_content: Markdown格式的视频帧分析内容
    :param api_key: API密钥
    :param base_url: API基础URL
    :param model: 使用的模型名称
    :return: 生成的解说文案
    """
    try:
        # 优先使用新的LLM服务架构
        logger.info("使用新的LLM服务架构生成解说文案")
        result = generate_narration_new(markdown_content, api_key, base_url, model)
        return result

    except Exception as e:
        logger.warning(f"使用新LLM服务失败，回退到旧实现: {str(e)}")

        # 回退到旧的实现以确保兼容性
        return _generate_narration_legacy(markdown_content, api_key, base_url, model)


def _generate_narration_legacy(markdown_content, api_key, base_url, model):
    """
    旧的解说文案生成实现 - 保留作为备用方案

    :param markdown_content: Markdown格式的视频帧分析内容
    :param api_key: API密钥
    :param base_url: API基础URL
    :param model: 使用的模型名称
    :return: 生成的解说文案
    """
    try:
        # 使用新的提示词管理系统构建提示词
        prompt = PromptManager.get_prompt(
            category="documentary",
            name="narration_generation",
            parameters={
                "video_frame_description": markdown_content
            }
        )







        prompt_obj = PromptManager.get_prompt_object(
            category="documentary",
            name="narration_generation",
        )
        system_prompt = (prompt_obj.get_system_prompt() if prompt_obj else None) or (
            "你是一位资深视频解说员，只输出合法 JSON，包含 items 数组。"
        )

        # 使用OpenAI SDK初始化客户端
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        llm_params = get_narration_script_llm_params()

        # 使用SDK发送请求
        if model not in ["deepseek-reasoner"]:
            # deepseek-reasoner 不支持 json 输出
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=llm_params["temperature"],
                max_tokens=llm_params["max_tokens"],
                response_format={"type": "json_object"},
            )
            # 提取生成的文案
            if response.choices and len(response.choices) > 0:
                narration_script = response.choices[0].message.content
                # 打印消耗的tokens
                logger.debug(f"消耗的tokens: {response.usage.total_tokens}")
                return narration_script
            raise RuntimeError("生成解说文案失败: 未获取到有效响应")
        else:
            # 不支持 json 输出，需要多一步处理 ```json ``` 的步骤
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=llm_params["temperature"],
                max_tokens=llm_params["max_tokens"],
            )
            # 提取生成的文案
            if response.choices and len(response.choices) > 0:
                narration_script = response.choices[0].message.content
                # 打印消耗的tokens
                logger.debug(f"文案消耗的tokens: {response.usage.total_tokens}")
                # 清理 narration_script 字符串前后的 ```json ``` 字符串
                narration_script = narration_script.replace("```json", "").replace("```", "")
                return narration_script
            raise RuntimeError("生成解说文案失败: 未获取到有效响应")
    
    except Exception as e:
        logger.error(f"调用API生成解说文案时出错: {traceback.format_exc()}")
        raise RuntimeError(f"调用API生成解说文案时出错: {e}") from e


if __name__ == '__main__':
    text_provider = 'openai'
    text_api_key = "sk-xxx"
    text_model = "deepseek-reasoner"
    text_base_url = "https://api.deepseek.com"
    video_frame_description_path = "/Users/apple/Desktop/home/NarratoAI/storage/temp/analysis/frame_analysis_20250508_1139.json"

    # 测试新的JSON文件
    test_file_path = "/Users/apple/Desktop/home/NarratoAI/storage/temp/analysis/frame_analysis_20250508_2258.json"
    markdown_output = parse_frame_analysis_to_markdown(test_file_path)
    # print(markdown_output)
    
    # 输出到文件以便检查格式
    output_file = "/Users/apple/Desktop/home/NarratoAI/storage/temp/家里家外1-5.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_output)
    # print(f"\n已将Markdown输出保存到: {output_file}")
    
    # # 生成解说文案
    # narration = generate_narration(
    #     markdown_output,
    #     text_api_key,
    #     base_url=text_base_url,
    #     model=text_model
    # )
    #
    # # 保存解说文案
    # print(narration)
    # print(type(narration))
    # narration_file = "/Users/apple/Desktop/home/NarratoAI/storage/temp/final_narration_script.json"
    # with open(narration_file, 'w', encoding='utf-8') as f:
    #     f.write(narration)
    # print(f"\n已将解说文案保存到: {narration_file}")
