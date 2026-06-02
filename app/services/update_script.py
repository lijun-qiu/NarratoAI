#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project: NarratoAI
@File   : update_script
@Author : Viccy同学
@Date   : 2025/5/6 下午11:00 
'''

import re
import os
import subprocess
from typing import Dict, List, Any, Tuple, Union

from loguru import logger

from app.utils import utils

# 统一裁剪输出: ost0_vid_00-00-00-000@00-00-20-250.mp4；旧版: vid_... / vid-...
_CLIP_FILENAME_TS_RE = re.compile(
    r"(?:ost[012]_)?vid[_-]"
    r"(\d{2})-(\d{2})-(\d{2})-(\d{3})@(\d{2})-(\d{2})-(\d{2})-(\d{3})\.mp4",
    re.IGNORECASE,
)
_CLIP_FILENAME_TS_LEGACY_RE = re.compile(
    r"(?:ost[012]_)?vid-(\d{2}-\d{2}-\d{2})-(\d{2}-\d{2}-\d{2})\.mp4",
    re.IGNORECASE,
)

MIN_VALID_VIDEO_BYTES = 4096


def extract_timestamp_from_video_path(video_path: str) -> str:
    """
    从视频文件路径中提取时间戳

    Args:
        video_path: 视频文件路径

    Returns:
        提取出的时间戳，格式为 'HH:MM:SS-HH:MM:SS' 或 'HH:MM:SS,sss-HH:MM:SS,sss'
    """
    filename = os.path.basename(video_path)

    match_new = _CLIP_FILENAME_TS_RE.search(filename)
    if match_new:
        start_h, start_m, start_s, start_ms = (
            match_new.group(1),
            match_new.group(2),
            match_new.group(3),
            match_new.group(4),
        )
        end_h, end_m, end_s, end_ms = (
            match_new.group(5),
            match_new.group(6),
            match_new.group(7),
            match_new.group(8),
        )
        return f"{start_h}:{start_m}:{start_s},{start_ms}-{end_h}:{end_m}:{end_s},{end_ms}"

    match_old = _CLIP_FILENAME_TS_LEGACY_RE.search(filename)
    if match_old:
        start_time = match_old.group(1).replace("-", ":")
        end_time = match_old.group(2).replace("-", ":")
        return f"{start_time}-{end_time}"

    return ""


def is_valid_video_file(video_path: str, min_bytes: int = MIN_VALID_VIDEO_BYTES) -> bool:
    """检查 mp4 是否包含可解码的视频流（排除中断/空壳文件）。"""
    if not video_path or not os.path.isfile(video_path):
        return False
    try:
        if os.path.getsize(video_path) < min_bytes:
            return False
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() == "video"
    except OSError:
        return False


def probe_media_duration(media_path: str) -> float:
    """用 ffprobe 读取媒体真实时长（秒），失败时返回 0。"""
    if not media_path or not os.path.isfile(media_path):
        return 0.0
    ext = os.path.splitext(media_path)[1].lower()
    if ext in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv"}:
        if not is_valid_video_file(media_path, min_bytes=512):
            return 0.0
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                media_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration = float(result.stdout.strip())
        return round(duration, 2) if duration > 0 else 0.0
    except (subprocess.CalledProcessError, ValueError, OSError):
        return 0.0


def _duration_to_source_time_range(duration_sec: float) -> str:
    """根据时长生成相对时间范围（用于无法从文件名解析时）。"""
    end_time = utils.seconds_to_time(duration_sec).replace(".", ",")
    return f"00:00:00,000-{end_time}"


def get_clip_info_from_video_path(video_path: str) -> Tuple[str, float]:
    """
    解析裁剪片段的源时间范围与真实时长。
    优先从文件名解析；失败则用 ffprobe 探测文件时长。
    """
    if not video_path:
        return "", 0.0

    probed = probe_media_duration(video_path) if os.path.isfile(video_path) else 0.0

    timestamp_range = extract_timestamp_from_video_path(video_path)
    if timestamp_range:
        duration = calculate_duration(timestamp_range)
        if probed > 0:
            duration = probed
        if duration > 0:
            return timestamp_range, duration

    if probed > 0:
        return _duration_to_source_time_range(probed), probed

    return "", 0.0


def format_edited_time_range(start_sec: float, end_sec: float) -> str:
    """成品时间轴范围，保留毫秒避免累积截断误差。"""
    return (
        f"{utils.seconds_to_time(start_sec).replace('.', ',')}-"
        f"{utils.seconds_to_time(end_sec).replace('.', ',')}"
    )


def probe_segment_video_duration(segment: Dict[str, Any]) -> float:
    """成片拼接时间轴：以裁剪后 mp4 的 ffprobe 时长为准。"""
    video_path = segment.get("video") or ""
    if video_path:
        probed = probe_media_duration(video_path)
        if probed > 0:
            return round(probed, 3)
    return 0.0


def rebuild_edited_time_ranges(script_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按脚本顺序、用实测视频时长重建 editedTimeRange，避免累积偏移。"""
    accumulated = 0.0
    for item in script_list:
        duration = probe_segment_video_duration(item)
        if duration <= 0:
            duration = float(item.get("duration") or 0)
        if duration <= 0:
            continue
        item["duration"] = round(duration, 3)
        start_sec = accumulated
        end_sec = accumulated + duration
        item["editedTimeRange"] = format_edited_time_range(start_sec, end_sec)
        accumulated = end_sec
    return script_list


def collect_processed_segment_durations(task_id: str, clip_count: int) -> List[float]:
    """读取 merger 重编码后的 processed_*.mp4 时长（与 video_clips 顺序一致）。"""
    if clip_count <= 0:
        return []
    temp_dir = os.path.join(utils.task_dir(task_id), "temp_videos")
    durations: List[float] = []
    for index in range(clip_count):
        processed_path = os.path.join(temp_dir, f"processed_{index}.mp4")
        probed = probe_media_duration(processed_path) if os.path.isfile(processed_path) else 0.0
        durations.append(probed if probed > 0 else 0.0)
    return durations


def rebuild_edited_time_ranges_from_processed(
    script_list: List[Dict[str, Any]],
    processed_durations: List[float],
) -> List[Dict[str, Any]]:
    """用 merger 重编码后的片段时长重建时间轴，与 merger.mp4 原声对齐。"""
    duration_index = 0
    accumulated = 0.0
    for item in script_list:
        video_path = item.get("video") or ""
        if not video_path or not is_valid_video_file(video_path):
            continue

        duration = 0.0
        if duration_index < len(processed_durations) and processed_durations[duration_index] > 0:
            duration = processed_durations[duration_index]
        if duration <= 0:
            duration = probe_segment_video_duration(item)
        if duration <= 0:
            duration = float(item.get("duration") or 0)
        duration_index += 1

        if duration <= 0:
            logger.warning(
                f"片段 {item.get('_id')} 无法确定成片时长，跳过 editedTimeRange 更新"
            )
            continue

        item["duration"] = round(duration, 3)
        start_sec = accumulated
        end_sec = accumulated + duration
        item["editedTimeRange"] = format_edited_time_range(start_sec, end_sec)
        accumulated = end_sec

    return script_list


def sync_script_timeline_after_video_merge(
    task_id: str,
    script_list: List[Dict[str, Any]],
    video_clip_count: int,
) -> List[Dict[str, Any]]:
    """视频合并后按 processed 片段时长校正成片时间轴。"""
    processed = collect_processed_segment_durations(task_id, video_clip_count)
    if not processed or all(value <= 0 for value in processed):
        logger.warning("未找到 processed 片段时长，保留裁剪后时间轴")
        return script_list

    before_total = sum(float(item.get("duration") or 0) for item in script_list)
    updated = rebuild_edited_time_ranges_from_processed(script_list, processed)
    after_total = sum(float(item.get("duration") or 0) for item in script_list)
    delta = round(after_total - before_total, 3)
    if abs(delta) >= 0.05:
        logger.info(
            f"成片时间轴已按 merger 重编码校正: {before_total:.3f}s -> {after_total:.3f}s (Δ{delta:+.3f}s)"
        )
    return updated


def resolve_segment_timeline_duration(
    segment: Dict[str, Any],
    *,
    video_duration: float = 0.0,
    tts_duration_by_id: Dict[Union[str, int], float] = None,
) -> float:
    """成片时间轴单段时长：与 merger.mp4 一致，统一使用裁剪视频实测时长。"""
    probed = probe_segment_video_duration(segment)
    if probed > 0:
        return probed
    if video_duration > 0:
        return round(video_duration, 3)
    return 0.0


def calculate_duration(timestamp: str) -> float:
    """
    计算时间戳范围的持续时间（秒）

    Args:
        timestamp: 格式为 'HH:MM:SS-HH:MM:SS' 或 'HH:MM:SS,sss-HH:MM:SS,sss' 的时间戳

    Returns:
        持续时间（秒）
    """
    try:
        start_time, end_time = timestamp.split('-')

        if ',' in start_time:
            start_parts = start_time.split(',')
            start_time_parts = start_parts[0].split(':')
            start_ms = float('0.' + start_parts[1]) if len(start_parts) > 1 else 0
            start_h, start_m, start_s = map(int, start_time_parts)
        else:
            start_h, start_m, start_s = map(int, start_time.split(':'))
            start_ms = 0

        if ',' in end_time:
            end_parts = end_time.split(',')
            end_time_parts = end_parts[0].split(':')
            end_ms = float('0.' + end_parts[1]) if len(end_parts) > 1 else 0
            end_h, end_m, end_s = map(int, end_time_parts)
        else:
            end_h, end_m, end_s = map(int, end_time.split(':'))
            end_ms = 0

        start_seconds = start_h * 3600 + start_m * 60 + start_s + start_ms
        end_seconds = end_h * 3600 + end_m * 60 + end_s + end_ms

        return round(end_seconds - start_seconds, 2)
    except (ValueError, AttributeError):
        return 0.0


def update_script_timestamps(
    script_list: List[Dict[str, Any]],
    video_result: Dict[Union[str, int], str],
    audio_result: Dict[Union[str, int], str] = None,
    subtitle_result: Dict[Union[str, int], str] = None,
    calculate_edited_timerange: bool = True,
    tts_duration_by_id: Dict[Union[str, int], float] = None,
) -> List[Dict[str, Any]]:
    """
    根据 video_result 中的视频文件更新 script_list 中的时间戳，添加持续时间，
    并根据 audio_result 添加音频路径，根据 subtitle_result 添加字幕路径
    """
    updated_script = []

    id_timestamp_mapping = {}
    for key, video_path in video_result.items():
        source_time_range, clip_duration = get_clip_info_from_video_path(video_path)
        if clip_duration > 0:
            id_timestamp_mapping[key] = {
                "new_timestamp": source_time_range,
                "video_path": video_path,
                "duration": clip_duration,
            }

    accumulated_duration = 0.0

    for item in script_list:
        item_copy = item.copy()
        item_id = item_copy.get('_id')
        orig_timestamp = item_copy.get('timestamp', '')

        item_copy['audio'] = ""
        item_copy['subtitle'] = ""
        item_copy['video'] = ""

        if audio_result:
            if item_id and item_id in audio_result:
                item_copy['audio'] = audio_result[item_id]
            elif orig_timestamp in audio_result:
                item_copy['audio'] = audio_result[orig_timestamp]

        if subtitle_result:
            if item_id and item_id in subtitle_result:
                item_copy['subtitle'] = subtitle_result[item_id]
            elif orig_timestamp in subtitle_result:
                item_copy['subtitle'] = subtitle_result[orig_timestamp]

        if item_id and item_id in video_result:
            item_copy['video'] = video_result[item_id]
        elif orig_timestamp in video_result:
            item_copy['video'] = video_result[orig_timestamp]

        video_duration = 0.0
        clip_mapping = None
        if item_id and item_id in id_timestamp_mapping:
            clip_mapping = id_timestamp_mapping[item_id]
        elif orig_timestamp in id_timestamp_mapping:
            clip_mapping = id_timestamp_mapping[orig_timestamp]

        if clip_mapping:
            item_copy["sourceTimeRange"] = clip_mapping["new_timestamp"] or orig_timestamp
            video_duration = clip_mapping["duration"]
        elif item_copy.get("video"):
            source_time_range, video_duration = get_clip_info_from_video_path(item_copy["video"])
            if video_duration > 0:
                item_copy["sourceTimeRange"] = source_time_range or orig_timestamp
            elif orig_timestamp:
                item_copy["sourceTimeRange"] = orig_timestamp
                video_duration = calculate_duration(orig_timestamp)
        elif orig_timestamp:
            item_copy["sourceTimeRange"] = orig_timestamp
            video_duration = calculate_duration(orig_timestamp)

        current_duration = resolve_segment_timeline_duration(
            item_copy,
            video_duration=video_duration,
            tts_duration_by_id=tts_duration_by_id,
        )
        if current_duration <= 0 and video_duration > 0:
            current_duration = video_duration
        if current_duration > 0:
            item_copy["duration"] = current_duration

        if calculate_edited_timerange and current_duration > 0:
            start_time_seconds = accumulated_duration
            end_time_seconds = accumulated_duration + current_duration
            item_copy["editedTimeRange"] = format_edited_time_range(
                start_time_seconds, end_time_seconds
            )
            accumulated_duration = end_time_seconds

        updated_script.append(item_copy)

    if calculate_edited_timerange:
        rebuild_edited_time_ranges(updated_script)

    return updated_script


if __name__ == '__main__':
    list_script = [
        {
            'picture': '【解说】好的，各位，欢迎回到我的频道！《庆余年 2》刚开播就给了我们一个王炸！范闲在北齐"死"了？这怎么可能！',
            'timestamp': '00:00:00,001-00:01:15,001',
            'narration': '好的各位，欢迎回到我的频道！《庆余年 2》刚开播就给了我们一个王炸！范闲在北齐"死"了？这怎么可能！上集片尾那个巨大的悬念，这一集就立刻揭晓了！范闲假死归来，他面临的第一个，也是最大的难关，就是如何面对他最敬爱的，同时也是最可怕的那个人——庆帝！',
            'OST': 0,
            '_id': 1
        },
    ]
    video_res = {
        1: '/Users/apple/Desktop/home/NarratoAI/storage/temp/clip_video/fc3db5844d1ba7d7d838be52c0dac1bd/vid_00-00-00-000@00-00-20-250.mp4',
    }
    audio_res = {1: '/Users/apple/Desktop/home/NarratoAI/storage/tasks/qyn2-2-demo/audio_00_00_00-00_01_15.mp3'}
    sub_res = {1: '/Users/apple/Desktop/home/NarratoAI/storage/tasks/qyn2-2-demo/subtitle_00_00_00-00_01_15.srt'}

    updated_list_script = update_script_timestamps(list_script, video_res, audio_res, sub_res)
    for item in updated_list_script:
        print(
            f"ID: {item['_id']} | SourceTimeRange: {item['sourceTimeRange']} | "
            f"EditedTimeRange: {item.get('editedTimeRange', '')} | Duration: {item['duration']} 秒"
        )
