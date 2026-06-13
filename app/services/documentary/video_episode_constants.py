#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""整片视频分析：时间格与上传转码参数（独立模块，避免循环导入）。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import toml

# 上传切段策略：time_chunk=按固定时长；scene_cut=按切镜/场景
SEGMENT_SPLIT_POLICY = "time_chunk"
SEGMENT_POLICY_TIME_CHUNK = "time_chunk"
SEGMENT_POLICY_SCENE_CUT = "scene_cut"
SCENE_MIN_MERGE_SECONDS = 3.0
SCENE_MIN_SEGMENT_SECONDS = 5.0
SCENE_MAX_SECONDS = 180.0
SCENE_DETECT_THRESHOLD = 0.35
SCENE_CUT_MODE = "environment_change"
SCENE_CANDIDATE_THRESHOLD = 0.25
SCENE_ENVIRONMENT_DIFF_THRESHOLD = 0.18
SCENE_FRAME_SAMPLE_BEFORE_SECONDS = 3.0
SCENE_FRAME_SAMPLE_AFTER_SECONDS = 1.0

# 兼容旧 adaptive_scene 策略与 JSON 字段
SEGMENT_MIN_SECONDS = 5
SEGMENT_MAX_SECONDS = 10

# 视频分析：切镜段内再按 5–10 秒窗口输出 episodic_segments
VIDEO_ANALYSIS_SUBSEGMENT_MIN_SECONDS = 5.0
VIDEO_ANALYSIS_SUBSEGMENT_MAX_SECONDS = 10.0

# 兼容旧 JSON / token 估算（固定格时代为 4）
SEGMENT_INTERVAL_SECONDS = 4

# 上传给视觉模型的转码档位（宽 px · fps · CRF · 音频 · x264 preset）
UPLOAD_TRANSCODE_PROFILES: Dict[str, Dict[str, Any]] = {
    "high": {
        "width": 720,
        "fps": 15,
        "crf": 26,
        "audio_bitrate": "64k",
        "preset": "fast",
    },
    "standard": {
        "width": 640,
        "fps": 15,
        "crf": 28,
        "audio_bitrate": "48k",
        "preset": "fast",
    },
    "compact": {
        "width": 480,
        "fps": 12,
        "crf": 32,
        "audio_bitrate": "32k",
        "preset": "veryfast",
    },
    # 整片网格快扫：优先体积，便于单次上传
    "grid_whole": {
        "width": 480,
        "fps": 10,
        "crf": 32,
        "audio_bitrate": "32k",
        "preset": "veryfast",
    },
    "grid_whole_compact": {
        "width": 360,
        "fps": 8,
        "crf": 34,
        "audio_bitrate": "24k",
        "preset": "veryfast",
    },
    "grid_whole_tiny": {
        "width": 320,
        "fps": 6,
        "crf": 36,
        "audio_bitrate": "16k",
        "preset": "veryfast",
    },
    "grid_whole_ultra": {
        "width": 240,
        "fps": 5,
        "crf": 38,
        "audio_bitrate": "0",
        "preset": "veryfast",
        "no_audio": True,
    },
}

VIDEO_EPISODE_UPLOAD_DEFAULTS: Dict[str, Any] = {
    # 单段上传体积上限（MB）
    "max_upload_mb": 24.0,
    # 分镜截取前整片压缩档位：high=720p（默认）
    "upload_transcode_profile": "high",
    # 上传切段：time_chunk | scene_cut
    "segment_split_policy": SEGMENT_SPLIT_POLICY,
    # 按时间切段时，每段时长（秒）；默认 15 分钟
    "chunk_seconds": 900.0,
    # 单次 API 最多输出的 5–10 秒情节窗条数；≥96 时 15 分钟上传段通常一批出完
    "max_segments_per_api_call": 96,
    # 整片视频分析时注入配对抽帧 frame_timeline（默认关闭，说话人由视频画面推断）
    "enable_frame_timeline_reference": False,
    "frame_timeline_reference_max_chars": 14000,
    "short_video_high_profile_sec": 300.0,
}

VIDEO_EPISODE_SCENE_DEFAULTS: Dict[str, Any] = {
    # environment_change=仅场景/环境明显变化才切段；edit_cut=每个硬切都切
    "scene_cut_mode": SCENE_CUT_MODE,
    # 候选硬切灵敏度（environment 模式下先找候选，再按画面差异过滤）
    "scene_candidate_threshold": SCENE_CANDIDATE_THRESHOLD,
    # edit_cut 模式直接使用的阈值
    "scene_detect_threshold": SCENE_DETECT_THRESHOLD,
    # 切点后稳定帧 vs 切点前参考帧 的画面差异 ≥ 此值才视为场景/环境切换
    "scene_environment_diff_threshold": SCENE_ENVIRONMENT_DIFF_THRESHOLD,
    # 参考帧：切点前若干秒（同场景对白反打在此仍相似，换景则差异大）
    "scene_frame_sample_before_seconds": SCENE_FRAME_SAMPLE_BEFORE_SECONDS,
    # 稳定帧：切点后若干秒（跳过转场叠化）
    "scene_frame_sample_after_seconds": SCENE_FRAME_SAMPLE_AFTER_SECONDS,
    # 切镜后短于该值的镜头并入相邻段
    "scene_min_merge_seconds": SCENE_MIN_MERGE_SECONDS,
    # 最终上传段最短时长（秒）
    "scene_min_segment_seconds": SCENE_MIN_SEGMENT_SECONDS,
    # 无切镜长镜头上限（秒）
    "scene_max_seconds": SCENE_MAX_SECONDS,
}

_PROFILE_FALLBACK_ORDER = ("high", "standard", "compact")


def _config_file_path() -> str:
    from app.config import config

    return config.config_file


def resolve_segment_split_policy(
    overrides: Dict[str, Any] | None = None,
) -> str:
    settings = get_video_episode_upload_settings(overrides)
    policy = str(settings.get("segment_split_policy") or SEGMENT_SPLIT_POLICY).strip().lower()
    if policy in ("time_chunk", "time", "fixed_time", "chunk"):
        return SEGMENT_POLICY_TIME_CHUNK
    if policy in ("scene_cut", "scene"):
        return SEGMENT_POLICY_SCENE_CUT
    if policy == "adaptive_scene":
        return "adaptive_scene"
    return SEGMENT_POLICY_TIME_CHUNK


def resolve_upload_chunk_seconds(
    overrides: Dict[str, Any] | None = None,
) -> float:
    settings = get_video_episode_upload_settings(overrides)
    try:
        return max(60.0, float(settings.get("chunk_seconds") or 900.0))
    except (TypeError, ValueError):
        return 900.0


def resolve_max_segments_per_api_call(
    overrides: Dict[str, Any] | None = None,
) -> int:
    """单次模型调用最多输出多少条 episodic_segments 时间窗；0 表示不拆分。"""
    settings = get_video_episode_upload_settings(overrides)
    try:
        return max(0, int(settings.get("max_segments_per_api_call") or 96))
    except (TypeError, ValueError):
        return 96


def get_video_episode_upload_settings(
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    settings = deepcopy(VIDEO_EPISODE_UPLOAD_DEFAULTS)
    try:
        from app.config.config import _cfg

        section = _cfg.get("video_episode_analysis", {})
    except Exception:
        try:
            section = toml.load(_config_file_path()).get("video_episode_analysis", {})
        except Exception:
            section = {}
    if isinstance(section, dict):
        for key in VIDEO_EPISODE_UPLOAD_DEFAULTS:
            if key in section and section[key] is not None:
                settings[key] = section[key]
    if overrides:
        for key, value in overrides.items():
            if key in VIDEO_EPISODE_UPLOAD_DEFAULTS and value is not None:
                settings[key] = value
    return settings


def resolve_upload_profile_chain(*, short_video: bool) -> tuple[str, ...]:
    """按片长返回尝试顺序：先清晰档，仍超限再降级 compact。"""
    if short_video:
        return _PROFILE_FALLBACK_ORDER
    return ("standard", "compact")


def get_upload_transcode_profile(name: str) -> Dict[str, Any]:
    profile = UPLOAD_TRANSCODE_PROFILES.get(name)
    if not profile:
        raise ValueError(f"未知上传转码档位: {name}")
    return dict(profile)


def get_video_episode_scene_settings(
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    settings = deepcopy(VIDEO_EPISODE_SCENE_DEFAULTS)
    try:
        from app.config.config import _cfg

        section = _cfg.get("video_episode_analysis", {})
    except Exception:
        try:
            section = toml.load(_config_file_path()).get("video_episode_analysis", {})
        except Exception:
            section = {}
    if isinstance(section, dict):
        for key in VIDEO_EPISODE_SCENE_DEFAULTS:
            if key in section and section[key] is not None:
                settings[key] = section[key]
    if overrides:
        for key, value in overrides.items():
            if key in VIDEO_EPISODE_SCENE_DEFAULTS and value is not None:
                settings[key] = value
    return settings


def resolve_upload_transcode_profile_name(
    overrides: Dict[str, Any] | None = None,
) -> str:
    """读取整片压缩档位名（默认 high=720p）。"""
    settings = get_video_episode_upload_settings(overrides)
    name = str(settings.get("upload_transcode_profile") or "high").strip()
    if name not in UPLOAD_TRANSCODE_PROFILES:
        return "high"
    return name
