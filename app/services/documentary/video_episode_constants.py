#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""整片视频分析：时间格与上传转码参数（独立模块，避免循环导入）。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import toml

# 分镜切段：按切镜点切分，每段单独上传分析（不再 1–10 秒随机采样）
SEGMENT_SPLIT_POLICY = "scene_cut"
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
SEGMENT_MIN_SECONDS = 1
SEGMENT_MAX_SECONDS = 10

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
}

VIDEO_EPISODE_UPLOAD_DEFAULTS: Dict[str, Any] = {
    # 单镜上传体积上限（MB）
    "max_upload_mb": 24.0,
    # 分镜截取前整片压缩档位：high=720p（默认）
    "upload_transcode_profile": "high",
    # 兼容旧配置（分镜模式下不再按固定时长切段）
    "chunk_seconds": 300.0,
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
