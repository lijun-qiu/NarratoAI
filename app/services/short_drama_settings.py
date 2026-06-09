#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""短剧解说规则与成片输出默认值。"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

import toml
from loguru import logger

SHORT_DRAMA_DEFAULTS: Dict[str, Any] = {
    # 解说为主：原声仅作短促点缀
    "narration_percent": 85,
    "original_audio_percent": 15,
    "narration_ratio_tolerance": 0.12,
    "narration_script_temperature": 0.4,
    # 成片总时长目标（分钟）
    "target_output_minutes_min": 8,
    "target_output_minutes_max": 13,
    # 原声 OST=1：仅情绪顶点，单段 ≤5 秒
    "ost1_duration_min": 2,
    "ost1_duration_max": 5,
    # 原声 OST=1 段数上限：0=不限制（按蓝图场景取舍）；>0 为后处理硬上限
    "ost1_max_segments": 0,
    # 播放顺序开头若干段内最多 1 段 OST=1（避免开篇连放两段原声）
    "opening_head_max_ost1": 1,
    "opening_head_segment_count": 3,
    "enable_opening_climax_chronological_replay": True,
    "ost0_duration_min": 5,
    "max_consecutive_ost1": 1,
    "ost0_lead_before_ost1_sec": 5,
    "narration_chars_min": 40,
    "narration_chars_max": 150,
    "max_ershi_per_script": 2,
    # 原声段 picture 旁白烧录（白字黑描边，短剧竖屏可读）
    "enable_picture_narration": True,
    "picture_narration_font_size": 28,
    "picture_narration_color": "#FFFFFF",
    "picture_narration_max_chars": 16,
    "picture_narration_duration": 2.0,
    "picture_wrap_double_quotes": True,
}


def compute_short_drama_segment_bounds(source_duration_sec: float) -> tuple[int, int]:
    """已弃用段数上下限；保留函数签名供旧代码兼容。"""
    _ = source_duration_sec
    return 0, 0


def _estimate_narration_duration_sec(text: str) -> float:
    import re

    chars = len(re.sub(r"\s+", "", text or ""))
    return max(3.0, chars * 0.35)


def _segment_playback_duration_sec(item: Dict[str, Any]) -> float:
    from app.services.update_script import calculate_duration

    ost = int(item.get("OST", 0) or 0)
    timestamp = str(item.get("timestamp") or "").strip()
    narration = str(item.get("narration") or "")

    if ost == 1 and timestamp and "-" in timestamp:
        duration = calculate_duration(timestamp)
        if duration > 0:
            return duration

    if ost == 2 and timestamp and "-" in timestamp:
        ts_duration = calculate_duration(timestamp)
        narr_duration = _estimate_narration_duration_sec(narration)
        if ts_duration > 0:
            return max(ts_duration, narr_duration)

    return _estimate_narration_duration_sec(narration)


def summarize_short_drama_playback(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """按 _id 播放顺序估算成片时长与解说/原声占比。"""
    ost0_sec = 0.0
    ost1_sec = 0.0
    ost2_sec = 0.0

    ordered = sorted(items, key=lambda item: int(item.get("_id") or 0))
    for item in ordered:
        duration = _segment_playback_duration_sec(item)
        if duration <= 0:
            continue
        ost = int(item.get("OST", 0) or 0)
        if ost == 1:
            ost1_sec += duration
        elif ost == 2:
            ost2_sec += duration
        else:
            ost0_sec += duration

    total_sec = ost0_sec + ost1_sec + ost2_sec
    narr_sec = ost0_sec + ost2_sec
    return {
        "total_sec": round(total_sec, 1),
        "ost0_sec": round(ost0_sec, 1),
        "ost1_sec": round(ost1_sec, 1),
        "ost2_sec": round(ost2_sec, 1),
        "narration_sec": round(narr_sec, 1),
        "original_sec": round(ost1_sec, 1),
        "narration_pct": round(narr_sec / total_sec * 100, 1) if total_sec > 0 else 0.0,
        "original_pct": round(ost1_sec / total_sec * 100, 1) if total_sec > 0 else 0.0,
    }


def _config_file_path() -> str:
    from app.config import config

    return config.config_file


def _read_short_drama_config_section() -> Dict[str, Any]:
    try:
        from app.config.config import _cfg

        section = _cfg.get("short_drama", {})
    except Exception:
        try:
            section = toml.load(_config_file_path()).get("short_drama", {})
        except Exception:
            section = {}
    return dict(section) if isinstance(section, dict) else {}


def get_short_drama_settings(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    settings = deepcopy(SHORT_DRAMA_DEFAULTS)
    for key, value in _read_short_drama_config_section().items():
        if key in SHORT_DRAMA_DEFAULTS:
            settings[key] = value
    if overrides:
        for key, value in overrides.items():
            if key in SHORT_DRAMA_DEFAULTS and value is not None:
                settings[key] = value
    return settings


def resolve_ost1_max_segments(settings: Optional[Dict[str, Any]] = None) -> int:
    """返回配置的原声段数上限；0 表示不限制。"""
    return int(get_short_drama_settings(settings).get("ost1_max_segments", 0) or 0)


def format_ost1_max_segments_rule(settings: Optional[Dict[str, Any]] = None) -> str:
    """生成提示词/说明中的原声段数规则文案。"""
    cap = resolve_ost1_max_segments(settings)
    if cap <= 0:
        return (
            "不设固定段数上限，按蓝图场景爆燃点按需增减"
            "（仍须解说为主，单段≤时长上限）"
        )
    return f"全片 ≤{cap} 段"


def compute_short_drama_ost_bounds(
    total_items: int,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """按 items 总数与 3:7 比例推算 OST=0/1 段数上下限。"""
    cfg = settings or get_short_drama_settings()
    total = max(1, int(total_items or 0))
    narr_pct = float(cfg.get("narration_percent", 30) or 30) / 100.0
    orig_pct = float(cfg.get("original_audio_percent", 70) or 70) / 100.0

    ost0_min_cfg = int(cfg.get("ost0_segment_min", 8) or 8)
    ost0_max_cfg = int(cfg.get("ost0_segment_max", 14) or 14)
    ost1_max_cfg = int(cfg.get("ost1_segment_max", 24) or 24)

    ost0_min = max(ost0_min_cfg, int(round(total * narr_pct * 0.85)))
    ost0_max = max(ost0_min, min(ost0_max_cfg, int(round(total * narr_pct * 1.15)) or ost0_min))
    ost1_max = min(ost1_max_cfg, max(1, int(round(total * orig_pct * 1.05))))
    ost1_min = max(1, int(round(total * orig_pct * 0.75)))

    return {
        "ost0_min": ost0_min,
        "ost0_max": ost0_max,
        "ost1_min": ost1_min,
        "ost1_max": ost1_max,
        "total": total,
    }


def get_short_drama_script_prompt_params(
    *,
    source_duration_sec: float = 0,
    expected_total_segments: int = 0,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """构建短剧解说 LLM 提示词参数（时长与占比为主，不限制段数）。"""
    cfg = get_short_drama_settings(settings)
    _ = source_duration_sec, expected_total_segments
    return {
        "narration_percent": str(int(cfg.get("narration_percent", 30))),
        "original_audio_percent": str(int(cfg.get("original_audio_percent", 70))),
        "target_output_minutes_min": str(int(cfg.get("target_output_minutes_min", 8))),
        "target_output_minutes_max": str(int(cfg.get("target_output_minutes_max", 13))),
        "narration_chars_min": str(int(cfg.get("narration_chars_min", 20))),
        "narration_chars_max": str(int(cfg.get("narration_chars_max", 120))),
        "max_consecutive_ost1": str(int(cfg.get("max_consecutive_ost1", 4))),
        "ost1_duration_min": str(int(cfg.get("ost1_duration_min", 2))),
        "ost1_duration_max": str(int(cfg.get("ost1_duration_max", 5))),
        "ost1_max_segments": str(resolve_ost1_max_segments(cfg)),
        "ost1_max_segments_rule": format_ost1_max_segments_rule(cfg),
        "ost0_duration_min": str(int(cfg.get("ost0_duration_min", 5))),
        "picture_narration_max_chars": str(int(cfg.get("picture_narration_max_chars", 16))),
        "max_ershi_per_script": str(int(cfg.get("max_ershi_per_script", 2))),
    }


def get_short_drama_video_output_overrides(
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """短剧解说成片时覆盖 [video_output] 旁白相关项。"""
    cfg = settings or get_short_drama_settings()
    return {
        "enable_picture_narration": bool(cfg.get("enable_picture_narration", True)),
        "picture_narration_font_size": int(cfg.get("picture_narration_font_size", 28)),
        "picture_narration_color": str(cfg.get("picture_narration_color", "#000000")),
        "picture_narration_max_chars": int(cfg.get("picture_narration_max_chars", 16)),
        "picture_narration_duration": float(cfg.get("picture_narration_duration", 2.0)),
    }


def resolve_video_output_for_script_mode(
    base: Optional[Dict[str, Any]] = None,
    *,
    script_path: str = "",
    workflow_mode: str = "",
) -> Dict[str, Any]:
    """按脚本模式合并成片输出配置。"""
    from app.services.video_output_settings import get_video_output_settings

    merged = dict(base or get_video_output_settings())
    path = (script_path or "").strip().lower()
    mode = (workflow_mode or "").strip().lower()
    # 短剧解说：生成中 path 为 summary；保存为 .json 后靠 workflow_mode 识别
    if path == "summary" or mode == "summary":
        merged.update(get_short_drama_video_output_overrides())
    return merged


def save_short_drama_settings_to_config(settings: Dict[str, Any]) -> bool:
    config_path = _config_file_path()
    try:
        if os.path.isfile(config_path):
            config_data = toml.load(config_path)
        else:
            config_data = {}

        section = {}
        for key in SHORT_DRAMA_DEFAULTS:
            section[key] = settings.get(key, SHORT_DRAMA_DEFAULTS[key])
        config_data["short_drama"] = section

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(toml.dumps(config_data))

        try:
            import app.config.config as config_py

            config_py._cfg["short_drama"] = section
        except Exception:
            pass

        logger.info("短剧解说配置已保存到 config.toml [short_drama]")
        return True
    except Exception as exc:
        logger.error(f"保存短剧解说配置失败: {exc}")
        return False
