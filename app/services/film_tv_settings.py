#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
影视解说规则参数：默认值、读取 config、UI 覆盖与持久化。
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Optional

import toml
from loguru import logger

from app.services.film_tv_rule_presets import (
    DEFAULT_PRESET_ID,
    FAZU2_RULES_MARKDOWN,
    apply_preset_to_settings,
    find_preset_id_for_rules_file,
    format_style_directive,
    list_film_tv_rules_markdown_files,
)
from app.services.picture_narration_builder import load_picture_narration_rules

TV_CONTENT_MOVIE = "movie"
TV_CONTENT_SERIES = "tv_series"

RULES_SOURCE_MARKDOWN = "markdown"
RULES_SOURCE_CUSTOM = "custom"

FILM_TV_DEFAULTS: Dict[str, Any] = {
    # 默认方案：《罚罪2》悬疑脉络（原声 40% / 解说 60%）
    "preset_id": DEFAULT_PRESET_ID,
    "target_duration_percent": 30,
    "ost1_duration_min": 5,
    "ost1_duration_max": 11,
    "ost1_duration_long_max": 14,
    "ost1_segment_min": 8,
    "ost1_segment_max": 14,
    "ost0_segment_min": 16,
    "ost0_segment_max": 24,
    "original_audio_percent": 40,
    "narration_percent": 60,
    "allow_consecutive_ost1": True,
    "enforce_narration_after_ost1": True,
    "narration_chars_min": 48,
    "narration_chars_max": 78,
    "opening_chars_max": 110,
    "max_consecutive_ost0": 3,
    "max_consecutive_ost1": 3,
    # 电视剧分集解说（content_type=tv_series 时生效）
    "content_type": "movie",
    "episode_number": 1,
    "tv_opening_line_template": "宝子们，我们开始《{film_name}》第{episode}集啦！",
    "tv_closing_line_template": "好啦宝子们，我们下集再见！",
    "tv_recap_prev_episode": True,
    "tv_recap_chars_min": 40,
    "tv_recap_chars_max": 80,
    # 规则来源：markdown=仅 MD 文件规则；custom=方案+页面滑块（互斥）
    "rules_source_mode": RULES_SOURCE_MARKDOWN,
    "rules_markdown_file": FAZU2_RULES_MARKDOWN,
}

# Markdown 模式下仍允许覆盖的分集/话术参数（不属于「自定义剪辑规则滑块」）
FILM_TV_OPERATIONAL_KEYS = (
    "content_type",
    "episode_number",
    "tv_opening_line_template",
    "tv_closing_line_template",
    "tv_recap_prev_episode",
    "tv_recap_chars_min",
    "tv_recap_chars_max",
)


def format_tv_line_template(template: str, film_name: str, episode: int) -> str:
    """将开场/收尾话术模板中的 {film_name}、{episode} 替换为实际值。"""
    text = str(template or "").strip()
    return (
        text.replace("{film_name}", (film_name or "").strip())
        .replace("{episode}", str(int(episode or 1)))
    )


def build_tv_series_prompt_block(
    film_name: str,
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """电视剧分集解说写入 LLM 提示词的规则块；电影模式返回空字符串。"""
    cfg = get_film_tv_settings(settings)
    if cfg.get("content_type") != TV_CONTENT_SERIES:
        return ""

    episode = max(1, int(cfg.get("episode_number") or 1))
    opening = format_tv_line_template(
        cfg.get("tv_opening_line_template") or FILM_TV_DEFAULTS["tv_opening_line_template"],
        film_name,
        episode,
    )
    closing = format_tv_line_template(
        cfg.get("tv_closing_line_template") or FILM_TV_DEFAULTS["tv_closing_line_template"],
        film_name,
        episode,
    )
    recap_min = int(cfg.get("tv_recap_chars_min") or 40)
    recap_max = int(cfg.get("tv_recap_chars_max") or 80)
    opening_max = int(cfg.get("opening_chars_max") or 110)

    lines = [
        f"## 电视剧分集解说（第 {episode} 集）",
        "- **全片第一段与最后一段必须是 OST=0 解说**（禁止以原声 OST=1 开场或收尾）。",
        f"- **第一段 OST=0 解说**必须先口播开场语：「{opening}」",
    ]
    if episode > 1 and cfg.get("tv_recap_prev_episode", True):
        prev = episode - 1
        lines.append(
            f"- 开场语之后、进入本集讲解之前，用约 **{recap_min}–{recap_max} 字**概括"
            f"**第 {prev} 集**主要剧情（人物、冲突、悬念），再衔接本集内容；"
            f"整段开场解说（含回顾）不超过 **{opening_max} 字**。"
        )
    else:
        lines.append("- 开场语之后直接进入本集剧情讲解。")
    lines.append(
        f"- **最后一段 OST=0 解说**在本集正文讲完后，必须口播收尾语：「{closing}」"
    )
    return "\n".join(lines)


def _config_file_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(root, "config.toml")


def _load_film_tv_from_config() -> Dict[str, Any]:
    settings = deepcopy(FILM_TV_DEFAULTS)
    try:
        from app.config.config import _cfg
        film_tv = _cfg.get("film_tv", {})
    except Exception:
        try:
            film_tv = toml.load(_config_file_path()).get("film_tv", {})
        except Exception:
            film_tv = {}

    for key in FILM_TV_DEFAULTS:
        if key in film_tv:
            settings[key] = film_tv[key]
    return settings


def get_film_tv_settings(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """合并 config.toml 与页面覆盖；Markdown 与自定义规则互斥。"""
    settings = _load_film_tv_from_config()
    ov = dict(overrides or {})
    mode = ov.get("rules_source_mode") or settings.get("rules_source_mode") or RULES_SOURCE_MARKDOWN
    settings["rules_source_mode"] = mode

    if mode == RULES_SOURCE_MARKDOWN:
        md_file = (
            ov.get("rules_markdown_file")
            or settings.get("rules_markdown_file")
            or FAZU2_RULES_MARKDOWN
        )
        settings["rules_markdown_file"] = md_file
        linked_preset = (
            find_preset_id_for_rules_file(md_file)
            or settings.get("preset_id")
            or DEFAULT_PRESET_ID
        )
        settings = apply_preset_to_settings(
            settings, linked_preset, use_rules_markdown=True
        )
        for key in FILM_TV_OPERATIONAL_KEYS:
            if key in ov and ov[key] is not None:
                settings[key] = ov[key]
    else:
        preset_id = ov.get("preset_id") or settings.get("preset_id") or DEFAULT_PRESET_ID
        settings = apply_preset_to_settings(
            settings, preset_id, use_rules_markdown=False
        )
        settings["rules_markdown_file"] = ""
        for key, value in ov.items():
            if value is None:
                continue
            if key in FILM_TV_DEFAULTS:
                settings[key] = value
            elif key == "preset_id" and value:
                settings = apply_preset_to_settings(
                    settings, value, use_rules_markdown=False
                )
    return settings


def _build_style_directive_params(
    cfg: Dict[str, Any],
    source_minutes: float,
    target_minutes: float,
    film_name: str,
) -> Dict[str, str]:
    """填充 rules/*.md 与 style_directive 模板中的 ${var} 占位符。"""
    return {
        "source_duration_minutes": f"{source_minutes:.1f}",
        "target_output_minutes": f"{target_minutes:.1f}",
        "target_duration_percent": str(int(cfg["target_duration_percent"])),
        "ost1_duration_min": str(int(cfg["ost1_duration_min"])),
        "ost1_duration_max": str(int(cfg["ost1_duration_max"])),
        "ost1_segment_min": str(cfg["ost1_segment_min"]),
        "ost1_segment_max": str(cfg["ost1_segment_max"]),
        "ost0_segment_min": str(cfg["ost0_segment_min"]),
        "ost0_segment_max": str(cfg["ost0_segment_max"]),
        "original_audio_percent": str(int(cfg["original_audio_percent"])),
        "narration_percent": str(int(cfg["narration_percent"])),
        "narration_chars_min": str(int(cfg["narration_chars_min"])),
        "narration_chars_max": str(int(cfg["narration_chars_max"])),
        "opening_chars_max": str(int(cfg["opening_chars_max"])),
        "film_name": (film_name or "").strip() or "罚罪2",
        "episode": str(max(1, int(cfg.get("episode_number") or 1))),
        "tv_recap_chars_min": str(int(cfg.get("tv_recap_chars_min") or 40)),
        "tv_recap_chars_max": str(int(cfg.get("tv_recap_chars_max") or 80)),
    }


def get_film_tv_script_prompt_params(
    source_duration_sec: Optional[float] = None,
    settings: Optional[Dict[str, Any]] = None,
    film_name: str = "",
) -> Dict[str, str]:
    """构建影视解说 LLM 提示词参数。"""
    cfg = get_film_tv_settings(settings)
    if source_duration_sec and source_duration_sec > 0:
        source_minutes = source_duration_sec / 60
        target_minutes = source_duration_sec * cfg["target_duration_percent"] / 100 / 60
    else:
        source_minutes = 5.0
        target_minutes = 2.0

    return {
        "source_duration_minutes": f"{source_minutes:.1f}",
        "target_output_minutes": f"{target_minutes:.1f}",
        "target_duration_percent": str(int(cfg["target_duration_percent"])),
        "ost1_duration_min": str(int(cfg["ost1_duration_min"])),
        "ost1_duration_max": str(int(cfg["ost1_duration_max"])),
        "ost1_duration_long_max": str(int(cfg["ost1_duration_long_max"])),
        "ost1_segment_min": str(cfg["ost1_segment_min"]),
        "ost1_segment_max": str(cfg["ost1_segment_max"]),
        "ost0_segment_min": str(cfg["ost0_segment_min"]),
        "ost0_segment_max": str(cfg["ost0_segment_max"]),
        "total_segment_min": str(int(cfg["ost1_segment_min"]) + int(cfg["ost0_segment_min"])),
        "original_audio_percent": str(int(cfg["original_audio_percent"])),
        "narration_percent": str(int(cfg["narration_percent"])),
        "narration_chars_min": str(int(cfg["narration_chars_min"])),
        "narration_chars_max": str(int(cfg["narration_chars_max"])),
        "opening_chars_max": str(int(cfg["opening_chars_max"])),
        "editing_mode_name": str(cfg.get("preset_name") or "均衡解说"),
        "editor_persona": str(cfg.get("editor_persona") or ""),
        "film_name": (film_name or "").strip(),
        "episode": str(max(1, int(cfg.get("episode_number") or 1))),
        "tv_recap_chars_min": str(int(cfg.get("tv_recap_chars_min") or 40)),
        "tv_recap_chars_max": str(int(cfg.get("tv_recap_chars_max") or 80)),
        "style_directive": format_style_directive(
            str(cfg.get("style_directive") or ""),
            _build_style_directive_params(cfg, source_minutes, target_minutes, film_name),
        ),
        "tv_series_rules": build_tv_series_prompt_block(film_name, settings=cfg),
        "picture_narration_rules": load_picture_narration_rules(),
    }


def save_film_tv_settings_to_config(settings: Dict[str, Any]) -> bool:
    """将当前参数写入 config.toml 的 [film_tv] 段。"""
    config_path = _config_file_path()
    try:
        if os.path.isfile(config_path):
            config_data = toml.load(config_path)
        else:
            config_data = {}

        film_tv = {}
        for key in FILM_TV_DEFAULTS:
            film_tv[key] = settings.get(key, FILM_TV_DEFAULTS[key])
        if settings.get("preset_id"):
            film_tv["preset_id"] = settings["preset_id"]
        config_data["film_tv"] = film_tv

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(toml.dumps(config_data))

        try:
            from app.config import config as config_module
            config_module.film_tv = film_tv
            import app.config.config as config_py
            config_py._cfg["film_tv"] = film_tv
            config_py.film_tv = film_tv
        except Exception:
            pass

        logger.info("影视解说规则已保存到 config.toml")
        return True
    except Exception as e:
        logger.error(f"保存影视解说配置失败: {e}")
        return False
