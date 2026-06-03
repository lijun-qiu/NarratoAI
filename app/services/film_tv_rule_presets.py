#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
影视解说规则方案（模块化预设）。

每个方案包含：数值参数 + 剪辑师人设 + 专项 AI 提示词片段。
页面勾选方案后，参数与提示词一并生效。
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

PRESET_BALANCED = "balanced_narration"
PRESET_ORIGINAL_HEAVY = "original_heavy"
PRESET_FAZU2 = "fazu2"

DEFAULT_PRESET_ID = PRESET_FAZU2

FAZU2_RULES_MARKDOWN = "fazu2_8min_master.md"

_RULES_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "prompts",
    "film_tv_narration",
    "rules",
)

# 仅数值类键（不含 preset_id / 提示词字段）
NUMERIC_SETTING_KEYS = (
    "target_duration_percent",
    "ost1_duration_min",
    "ost1_duration_max",
    "ost1_duration_long_max",
    "ost1_segment_min",
    "ost1_segment_max",
    "ost0_segment_min",
    "ost0_segment_max",
    "original_audio_percent",
    "narration_percent",
    "allow_consecutive_ost1",
    "enforce_narration_after_ost1",
    "narration_chars_min",
    "narration_chars_max",
    "opening_chars_max",
)

# 方案可一并写入 settings 的非数值键
PRESET_EXTRA_SETTING_KEYS = (
    "content_type",
    "episode_number",
    "tv_opening_line_template",
    "tv_closing_line_template",
    "tv_recap_prev_episode",
)


def load_film_tv_rules_markdown(filename: str) -> str:
    """读取 rules/*.md 作为 style_directive 正文。"""
    path = os.path.join(_RULES_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _resolve_preset_style_directive(preset: Dict[str, Any]) -> str:
    rules_file = preset.get("rules_markdown_file")
    if rules_file:
        content = load_film_tv_rules_markdown(str(rules_file))
        if content:
            return content
    return str(preset.get("style_directive") or "")

_PRESETS: Dict[str, Dict[str, Any]] = {
    PRESET_BALANCED: {
        "id": PRESET_BALANCED,
        "name": "均衡解说",
        "subtitle": "解说与原声各半 · 新默认",
        "description": (
            "适合大多数剧集：原声保留名场面与爆点台词，解说负责串线、立人、过渡。"
            "相比旧版「原声 80%」，解说明显增多，成片更好懂。"
        ),
        "default_work_name": "",
        "editor_persona": (
            "你是一位**从业二十年的资深影视剪辑师**，擅长「解说牵引 + 原声点睛」的精剪。"
            "你深知：观众需要解说帮他们把复杂剧情串起来，原声只留给最值钱的 moment。"
        ),
        "style_directive": (
            "## 均衡解说剪辑要点\n"
            "- 原声段：对峙、反转、金句、表演高光；每段 ${ost1_duration_min}–${ost1_duration_max} 秒\n"
            "- 解说段：开场钩子、换线过渡、关系梳理、段末悬念；每段 ${narration_chars_min}–${narration_chars_max} 字\n"
            "- 节奏：原声 2–3 段为一组 → 1 段解说过渡 → 再进下一组原声\n"
            "- 禁止用大段原声堆时长；禁止解说复述刚播过的台词"
        ),
        "settings": {
            "target_duration_percent": 28,
            "ost1_duration_min": 6,
            "ost1_duration_max": 12,
            "ost1_duration_long_max": 16,
            "ost1_segment_min": 14,
            "ost1_segment_max": 22,
            "ost0_segment_min": 12,
            "ost0_segment_max": 18,
            "original_audio_percent": 52,
            "narration_percent": 48,
            "allow_consecutive_ost1": True,
            "enforce_narration_after_ost1": True,
            "narration_chars_min": 42,
            "narration_chars_max": 72,
            "opening_chars_max": 95,
        },
    },
    PRESET_ORIGINAL_HEAVY: {
        "id": PRESET_ORIGINAL_HEAVY,
        "name": "原声燃剪",
        "subtitle": "原声为主 · 解说点睛",
        "description": (
            "旧版高燃方案：成片约 80% 原声、20% 解说，适合对白本身极强的片段。"
            "原声段数多、单段偏长，解说仅作短过渡。"
        ),
        "default_work_name": "",
        "editor_persona": (
            "你是一位**专家级影视剪辑师**（10 年+ 精剪经验），精通「原声为主、解说点睛」的高燃精剪风格。"
            "你像院线预告片剪辑师一样选 moment、控节奏：成片以原片对白和名场面为主，解说只做简短串联。"
        ),
        "style_directive": (
            "## 原声燃剪要点\n"
            "- 多安排 OST=1（${ost1_segment_min}–${ost1_segment_max} 段），每段 ${ost1_duration_min}–${ost1_duration_max} 秒\n"
            "- OST=0 仅 ${ost0_segment_min}–${ost0_segment_max} 段，每段 ${narration_chars_min}–${narration_chars_max} 字，点到为止\n"
            "- 允许连续 2–3 段原声后再插一句短解说\n"
            "- 禁止写成「长解说 → 短原声」的解说主导结构"
        ),
        "settings": {
            "target_duration_percent": 25,
            "ost1_duration_min": 8,
            "ost1_duration_max": 15,
            "ost1_duration_long_max": 20,
            "ost1_segment_min": 28,
            "ost1_segment_max": 40,
            "ost0_segment_min": 6,
            "ost0_segment_max": 10,
            "original_audio_percent": 80,
            "narration_percent": 20,
            "allow_consecutive_ost1": True,
            "enforce_narration_after_ost1": True,
            "narration_chars_min": 35,
            "narration_chars_max": 60,
            "opening_chars_max": 80,
        },
    },
    PRESET_FAZU2: {
        "id": PRESET_FAZU2,
        "name": "《罚罪2》悬疑脉络",
        "subtitle": "8 分钟大师版 · 情绪曲线 · 宝子们片头片尾",
        "description": (
            "40 分钟正片 → 约 8 分钟成片：解说 12–16 段（52%），原声 10–14 段（48%）。"
            "规则全文见 rules/fazu2_8min_master.md，含片头片尾「宝子们」固定话术。"
        ),
        "default_work_name": "罚罪2",
        "rules_markdown_file": FAZU2_RULES_MARKDOWN,
        "editor_persona": (
            "你是一位**《罚罪2》影视解说脚本生成大师**（二十年剪辑经验），"
            "严格按「8 分钟版大师规则」输出 JSON：缓起→渐升→高潮→余韵；"
            "解说带情绪钩子与信息密度，原声只留金句/反转/爆点；"
            "第一段 OST=0 必须以「宝子们，我们开始…」开场，"
            "最后一段必须以「好啦宝子们，我们下集再见！」收尾。"
        ),
        "style_directive": "",
        "inline_style_directive": (
            "## 《罚罪2》自定义剪辑要点（非 MD 模式）\n"
            "- 原声 OST=1：${ost1_segment_min}–${ost1_segment_max} 段，每段 ${ost1_duration_min}–${ost1_duration_max} 秒，只留爆点对峙与信息炸弹\n"
            "- 解说 OST=0：${ost0_segment_min}–${ost0_segment_max} 段，每段 ${narration_chars_min}–${narration_chars_max} 字，带情绪钩子\n"
            "- 原声/解说穿插，禁止连续超过 3 段同类型；picture 写画面/神情，禁止复读对白"
        ),
        "settings": {
            "target_duration_percent": 20,
            "ost1_duration_min": 8,
            "ost1_duration_max": 12,
            "ost1_duration_long_max": 12,
            "ost1_segment_min": 10,
            "ost1_segment_max": 14,
            "ost0_segment_min": 12,
            "ost0_segment_max": 16,
            "original_audio_percent": 48,
            "narration_percent": 52,
            "allow_consecutive_ost1": True,
            "enforce_narration_after_ost1": True,
            "narration_chars_min": 48,
            "narration_chars_max": 72,
            "opening_chars_max": 130,
            "content_type": "tv_series",
            "episode_number": 1,
            "tv_opening_line_template": "宝子们，我们开始《{film_name}》第{episode}集啦！",
            "tv_closing_line_template": "好啦宝子们，我们下集再见！",
            "tv_recap_prev_episode": True,
        },
    },
}


def list_film_tv_rules_markdown_files() -> List[Dict[str, str]]:
    """扫描 rules 目录下可用的 Markdown 规则文件。"""
    items: List[Dict[str, str]] = []
    try:
        for name in sorted(os.listdir(_RULES_DIR)):
            if not name.lower().endswith(".md"):
                continue
            linked = find_preset_id_for_rules_file(name)
            preset = _PRESETS.get(linked) if linked else None
            items.append(
                {
                    "filename": name,
                    "label": name[:-3].replace("_", " "),
                    "subtitle": str((preset or {}).get("subtitle") or ""),
                }
            )
    except OSError:
        pass
    return items


def find_preset_id_for_rules_file(filename: str) -> Optional[str]:
    """查找与 Markdown 规则文件绑定的方案 id。"""
    if not filename:
        return None
    for pid, preset in _PRESETS.items():
        if preset.get("rules_markdown_file") == filename:
            return pid
    return None


def list_film_tv_presets() -> List[Dict[str, Any]]:
    """返回所有方案（供 UI 展示）。"""
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "subtitle": p["subtitle"],
            "description": p["description"],
            "default_work_name": (p.get("default_work_name") or "").strip(),
        }
        for p in _PRESETS.values()
    ]


def get_film_tv_preset(
    preset_id: Optional[str],
    *,
    use_rules_markdown: bool = True,
) -> Optional[Dict[str, Any]]:
    pid = preset_id or DEFAULT_PRESET_ID
    preset = deepcopy(_PRESETS.get(pid))
    if not preset:
        return None
    if use_rules_markdown and preset.get("rules_markdown_file"):
        preset["style_directive"] = _resolve_preset_style_directive(preset)
    else:
        preset["style_directive"] = str(
            preset.get("inline_style_directive") or preset.get("style_directive") or ""
        )
    return preset


def get_default_preset_id() -> str:
    return DEFAULT_PRESET_ID


def get_preset_default_work_name(preset_id: Optional[str]) -> str:
    """专题方案绑定的默认作品名（无则返回空字符串）。"""
    preset = get_film_tv_preset(preset_id)
    if not preset:
        return ""
    return str(preset.get("default_work_name") or "").strip()


def apply_preset_to_settings(
    settings: Optional[Dict[str, Any]] = None,
    preset_id: Optional[str] = None,
    *,
    use_rules_markdown: bool = True,
) -> Dict[str, Any]:
    """将方案数值与元数据合并进 settings。"""
    merged = deepcopy(settings) if settings else {}
    preset = get_film_tv_preset(
        preset_id or merged.get("preset_id") or DEFAULT_PRESET_ID,
        use_rules_markdown=use_rules_markdown,
    )
    if not preset:
        preset = get_film_tv_preset(DEFAULT_PRESET_ID, use_rules_markdown=use_rules_markdown)
    assert preset is not None

    merged["preset_id"] = preset["id"]
    merged["preset_name"] = preset["name"]
    merged["editor_persona"] = preset["editor_persona"]
    merged["style_directive"] = preset["style_directive"]
    for key in NUMERIC_SETTING_KEYS:
        if key in preset["settings"]:
            merged[key] = preset["settings"][key]
    for key in PRESET_EXTRA_SETTING_KEYS:
        if key in preset.get("settings", {}):
            merged[key] = preset["settings"][key]
    return merged


def format_style_directive(template: str, prompt_params: Dict[str, str]) -> str:
    """将方案内 ${var} 占位符替换为当前数值参数。"""
    if not template:
        return ""
    result = template
    for key, value in prompt_params.items():
        result = result.replace(f"${{{key}}}", str(value))
    return result
