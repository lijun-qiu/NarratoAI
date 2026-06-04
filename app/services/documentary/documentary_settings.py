#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""逐帧解说（画面解说）规则参数。"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Optional

import toml
from loguru import logger

DOCUMENTARY_DEFAULTS: Dict[str, Any] = {
    # 在爆燃、恐怖、尖叫、激烈冲突等场面自动插入 OST=1 纯原声段
    "enable_original_audio_highlights": True,
    "ost1_duration_min": 4,
    "ost1_duration_max": 12,
    "max_ost1_segments": 6,
    # 非高光片段默认 OST：2=解说+环境原声，0=纯解说无原声
    "default_narration_ost": 2,
    # 解说风格：幽默风趣、动作表情修饰、上下文物料预判、反常理吐槽
    "enable_humor_narration": True,
    "context_window_sec": 30,
    "enable_action_expression_modifiers": True,
    "enable_logic_roast": True,
    # 全片覆盖与解说字数
    "enable_full_timeline_coverage": True,
    "coverage_interval_sec": 30,
    "narration_chars_min": 20,
    "narration_chars_max": 40,
    "default_custom_prompt": (
        "尽量覆盖全片时间线，每 30 秒至少一段解说，不要大段跳过。"
    ),
}


def _config_file_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(root, "config.toml")


def _read_documentary_config_section() -> Dict[str, Any]:
    try:
        from app.config.config import _cfg
        section = _cfg.get("documentary", {})
    except Exception:
        try:
            section = toml.load(_config_file_path()).get("documentary", {})
        except Exception:
            section = {}
    return dict(section) if isinstance(section, dict) else {}


def get_documentary_settings(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    settings = deepcopy(DOCUMENTARY_DEFAULTS)
    for key, value in _read_documentary_config_section().items():
        if key in DOCUMENTARY_DEFAULTS and value is not None:
            settings[key] = value
    if overrides:
        for key, value in overrides.items():
            if key in DOCUMENTARY_DEFAULTS and value is not None:
                settings[key] = value
    return settings


def build_coverage_instructions(settings: Optional[Dict[str, Any]] = None) -> str:
    """全片时间线覆盖与解说密度规则。"""
    cfg = get_documentary_settings(settings)
    if not cfg.get("enable_full_timeline_coverage", True):
        return ""

    interval = int(cfg.get("coverage_interval_sec", 30))
    chars_min = int(cfg.get("narration_chars_min", 20))
    chars_max = int(cfg.get("narration_chars_max", 40))

    return f"""## 全片覆盖（必须遵守）

- **尽量覆盖全片时间线**，从开头到结尾连贯推进，**不要大段跳过**未解说的空白区间
- 原片时间轴上**每 {interval} 秒至少 1 段**解说（OST=2 或 OST=1），`_id` 按时间顺序递增
- 时间戳必须落在 `<video_frame_description>` 已有范围内，**严禁重叠**，后段开始 ≥ 前段结束
- 长视频按原片时长估算：`items` 数量 ≈ 原片秒数 ÷ {interval}（40 分钟约 **80 段以上**），不得因篇幅偷懒而合并成少量片段
- 解说段（OST=0/2）的 `narration` 每段 **{chars_min}–{chars_max} 字**（OST=1 原声段除外）
- 允许同一批次拆成多段解说，但不得跳过整段批次未覆盖的时间范围
"""


def resolve_documentary_custom_prompt(
    user_prompt: str,
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """合并 config 默认补充提示与用户自定义提示。"""
    cfg = get_documentary_settings(settings)
    default_prompt = str(cfg.get("default_custom_prompt") or "").strip()
    user_text = (user_prompt or "").strip()
    if default_prompt and user_text:
        if default_prompt in user_text:
            return user_text
        return f"{default_prompt}\n{user_text}"
    return user_text or default_prompt


def build_ost_instructions(settings: Optional[Dict[str, Any]] = None) -> str:
    """生成写入解说提示词的 OST 规则说明。"""
    cfg = get_documentary_settings(settings)
    if not cfg.get("enable_original_audio_highlights", True):
        return (
            "## 音频模式\n"
            "- 所有片段统一使用 `\"OST\": 2`（解说配音 + 保留环境原声）\n"
            "- 不要使用 OST=0 或 OST=1\n"
        )

    ost_min = int(cfg.get("ost1_duration_min", 4))
    ost_max = int(cfg.get("ost1_duration_max", 12))
    max_seg = int(cfg.get("max_ost1_segments", 6))
    default_ost = int(cfg.get("default_narration_ost", 2))

    return f"""## 音频模式（必须遵守）

成片按 `_id` 顺序播放。每个片段必须包含整数 `OST` 字段：

| OST | 含义 | 何时使用 |
|-----|------|----------|
| **1** | **纯原声**，无 AI 解说 | 爆燃、爆炸、追逐、尖叫、恐怖 jump scare、激烈打斗、经典原声台词、音乐/音效高潮 |
| **{default_ost}** | 解说 + 环境原声 | 普通叙述、铺垫、过渡（默认） |
| **0** | 纯解说，去掉原声 | 极少使用，仅当原声严重干扰时使用 |

### 原声高光段（OST=1）规则
- 全片 **最多 {max_seg} 段** OST=1，只选最冲击的 moment，不要滥用
- 每段时间戳跨度 **{ost_min}–{ost_max} 秒**，必须完整框住高潮画面/音效
- `narration` 固定写 `播放原片` + 序号（如 `播放原片1`），不要写解说词
- `picture` 写 12 字以内的画面/氛围描述（会显示为旁白字幕）
- **禁止**在 OST=1 播放期间安排解说；一组连续 OST=1 播完后，再用 OST={default_ost} 过渡

### 推荐节奏
```
OST={default_ost} 解说铺垫 → OST=1 原声高潮 → OST={default_ost} 点评 → OST=1 原声 → …
```

### 输出 JSON 示例
```json
{{
  "items": [
    {{
      "_id": 1,
      "timestamp": "00:00:00,000-00:00:08,000",
      "picture": "主角踏入荒原",
      "narration": "谁能想到，这片看似平静的土地，藏着致命危机。",
      "OST": {default_ost}
    }},
    {{
      "_id": 2,
      "timestamp": "00:00:18,000-00:00:26,000",
      "picture": "爆炸火光冲天",
      "narration": "播放原片1",
      "OST": 1
    }}
  ]
}}
```
"""


def build_frame_highlight_hint(settings: Optional[Dict[str, Any]] = None) -> str:
    """视觉分析阶段：标记高能量场面，供后续脚本选用 OST=1。"""
    cfg = get_documentary_settings(settings)
    hints: list[str] = []
    if cfg.get("enable_original_audio_highlights", True):
        hints.append(
            "若某帧出现爆炸、追逐、尖叫、恐怖、激烈冲突、名场面台词或音效高潮，"
            "请在 observation 末尾标注 `[高光原声]`。"
        )
    if cfg.get("enable_action_expression_modifiers", True):
        hints.append(
            "描述每一帧时，务必写出人物表情（如懵圈、瞳孔地震、强装镇定）"
            "和肢体动作修饰（如慢半拍回头、教科书式送人头、走位像开了导航）。"
        )
    if cfg.get("enable_logic_roast", True):
        hints.append(
            "若人物行为明显违背常理（明明能躲却不躲、明知有坑还踩、"
            "反常识操作），请在 observation 末尾标注 `[可吐槽]` 并简述槽点。"
        )
    return " ".join(hints)


def build_narration_style_instructions(settings: Optional[Dict[str, Any]] = None) -> str:
    """生成写入解说提示词的幽默/预判/吐槽风格规则。"""
    cfg = get_documentary_settings(settings)
    if not cfg.get("enable_humor_narration", True):
        return ""

    window = int(cfg.get("context_window_sec", 30))
    chars_min = int(cfg.get("narration_chars_min", 20))
    chars_max = int(cfg.get("narration_chars_max", 40))
    lines = [
        "## 解说风格（必须遵守）",
        "",
        f"### 上下文预判（前后约 {window} 秒）",
        f"- 写每一段解说前，纵览该片段**前后约 {window} 秒**内的帧描述与批次摘要",
        "- 对即将发生的行为、冲突、反转做**提前铺垫**（「接下来这位就要……」「三秒后全场沉默」）",
        "- 铺垫必须基于已有画面分析，严禁编造未出现的剧情",
        "",
        "### 语气：幽默风趣、像损友陪看",
        "- 口语化、有节奏，**鼓励多用网络梗、热词和流行句式**（如「主打一个」「蚌埠住了」「这很难评」），让解说有网感",
        "- 全片可穿插 5–10 处自然玩梗，与画面/行为贴合，避免生硬堆砌同一句式",
        f"- 整段仍保持 **{chars_min}–{chars_max} 字/句** 为主，梗要服务于笑点，不能牺牲信息清晰度",
        "- 可夸张修辞、密集玩梗，但不得歪曲画面事实",
    ]

    if cfg.get("enable_action_expression_modifiers", True):
        lines.extend(
            [
                "",
                "### 动作与表情修饰",
                "- 解说里带上可见的动作、表情细节，用生动修饰词增强画面感",
                "- 示例：「一脸『这题我会』的自信」「走位丝滑，就是方向反了」「表情管理当场崩盘」",
                "- `picture` 字段也可写入简短的动作/表情关键词（供字幕展示）",
            ]
        )

    if cfg.get("enable_logic_roast", True):
        lines.extend(
            [
                "",
                "### 反常理剧情 · 适度吐槽",
                "- 对标注 `[可吐槽]` 或明显不合逻辑的行为，用**一两句**幽默吐槽点破",
                "- 吐槽角度示例（择一，勿全用）：",
                "  - 「导演不让躲，剧本写死了」",
                "  - 「人家有自己的想法，主打一个头铁」",
                "  - 「这操作，观众都替他着急」",
                "  - 「明明有 safer 选项，他偏要选节目效果」",
                "- 吐槽要尖而不毒，全片 2–4 处即可，不要段段都在骂",
                "- 仍须尊重画面：只吐槽**画面已呈现**的迷惑行为，不虚构情节",
            ]
        )

    lines.extend(
        [
            "",
            "### OST=1 原声段",
            "- 进入纯原声高光前，上一段可用一句短铺垫或吐槽收尾；原声段本身不写解说词",
        ]
    )

    return "\n".join(lines)
