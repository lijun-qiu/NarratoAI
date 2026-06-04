#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""逐帧解说（画面解说）规则参数。"""

from __future__ import annotations

import math
import os
from copy import deepcopy
from typing import Any, Dict, Optional

import toml
from loguru import logger

DOCUMENTARY_DEFAULTS: Dict[str, Any] = {
    # 在爆燃、恐怖、尖叫、激烈冲突等场面自动插入 OST=1 纯原声段
    "enable_original_audio_highlights": True,
    "ost1_duration_min": 3,
    "ost1_duration_max": 12,
    "ost1_every_n_segments": 10,
    "max_ost1_segments": 0,
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
    # 字幕 × 抽帧：有 SRT 时与画面分析交叉验证
    "enable_subtitle_enrichment": True,
    "subtitle_max_chars": 15000,
    "subtitle_analysis_max_frame_chars": 8000,
    "subtitle_batch_pad_sec": 5,
    # 逐帧解说/精剪默认抽帧间隔（秒）
    "frame_interval_input": 3,
    # 解说生成：超长输入自动分块（避免超出文本模型上下文）
    "narration_input_max_chars": 65000,
    "narration_input_max_tokens": 85000,
    "narration_compact_markdown_chars": 120000,
    "narration_chunk_max_chars": 50000,
    "narration_chunk_max_sections": 12,
    # 单次 LLM 调用可靠输出的 items 上限，超过则强制分块生成
    "narration_chunk_max_items_per_call": 20,
    # 逐帧解说/精剪脚本生成：输出 token 上限与温度（分块时每块一次调用）
    "narration_script_max_tokens": 16000,
    "narration_script_temperature": 0.4,
}

# 逐帧精剪：故事讲述型（35–45 段，原声 ≤6，解说 30–100 字/段，台词融入叙述）
DOCUMENTARY_COMPACT_OVERRIDES: Dict[str, Any] = {
    "documentary_compact_mode": True,
    "documentary_compact_style": "fazu2",
    "default_narration_ost": 0,
    "enable_full_timeline_coverage": False,
    "coverage_interval_sec": 30,
    "target_output_ratio": 0.3,
    "target_output_minutes": 12,
    "frame_interval_input": 3,
    "min_total_segments": 35,
    "max_total_segments": 45,
    "ost0_segment_min": 30,
    "enable_picture_narration": False,
    "enable_humor_narration": False,
    "enable_logic_roast": False,
    "context_window_sec": 45,
    "narration_chars_min": 30,
    "narration_chars_max": 100,
    "ost1_duration_min": 3,
    "ost1_duration_max": 8,
    "ost1_duration_hard_max": 10,
    "max_ost1_segments": 6,
    "ost1_every_n_segments": 999,
    "fazu2_core_theme": "",
    "default_custom_prompt": (
        "故事讲述型精剪：像说书人一样讲清剧情，把对白自然写进解说；"
        "35–45 段，原声≤6 段；必须使用具体人名，禁止警员1/警员2/说话人1；"
        "禁止分析镜头/导演手法；禁止然后/接着/我们可以看到。"
    ),
}

DOCUMENTARY_SETTING_KEYS = frozenset(
    set(DOCUMENTARY_DEFAULTS)
    | set(DOCUMENTARY_COMPACT_OVERRIDES)
    | {
        "enable_subtitle_enrichment",
        "subtitle_max_chars",
        "subtitle_analysis_max_frame_chars",
        "subtitle_batch_pad_sec",
        "frame_interval_input",
        "min_total_segments",
        "max_total_segments",
        "enable_picture_narration",
        "narration_input_max_chars",
        "narration_input_max_tokens",
        "narration_compact_markdown_chars",
        "narration_chunk_max_chars",
        "narration_chunk_max_sections",
        "narration_chunk_max_items_per_call",
        "narration_script_max_tokens",
        "narration_script_temperature",
        "ost1_every_n_segments",
        "documentary_compact_mode",
        "documentary_compact_style",
        "fazu2_core_theme",
        "coverage_interval_sec",
        "target_output_ratio",
        "target_output_minutes",
        "ost0_segment_min",
        "ost1_duration_hard_max",
    }
)


def is_fazu2_compact_settings(settings: Optional[Dict[str, Any]] = None) -> bool:
    cfg = settings or get_documentary_compact_settings()
    return is_compact_documentary_settings(cfg) and str(
        cfg.get("documentary_compact_style") or "fazu2"
    ).lower() in (
        "fazu2",
        "罚罪2",
        "film_tv_highlight",
        "story_telling",
        "story",
        "故事讲述",
    )


FAZU2_FORBIDDEN_NARRATION_PHRASES: tuple[str, ...] = (
    "然后，",
    "接着，",
    "接下来，",
    "我们可以看到",
    "这里讲的是",
    "镜头语言",
    "导演手法",
    "社会隐喻",
    "权力关系",
    "不禁猜测",
    "暗藏波澜",
    "让人窒息",
    "仿佛在说",
    "警员1",
    "警员2",
    "警员3",
    "说话人1",
    "说话人2",
)


def resolve_fazu2_core_theme(
    video_theme: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """罚罪2 精剪核心主题：config > 视频主题 > 空（由模型自拟）。"""
    cfg = settings or get_documentary_compact_settings()
    for candidate in (
        str(cfg.get("fazu2_core_theme") or "").strip(),
        (video_theme or "").strip(),
    ):
        if candidate:
            return candidate
    return ""


def build_fazu2_narration_copy_hard_requirements(
    core_theme: str = "",
) -> str:
    """逐帧精剪：故事讲述型脚本规则（写入脚本生成提示）。"""
    return build_compact_story_script_rules(core_theme)


def build_compact_story_script_rules(core_theme: str = "") -> str:
    """逐帧精剪 · 故事讲述型：角色设定与创作规则。"""
    work_hint = (core_theme or "").strip() or "（根据字幕与视频主题填写剧名/集数）"
    return f"""## 逐帧精剪 · 故事讲述型脚本规则（优先级最高）

### 角色设定
你是一位**资深影视解说文案撰写人**，擅长把电视剧剧情转成流畅、有情绪、易传播的**故事讲述型**短视频脚本。
受众是普通观众：他们要快速看懂**剧情脉络、人物冲突、情感高潮**，**不要**分析镜头语言、导演手法或社会隐喻。

### 任务目标
根据 `<subtitles>` / 抽帧画面与**精确字幕时间戳**，生成解说脚本。作品：**{work_hint}**

### 输出格式（必须严格遵守）
只输出 JSON：`{{"items":[{{"_id", "timestamp", "picture", "narration", "OST"}}]}}`

### 1. 整体风格
- **故事讲述型**：像说书人从头到尾讲清剧情，有细节、情绪、转折
- **贴合时间轴**：`timestamp` **必须从原字幕逐字复制**（`HH:MM:SS,mmm-HH:MM:SS,mmm`），禁止编造整分等间隔
- **语言通俗**：短句、口语（如「结果到了地方，根本不是狗贩子」）
- **情绪递进**：紧张/愤怒/悲伤/希望；可用小钩子（如「可谁也没想到……」）

### 2. 人物称呼（硬性）
- `narration` 与 `picture` 必须使用**具体人名**（如胡小月、秦枫、伟业、罗博、马金），与字幕/剧情一致
- **禁止**编号式称呼：❌ 警员1/警员2、说话人1、男子1、女子A、黑衣人1 等
- 字幕只有职务时：用剧情已出现的**姓名**；首次可写「胡小月的师弟秦枫」等，**禁止**用匿名编号代替具名角色
- 全片人名前后统一，不要同一人又叫「年轻警员」又叫「警员2」

### 3. 解说段 OST=0（默认，全片 ≥30 段）
- 每段 **30–100 字**，朗读约 4–15 秒；**一段一个情节点**，不要塞太多信息
- **把人物对白融入解说**：用「秦枫开口就说」「胡小月突然吼了出来」等转述，对白用引号嵌入正文
- **禁止流水账词**：❌ 然后、接着、接下来、我们可以看到、这里讲的是
- **禁止拉片腔**：❌ 导演、镜头语言、社会隐喻、权力博弈、试探底线（改写成讲故事）
- 段与段**自然过渡**（例：「可谁也没想到，真正的麻烦还在后面。」）

### 4. 原声段 OST=1（全片 ≤6 段，通常 3–5 段）
- **仅**标志性金句/情绪爆点/绝望叹息（如「天就快亮了」「有意思」）
- 每段 **3–8 秒**（最长 ≤10 秒），`timestamp` **精确取自字幕**
- `narration` **只写这一句台词**（引号包裹）；前面须有 OST=0 铺垫
- **严禁**相邻两段 OST=1；**禁止**「播放原片N」

### 5. picture（10–20 字）
- **写出人物姓名** + 动作 + 场景（例：「秦枫冷冷看着罗博，灵堂内气氛压抑」）
- ❌ 「警员1站在门口」→ ✅ 「秦枫站在灵堂门口」

### 6. 时间戳与 `_id`
- `_id` = **成片播放顺序**（1→2→3…），按**剧情推进**编排，不必按原片时间先后
- 解说段时长 ≈ 字数÷10×1.5 秒；段与段可略有重叠，**禁止大段空白跳跃**

### 7. 全片指标（约 40 分钟原片 → 12–15 分钟成片）
| 指标 | 要求 |
|------|------|
| 总段数 | **35–45 段** |
| 解说 OST=0 | **≥30 段** |
| 原声 OST=1 | **≤6 段**，原声总时长 **≤50 秒** |
| 解说总时长 | 约 11–14 分钟 |
"""


# 故事讲述型：标准输出参考（《罚罪2》第一集节选）
FAZU2_SCRIPT_REFERENCE_ITEMS_JSON = """[
  {
    "_id": 1,
    "timestamp": "00:00:01,940-00:00:12,740",
    "picture": "办公室内，领导与伟业对坐，气氛凝重",
    "narration": "《罚罪2》开场，省厅领导把伟业叫到办公室。领导开口就说：你都到厅级了，干嘛非要回去当局长？想清楚了？伟业没有接话，他只说了一句：胡晓月是我的徒弟。",
    "OST": 0
  },
  {
    "_id": 2,
    "timestamp": "00:00:12,820-00:00:22,500",
    "picture": "领导眉头紧锁，语气沉重",
    "narration": "领导叹了口气，说我知道，但她确实是死于自杀，还有一些关于举报她的材料也正在核实。伟业当场就反驳了。",
    "OST": 0
  },
  {
    "_id": 33,
    "timestamp": "00:20:05,120-00:20:06,240",
    "picture": "胡小月抬头看向夜空",
    "narration": "「天就快亮了。」",
    "OST": 1
  },
  {
    "_id": 37,
    "timestamp": "00:28:22,530-00:28:23,490",
    "picture": "罗博咬碎后槽牙，冷笑",
    "narration": "罗博咬着牙挤出两个字：「有意思。」",
    "OST": 1
  }
]"""


def build_fazu2_script_output_reference() -> str:
    """故事讲述型 JSON 参考模板。"""
    return f"""## 输出 JSON 参考模板（故事讲述型 · 必须严格仿照）

```json
{{"items": {FAZU2_SCRIPT_REFERENCE_ITEMS_JSON}}}
```

{build_fazu2_generation_anti_patterns()}

### 字段要点
| 字段 | OST=0 | OST=1 |
|------|-------|-------|
| `narration` | 30–100 字，**讲故事+嵌入对白**，禁止拉片分析 | **仅一句**金句台词（引号） |
| `timestamp` | 字幕/画面真实毫秒范围 | 字幕对白**精确**起止 |
| `picture` | 10–20 字，人物+动作+场景 | 说话人画面 |
| `OST` | 0（默认） | 1（≤6 段/全片） |
"""


def build_fazu2_generation_anti_patterns() -> str:
    """故事讲述型：错误 vs 正确对照。"""
    return """## 错误示范 vs 正确示范

### ❌ 禁止
- 匿名编号人物：「警员1说」「警员2追问」「说话人1开口」
- 拉片分析：「注意领导的表情——他在试探」「导演用特写捕捉心理崩塌」
- 编造时间戳：`00:06:00,000-00:06:12,000` 等整分铺段
- 金句标 OST=0 却只剩一句裸台词（应 OST=1 或写入前后解说正文）
- 流水账：然后、接着、接下来、我们可以看到

### ✅ 必须
- 具名人名：「秦枫逼问」「胡小月吼了出来」「罗博冷笑」
- OST=0 示例：「领导开口就说：你都到厅级了……伟业没有接话，他只说了一句：胡晓月是我的徒弟。」
- OST=1 示例：仅「天就快亮了。」等 3–8 秒金句，前后有解说铺垫
- 时间戳：`00:00:01,940-00:00:12,740` 这类字幕原样复制
- `_id` 按剧情顺序 1→2→3… 讲完整集故事线
"""


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


def _read_documentary_compact_config_section() -> Dict[str, Any]:
    try:
        from app.config.config import _cfg
        section = _cfg.get("documentary_compact", {})
    except Exception:
        try:
            section = toml.load(_config_file_path()).get("documentary_compact", {})
        except Exception:
            section = {}
    return dict(section) if isinstance(section, dict) else {}


def get_documentary_settings(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    settings = deepcopy(DOCUMENTARY_DEFAULTS)
    for key, value in _read_documentary_config_section().items():
        if key in DOCUMENTARY_SETTING_KEYS and value is not None:
            settings[key] = value
    if overrides:
        for key, value in overrides.items():
            if key in DOCUMENTARY_SETTING_KEYS and value is not None:
                settings[key] = value
    return settings


def get_narration_script_llm_params(
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """逐帧脚本生成 LLM 参数（分块/补段共用）。"""
    cfg = get_documentary_settings(settings)
    max_tokens = int(cfg.get("narration_script_max_tokens", 0) or 0)
    temperature = float(cfg.get("narration_script_temperature", 0) or 0)
    if max_tokens <= 0:
        try:
            from app.config import config

            max_tokens = int(config.app.get("narration_script_max_tokens", 16000) or 16000)
        except Exception:
            max_tokens = 16000
    if temperature <= 0:
        try:
            from app.config import config

            temperature = float(config.app.get("narration_script_temperature", 0.4) or 0.4)
        except Exception:
            temperature = 0.4
    temperature = max(0.0, min(1.0, temperature))
    return {"max_tokens": max_tokens, "temperature": temperature}


def get_documentary_compact_settings(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """逐帧精剪：默认故事讲述型（35–45 段，原声 ≤6，解说 30–100 字/段）。"""
    settings = get_documentary_settings()
    for key, value in DOCUMENTARY_COMPACT_OVERRIDES.items():
        settings[key] = value
    for key, value in _read_documentary_compact_config_section().items():
        if key in DOCUMENTARY_SETTING_KEYS and value is not None:
            settings[key] = value
    if overrides:
        for key, value in overrides.items():
            if key in DOCUMENTARY_SETTING_KEYS and value is not None:
                settings[key] = value
    return settings


def is_compact_documentary_settings(settings: Optional[Dict[str, Any]] = None) -> bool:
    cfg = settings or get_documentary_settings()
    if cfg.get("documentary_compact_mode"):
        return True
    # 兼容旧配置：未显式标记时，仍以「非全片覆盖」视为精剪
    return not cfg.get("enable_full_timeline_coverage", True)


def compute_compact_segment_bounds(
    settings: Optional[Dict[str, Any]] = None,
    source_duration_sec: Optional[float] = None,
) -> tuple[int, int, int]:
    """返回精剪模式 (最少段数, 目标段数, 上限段数)。"""
    cfg = settings or get_documentary_compact_settings()
    min_floor = max(5, int(cfg.get("min_total_segments", 5)))
    max_cap_cfg = int(cfg.get("max_total_segments", 0) or 0)

    if cfg.get("enable_full_timeline_coverage", True):
        interval = max(1, int(cfg.get("coverage_interval_sec", 30)))
        if source_duration_sec and source_duration_sec > 0:
            coverage_min = max(min_floor, int(math.ceil(source_duration_sec / interval)))
        else:
            coverage_min = min_floor
        target = coverage_min
        if max_cap_cfg > 0:
            max_cap = max(coverage_min, max_cap_cfg)
        else:
            slack = max(8, int(round(coverage_min * 0.12)))
            max_cap = coverage_min + slack
        return coverage_min, target, max_cap

    if max_cap_cfg > 0:
        max_cap = max(min_floor, max_cap_cfg)
    else:
        max_cap = max(min_floor, 45)
    target = max(min_floor, min(max_cap, (min_floor + max_cap) // 2))

    if is_fazu2_compact_settings(cfg):
        return min_floor, target, max_cap

    ratio = float(cfg.get("target_output_ratio", 0.3))
    chars_min = int(cfg.get("narration_chars_min", 20))
    chars_max = int(cfg.get("narration_chars_max", 40))
    avg_chars = (chars_min + chars_max) / 2
    avg_tts_sec = max(1.0, avg_chars / 4.0)

    if source_duration_sec and source_duration_sec > 0:
        target_output_sec = source_duration_sec * ratio
        estimated = int(round(target_output_sec / avg_tts_sec))
        target = max(min_floor, min(max_cap, estimated))

    return min_floor, target, max_cap


def compute_max_ost1_segments(
    total_items: int,
    settings: Optional[Dict[str, Any]] = None,
) -> int:
    """按「约每 N 段 1 原声」估算 OST=1 上限；max_ost1_segments>0 时为硬封顶。"""
    cfg = settings or get_documentary_settings()
    every_n = max(1, int(cfg.get("ost1_every_n_segments", 10) or 10))
    hard_cap = int(cfg.get("max_ost1_segments", 0) or 0)
    auto_max = max(1, round(total_items / every_n)) if total_items > 0 else 1
    if hard_cap > 0:
        return min(auto_max, hard_cap)
    return auto_max


def format_ost1_segment_hint(
    settings: Optional[Dict[str, Any]] = None,
    *,
    estimated_items: Optional[int] = None,
) -> str:
    cfg = settings or get_documentary_settings()
    if not cfg.get("enable_original_audio_highlights", True):
        return ""
    ost_min = int(cfg.get("ost1_duration_min", 3))
    ost_max = int(cfg.get("ost1_duration_max", 12))
    every_n = max(1, int(cfg.get("ost1_every_n_segments", 10) or 10))
    if estimated_items and estimated_items > 0:
        target = compute_max_ost1_segments(estimated_items, cfg)
        count_hint = (
            f"本脚本约 **{target} 段** OST=1"
            f"（总量 {estimated_items} 段、约每 {every_n} 段 1 原声）"
        )
    else:
        count_hint = f"全片 **约每 {every_n} 段 items 插入 1 段** OST=1"
    if is_fazu2_compact_settings(cfg):
        max_ost1 = int(cfg.get("max_ost1_segments", 6) or 6)
        return (
            f"- **原声点睛**：全片 OST=1 **≤{max_ost1} 段**（通常 3–5 段），每段 **{ost_min}–{ost_max} 秒**，总时长 **≤50 秒**\n"
            f"- 仅用于标志性金句/情绪爆点；**前面至少 1 段 OST=0 铺垫**；**严禁相邻原声**\n"
            f"- OST=1：`narration` **只写该句台词**（引号）；其余对白**写入 OST=0 解说正文**\n"
        )
    if is_compact_documentary_settings(cfg):
        return (
            f"- OST=1 用于金句/爆点；{count_hint}，每段 **{ost_min}–{ost_max} 秒**\n"
        )
    return (
        f"- 标记 `[高光原声]` 的爆燃/恐怖/尖叫等场面，可设 **OST=1** 纯原声"
        f"（{count_hint}，每段 **{ost_min}–{ost_max} 秒**）\n"
    )


def build_compact_coverage_instructions(
    settings: Optional[Dict[str, Any]] = None,
    source_duration_sec: Optional[float] = None,
) -> str:
    """精剪模式覆盖规则（全片时间线或按成片比例精选）。"""
    cfg = settings or get_documentary_compact_settings()
    chars_min = int(cfg.get("narration_chars_min", 20))
    chars_max = int(cfg.get("narration_chars_max", 40))
    avg_chars = (chars_min + chars_max) / 2
    chars_per_sec = 4.0
    avg_tts_sec = max(1.0, avg_chars / chars_per_sec)
    min_segments, target_segments, max_segments = compute_compact_segment_bounds(
        cfg, source_duration_sec
    )
    ost1_hint = format_ost1_segment_hint(cfg, estimated_items=target_segments)
    interval = max(1, int(cfg.get("coverage_interval_sec", 30)))

    duration_hint = ""
    if source_duration_sec and source_duration_sec > 0:
        source_minutes = source_duration_sec / 60
        est_items = int(math.ceil(source_duration_sec / interval))
        duration_hint = (
            f"- 原片约 **{source_minutes:.1f} 分钟**，按每 {interval} 秒 1 段估算约 **{est_items} 段**以上\n"
        )

    if cfg.get("enable_full_timeline_coverage", True):
        return f"""## 精剪覆盖（必须遵守）

- **尽量覆盖全片时间线**，从开头到结尾连贯推进，**不要大段跳过**未解说的空白区间
- 原片时间轴上**每 {interval} 秒至少 1 段** items（OST=0 或 OST=1），`_id` 按时间顺序递增
{duration_hint}{ost1_hint}- 普通解说段默认 **OST=0**，每段 **{chars_min}–{chars_max} 字**，约 **{avg_tts_sec:.0f} 秒**配音
- **items 总数硬性范围：{min_segments}–{max_segments} 段**（低于 {min_segments} 或超过 {max_segments} 均无效）
- **items 目标数量：约 {target_segments} 段**（长片不得因篇幅偷懒只写十几段）
- 时间戳必须落在 `<video_frame_description>` / 字幕已有范围内，**严禁重叠**，后段开始 ≥ 前段结束
- **精剪≠梗概**：每段须有深度拉片观察，禁止流水账复述；优先华彩镜头，但**不得**为省段数跳过整段 {interval} 秒未覆盖区间
- **声画对位（何止电影）**：解说抛观点后，可紧接 OST=1 切入能印证该观点的原声对白（时间戳以字幕为准）
- **禁止** OST=2；本模式只用 OST=0 与 OST=1
"""

    target_minutes = float(cfg.get("target_output_minutes", 12) or 12)
    ost0_min = int(cfg.get("ost0_segment_min", 30) or 30)
    max_ost1 = int(cfg.get("max_ost1_segments", 6) or 6)

    if is_fazu2_compact_settings(cfg):
        return f"""## 精剪覆盖（必须遵守 · 故事讲述型）

### 风格目标
- 像说书人**从头到尾讲清剧情**，有细节、情绪、转折；**不要**拉片/镜头分析
- **OST=0** 为主体：对白融入叙述；**OST=1** 仅 3–5 个金句爆点

### 全片指标（目标成片约 {target_minutes:.0f} 分钟）
| 指标 | 要求 |
|------|------|
| 总片段数 | **{min_segments}–{max_segments} 段**（目标约 {target_segments}） |
| 解说 OST=0 | **≥{ost0_min} 段**，每段 **{chars_min}–{chars_max} 字** |
| 原声 OST=1 | **≤{max_ost1} 段**，总时长 **≤50 秒** |

{ost1_hint}- 按**剧情顺序**选情节点，允许跳剪；`_id` 为播放顺序；时间戳**必须从字幕复制**
- 遵守「逐帧精剪 · 故事讲述型脚本规则」；**禁止 OST=2**
"""

    ratio = float(cfg.get("target_output_ratio", 0.3))
    ratio_percent = int(ratio * 100)
    target_segments_hint = ""
    if source_duration_sec and source_duration_sec > 0:
        target_minutes_est = source_duration_sec * ratio / 60
        target_segments_hint = (
            f"- 成片播放目标约 **{target_minutes_est:.1f} 分钟**（约原片 {ratio_percent}%）\n"
        )

    return f"""## 精剪覆盖（必须遵守）

- **成片总时长约为原片的 {ratio_percent}%**
{target_segments_hint}{ost1_hint}- 解说段 **OST=0**，每段 **{chars_min}–{chars_max} 字**
- **items 总数：{min_segments}–{max_segments} 段**；**禁止** OST=2
"""


def build_coverage_instructions(settings: Optional[Dict[str, Any]] = None) -> str:
    """全片时间线覆盖与解说密度规则。"""
    cfg = settings or get_documentary_settings()
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


def build_compact_ost_instructions(settings: Optional[Dict[str, Any]] = None) -> str:
    """精剪模式 OST 规则。"""
    cfg = settings or get_documentary_compact_settings()
    if not cfg.get("enable_original_audio_highlights", True):
        return (
            "## 音频模式（必须遵守）\n"
            "- 所有片段统一使用 `\"OST\": 0`（纯 AI 解说）\n"
            "- **禁止** OST=1 或 OST=2\n"
        )

    ost_min = int(cfg.get("ost1_duration_min", 3))
    ost_max = int(cfg.get("ost1_duration_max", 8))
    max_ost1 = int(cfg.get("max_ost1_segments", 6) or 6)
    chars_min = int(cfg.get("narration_chars_min", 80))
    chars_max = int(cfg.get("narration_chars_max", 150))

    if is_fazu2_compact_settings(cfg):
        return f"""## 音频模式（必须遵守 · 故事讲述型）

| OST | 含义 |
|-----|------|
| **0** | 解说讲故事（TTS），**对白写在 narration 里** |
| **1** | 纯原声金句（极少），无 TTS |
| **2** | **禁止** |

### OST=0（全片 ≥30 段，每段 {chars_min}–{chars_max} 字）
- 用「谁对谁说 / 谁喊了一句」把对白自然写进解说；**不要**拆成大量裸台词段
- 一段一事；段末可留情绪或过渡（如「可谁也没想到……」）
- `timestamp` 取自字幕，时长 ≈ 字数÷10×1.5 秒

### OST=1（全片 ≤{max_ost1} 段，每段 {ost_min}–{ost_max} 秒）
- 仅标志性台词（如「天就快亮了」）；`narration` **只写该句**；`timestamp` 精确取自字幕
- 前面至少 1 段 OST=0 铺垫；**禁止相邻 OST=1**

{build_fazu2_script_output_reference()}
只输出 `{{"items":[...]}}`，不要 markdown 代码块包裹。
"""

    picture_line = "- `picture` 写画面/人物备注"
    return f"""## 音频模式（必须遵守）

| OST | 含义 |
|-----|------|
| **0** | 纯 AI 解说（默认） |
| **1** | 纯原声，≤{max_ost1} 段，{ost_min}–{ost_max} 秒 |
| **2** | **禁止** |

{picture_line}
"""


def build_ost_instructions(settings: Optional[Dict[str, Any]] = None) -> str:
    """生成写入解说提示词的 OST 规则说明。"""
    cfg = settings or get_documentary_settings()
    if is_compact_documentary_settings(cfg):
        return build_compact_ost_instructions(cfg)
    default_ost = int(cfg.get("default_narration_ost", 2))
    if default_ost not in (0, 2):
        default_ost = 2

    if not cfg.get("enable_original_audio_highlights", True):
        if default_ost == 0:
            return (
                "## 音频模式（必须遵守）\n"
                "- 所有片段统一使用 `\"OST\": 0`（纯 AI 解说，**去掉原片声音**）\n"
                "- **禁止**使用 OST=1 或 OST=2\n"
                "- 每段 `narration` 写解说词，不要写「播放原片」\n"
            )
        return (
            "## 音频模式\n"
            "- 所有片段统一使用 `\"OST\": 2`（解说配音 + 保留环境原声）\n"
            "- 不要使用 OST=0 或 OST=1\n"
        )

    ost_min = int(cfg.get("ost1_duration_min", 3))
    ost_max = int(cfg.get("ost1_duration_max", 12))
    every_n = max(1, int(cfg.get("ost1_every_n_segments", 10) or 10))
    if cfg.get("enable_picture_narration", True):
        picture_line = "- `picture` 写 12 字以内的画面/氛围描述（会显示为旁白字幕）"
    else:
        picture_line = "- `picture` 可写简短画面备注（**成片原声段不显示旁白字幕**）"

    if default_ost == 0:
        narration_ost_row = (
            f"| **0** | **纯 AI 解说**，无原片声音 | 普通叙述、铺垫、过渡（**默认**） |\n"
            f"| **1** | **纯原声**，无 AI 解说 | 爆燃、爆炸、追逐、尖叫、恐怖 jump scare、激烈打斗、经典原声台词、音乐/音效高潮 |\n"
            f"| **2** | 解说 + 环境原声 | **禁止使用** |"
        )
        ost2_note = "- **禁止**使用 OST=2\n"
    else:
        narration_ost_row = (
            f"| **1** | **纯原声**，无 AI 解说 | 爆燃、爆炸、追逐、尖叫、恐怖 jump scare、激烈打斗、经典原声台词、音乐/音效高潮 |\n"
            f"| **{default_ost}** | 解说 + 环境原声 | 普通叙述、铺垫、过渡（默认） |\n"
            f"| **0** | 纯解说，去掉原声 | 极少使用，仅当原声严重干扰时使用 |"
        )
        ost2_note = ""

    return f"""## 音频模式（必须遵守）

成片按 `_id` 顺序播放。每个片段必须包含整数 `OST` 字段：

| OST | 含义 | 何时使用 |
|-----|------|----------|
{narration_ost_row}

{ost2_note}### 原声高光段（OST=1）规则
- 全片 **约每 {every_n} 段 items 插入 1 段** OST=1（如 100 段约 10 段原声），只选最冲击的 moment
- 每段时间戳跨度 **{ost_min}–{ost_max} 秒**，必须完整框住高潮画面/音效
- `narration` 固定写 `播放原片` + 序号（如 `播放原片1`），不要写解说词
- {picture_line}
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
    cfg = settings or get_documentary_settings()
    hints: list[str] = []
    if is_fazu2_compact_settings(cfg):
        hints.append(
            "画面描述须写出**人物姓名**（从字幕/剧情推断），禁止警员1、说话人2等编号；"
            "人物+动作+场景，10–20 字，供 picture 字段引用。"
        )
        hints.append(
            "若该帧有可作 OST=1 的标志性台词，标注 `[金句原声]` 并摘录对白+字幕时间。"
        )
    elif is_compact_documentary_settings(cfg):
        hints.append(
            "写出景别、运镜、构图与人物动作，供解说引用。"
        )
    if cfg.get("enable_original_audio_highlights", True):
        hints.append(
            "若某帧出现爆炸、追逐、尖叫、恐怖、激烈冲突、名场面台词或音效高潮，"
            "请在 observation 末尾标注 `[高光原声]`。"
        )
    if cfg.get("enable_action_expression_modifiers", True):
        hints.append(
            "描述每一帧时，写出人物表情、肢体动作与场景氛围，"
            "用具体可见的细节帮助后续解说理解画面。"
        )
    if cfg.get("enable_logic_roast", True) and not is_compact_documentary_settings(cfg):
        hints.append(
            "若人物行为明显违背常理或令人费解，"
            "请在 observation 末尾标注 `[可吐槽]` 并简述原因（供解说员适度点评）。"
        )
    return " ".join(hints)


def build_compact_narration_style_instructions(
    settings: Optional[Dict[str, Any]] = None,
    core_theme: str = "",
) -> str:
    """逐帧精剪解说风格。"""
    cfg = settings or get_documentary_compact_settings()
    window = int(cfg.get("context_window_sec", 45))
    chars_min = int(cfg.get("narration_chars_min", 80))
    chars_max = int(cfg.get("narration_chars_max", 150))

    if is_fazu2_compact_settings(cfg):
        theme = resolve_fazu2_core_theme(core_theme, cfg)
        min_segments, _, max_segments = compute_compact_segment_bounds(cfg)
        ost0_min = int(cfg.get("ost0_segment_min", 30) or 30)
        rules_block = build_compact_story_script_rules(theme)
        return f"""{rules_block}

## 解说风格（必须遵守 · 故事讲述型）

- 全片 **{min_segments}–{max_segments} 段**，解说 ≥{ost0_min}，原声 ≤6
- OST=0：每段 **{chars_min}–{chars_max} 字**，讲故事、嵌入对白，禁止拉片分析
- OST=1：仅 3–5 个金句，每句单独成段，前后有解说铺垫
- 纵览前后约 {window} 秒画面与字幕，保持剧情连贯，像给观众**讲一集电视剧**
"""

    return f"""## 解说风格（必须遵守）
- 单段 **{chars_min}–{chars_max} 字**；禁止流水账
"""


def build_narration_style_instructions(
    settings: Optional[Dict[str, Any]] = None,
    core_theme: str = "",
) -> str:
    """生成写入解说提示词的解说风格规则（全片覆盖 vs 精剪拉片）。"""
    cfg = settings or get_documentary_settings()
    if is_compact_documentary_settings(cfg):
        return build_compact_narration_style_instructions(cfg, core_theme=core_theme)

    window = int(cfg.get("context_window_sec", 30))
    chars_min = int(cfg.get("narration_chars_min", 20))
    chars_max = int(cfg.get("narration_chars_max", 40))
    lines = [
        "## 解说风格（必须遵守）",
        "",
        "### 人设：30 年经验的资深解说员",
        "- 像一位看过无数片子的老练讲片人，**亲民、清楚、有分寸**",
        "- 目标是帮观众**看懂画面、跟上剧情**；大白话为主，轻松处可**适当**点缀贴切网络梗",
        "- 语言避免书面语、术语腔和生僻词，别堆梗、别尬玩",
        "",
        "### 灵活应变：随剧情与画面调整讲法",
        "- **不要全片同一种语气**；写每段前先判断当前是铺垫、冲突、反转、情感还是高潮",
        "- 紧张处：句子更短、信息更集中；情感处：放慢、用词走心；日常处：轻松唠几句",
        "- 该点破就点破，该留白就留白，让讲法**服务于画面**",
        "",
        f"### 上下文衔接（前后约 {window} 秒）",
        f"- 写每段前，纵览**前后约 {window} 秒**的帧描述与批次摘要",
        "- 对即将发生的转折做**适度铺垫**，但必须基于已有分析，严禁虚构",
        f"- 单句 **{chars_min}–{chars_max} 字** 为主，长短可随节奏微调",
    ]

    if cfg.get("enable_action_expression_modifiers", True):
        lines.extend(
            [
                "",
                "### 画面细节",
                "- 点出观众能看到的动作、表情、环境变化，增强代入感",
                "- 示例：「他犹豫了一下才推门」「屋里一下子安静了」",
                "- `picture` 字段可写简短的动作/氛围关键词",
            ]
        )

    if cfg.get("enable_humor_narration", True):
        lines.extend(
            [
                "",
                "### 适当网络梗（点缀即可）",
                "- 在轻松、日常、可吐槽的段落，**适当**用贴切网络梗或流行语（如「主打一个」「蚌埠住了」）",
                "- 全片 **3–6 处**为宜，必须贴合画面；情感/严肃/紧张段落**不要用梗**",
            ]
        )

    if cfg.get("enable_logic_roast", True) and cfg.get("enable_humor_narration", True):
        lines.extend(
            [
                "",
                "### 适度点评（可选，勿滥用）",
                "- 对 `[可吐槽]` 或明显费解的行为，**偶尔**用一两句轻松点破，全片 2–4 处",
                "- 示例：「这会儿正常人都会先躲一下」「他偏要硬刚，后面就有得受了」",
                "- 点评是为了帮观众理解，不是段段开玩笑",
            ]
        )

    lines.extend(
        [
            "",
            "### OST=1 原声段",
            "- 进入纯原声前，上一段可用一句短铺垫收束；原声段本身不写解说词",
        ]
    )

    return "\n".join(lines)
