#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""逐帧解说（画面解说）规则参数。"""

from __future__ import annotations

import json
import math
import re
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
    # 对照抽帧校正 ASR 字幕（输出 *_refined.srt）
    "enable_subtitle_refinement": True,
    "subtitle_refinement_max_entries_per_call": 25,
    "subtitle_refinement_temperature": 0.3,
    "subtitle_refinement_min_similarity": 0.5,
    "subtitle_refinement_max_length_ratio_delta": 0.4,
    # 硬字幕 OCR 校准（裁剪关键帧底部，输出 *_ocr_refined.srt）
    "enable_hard_subtitle_ocr": True,
    "auto_subtitle_calibration_on_frame_analysis": False,
    "subtitle_ocr_min_similarity": 0.5,
    "subtitle_ocr_max_length_ratio_delta": 0.35,
    "subtitle_ocr_crop_ratio": 0.22,
    "subtitle_ocr_batch_size": 10,
    "subtitle_ocr_max_concurrency": 2,
    "subtitle_ocr_match_pad_ms": 1500,
    "subtitle_ocr_min_confidence_frames": 1,
    # 逐帧解说/精剪默认抽帧间隔（秒）
    "frame_interval_input": 3,
    # 解说生成：超长输入自动分块（避免超出文本模型上下文）
    "narration_input_max_chars": 65000,
    "narration_input_max_tokens": 85000,
    "narration_compact_markdown_chars": 120000,
    # 抽帧 JSON 保存（v4）：去掉 raw_response / fallback_summary 等调试字段；批次用 frame_files + keyframe_cache_key
    "strip_frame_analysis_debug_fields": True,
    # 同一场景连续片段不重复写入 scene / key_visual / emotion / 观察句
    "dedupe_scene_environment": True,
    # 抽帧 JSON 保存为紧凑单行（无 indent，体积更小）
    "compact_analysis_json": False,
    # 抽帧落盘时是否剥离 frame_observations.observation / batches 调试字段（默认保留完整 JSON）
    "compress_frame_analysis_on_save": False,
    # 抽帧参照图 token 优化：仅首批发送、缩小、多头像拼图
    "frame_reference_token_saver": True,
    "frame_reference_attach_mode": "first_batch",
    "frame_reference_max_edge": 384,
    "frame_reference_use_collage": True,
    "frame_reference_individual_max_heads": 4,
    "frame_reference_collage_max_heads": 4,
    # 抽帧视觉分析：默认不注入剧集人物关系知识库（避免脑补全剧名场面）
    "enable_frame_analysis_drama_knowledge": False,
    # 抽帧 scene_segments 硬性规则：仅可见画面、同批同景合并、跨场景重叠剔除
    "enable_frame_strict_scene_rules": True,
    "frame_cross_scene_overlap_prune_ratio": 0.5,
    "frame_max_segment_duration_sec": 30,
    "narration_chunk_max_chars": 50000,
    "narration_chunk_max_sections": 30,
    # 单次 LLM 调用可靠输出的 items 上限；分块数以段数容量为下限，仅单块超限时才增块
    "narration_chunk_max_items_per_call": 15,
    # 精剪段数校验：首次生成失败后最多重试次数
    "narration_segment_max_retries": 3,
    # 逐帧解说/精剪脚本生成：输出 token 上限与温度
    "narration_script_max_tokens": 16000,
    "narration_script_temperature": 0.4,
    # WebUI「视频主题」默认值
    "default_video_theme": "罚罪2",
    # 叠加在自定义提示词之后的本集/本片专属要求
    "append_custom_prompt": "",
}

FAZU2_WRONG_CHARACTER_NAMES: tuple[tuple[str, str], ...] = (
    ("胡小月", "胡小跃"),
    ("胡晓月", "胡小跃"),
    ("小月", "小跃"),
    ("伟叶", "伟业"),
    ("秦峰", "秦枫"),
    ("罗伯", "罗博"),
)

# 抽帧 characters：无法确认姓名时使用
FRAME_UNKNOWN_CHARACTER_MALE = "未名人员(男)"
FRAME_UNKNOWN_CHARACTER_FEMALE = "未名人员(女)"
FRAME_UNKNOWN_CHARACTER_UNKNOWN = "未名人员(不明)"
FRAME_FACE_MATCH_SIMILARITY_HINT = (
    "须逐脸对照定妆照，五官轮廓高度一致（约90%以上）方可写规范姓名；"
    "未达阈值或仅侧背/模糊/夜景压缩时禁止猜名"
)

# 剧情人物参考（写每段前须与当段抽帧/字幕核对，勿凭印象套性别）
FAZU2_CHARACTER_ROLES: tuple[tuple[str, str, str], ...] = (
    ("胡小跃", "刑警（男）", "字幕/抽帧确认为男性时用「他」，勿写成女性"),
    ("小跃", "刑警（男）", "胡小跃简称，同上"),
    ("伟业", "局长/警察（男）", "专属人名；与之对话的上级不是伟业，勿混称"),
    ("老叶", "长辈/领导（男）", "硬字幕出现「老叶」时可写；与伟业(男)是两人"),
    ("叶天佑", "局长（男）", "正式姓名；硬字幕有时标「老叶」或「叶天佑」"),
    ("秦枫", "刑警（男）", "二师兄，禁止秦峰"),
    ("罗博", "反派", "禁止罗伯"),
    ("常征", "警察", "—"),
    ("赵鹏超", "—", "—"),
    ("马金", "—", "—"),
)

FAZU2_DEFAULT_OPENING_CLIMAX_HINT = (
    "胡小跃楼顶跳楼牺牲名场面（夜色楼顶纵身跃下），"
    "原声金句优先「天就快亮了。」（时间戳从字幕原样复制）。"
    "禁止用中段抓捕/争吵（如「你跟我说这是狗贩子」）顶替第 1 段；此类放正叙 OST=1。"
)

# 逐帧精剪：罚罪2 高潮前置版 V2（30–55 段，原声/解说约 1:1 穿插，解说 30–100 字/段）
DOCUMENTARY_COMPACT_OVERRIDES: Dict[str, Any] = {
    "documentary_compact_mode": True,
    "documentary_compact_style": "fazu2",
    "default_narration_ost": 0,
    "enable_full_timeline_coverage": False,
    "coverage_interval_sec": 30,
    "target_output_ratio": 0.3,
    "target_output_minutes": 12,
    "frame_interval_input": 3,
    "min_total_segments": 30,
    "max_total_segments": 55,
    "ost0_segment_min": 15,
    "original_audio_ratio": 0.5,
    "enable_picture_narration": False,
    "enable_humor_narration": False,
    "enable_logic_roast": False,
    "context_window_sec": 45,
    "narration_chars_min": 30,
    "narration_chars_max": 100,
    "ost1_duration_min": 8,
    "ost1_duration_max": 18,
    "ost1_duration_hard_max": 22,
    "min_ost1_segments": 15,
    "max_ost1_segments": 28,
    "ost1_every_n_segments": 2,
    # OST=0 引出下一段 OST=1 时，取画起点 = 下一段原声开始时间 − 该秒数
    "ost0_lead_before_ost1_sec": 10,
    "fazu2_core_theme": "",
    "fazu2_opening_climax_hint": FAZU2_DEFAULT_OPENING_CLIMAX_HINT,
    "enable_opening_closing_hook": True,
    "enable_opening_climax_chronological_replay": True,
    "opening_hook_template": "",
    # 仅第 1 集开头高潮末尾使用；第 2 集起由模型按当集名场面自拟转场
    "transition_hook_template": "故事，得从头讲起。",
    "closing_hook_template": "宝子们，我们下期再见！",
    "default_custom_prompt": "",
    "default_video_theme": "罚罪2",
    "append_custom_prompt": "",
    # 逐帧精剪：生成脚本前必须有字幕，并完成字幕×抽帧对照分析
    "require_subtitle_for_script": True,
    "subtitle_analysis_max_frame_chars": 20000,
    "subtitle_analysis_max_subtitle_chars": 12000,
    "subtitle_analysis_min_chars": 500,
    "subtitle_analysis_max_tokens": 4096,
}

DOCUMENTARY_SETTING_KEYS = frozenset(
    set(DOCUMENTARY_DEFAULTS)
    | set(DOCUMENTARY_COMPACT_OVERRIDES)
    | {
        "enable_subtitle_enrichment",
        "subtitle_max_chars",
        "subtitle_analysis_max_frame_chars",
        "subtitle_analysis_max_subtitle_chars",
        "subtitle_analysis_min_chars",
        "subtitle_analysis_max_tokens",
        "subtitle_batch_pad_sec",
        "enable_subtitle_refinement",
        "subtitle_refinement_max_entries_per_call",
        "subtitle_refinement_temperature",
        "subtitle_refinement_min_similarity",
        "subtitle_refinement_max_length_ratio_delta",
        "enable_hard_subtitle_ocr",
        "auto_subtitle_calibration_on_frame_analysis",
        "subtitle_ocr_crop_ratio",
        "subtitle_ocr_batch_size",
        "subtitle_ocr_max_concurrency",
        "subtitle_ocr_match_pad_ms",
        "subtitle_ocr_min_confidence_frames",
        "subtitle_ocr_min_similarity",
        "subtitle_ocr_max_length_ratio_delta",
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
        "narration_segment_max_retries",
        "narration_script_max_tokens",
        "narration_script_temperature",
        "ost1_every_n_segments",
        "documentary_compact_mode",
        "documentary_compact_style",
        "fazu2_core_theme",
        "fazu2_opening_climax_hint",
        "original_audio_ratio",
        "enable_opening_closing_hook",
        "opening_hook_template",
        "transition_hook_template",
        "closing_hook_template",
        "coverage_interval_sec",
        "target_output_ratio",
        "target_output_minutes",
        "ost0_segment_min",
        "ost0_lead_before_ost1_sec",
        "ost1_duration_hard_max",
        "min_ost1_segments",
        "max_ost1_segments",
        "require_subtitle_for_script",
        "enable_frame_analysis_drama_knowledge",
        "frame_analysis_drama_knowledge_max_chars",
        "enable_subtitle_analysis_drama_knowledge",
        "subtitle_analysis_drama_knowledge_max_chars",
        "enable_drama_knowledge",
        "enable_frame_strict_scene_rules",
        "frame_cross_scene_overlap_prune_ratio",
        "frame_max_segment_duration_sec",
        "frame_reference_token_saver",
        "frame_reference_attach_mode",
        "frame_reference_max_edge",
        "frame_reference_use_collage",
    }
)


def build_compact_pre_script_workflow_instructions(
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """逐帧精剪：生成 JSON 脚本前的材料与分析要求。"""
    cfg = settings or get_documentary_compact_settings()
    min_ost1, max_ost1 = compute_ost1_segment_bounds(settings=cfg)
    return f"""## 脚本生成工作流（前置条件 · 必须遵守）

本任务分两步，**不可跳过第 1 步直接写 JSON**：

### 素材优先级（硬性）
| 优先级 | 素材 | 用途 |
|--------|------|------|
| **主** | **原始字幕** | 剧情主线、`narration` 内容、`original_line` 台词、人名、**所有 `timestamp`** |
| **辅** | **抽帧画面分析** | 仅写 `picture` 画面描述；截取时间范围的**对齐参考**（须落在字幕区间内） |
| **辅** | **字幕×抽帧对照分析** | 策划蓝图（情节点、OST=1 清单等），剧情与时间戳仍以字幕为准 |

### 第 1 步：充分阅读已有材料（生成前必做）
1. **原始字幕**（`<subtitles>` / 下文「原始字幕」）— **第一依据**
   - **所有 `timestamp` 必须从此处逐字复制**，禁止编造
   - 剧情推进、对白引用、OST=1 金句选取均以字幕为准
2. **字幕×抽帧对照分析**（下文「字幕×抽帧 对照分析」）
   - 策划蓝图：人物表、开头高潮、正叙时间线、OST=1 清单、高潮复现、下集钩子
   - 若该节为空，说明前置分析未完成，**不得臆造剧情**
3. **抽帧画面分析**（下文「抽帧画面分析」）— **画面参考**
   - 仅供 `picture` 字段：人物动作、表情、场景、昼夜/光线
   - 截取时间可参考抽帧 moment，但**起止须落在对应字幕时间范围内**

### 第 2 步：严格依据上述材料 + 高潮前置版规则输出 JSON
- 若存在**本集追加要求**且指定开头高潮 → 第 1 个 item（OST=1）**必须以追加为准**
- 对照分析中的**开头高潮方案** → 第 1 个 item（OST=1）
- 对照分析中的**正叙时间线** → 主体 OST=0 段（段数见下方配置范围）
- 对照分析中的 **OST=1 金句清单（{min_ost1}–{max_ost1} 条）** → 逐一落实，时间戳与字幕一致
- 对照分析中的**高潮复现 / 下集钩子** → 对应收尾 items
- **禁止**脱离字幕编造剧情、台词、时间戳、人名；`picture` 环境氛围须与抽帧对照，但不得覆盖字幕剧情
"""


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

def resolve_fazu2_opening_climax_hint(
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    cfg = settings or get_documentary_compact_settings()
    return str(
        cfg.get("fazu2_opening_climax_hint") or FAZU2_DEFAULT_OPENING_CLIMAX_HINT
    ).strip()


def build_fazu2_picture_narration_sync_section() -> str:
    """解说以字幕为主，画面对位参考抽帧。"""
    return """### 声画对位（硬性 · 字幕为主，抽帧为辅）
- **`timestamp`**：必须从**字幕**原样复制；可参考抽帧 moment 在字幕区间内微调起止，**禁止**编造或超出字幕范围
- **`narration`**：剧情、对白复述、点评均以**字幕**为准；勿凭抽帧臆造字幕未出现的台词或情节
- **`picture`**：以**抽帧**可见画面为准（人物、动作、场景、昼夜/光线）；与字幕剧情不矛盾即可
- 写每段：先定字幕时间范围与剧情 → 再查同段抽帧补 `picture` → 最后写 `narration`
- **示例（正确）**：
  - 字幕：伟业与领导对峙，台词「胡小跃是我的徒弟」
  - picture（抽帧）：办公室内，伟业目光坚定
  - narration（字幕）：领导把材料摔在桌上。伟业一字一句回道……
- **示例（错误）**：抽帧是蹲守画面，解说却写「正在激烈抓捕」——动作阶段与画面对不上
- OST=1：`original_line` 台词来自字幕；`timestamp` 须覆盖该句字幕**完整起止**（可含前后连贯短条），勿截在半句
"""


def build_fazu2_ost_interleave_section(
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """原文与解说约 1:1 穿插，解说须在原声播完后开始。"""
    cfg = settings or get_documentary_compact_settings()
    ratio = float(cfg.get("original_audio_ratio", 0.5) or 0.5)
    pct = int(round(ratio * 100))
    narr_pct = 100 - pct
    min_ost1, max_ost1 = compute_ost1_segment_bounds(settings=cfg)
    ost_dur_min = int(cfg.get("ost1_duration_min", 8))
    ost_dur_max = int(cfg.get("ost1_duration_max", 18))
    return f"""### 原文与解说穿插（约 **{pct}:{narr_pct}** · 硬性）
| 要求 | 说明 |
|------|------|
| 段数比例 | OST=1 原声与 OST=0 解说各约 **{pct}%**（全片 **{min_ost1}–{max_ost1}** 段原声） |
| 原声时长 | 每段 OST=1 **{ost_dur_min}–{ost_dur_max} 秒**；金句/名场面偏长，勿切成 2–3 秒碎片 |
| 播放顺序 | 按 `_id` 依次播放；**一段播完再进下一段** |
| 时间戳 | **后一段 `timestamp` 开始 ≥ 前一段结束**；须等 OST=1 **当前台词/片段说完**再切解说 |
| 原声完整性 | OST=1 的 `timestamp` 须覆盖字幕**整句或连续对白**，**禁止**半句、半段突然切走 |
| 穿插节奏 | 第1段 OST=1（开头高潮）→ 第2段 OST=0 → 第3段起可 **连续 2–3 段 OST=1** 组成原声块，整块播完再接解说 |
| OST=1 段 | 仅原片声音，`narration`=「播放原片」+ `original_line`；**禁止**段内夹解说 |
| OST=0 段 | 仅 TTS 解说；须等**上一段（及同组连续）OST=1 原声完全结束**后再起止 |

- **禁止** OST=1 → OST=0 → OST=1 模式中解说**打断**未播完的原声（解说插在两个原声之间）
- **允许** 连续多段 OST=1；整块原声播完后再接 OST=0 点评
- **推荐** **OST=0 → OST=1**：解说先**埋伏、铺垫或制造悬疑**，引出下一段原声金句
- **禁止** 连续过多 OST=0 破坏原声/解说约 1:1 比例

### 解说引出原声（推荐写法）
| 手法 | 解说（OST=0）示例 | 下一段原声（OST=1） |
|------|-------------------|---------------------|
| **铺垫** | 领导叹口气，举报材料堆成山，伟业脸刷地白了。 | 「胡小跃是我的徒弟。」 |
| **埋伏** | 他嘴上答应配合，心里却另有盘算。下一秒—— | 原片对峙/反转台词 |
| **悬疑** | 您猜他怎么回？往下看！ / 更狠的还在后面。 | 金句原声 |
| **钩子** | 注意看，这群人已经蹲守三天了。且看他们怎么收网。 | 行动现场原声 |

- 引出原声的 OST=0 段：`timestamp` **起点** = 下一段 OST=1 的 `timestamp` 开始时间 **往前约 {int(cfg.get("ost0_lead_before_ost1_sec", 10))} 秒**（取该时段剧情画面作 B-roll）；**不要**与下一段 OST=1 时间重叠
- **禁止** OST=0 铺垫段长期停留在片头同一区间（如全片复用 `00:00:01`）；铺垫写什么剧情，`timestamp` 就对准下一段原声前后
- 原声播完后，再用 OST=0 **点评/承接**（`timestamp` 取**上一段 OST=1** 同场景区间）；形成「铺垫引出原声 → 原声整句播完 → 点评」节奏
- **突兀感红线**：观众应听完一句完整原台词或一个完整原声片段，再听到解说；勿在话说到一半切走
"""


def build_fazu2_character_roles_section() -> str:
    rows = "\n".join(
        f"| {name} | {role} | {note} |"
        for name, role, note in FAZU2_CHARACTER_ROLES
    )
    return f"""### 角色身份与性别（须画面对照 · 勿凭印象写错）
| 姓名 | 剧情参考 | 写作注意 |
|------|----------|----------|
{rows}

- **性别、职级、人称（他/她）必须与当段抽帧画面和字幕一致**；画面是男警察就写男/他，画面是女警察才可写女/她
- **不是禁止「女警」这个词**——若本段画面里确实是女警，可以照实写
- 写每段前先查抽帧：这段画面里**是谁**、**什么性别**、**什么职级**；勿把 A 角色的性别套到 B 身上
- 例：胡小跃是男刑警 → 指胡小跃时用「他」「刑警」；勿在胡小跃段落写成女警（与画面/剧情不符）
"""


def _character_gender_from_role(role: str) -> str:
    text = str(role or "")
    if "（男）" in text or "(男)" in text:
        return "男"
    if "（女）" in text or "(女)" in text:
        return "女"
    return ""


def build_fazu2_frame_character_gender_reference() -> str:
    """抽帧视觉分析：已知人物性别参考（须与画面对照，画面优先）。"""
    refs = [
        f"{name}={gender}"
        for name, role, _ in FAZU2_CHARACTER_ROLES
        if (gender := _character_gender_from_role(role))
    ]
    if not refs:
        return ""
    return (
        "剧情人物性别参考（须在**头像/定妆照面孔匹配成功并写入姓名后**用来核对性别，**画面可见性别优先**）："
        f"{', '.join(refs)}。"
        "未完成面孔匹配前勿写规范姓名；匹配成功后须标「姓名(男/女)」，勿写成女警/她。"
    )


def build_frame_visible_content_hint(
    settings: Optional[Dict[str, Any]] = None,
    *,
    frame_count: int = 0,
) -> str:
    """抽帧视觉分析：仅描述本批次帧内可见内容（硬性）。"""
    from app.services.documentary.frame_extraction_rules import (
        resolve_frame_max_segment_duration_sec,
    )

    cfg = settings or get_documentary_settings()
    if cfg.get("enable_frame_strict_scene_rules") is False:
        return ""
    count_text = str(frame_count) if frame_count > 0 else "本批次"
    max_seg_sec = resolve_frame_max_segment_duration_sec(cfg)
    return (
        f"**仅可见画面（硬性）**：scene_segments 只能描述本批次 {count_text} 张图片中实际可见的内容；"
        "禁止编造未出现在画面中的地点、人物、闪回、航拍、牺牲、追车、仓库突袭等「印象名场面」。"
        "**仅**连续同地点、同动作链（如整段天台对话）可合并为 1 条 segment；"
        "若本批含地点/动作阶段变化（停车场→车顶→地面奔跑等），**必须**输出多条 scene_segments；"
        f"单条 segment 时长不得超过约 {max_seg_sec} 秒，跨场景须拆成多条；"
        "**scene 必填**（如「楼顶天台」「废弃停车场」「车顶」），禁止留空；"
        "车内/车顶/车外须据可见结构区分，夜间特写无内饰时勿默认写车内；"
        "须填写 shot_scale / lighting_time / edit_role，方便后期选 OST=1 与 picture；"
        "同一批次内 timestamp 不得重叠；不同地点/不同场景不得拆成多条重叠时间段。"
        f"人名写入：**仅**本批画面清晰可见且与定妆照/头像对照匹配（{FRAME_FACE_MATCH_SIMILARITY_HINT}）→ 写规范姓名；"
        f"硬字幕/SRT **不得**猜人；无法匹配 → 「{FRAME_UNKNOWN_CHARACTER_MALE}」「{FRAME_UNKNOWN_CHARACTER_FEMALE}」。"
        "勾选头像不是默认全员在场；每一姓名须对应本批某帧面孔与参照图达到相似度阈值。"
    )


def build_frame_character_naming_hint(settings: Optional[Dict[str, Any]] = None) -> str:
    """视觉分析阶段：人名须有据，无据用未名人员+性别。"""
    cfg = settings or get_documentary_settings()
    hints: list[str] = [
        f"人名/称呼写入：**仅**本批可见面孔与定妆照/头像对照匹配时可写规范姓名（{FRAME_FACE_MATCH_SIMILARITY_HINT}）；",
        "硬字幕/SRT/subtitle_entries 中的姓名、称呼（老叶、二师兄、叶局等）**不得**用于推断画面人物；",
        "关系表/关系图**不能**作为猜人依据；",
        "**两人姓名均已由面孔匹配写入**时，可补明显师徒/父子/上下级等关系词；",
        "subtitle_entries 须**原样**摘录对白原文（含 ASR 错字）；无面孔匹配时写带特征的暂称或未名人员；",
        "后帧头像匹配成功后，前序帧仅当**同一身形+同一服装**可确认同一人时才回溯写规范名，否则保留暂称；",
        f"无法完成头像匹配时，characters 不写该项或用暂称；描述文本用「{FRAME_UNKNOWN_CHARACTER_MALE}」等仅当必要，"
        f"**禁止**用「领导」「警员」「男子A」等泛称作姓名；",
        "只允许在 characters 写**已上传头像名单内**、且本批面孔匹配成功的规范姓名；"
        "禁止写名单外旧称（如伟业、老叶等历史解说剧本人名）。",
        "observation/action/key_visual **禁止写姓名(男/女)**，人名只进 characters 数组。",
    ]
    if is_fazu2_compact_settings(cfg):
        hints.append(
            "示例：observation「楼顶天台，并肩对峙，阴天冷色调」，characters: [\"叶天佑\"]；"
            f"另一人无法匹配时不写入 characters，描述可用服装特征暂称。"
        )
    return " ".join(hints)


def build_frame_gender_hint(settings: Optional[Dict[str, Any]] = None) -> str:
    """视觉分析阶段：人物性别须据画面判断，供抽帧 prompt 注入。"""
    cfg = settings or get_documentary_settings()
    hints: list[str] = [
        "人物性别**仅据画面**判断（面容、发型、体型、着装、胡须等可见特征），"
        "勿凭姓名谐音、剧情印象或字幕语气臆测；",
        "characters 使用「姓名(男)」「姓名(女)」"
        f"「{FRAME_UNKNOWN_CHARACTER_MALE}」「{FRAME_UNKNOWN_CHARACTER_FEMALE}」「姓名(不明)」格式；"
        f"有面孔匹配写姓名(性别)，无匹配写{FRAME_UNKNOWN_CHARACTER_MALE}/{FRAME_UNKNOWN_CHARACTER_FEMALE}；"
        "**禁止**凭字幕中的称呼写规范姓名；",
        "frame_observations 须用 characters 数组列出本帧可见人物的规范姓名（面孔匹配后）；",
        "observation/action/key_visual 只写地点、动作、光线，**禁止写姓名(男/女)或代称**；",
        "同一批次内同一人物的性别须前后一致；仅见背影/侧脸无法确认时标「不明」，勿猜测。",
    ]
    if is_fazu2_compact_settings(cfg):
        reference = build_fazu2_frame_character_gender_reference()
        if reference:
            hints.append(reference)
    return " ".join(hints)


def warn_frame_analysis_gender_mismatch(
    *,
    scene_segments: list[dict[str, Any]],
    frame_observations: list[dict[str, Any]],
    batch_index: int = 0,
    time_range: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> None:
    """抽帧结果中已知男性角色被标成女性时打日志，便于排查视觉模型误判。"""
    if not is_fazu2_compact_settings(settings):
        return

    male_names = [
        name
        for name, role, _ in FAZU2_CHARACTER_ROLES
        if _character_gender_from_role(role) == "男"
    ]
    if not male_names:
        return

    texts: list[str] = []
    for segment in scene_segments:
        if isinstance(segment, dict):
            texts.append(json.dumps(segment, ensure_ascii=False))
    for observation in frame_observations:
        if isinstance(observation, dict):
            texts.append(str(observation.get("observation") or ""))

    location = f"批次 #{batch_index} · {time_range}".strip(" ·")
    for text in texts:
        for name in male_names:
            if name not in text:
                continue
            female_markers = (
                f"{name}(女)",
                f"{name}（女）",
                f"女警{name}",
                f"女刑警{name}",
                f"女警察{name}",
            )
            if any(marker in text for marker in female_markers):
                logger.warning(
                    f"抽帧分析{location}：{name} 被标为女性或与女警混用，"
                    "请对照画面重新抽帧或重跑该批次"
                )
                continue
            if re.search(rf"{re.escape(name)}[^。；，,\n]{{0,12}}她", text):
                logger.warning(
                    f"抽帧分析{location}：{name} 附近出现「她」，疑似性别与画面对不上，"
                    "请对照画面重新抽帧或重跑该批次"
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
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """逐帧精剪：高潮前置版脚本规则（写入脚本生成提示）。"""
    return build_compact_story_script_rules(core_theme, settings=settings)


def _build_compact_hooks_rules_section(
    settings: Optional[Dict[str, Any]] = None,
    work_hint: str = "",
) -> str:
    """罚罪2 V2：开篇、正叙开场与结尾硬性约束。"""
    cfg = settings or get_documentary_compact_settings()
    transition_tpl = str(
        cfg.get("transition_hook_template") or "故事，得从头讲起。"
    ).strip()
    closing_tpl = str(cfg.get("closing_hook_template") or "宝子们，我们下期再见！")
    hook_enabled = cfg.get("enable_opening_closing_hook", True)
    closing_line = (
        f"- **最后一段**（OST=0 或 OST=1）须含结束语：**{closing_tpl}**"
        if hook_enabled
        else "- **已关闭**固定结尾模板；最后一段按剧情自然收束即可"
    )
    return f"""### 开篇、正叙与结尾（硬性）

| 段落 | OST | 要求 |
|------|-----|------|
| 第 1 段（开头高潮） | 1 | **纯原声**，`narration` 固定「播放原片」；**禁止**旁白、**禁止**「宝子们」 |
| 第 2 段（转场+正叙） | 0 | **必须以「宝子们」开头**，接「{transition_tpl}」进入正叙 |
| 最后一段 | 0 或 1 | 可复现开头名场面；须含道别语 |

- **「宝子们」**仅出现在第 2 段与最后一段，第 1 段不出现
{closing_line}
- 成片后处理会补全缺失的「宝子们」开场与结尾道别，模型生成时尽量自带
"""


def build_compact_story_script_rules(
    core_theme: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """逐帧精剪 · 罚罪2 高潮前置版 V2：角色设定与创作规则。"""
    cfg = settings or get_documentary_compact_settings()
    work_hint = (core_theme or "").strip() or "（填写「视频主题」如《罚罪2》第1集）"
    hooks_section = _build_compact_hooks_rules_section(cfg, work_hint)
    transition_tpl = str(
        cfg.get("transition_hook_template") or "故事，得从头讲起。"
    ).strip()
    min_seg = int(cfg.get("min_total_segments", 30))
    max_seg = int(cfg.get("max_total_segments", 55))
    ost0_min = int(cfg.get("ost0_segment_min", 24) or 24)
    min_ost1, max_ost1 = compute_ost1_segment_bounds(settings=cfg)
    ost_dur_min = int(cfg.get("ost1_duration_min", 3))
    ost_dur_max = int(cfg.get("ost1_duration_hard_max", 10) or 10)
    target_min = float(cfg.get("target_output_minutes", 12))
    pre_workflow = build_compact_pre_script_workflow_instructions(cfg)
    char_name_rows = "\n".join(
        f"| {wrong} | {correct} |"
        for wrong, correct in FAZU2_WRONG_CHARACTER_NAMES
    )
    return f"""{pre_workflow}

## 《罚罪2》解说脚本制作规则（V2 · 优先级最高）

### 角色设定
你是一位**30 年经验的资深说书人**，擅长《罚罪2》类悬疑刑侦剧的**高潮前置型**短视频脚本。
当前数据可能不是整集内容（如某一集的一半），分析时应结合整集/整部剧来把握情节。
受众是普通观众：开头先给最炸裂名场面勾住人，再从头讲清剧情，结尾形成闭环。
**不要**分析镜头语言、导演手法或社会隐喻。

### 任务目标
在**已充分理解**下文「抽帧画面分析」「原始字幕」「字幕×抽帧 对照分析」之后，生成解说脚本。作品：**{work_hint}**
对照分析是策划蓝图，JSON 是其执行结果，二者须一致。

### 输出格式（必须严格遵守）
- 只输出纯 JSON **数组**：`[{{"_id", "timestamp", "picture", "narration", "OST"}}, ...]`
- OST=1 的 item **必须**额外包含 `"original_line"` 字段
- **不要**使用 Markdown 代码块，直接输出 JSON 文本
- **不要**添加任何注释、前后缀或解释文字；不要用 `{{"items":[...]}}` 包裹

### 一、核心数据指标
| 项目 | 标准 |
|------|------|
| 单集总时长 | 约 {target_min:.0f} 分钟 |
| 总段落数 | **{min_seg}–{max_seg} 段**（可微调） |
| 原文台词占比 | **约 50%**（OST=1 与原片声音，段数约一半） |
| 解说占比 | **约 50%**（OST=0，段数约一半） |
| 每段时长 | 平均 **15–20 秒**，根据内容灵活调整 |
| 时间戳依据 | **字幕**中的实际时间位置（抽帧仅作截取对齐参考） |
| 剪辑顺序 | 开头高潮可取自剧中任意位置，正叙部分按时间线排列 |

### 二、OST 字段定义
| OST | 含义 | narration 要求 | 额外字段 |
|-----|------|----------------|----------|
| **0** | 旁白解说 | 完整解说词（非空） | 无 |
| **1** | 原声播放（原文台词） | 固定填写 **「播放原片」**（非空） | `"original_line": "「原台词」"` |

**特殊约束：**
- 第 1 段（开头高潮）**必须** OST=1（播放原片，**不夹杂旁白**）
- 最后一段可以是 OST=0 或 OST=1，但**必须**含结束语「宝子们，我们下期再见！」
- 「宝子们」出现在第 2 段与最后一段，**第 1 段不出现**

### 三、整体结构（按段落顺序）
| 部分 | 段落位置 | 内容要点 | OST |
|------|----------|----------|-----|
| ① 开头高潮 | 第 1 段 | **默认跳楼牺牲**（胡小跃楼顶纵身跃下，金句「天就快亮了。」），**纯原声，无旁白** | 1 |
| ② 转场+正叙开始 | 第 2 段 | 以「宝子们」开头，接「{transition_tpl}」，然后进入正叙 | 0 |
| ③ 正叙剧情 | 第 3 段至倒数第 2 段 | 按时间线推进；**正叙走到第 1 段开篇高潮的原片时刻，须再插入同一片段 OST=1（【复现】）**；推荐 OST=0 铺垫后接 OST=1 原声 | 混合 |
| ④ 高潮复现+后续+结束语 | 最后一段 | 可再次呼应开头名场面；须含道别语收尾 | 0 或 1 |

**`_id` = 成片播放顺序**（1→2→3…）。① 倒叙开篇高潮 → ② 转场正叙 → ③ 正叙推进至第 1 段时间戳处**必须复现同一片段** → ④ 收尾。

### ③ 开头高潮选取（第 1 段 OST=1 · 硬性）
- **默认优先**：{resolve_fazu2_opening_climax_hint(cfg)}
- 用户「追加提示词」若指定开头名场面 → **以追加为准**
- 中段冲突台词（如狗贩子争吵、掏枪对峙）可作正叙 **OST=1**，**不得**顶替跳楼作第 1 段
- **正叙复现（硬性）**：播放顺序推进到第 1 段 `timestamp` 对应的原片时刻时，**必须再插入一段与第 1 段完全相同的 OST=1**（同 timestamp / original_line / picture，picture 可加「【复现】」前缀）
- 最后一段可再次呼应开头名场面，但**不能替代**正叙中的这次复现

{build_fazu2_character_roles_section()}
{build_fazu2_picture_narration_sync_section()}
{build_fazu2_ost_interleave_section(cfg)}
### 四、原文台词处理细则
- 标点：使用 **「 」** 将原台词括起来（写在 `original_line` 中）
- 选取标准：冲突爆发、情感高潮、反转时刻、反派嚣张名场面
- 占比控制：OST=1 与 OST=0 **各约 50%**；第 3 段起形成「铺垫引出原声 → 原声 → 点评」节奏
- **引出原声**：关键金句前优先用 OST=0 **埋伏、铺垫、悬疑**（如「您猜他怎么回？」「且看他们怎么收网」），再接 OST=1
- OST=1 段：`narration` **固定**「播放原片」，台词原文写在 `original_line`
- **切解说原则**：下一段 OST=0 只能在上一段 OST=1 **这句话说完、这个原声片段播完**后开始；半句切走极突兀

### 五、解说词写作风格
- 解说开头：第 2 段以「宝子们」开头，然后叙述
- 情绪词：好家伙、您品、憋屈、头皮发麻、鼻子一酸、这嘴脸、恨不得抽他
- 节奏：平静叙述 → **铺垫/悬疑引出原声** → 原声金句 → 点评承接
- 小钩子：每 1–2 分钟加一句「您猜怎么着？」「接下来更狠」，**优先放在 OST=1 原声之前**作引出
- **禁止**：猎奇式调侃、过度玩梗破坏沉重氛围
- **禁止流水账词**：❌ 然后、接着、接下来、我们可以看到

### 六、角色名规范（依据字幕文件）
| 正确 | 错误（禁止） |
|------|-------------|
{char_name_rows}

- `narration` 与 `picture` 必须用**具体人名**；**禁止**警员1/说话人1 等编号

### 七、声画对位（硬性 · 字幕为主，抽帧为辅）
- **剧情、对白、人名、时间戳** → 以**字幕**为准
- **`picture`** → 以当段**抽帧**可见画面为准（场景、光线、昼夜、人物动作）
- **人物性别、职级、人称**：人名来自字幕，性别/职级与抽帧画面对照；勿张冠李戴
- **`narration` 讲什么**由字幕剧情决定；动作阶段（蹲守/抓捕等）须与 `picture` 一致，可用「注意看」引导
- 情绪靠**字幕台词、画面动作、表情**表达，不靠臆造天气/光线

### 八、段数与时长
| 指标 | 要求 |
|------|------|
| 总段数 | **{min_seg}–{max_seg} 段** |
| 解说 OST=0 | **≥{ost0_min} 段**，每段 **30–100 字** |
| 原声 OST=1 | **{min_ost1}–{max_ost1} 段**，每段 **{ost_dur_min}–{ost_dur_max} 秒** |
| 结构 | ①开头高潮 → ②转场正叙 → ③正叙 → ④收尾 |

{hooks_section}
"""


def get_compact_custom_prompt_display(
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """逐帧精剪：供 WebUI「自定义提示词」反显的完整规则文案（可编辑）。"""
    cfg = get_documentary_compact_settings(settings)
    chars_min = int(cfg.get("narration_chars_min", 30))
    chars_max = int(cfg.get("narration_chars_max", 100))
    min_seg = int(cfg.get("min_total_segments", 35))
    max_seg = int(cfg.get("max_total_segments", 50))
    min_ost1, max_ost1 = compute_ost1_segment_bounds(settings=cfg)
    ost_min = int(cfg.get("ost1_duration_min", 3))
    ost_max = int(cfg.get("ost1_duration_max", 8))
    target_min = float(cfg.get("target_output_minutes", 12))
    ep1_transition_tpl = str(
        cfg.get("transition_hook_template") or "故事，得从头讲起。"
    ).strip()
    closing_tpl = str(cfg.get("closing_hook_template") or "宝子们，我们下期再见！")
    auto_hook = "开启" if cfg.get("enable_opening_closing_hook", True) else "关闭"

    rules = build_compact_story_script_rules(
        str(cfg.get("fazu2_core_theme") or "").strip()
        or "（填写「视频主题」如《罚罪2》第1集）",
        settings=cfg,
    )
    return f"""# 逐帧精剪 · 罚罪2 脚本规则（V2）
（本框内容会作为「补充创作要求」参与生成；与系统内置规则一致，可直接修改）

{rules}

## 当前配置摘要（[documentary_compact]）
| 项 | 值 |
|----|-----|
| 总段数 | {min_seg}–{max_seg} |
| 解说字数/段 | {chars_min}–{chars_max} |
| 原声 OST=1 | **{min_ost1}–{max_ost1} 段**，每段 {ost_min}–{ost_max} 秒 |
| 目标成片 | 约 {target_min:.0f} 分钟 |
| 第1集转场句 | 「{ep1_transition_tpl}」（第2集起按当集高潮自拟） |
| 结尾道别 | {auto_hook}；「{closing_tpl}」 |

## JSON 输出示例（结构参考）
```json
{FAZU2_SCRIPT_REFERENCE_ITEMS_JSON}
```

---
（本集/本片专属要求请写在 WebUI「追加提示词」框，会叠加在本规则之后参与生成）
"""
FAZU2_SCRIPT_REFERENCE_ITEMS_JSON = """[
  {
    "_id": 1,
    "timestamp": "00:20:05,000-00:20:13,000",
    "picture": "夜色楼顶，胡小跃站在边缘，纵身跃下",
    "narration": "播放原片",
    "OST": 1,
    "original_line": "「天就快亮了。」"
  },
  {
    "_id": 2,
    "timestamp": "00:00:01,940-00:00:09,940",
    "picture": "楼顶天台，老叶与伟业并肩",
    "narration": "宝子们，故事得从头讲起。伟业，厅级干部，放着舒服日子不过，非要回汉州当局长……",
    "OST": 0
  },
  {
    "_id": 3,
    "timestamp": "00:00:11,000-00:00:16,890",
    "picture": "领导叹气，语重心长",
    "narration": "领导叹口气：胡小跃是你徒弟，我知道。可人家是自杀，举报材料堆成山。伟业脸刷地白了。您猜他怎么回？往下看！",
    "OST": 0
  },
  {
    "_id": 4,
    "timestamp": "00:00:16,940-00:00:26,700",
    "picture": "伟业目光坚定，一字一句",
    "narration": "播放原片",
    "OST": 1,
    "original_line": "「胡小跃是我的徒弟。」"
  },
  {
    "_id": 5,
    "timestamp": "00:00:26,700-00:00:36,020",
    "picture": "伟业情绪激动，反驳领导",
    "narration": "一句话把领导噎住了。伟业接着崩了：我了解小跃，她不是对组织失去信心，更不是害怕逃避。她是不甘心被陷害，想用自己的命加速破案。听得我后背发凉。这师傅够硬！",
    "OST": 0
  },
  {
    "_id": 45,
    "timestamp": "00:20:05,000-00:20:13,000",
    "picture": "【复现】夜色楼顶，胡小跃纵身跃下",
    "narration": "还记得开头吗？胡小跃从楼顶一跃而下，「天就快亮了。」一个刑警，用自己的命换来了重启调查的机会。宝子们，汉州的天，是该亮了。我们下期再见！",
    "OST": 0
  }
]"""


def build_fazu2_script_output_reference(
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """高潮前置版 JSON 参考模板。"""
    cfg = get_documentary_compact_settings(settings)
    min_ost1, max_ost1 = compute_ost1_segment_bounds(settings=cfg)
    return f"""## 输出 JSON 参考模板（V2 · 必须严格仿照）

```json
{FAZU2_SCRIPT_REFERENCE_ITEMS_JSON}
```

{build_fazu2_generation_anti_patterns(cfg)}

### 字段要点
| 字段 | OST=0 | OST=1 |
|------|-------|-------|
| `narration` | 30–100 字，**以字幕剧情为准** | **固定**「播放原片」 |
| `original_line` | 无 | **必填**，字幕原台词 `"「…」"` |
| `timestamp` | **从字幕复制**（抽帧仅作对齐参考） | 字幕对白**精确**起止 |
| `picture` | 参考抽帧：人物+动作+场景+光线 | 参考抽帧：说话人/名场面 |
| `OST` | 0（约 50%） | 1（**{min_ost1}–{max_ost1}** 段，约 50%） |
| `_id` | **播放顺序**：①开头高潮(OST=1) → ②转场正叙(OST=0) → ③正叙 → ④收尾 |
"""


def build_fazu2_generation_anti_patterns(
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """罚罪2 V2：错误 vs 正确对照。"""
    cfg = settings or get_documentary_compact_settings()
    ost_dur_min = int(cfg.get("ost1_duration_min", 8))
    ost_dur_max = int(cfg.get("ost1_duration_max", 18))
    lead_sec = int(cfg.get("ost0_lead_before_ost1_sec", 10) or 10)
    return f"""## 错误示范 vs 正确示范

### ❌ 禁止
- 第 1 段 OST=0 或含旁白/「宝子们」
- 第 2 段没有「宝子们」开头
- OST=0 铺垫段 `timestamp` 全片复用片头同一区间（如全是 `00:00:01`），与下一段 OST=1 画面相隔数分钟
- OST=1 的 `narration` 为空或写解说词（应固定「播放原片」）
- OST=1 缺少 `original_line` 字段
- 在 `narration` 中用 `**「金句」**` 代替 `original_line`（旧版格式，已废弃）
- 原声段过短（应 **{ost_dur_min}–{ost_dur_max} 秒**）；原声与解说比例失衡（应约 **1:1**）
- OST=1 未播完即开始下一段 OST=0 解说（时间戳重叠、夹在两个原声之间、或原声半句被截断）
- 人名错误（胡小月/胡晓月/小月/秦峰/罗伯/伟叶）；性别/职级与画面对不上（如胡小跃段落写成女警）
- 第 1 段不用跳楼而用中段台词（如「你跟我说这是狗贩子」）作开头高潮
- 匿名编号人物；拉片分析；流水账词（然后、接着、我们可以看到）
- 编造时间戳；用 `{{"items":[...]}}` 包裹

### ✅ 必须
- 第 1 段 OST=1：跳楼 sacrifice +「天就快亮了。」类金句；`播放原片` + `original_line`；纯原声无旁白
- 第 2 段 OST=0：以「宝子们」开头，接「故事，得从头讲起」进入正叙
- OST=0 铺垫下一段 OST=1：`timestamp` **起点** = 下一段原声开始 **− 约 {lead_sec} 秒**（取画与解说/原声内容一致）
- 关键台词前 OST=0 埋伏/悬疑引出 → OST=1：`"narration": "播放原片"` + `"original_line": "「台词」"` → 下段 OST=0 点评
- 最后一段含「宝子们，我们下期再见！」；可复现开头名场面（OST=0 时台词用「」嵌入 narration）
- 输出顶层 JSON 数组 `[...]`；`_id` 为播放顺序；人名严格按字幕
"""


def _config_file_path() -> str:
    from app.config import config

    return config.config_file


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
    """逐帧精剪：默认高潮前置版（35–50 段，原声 ≤20 段，解说 30–100 字/段）。"""
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


DOCUMENTARY_COMPACT_CONFIG_KEYS = (
    "enable_opening_closing_hook",
    "opening_hook_template",
    "transition_hook_template",
    "closing_hook_template",
    "fazu2_core_theme",
    "default_custom_prompt",
    "default_video_theme",
    "append_custom_prompt",
)


def save_documentary_compact_settings_to_config(settings: Dict[str, Any]) -> bool:
    """将逐帧精剪参数写入 config.toml 的 [documentary_compact] 段。"""
    config_path = _config_file_path()
    try:
        if os.path.isfile(config_path):
            config_data = toml.load(config_path)
        else:
            config_data = {}
        section = dict(config_data.get("documentary_compact") or {})
        for key in DOCUMENTARY_COMPACT_CONFIG_KEYS:
            if key in settings:
                section[key] = settings[key]
        config_data["documentary_compact"] = section
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(toml.dumps(config_data))
        try:
            import app.config.config as config_py

            config_py._cfg["documentary_compact"] = section
            config_py.documentary_compact = section
        except Exception:
            pass
        logger.info("逐帧精剪规则已保存到 config.toml [documentary_compact]")
        return True
    except Exception as e:
        logger.error(f"保存逐帧精剪配置失败: {e}")
        return False


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


def compute_ost1_segment_bounds(
    total_items: int = 0,
    settings: Optional[Dict[str, Any]] = None,
) -> tuple[int, int]:
    """返回 OST=1 原声段数 (最少, 最多)。"""
    cfg = settings or get_documentary_settings()
    min_cfg = int(cfg.get("min_ost1_segments", 0) or 0)
    max_cfg = int(cfg.get("max_ost1_segments", 0) or 0)

    if is_fazu2_compact_settings(cfg):
        ratio = float(cfg.get("original_audio_ratio", 0.5) or 0.5)
        ratio = max(0.2, min(0.8, ratio))
        if total_items > 0:
            target = max(1, round(total_items * ratio))
            slack = max(2, int(total_items * 0.06))
            min_ost1 = max(1, target - slack)
            max_ost1 = min(total_items - 1, target + slack)
        else:
            min_seg = int(cfg.get("min_total_segments", 30))
            target = max(1, round(min_seg * ratio))
            min_ost1 = max(1, min_cfg if min_cfg > 0 else target - 2)
            max_ost1 = max(min_ost1, max_cfg if max_cfg > 0 else target + 3)
        return min_ost1, max_ost1

    every_n = max(1, int(cfg.get("ost1_every_n_segments", 10) or 10))
    auto_max = max(1, round(total_items / every_n)) if total_items > 0 else 1
    if max_cfg > 0:
        max_ost1 = min(auto_max, max_cfg)
    else:
        max_ost1 = auto_max
    min_ost1 = max(1, min_cfg) if min_cfg > 0 else 1
    return min(min_ost1, max_ost1), max_ost1


def compute_max_ost1_segments(
    total_items: int,
    settings: Optional[Dict[str, Any]] = None,
) -> int:
    """OST=1 原声段数上限。"""
    _, max_ost1 = compute_ost1_segment_bounds(total_items, settings)
    return max_ost1


def compute_min_ost1_segments(
    total_items: int = 0,
    settings: Optional[Dict[str, Any]] = None,
) -> int:
    """OST=1 原声段数下限。"""
    min_ost1, _ = compute_ost1_segment_bounds(total_items, settings)
    return min_ost1


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
        min_ost1, max_ost1 = compute_ost1_segment_bounds(
            estimated_items or 0, cfg
        )
        ratio = float(cfg.get("original_audio_ratio", 0.5) or 0.5)
        pct = int(round(ratio * 100))
        return (
            f"- **原声 OST=1**：全片 **{min_ost1}–{max_ost1} 段**（约 {pct}%），每段 **{ost_min}–{ost_max} 秒**\n"
            f"- `narration`=「播放原片」+ `original_line`；推荐 **OST=0 铺垫/悬疑引出 → OST=1 → OST=0 点评**\n"
            f"- 原声须等当前台词/片段说完再切解说；timestamp 覆盖字幕整句；禁止半句截断\n"
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
- 时间戳必须落在**字幕**已有范围内，可参考抽帧对齐画面 moment；**严禁重叠**，后段开始 ≥ 前段结束
- **精剪≠梗概**：每段须有深度拉片观察，禁止流水账复述；优先华彩镜头，但**不得**为省段数跳过整段 {interval} 秒未覆盖区间
- **声画对位（何止电影）**：解说抛观点后，可紧接 OST=1 切入能印证该观点的原声对白（时间戳以字幕为准）
- **禁止** OST=2；本模式只用 OST=0 与 OST=1
"""

    target_minutes = float(cfg.get("target_output_minutes", 12) or 12)
    ost0_min = int(cfg.get("ost0_segment_min", 30) or 30)
    min_ost1, max_ost1 = compute_ost1_segment_bounds(target_segments, cfg)

    if is_fazu2_compact_settings(cfg):
        return f"""## 精剪覆盖（必须遵守 · 罚罪2 V2）

### 风格目标
- **高潮前置**：第 1 段纯原声名场面 → 第 2 段「宝子们」转场正叙 → 按时间线推进 → 最后一段收尾道别
- **OST=0 与 OST=1 约 1:1**（各约 50%，原声 **{min_ost1}–{max_ost1}** 段），第 3 段起交替穿插

### 全片指标（目标成片约 {target_minutes:.0f} 分钟）
| 指标 | 要求 |
|------|------|
| 总片段数 | **{min_segments}–{max_segments} 段**（目标约 {target_segments}） |
| 解说 OST=0 | **≥{ost0_min} 段**，每段 **{chars_min}–{chars_max} 字** |
| 原声 OST=1 | **{min_ost1}–{max_ost1} 段**（约 50%） |
| 每段时长 | 平均 **15–20 秒** |
| 结构 | ①开头高潮(纯原声) → ②转场正叙 → ③正叙 → ④收尾 |

{ost1_hint}- `_id` 为**播放顺序**（可倒叙开场）；时间戳**必须从字幕复制**
- 遵守「罚罪2 脚本规则 V2」；OST=1 须含 `original_line`；**禁止 OST=2**
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


def resolve_append_custom_prompt(
    append_prompt: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """解析本集追加提示词（WebUI 输入优先于 config）。"""
    text = (append_prompt or "").strip()
    if text:
        return text
    cfg = get_documentary_settings(settings)
    return str(cfg.get("append_custom_prompt") or "").strip()


def build_append_requirements_section(
    append_prompt: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """本集追加要求独立区块（置于 prompt 首位，优先级高于默认规则）。"""
    append = resolve_append_custom_prompt(append_prompt, settings)
    if not append:
        return ""
    return f"""## 本集追加要求（最高优先级 · 必须遵守）

以下内容由用户在 WebUI「追加提示词」填写，**覆盖**默认开头高潮示例与其它通用建议中冲突的部分：

{append}

硬性要求：
- 若指定了开头高潮/爆燃名场面（人物、台词、场景），第 1 个 item（OST=1）**必须**使用该场面，`timestamp` 从字幕原样复制
- 不得改用其他角色的台词或桥段顶替用户指定的开头高潮
- 开头段后，再按时间线正叙展开；高潮复现段可再次呼应同一金句（若追加要求未禁止）"""


def build_effective_documentary_prompt(
    user_prompt: str = "",
    append_prompt: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    """合并 config 默认补充与自定义提示词（不含追加提示词，追加由独立区块注入）。"""
    cfg = get_documentary_settings(settings)
    return resolve_documentary_custom_prompt(user_prompt, cfg)


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
    min_ost1, max_ost1 = compute_ost1_segment_bounds(settings=cfg)
    chars_min = int(cfg.get("narration_chars_min", 80))
    chars_max = int(cfg.get("narration_chars_max", 150))

    if is_fazu2_compact_settings(cfg):
        return f"""## 音频模式（必须遵守 · 罚罪2 V2）

| OST | 含义 | narration | 额外字段 |
|-----|------|-----------|----------|
| **0** | 旁白解说（约 50%） | 完整解说词，与 picture 一致 | 无 |
| **1** | 原声播放（约 50%） | **固定**「播放原片」 | `"original_line": "「原台词」"` |
| **2** | **禁止** | — | — |

### OST=0（全片解说段数见配置下限，每段 {chars_min}–{chars_max} 字）
- 正叙、点评、钩子、道别；复述对白用「」嵌入 narration
- 金句 OST=1 之后的点评段用 OST=0

### OST=1（全片 **{min_ost1}–{max_ost1} 段**，约 50%，每段 {ost_min}–{ost_max} 秒）
- `narration` **固定**「播放原片」；台词写在 `original_line`
- 第 1 段必须为纯原声；第 3 段起可连续多段 OST=1，整块播完再接 OST=0
- **禁止** OST=1 未播完即开始下一段 OST=0 解说（时间戳不得重叠；禁止夹在两个原声之间）

{build_fazu2_script_output_reference(cfg)}
只输出 JSON 数组 `[...]`，不要 markdown 代码块包裹。
"""

    picture_line = "- `picture` 写画面/人物备注"
    return f"""## 音频模式（必须遵守）

| OST | 含义 |
|-----|------|
| **0** | 纯 AI 解说（默认） |
| **1** | 纯原声，**{min_ost1}–{max_ost1} 段**，{ost_min}–{ost_max} 秒 |
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
    hints: list[str] = [
        "剪辑师标注：激烈冲突、名场面台词、情绪爆发、静默压迫 → edit_role 标「高潮/动作/反应」，"
        "importance 标「高」，audio_cue 写可听见的原声线索；"
        "平缓对话/定场 → edit_role「对话/定场」，importance「中/低」。",
    ]
    if is_fazu2_compact_settings(cfg):
        hints.append(
            "画面描述：人物+动作+场景+可见性别/职级+光线/昼夜/天气+情绪，15–30 字；"
            f"有字幕真名写「姓名(性别)」，无真名写「{FRAME_UNKNOWN_CHARACTER_MALE}/{FRAME_UNKNOWN_CHARACTER_FEMALE}」，禁止警员1、说话人2等编号；"
            "禁止把「领导」当人物姓名；伟业是局长专名，勿与上级混称；性别须据画面判断。"
        )
        hints.append(
            "若该帧有可作 OST=1 的标志性台词，在 audio_cue 标注 `[金句原声]` 并确保 timestamp 对齐字幕。"
        )
    elif is_compact_documentary_settings(cfg):
        hints.append(
            "写出景别、运镜、构图与人物动作，供解说 picture 引用。"
        )
    if cfg.get("enable_original_audio_highlights", True):
        hints.append(
            "爆炸、追逐、尖叫、恐怖、激烈冲突、名场面台词或音效高潮 → audio_cue 须具体写出，importance 为「高」。"
        )
    if cfg.get("enable_action_expression_modifiers", True):
        hints.append(
            "action / emotion / key_visual 须写出可见表情、肢体与氛围；"
            "key_visual 同时写清光线+景别+构图，避免空泛「画面紧张」。"
        )
    if cfg.get("enable_logic_roast", True) and not is_compact_documentary_settings(cfg):
        hints.append(
            "若人物行为明显违背常理或令人费解，"
            "请在 key_visual 或 importance 中标注「可吐槽」并简述原因（供解说员适度点评）。"
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
        min_ost1, max_ost1 = compute_ost1_segment_bounds(min_segments, cfg)
        rules_block = build_compact_story_script_rules(theme, settings=cfg)
        return f"""{rules_block}

## 解说风格（必须遵守 · 罚罪2 V2）

- 全片 **{min_segments}–{max_segments} 段**，解说与原声各约 **50%**（OST=0 ≥{ost0_min}，OST=1 **{min_ost1}–{max_ost1}**）
- 结构：①开头高潮(纯原声) → ②「宝子们」转场 → ③ **铺垫/悬疑引出原声 → 原声 → 点评** → ④收尾
- OST=0：每段 **{chars_min}–{chars_max} 字**；可埋伏、铺垫、制造悬疑引出下一段 OST=1
- OST=1：「播放原片」+ `original_line`；须整句/整段播完，下一段 OST=0 再点评承接
- 纵览前后约 {window} 秒画面与字幕，正叙按时间线连贯推进
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
            "- **推荐**上一段 OST=0 用埋伏、铺垫或悬疑引出原声（「您猜他怎么回？」「且看他们怎么收网」等）",
            "- 原声播完后，下一段 OST=0 点评承接；原声段本身不写解说词",
        ]
    )

    return "\n".join(lines)
