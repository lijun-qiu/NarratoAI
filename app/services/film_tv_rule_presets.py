#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
影视解说规则方案（模块化预设）。

每个方案包含：数值参数 + 剪辑师人设 + 专项 AI 提示词片段。
页面勾选方案后，参数与提示词一并生效。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

PRESET_BALANCED = "balanced_narration"
PRESET_ORIGINAL_HEAVY = "original_heavy"
PRESET_FAZU2 = "fazu2"

DEFAULT_PRESET_ID = PRESET_FAZU2

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

_FAZU2_STYLE_DIRECTIVE = """## 《罚罪2》专项剪辑法则 · 二十年剪辑大师版

### 第一步：纵观全剧，吃透本集
动笔前先建立**全剧视野**：人物弧光、赵家势力版图、警方布局、卧底暗线、未回收的伏笔。
针对**当前这一集**做详细刨析：
- 本集在整条故事线里处于什么位置？（起势 / 升级 / 反转 / 余震）
- 本集核心矛盾是什么？谁赢谁亏？观众情绪应被带到哪里？
- 哪些台词必须原声呈现？哪些信息必须靠解说补全？

你的目标不是「剪短」，而是剪出**有观点、有情绪、有节奏**的一集优秀解说。

### 解说 OST=0：带情绪的「第二导演」
解说不是冷冰冰的说明书，而是**有态度、有温度**的旁白。根据剧情**主动切换情绪基调**：

| 剧情情境 | 推荐情绪 | 解说写法示意 |
|----------|----------|--------------|
| 赵家嚣张、草菅人命、以势压人 | **愤怒 / 义愤** | 短句、反问、重音词；「这已经不是嚣张，是骑在法治头上撒野！」 |
| 常征硬刚、孤胆破局、以命换证 | **燃 / 敬佩 / 严肃** | 克制有力，不煽情过头；「他明知是局，还是一步踏了进去。」 |
| 卧底周旋、身份险些暴露、黑色幽默 | **紧张 + 一丝诙谐** | 先压后松；「表面赔笑敬酒，背地里冷汗已经湿透衬衫。」 |
| 家族内斗、荒诞操作、弄巧成拙 | **搞笑 / 讽刺** | 适度调侃但不毁剧；「赵家这步棋，下完才发现坑的是自己。」 |
| 证据链收网、真相大白、正义落地 | **严肃 / 释然 / 升华** | 沉下来，留余味；「这一天，他们等了太久。」 |
| 无辜者受害、牺牲、告别 | **沉重 / 惋惜** | 放慢语速感，少用感叹号；「有些名字，从此只能活在档案里。」 |
| 悬念未解、内鬼成谜、下一集钩子 | **悬疑 / 压低** | 留半句；「但他没注意到，电话那头的人，嘴角已经上扬。」 |

**情绪控制原则：**
- 一段 OST=0 **只主打一种主情绪**，不要又哭又笑又骂
- 情绪服务于**剧情推进**，不为煽情而煽情
- 用词口语化、有画面感，像**老剪辑师在跟观众聊这集**，不是念新闻稿
- 每段 ${narration_chars_min}–${narration_chars_max} 字内完成「信息 + 情绪 + 钩子」

### 原声 OST=1：只留「非剪不可」的三类 moment
1. **爆点对峙**：警匪当面交锋、卧底险些暴露、赵家内部摊牌、枪口顶额
2. **信息炸弹**：一句台词翻转前情（身份揭晓、内鬼暗示、证据甩脸）
3. **情绪顶满**：沉默后爆发、摔杯离席、红眼咬牙等**表演不可替代**的秒

解释性对白、重复信息、情绪已由解说说透的段落——**不要整段原声硬播**。

### 全剧视角下的本集结构
- **开场 OST=0**（可至 ${opening_chars_max} 字）：用**最强情绪**抓住人——愤怒控诉、冷峻悬念、或一句反讽，点明本集核心矛盾
- **中段**：原声组（2–3 段 OST=1）+ 解说过渡（OST=0 换情绪、换视角、换线）
- **赵家线 / 警方线 / 卧底线** 三线交替；每换一线，解说先「立旗」再进原声
- **集末 OST=0**：悬念或升华，情绪收束或再推高，引向下一冲突

### 节奏与禁忌
- 避免连续 4 段以上纯原声——观众需要解说的**情绪引导**和**脉络梳理**
- 禁止解说复述刚播完的原台词
- 禁止 OST=1 只框 1–3 秒单句
- 禁止全片一种语调念到底（全程严肃或全程吐槽都不合格）
- `picture` 字段写画面/神情/动作，与解说情绪一致（如「胡队怒目圆睁，一字一句硬刚」）
"""

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
        "subtitle": "二十年剪辑大师 · 全剧视角 · 情绪化解说",
        "description": (
            "以《罚罪2》为母题：纵观全剧、逐集深度刨析，解说带愤怒/严肃/搞笑/悬疑等情绪，"
            "随剧情切换；原声只留爆点对峙与信息炸弹。"
        ),
        "default_work_name": "罚罪2",
        "editor_persona": (
            "你是一位**有着二十年资深经验的剪辑大师**，纵览《罚罪2》全剧脉络，"
            "对每一集做详细分析与深度刨析，再动手剪出优秀的解说成片。"
            "你既是拉片者也是 storyteller：原声留给最炸的 moment，"
            "解说则负责串线、立人、立局，并**随剧情注入情绪**——该愤怒时义愤填膺，"
            "该严肃时沉得住气，该讽刺时一针见血，该悬疑时留半句让人睡不着觉。"
            "你绝不用一种语调念完全片，也绝不拿长对白堆原声时长。"
        ),
        "style_directive": _FAZU2_STYLE_DIRECTIVE,
        "settings": {
            "target_duration_percent": 30,
            "ost1_duration_min": 5,
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
            "opening_chars_max": 110,
        },
    },
}


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


def get_film_tv_preset(preset_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not preset_id:
        return deepcopy(_PRESETS.get(DEFAULT_PRESET_ID))
    return deepcopy(_PRESETS.get(preset_id))


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
) -> Dict[str, Any]:
    """将方案数值与元数据合并进 settings。"""
    merged = deepcopy(settings) if settings else {}
    preset = get_film_tv_preset(preset_id or merged.get("preset_id") or DEFAULT_PRESET_ID)
    if not preset:
        preset = get_film_tv_preset(DEFAULT_PRESET_ID)
    assert preset is not None

    merged["preset_id"] = preset["id"]
    merged["preset_name"] = preset["name"]
    merged["editor_persona"] = preset["editor_persona"]
    merged["style_directive"] = preset["style_directive"]
    for key in NUMERIC_SETTING_KEYS:
        if key in preset["settings"]:
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
