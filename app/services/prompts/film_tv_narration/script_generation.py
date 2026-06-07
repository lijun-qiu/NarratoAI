#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project: NarratoAI
@File   : script_generation.py
@Description: 电影/电视剧解说脚本生成提示词
"""

from ..base import ParameterizedPrompt, PromptMetadata, ModelType, OutputFormat


class ScriptGenerationPrompt(ParameterizedPrompt):
    """电影/电视剧解说脚本生成提示词"""

    def __init__(self):
        metadata = PromptMetadata(
            name="script_generation",
            category="film_tv_narration",
            version="v1.5",
            description="模块化剪辑方案：均衡解说 / 原声燃剪 / 专题规则",
            model_type=ModelType.TEXT,
            output_format=OutputFormat.JSON,
            tags=["电影", "电视剧", "影视解说", "解说脚本", "原声片段", "罚罪2"],
            parameters=["film_name", "plot_analysis", "subtitle_content", "work_brief",
                        "source_duration_minutes", "target_output_minutes", "target_duration_percent",
                        "ost1_duration_min", "ost1_duration_max", "ost1_duration_long_max",
                        "ost1_segment_min", "ost1_segment_max",
                        "ost0_segment_min", "ost0_segment_max",
                        "original_audio_percent", "narration_percent",
                        "narration_chars_min", "narration_chars_max", "opening_chars_max",
                        "total_segment_min", "total_segment_max", "max_total_segments",
                        "min_total_segments",
                        "picture_chars_max",
                        "editing_mode_name", "editor_persona", "style_directive"],
        )
        super().__init__(metadata, required_parameters=["film_name", "plot_analysis"])

        self._system_prompt = (
            "你是一位资深影视剪辑师，根据当前选定的剪辑方案生成 JSON 脚本。"
            "原声 OST=1 播放期间禁止插入解说，解说 OST=0 必须等当前原声段完全结束后再出现。"
            "你必须理解：OST=0 时长由解说字数决定，OST=1 时长由时间戳跨度决定。"
            "你必须严格按照 JSON 格式输出，绝不能包含任何其他文字、说明或代码块标记。"
        )

    def get_template(self) -> str:
        return """# 影视解说脚本创作任务 · 方案「${editing_mode_name}」

## 你的身份
${editor_persona}
即将为《${film_name}》输出可直接交给剪辑引擎执行的 JSON 脚本。
请先回顾作品调研与剧情分析，再动手选段——像拉片一样精准，像预告片一样抓人。

${style_directive}

## 作品背景调研
<work_brief>
${work_brief}
</work_brief>

## 任务目标
为《${film_name}》创作解说脚本：**原声约 ${original_audio_percent}%、解说约 ${narration_percent}%**（按当前方案执行，不要偏离比例目标）。

## ⚠️ 成片时长目标（硬性要求）
- 原片时长约 **${source_duration_minutes} 分钟**
- **成片总时长必须达到约 ${target_output_minutes} 分钟**（约为原片的 ${target_duration_percent}%）
- 生成前请自行估算：所有 OST=1 时间戳跨度之和 + 所有 OST=0 解说 TTS 时长之和 ≈ ${target_output_minutes} 分钟
- **严禁**生成总时长不足 1 分钟的脚本（原声段过短是主要原因）

## 核心比例（成片时长，必须尽量达到）

| 类型 | 目标占比 | 说明 |
|------|----------|------|
| **原声 OST=1** | **约 ${original_audio_percent}%** | 成片的大部分时间播放原片 |
| **解说 OST=0** | **约 ${narration_percent}%** | 仅作短旁白串联 |

**实现方式（在遵守剪辑引擎规则的前提下）：**
- **OST=1** 段 **${ost1_segment_min}–${ost1_segment_max} 段**，每段 **${ost1_duration_min}–${ost1_duration_max} 秒**
- **OST=0** 段 **${ost0_segment_min}–${ost0_segment_max} 段**、每段 **${narration_chars_min}–${narration_chars_max} 字**（开场可至 ${opening_chars_max} 字）

## 素材信息

### 剧情分析
<plot>
${plot_analysis}
</plot>

（剧情分析已融合字幕对白与约每 30 秒一帧的视觉拉片观察；写脚本时请优先采用分析中「原声保留建议」及字幕×画面对照结论。）

### 原始字幕
<subtitles>
${subtitle_content}
</subtitles>

---

## ⚠️ 剪辑引擎规则（必读）

| OST | 成片时长由什么决定 |
|-----|-------------------|
| **0** 解说 | **解说字数 → TTS 长度**（时间戳结束时间**不**控制成片长度） |
| **1** 原声 | **时间戳起止差值**（写多长播多长） |

因此：
- 要**增加原声占比** → 增加 OST=1 段数（**${ost1_segment_min}–${ost1_segment_max} 段**）、拉长每段 OST=1 时间戳（**${ost1_duration_min}–${ost1_duration_max} 秒**）
- 要**减少解说占比** → 减少 OST=0 段数、缩短每段解说（${narration_chars_min}–${narration_chars_max} 字），不要写超过 ${opening_chars_max} 字的长旁白
- **致命错误**：OST=1 只框 1–3 秒的单句台词 → 成片会极短，必须框住完整对话（${ost1_duration_min} 秒以上）

其他硬性规则：
- 所有片段在原片时间轴上**绝对不能重叠**
- OST=1 必须**单独成段**，禁止嵌套在 OST=0 的大时间窗内
- 按 `_id` 顺序，原片时间单调向前

## ⚠️ 原声与解说交替规则（成片播放顺序，必须遵守）

成片按 `_id` **顺序播放**，观众听到的声音如下：
- **OST=1 播放期间**：只有原片声音，**禁止**任何解说配音或解说字幕
- **OST=0 解说**：必须等**前一段 OST=1 当前台词/片段完全说完、播完**后才能开始；禁止半句突然切走

**禁止的排列（原声被解说打断）：**
```
OST=1 → OST=0 → OST=1   ❌ 中间插入的解说了打断原声
```

**正确的排列：**
```
OST=1 → OST=1 → OST=0 → OST=1 → OST=1 → OST=0   ✅ 连续原声播完，再插解说
短解说(铺垫) → 原声 → 原声 → 短解说(转折) → …
```

**执行要点：**
- 同一场对峙/对话的多个 OST=1 段应**连续排列**，中间**不要**夹 OST=0
- OST=0 只出现在**一组连续 OST=1 结束之后**，用作过渡或点评
- **禁止使用 OST=2**（解说+原声混合）；本模式只有 OST=0 和 OST=1

---

## 推荐结构（原声为主）

### 片段数量（参考）
- **OST=1 原声：${ost1_segment_min}–${ost1_segment_max} 段**（高光原声，**严禁超过 ${ost1_segment_max} 段**）
- **OST=0 解说：${ost0_segment_min}–${ost0_segment_max} 段**（串线主力，与 OST=1 **数量相当**，段数不足会导致生成失败）
- **总段数：${min_total_segments}–${max_total_segments} 段**（OST=0+OST=1 合计，低于 ${min_total_segments} 或超过 ${max_total_segments} 均无效）
- 总 `items` 建议 **${min_total_segments}–${max_total_segments} 个**，原声与解说各约一半

### 推荐节奏
```
短解说(铺垫/立人) → 原声 → 原声 → 短解说(过渡) → 原声 → 短解说(转折) → …
```
- 允许 **连续 2–3 段 OST=1**，**整组播完后**再用 OST=0 解说串线
- **禁止**在 OST=1 之间插入 OST=0（原声播放时解说必须等待）
- 当解说占比 ≥ 45% 时：**不要**写成「长原声、短解说」；解说要承担脉络，原声只留高光

### 原声段（OST=1）怎么选
- 完整对白、争吵、威胁、反转、告白、名场面
- 每段包住**一句或多句连贯对话**，跨度 **${ost1_duration_min}–${ost1_duration_max} 秒**
- 重要冲突可给到 **12–18 秒**（仍须来自字幕真实范围）
- `narration` 固定：`播放原片+序号`
- `picture` 字段（OST=1 原声旁白）：**${picture_chars_max} 字以内**，精简承上启下，写画面/神情/动作/氛围，**禁止复述对白、禁止长句**

### 解说段（OST=0）怎么写
- 每段 **${narration_chars_min}–${narration_chars_max} 字**（开场可至 ${opening_chars_max} 字）
- 负责：开场钩子、人物关系、段落过渡、局势点评、结尾悬念
- 禁止复述刚播完的原台词；禁止用超长旁白代替原声（该给原声的高光必须 OST=1）

---

## 时间戳规范

- 格式：`HH:MM:SS,mmm-HH:MM:SS,mmm`
- **严禁重叠**；后段开始 ≥ 前段结束
- **OST=0**：开始时间为画面起点；结束时间仅作参考，跨度建议 **10–25 秒**
- **OST=1**：必须准确框住对白，跨度 **${ost1_duration_min}–${ost1_duration_max} 秒**（核心台词优先取够长度，**禁止 1–3 秒的极短片段**）

---

## 输出格式

只输出 JSON：

{
  "items": [
    {
        "_id": 1,
        "timestamp": "00:00:01,000-00:00:15,000",
        "picture": "开场画面",
        "narration": "停职只是开始！这名硬汉警察因追查真相被组织盯上，接下来这场对峙，将彻底改变一切。",
        "OST": 0
    },
    {
        "_id": 2,
        "timestamp": "00:00:16,150-00:00:28,500",
        "picture": "组织成员围住胡队施压",
        "narration": "播放原片2",
        "OST": 1
    },
    {
        "_id": 3,
        "timestamp": "00:00:30,000-00:00:42,800",
        "picture": "反派摊牌威胁",
        "narration": "播放原片3",
        "OST": 1
    },
    {
        "_id": 4,
        "timestamp": "00:00:45,000-00:01:00,000",
        "picture": "胡队沉默后抬头",
        "narration": "然而胡队没有退让。",
        "OST": 0
    },
    {
        "_id": 5,
        "timestamp": "00:01:02,000-00:01:16,500",
        "picture": "胡队硬刚宣言",
        "narration": "播放原片5",
        "OST": 1
    }
  ]
}

---

## 质量标准

### 必须达到
- 成片时长构成：**原声约 ${original_audio_percent}%，解说约 ${narration_percent}%**
- OST=1：**${ost1_segment_min}–${ost1_segment_max} 段**，每段 **${ost1_duration_min}–${ost1_duration_max} 秒**
- OST=0：**${ost0_segment_min}–${ost0_segment_max} 段**，每段 **${narration_chars_min}–${narration_chars_max} 字**（开场不超过 ${opening_chars_max} 字）
- **硬性下限（不满足则输出无效）**：items 总数 **≥ ${min_total_segments}**（OST=1 **≥ ${ost1_segment_min}** 且 OST=0 **≥ ${ost0_segment_min}**）
- **硬性上限（不满足则输出无效）**：OST=1 **≤ ${ost1_segment_max}**、OST=0 **≤ ${ost0_segment_max}**、items 总数 **≤ ${max_total_segments}**
- **首段 OST=0**：「宝子们」简短招呼后，立刻进入**悬念式剧情解说**（抛冲突、抛反常、抛问题），禁止写「开看之前先捋主线/人物/本集定位」等元叙述
- **末段 OST=0**：必须先**总结本集核心冲突与悬念**，再写道别收束，不可只有「下期再见」

### 输出前自检
- [ ] OST=1 段数 ${ost1_segment_min}–${ost1_segment_max}（**不得超过 ${ost1_segment_max}**）
- [ ] OST=0 段数 ${ost0_segment_min}–${ost0_segment_max}
- [ ] items 总段数 **${min_total_segments}–${max_total_segments}**
- [ ] 所有 OST=1 的 picture 不超过 ${picture_chars_max} 字，短句承上启下
- [ ] 无 OST=0 段超过 ${opening_chars_max} 字
- [ ] 无 OST=1 段短于 ${ost1_duration_min} 秒
- [ ] 估算成片总时长 ≥ ${target_output_minutes} 分钟
- [ ] 原片时间戳无重叠
- [ ] 无「OST=1 → OST=0 → OST=1」打断原声的排列
- [ ] 未使用 OST=2

### 创作原则
1. 只输出 JSON
2. 严格基于剧情与字幕，不虚构
3. **让原片说话，解说只做导游**

现在请为《${film_name}》创作影视解说脚本："""
