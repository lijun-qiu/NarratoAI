#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project: 短剧解说-文案画面匹配
@File   : script_generation.py
@Author : viccy同学
@Date   : 2025/1/7
@Description: 短剧解说脚本生成提示词 - 优化版本
"""

from ..base import ParameterizedPrompt, PromptMetadata, ModelType, OutputFormat
from app.services.documentary.video_episode_segment_schedule import segment_policy_summary


class ScriptGenerationPrompt(ParameterizedPrompt):
    """短剧解说脚本生成提示词 - 优化版本"""

    def __init__(self):
        metadata = PromptMetadata(
            name="script_generation",
            category="short_drama_narration",
            version="v2.7",
            description="短剧解说：场景分段蓝图×视频分析×字幕、开篇爆燃单原声、旁白贴合",
            model_type=ModelType.TEXT,
            output_format=OutputFormat.JSON,
            tags=["短剧", "解说脚本", "文案生成", "原声片段", "黄金开场", "爽点放大", "个性吐槽", "悬念预埋"],
            parameters=[
                "drama_name",
                "plot_analysis",
                "subtitle_content",
                "subtitle_frame_analysis",
                "video_episode_analysis",
                "picture_narration_max_chars",
                "output_duration_hint",
                "narration_percent",
                "original_audio_percent",
                "target_output_minutes_min",
                "target_output_minutes_max",
                "narration_chars_min",
                "narration_chars_max",
                "max_consecutive_ost1",
                "ost1_duration_min",
                "ost1_duration_max",
                "ost1_max_segments",
                "ost1_max_segments_rule",
                "ost0_duration_min",
                "max_ershi_per_script",
            ]
        )
        super().__init__(metadata, required_parameters=["drama_name", "plot_analysis"])
        
        self._system_prompt = (
            "你是一位顶级影视解说剪辑师，风格以「解说旁白驱动叙事」为主，"
            "原片对白仅作≤5秒的情绪点缀。每句旁白须升华信息或情绪；"
            "picture 须写可执行的景别+画面。输出严格 JSON，禁止任何多余文字。"
        )
        
    def get_template(self) -> str:
        grid_header = f"### 整片视频分析（{segment_policy_summary()} · 画面/旁白/环境）"
        template = """# 短剧解说脚本创作任务

## 任务目标
我是一位专业的短剧解说up主，需要为短剧《${drama_name}》创作一份高质量的解说脚本。目标是让观众在短时间内了解剧情精华，并产生强烈的继续观看欲望。

## 素材信息

### 场景分段蓝图（已确认 · 最高优先级）
<plot>
${plot_analysis}
</plot>

### __VIDEO_GRID_HEADER__
<video_episode_analysis>
${video_episode_analysis}
</video_episode_analysis>

### 原始字幕（含精确时间戳）
<subtitles>
${subtitle_content}
</subtitles>

### 蓝图执行说明
<blueprint_execution>
${subtitle_frame_analysis}
</blueprint_execution>

**素材优先级（硬性）：场景分段蓝图 > SRT 时间戳 > 整片视频分析画面**

### 核心风格（硬性 · 解说为主）
- **默认 OST=0 解说旁白**推进全片；用旁白**概括**对话与动作，不要让观众长时间听原片
- **OST=1 原声**作**短促点缀**：${ost1_max_segments_rule}，每段 **≤${ost1_duration_max} 秒**，按**场景**在情绪顶点保留
- **场景配对**：同一爆燃场景先 **OST=0 解说铺垫** → 再 **OST=1 原声金句**（形成「解说—原声」对应）；能概括的一律写进 OST=0
- 成片时长占比：解说 **≈${narration_percent}%**，原声 **≈${original_audio_percent}%**

### 片段裁切
- 蓝图场景 **时间窗** = OST=0 取画范围；**关键对白**若保留原声，只截取**最短金句**（≤${ost1_duration_max} 秒）
- **禁止**连续多段 OST=1（`max_consecutive_ost1=${max_consecutive_ost1}`）；原声后必须用解说承接
- OST=0 取画 timestamp 跨度须 **≥ 解说 TTS 估算时长**

### 开篇结构
- **第 1 段 OST=1 ≤${ost1_duration_max} 秒**：开篇爆燃原声（可倒叙片尾高潮），`picture`/`timestamp`/`original_line` 必须同一镜头
- **第 2 段 OST=0**：`narration` 必须以「宝子们，我们开始看${drama_name}。」开头
- **开头禁止连放两段 OST=1**：播放顺序前 3 段内最多 1 段原声
- **正叙走到同场景时**：可再插入同一片段的 OST=1 复现（与第 1 段 timestamp 相同），实现「先钩子、后解说、再原声对应」

### timestamp 硬性约束
- 格式固定：`HH:MM:SS,mmm-HH:MM:SS,mmm`（**结束时间必须严格大于开始时间**）
- **内容对位（通用）**：写到哪段剧情，`timestamp` 就必须来自该段在 **SRT 字幕** 或 **蓝图场景时间窗** 中的位置
- OST=1：`original_line` 须在 SRT 中定位到同句起止；`picture`/`original_line`/`timestamp` 必须同场景
- OST=0：取画范围须落在对应蓝图场景 `timestamp_ranges` 内
- **禁止零时长**：不得出现起止相同，如 `00:01:02,000-00:01:02,000`
- OST=1 每段 **${ost1_duration_min}–${ost1_duration_max} 秒**，只框**最短金句**，禁止长对白原声
- **原片时间轴**：各段 `timestamp` **禁止重叠**；正叙推进时前后段尽量首尾相接（允许 ≤1 秒过渡）
- **`_id` = 成片播放顺序**；`timestamp` 可倒叙/闪回，解说须帮观众理清时间

### 播放顺序 vs 原片时间
- 段落 `_id` 决定观众观看顺序；各段 `timestamp` 可跳跃
- 正叙时：后段开始时间 ≥ 前段结束时间（同线叙事尽量不跳剪）
- 倒叙/闪回：用「故事，得从头讲起。」「画面切回三天前——」等解说点明

### 成片时长
${output_duration_hint}

**叙事节奏**：按蓝图场景顺序，**以 OST=0 串联**；原声只在情绪顶点「点一下」。

## 解说文案（硬性 · 每句须有新信息或升华情绪）

**禁止空泛复述画面：**
- ❌ 「而这时，秦枫勇斗歹徒，激烈搏斗。」
- ❌ 「随后，特写：楚青桐神情严肃，语气沉重。」（只是在念 picture）
- ✅ 「秦枫一个人冲进窝点，与多名歹徒肉搏。他身手矫健，但拳拳到肉的打斗也暗示着，这绝不是普通的偷狗案。」

**转折词多样化**（全片「而这时」不超过 ${max_ershi_per_script} 次）：
- 可用：随后、另一边、与此同时、更让人揪心的是、紧接着、镜头一转、谁也没想到、偏偏在这时

每段 OST=0：**${narration_chars_min}–${narration_chars_max} 字**；提供新信息、因果、动机或情绪升华，禁止流水账。

## 画面说明 picture（硬性 · 可执行）

**每条 item 的 `picture` 须写清景别 + 主体 + 动作/神态 + 环境**，供剪辑直接取画。示例：
- `特写：胡小跃站在楼顶边缘，警服被风吹起，他闭眼后仰`
- `中景：叶天佑在天台握紧拳头，眼神愤怒`
- `航拍：龙湾村祠堂广场，人群聚集，黑轿车驶入`

OST=1 的 picture 另可写 ≤${picture_narration_max_chars} 字烧录旁白（双引号包裹）；OST=0 的 picture 写完整取画说明（不限于 ${picture_narration_max_chars} 字）。
- **地点**：素材未写明归属时写「室内·家中」，禁止臆测「罗博家中」「秦枫家中」等

## 原声 OST=1 使用规范（按场景 · 解说后播放）

- ${ost1_max_segments_rule} OST=1，每段 **≤${ost1_duration_max} 秒**
- **开篇**：播放顺序第 1 段可为 OST=1 爆燃钩子；**禁止**第 1、2 段连续 OST=1
- **中段爆燃**：须 **OST=0 解说铺垫 → OST=1 原声金句** 成对出现（按蓝图场景取舍，非每场景都要）
- `timestamp`、`picture`、`original_line` 必须同镜；`narration` 固定「播放原片+序号」

## 开篇与收尾

| 段落 | OST | 要求 |
|------|-----|------|
| **第 1 段** | **1** | 开篇爆燃原声 ≤${ost1_duration_max} 秒；picture/timestamp/original_line 同镜 |
| **第 2 段** | **0** | 「宝子们，我们开始看${drama_name}。」开头 |
| **中段** | **0→1 配对** | 按场景：解说铺垫后接 ≤${ost1_duration_max} 秒原声（非每场景都要） |
| **末段** | **0** | 含「宝子们，我们下期再见！」 |

## 创作要点（精简）
- 主线提炼：舍弃支线，每段解说推进因果或冲突
- 上帝视角：点破动机、预判后果，禁止复述画面
- 悬念：OST=0 埋伏 → 可选 ≤5 秒 OST=1 金句 → OST=0 点评

## 时间戳（绝对不能违反）
- 格式：`HH:MM:SS,mmm-HH:MM:SS,mmm`；**禁止零时长**
- **禁止任何两段 timestamp 在原片时间轴上重叠**
- 正叙段落：后段开始 ≥ 前段结束（允许 ≤1 秒黑场过渡）
- 所有 timestamp 须来自 SRT / 蓝图场景时间窗

## 输出格式（JSON only）

{
  "items": [
    {
        "_id": 1,
        "timestamp": "00:19:51,659-00:19:55,659",
        "picture": "\"胡小跃天台对峙，语气决绝\"",
        "narration": "播放原片1",
        "original_line": "「我死了，你一定会给我陪葬。」",
        "OST": 1
    },
    {
        "_id": 2,
        "timestamp": "00:00:01,000-00:00:18,000",
        "picture": "中景：警局走廊，胡小跃与叶天佑并肩前行，气氛压抑",
        "narration": "宝子们，我们开始看${drama_name}。故事，得从头讲起。胡小跃本是刑警骨干，却因追查真相被内部盯上……",
        "OST": 0
    },
    {
        "_id": 3,
        "timestamp": "00:05:22,100-00:05:35,000",
        "picture": "中景：审讯室，胡小跃拍案而起，秦枫震惊",
        "narration": "胡小跃终于爆发了。他历数金鼎集团马金和罗博的罪行：二十八起恶性案件，五人失踪，一人死亡。",
        "OST": 0
    },
    {
        "_id": 4,
        "timestamp": "00:10:23,000-00:10:28,000",
        "picture": "\"胡小跃拍案控诉，情绪爆发\"",
        "narration": "播放原片4",
        "original_line": "「二十八起恶性案件！」",
        "OST": 1
    },
    {
        "_id": 5,
        "timestamp": "00:45:00,000-00:46:30,000",
        "picture": "航拍：龙湾村祠堂广场，人群聚集，黑轿车驶入",
        "narration": "胡队这一局还没赢，更大的风暴在后头。宝子们，我们下期再见！",
        "OST": 0
    }
  ]
}

## 输出前自检
- [ ] 播放顺序**前 3 段内最多 1 段** OST=1；${ost1_max_segments_rule}，每段 **≤${ost1_duration_max} 秒**
- [ ] 中段 OST=1 前一段为 OST=0 解说铺垫（场景配对）
- [ ] 第 1 段 `picture`/`timestamp`/`original_line` 同镜对齐
- [ ] 成片解说:原声时长约 **${narration_percent}:${original_audio_percent}**
- [ ] 第 2 段以「宝子们，我们开始看${drama_name}。」开头；末段含「宝子们，我们下期再见！」
- [ ] 每段 OST=0 提供新信息/升华情绪；「而这时」全片 **≤${max_ershi_per_script} 次**
- [ ] 每条 `picture` 含景别（特写/中景/全景/航拍等）+ 可执行画面描述
- [ ] **无 timestamp 重叠**；正叙段尽量首尾相接
- [ ] 只输出 JSON，无 Markdown 包裹

现在请为短剧《${drama_name}》创作解说脚本："""
        return template.replace("### __VIDEO_GRID_HEADER__", grid_header)
