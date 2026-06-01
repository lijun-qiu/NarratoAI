#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
智能混剪解说专用脚本生成提示词：成片规则随上传视频时长动态计算。
"""

from ..base import ParameterizedPrompt, PromptMetadata, ModelType, OutputFormat


class ScriptGenerationEnhancedPrompt(ParameterizedPrompt):
    """智能混剪解说脚本 — 时长规则由上传视频动态推导"""

    def __init__(self):
        metadata = PromptMetadata(
            name="script_generation_enhanced",
            category="short_drama_narration",
            version="v1.2",
            description="智能混剪解说：根据原片秒数动态计算成片目标时长与片段数量",
            model_type=ModelType.TEXT,
            output_format=OutputFormat.JSON,
            tags=["智能混剪", "压缩混剪", "动态时长", "解说脚本"],
            parameters=[
                "drama_name",
                "plot_analysis",
                "subtitle_content",
                "video_duration",
                "video_duration_sec",
                "min_duration",
                "target_duration",
                "max_duration",
                "min_segment_count",
                "max_segment_count",
                "narration_span_range",
                "ost_span_range",
                "duration_plan_summary",
                "source_timeline_pick_ratio",
            ],
        )
        super().__init__(
            metadata,
            required_parameters=[
                "drama_name",
                "plot_analysis",
                "video_duration",
                "target_duration",
                "min_duration",
            ],
        )
        self._system_prompt = (
            "你是智能混剪解说导演。你的任务是把长原片压缩成约一半时长的混剪成片："
            "跳跃选取关键情节，每段解说文案 20 字以内、精简准确，严格输出 JSON，"
            "不得包含任何说明或代码块。"
        )

    def get_template(self) -> str:
        return """# 智能混剪解说脚本（压缩混剪模式）

## 时长计划（已根据上传视频自动计算，必须遵守）
${duration_plan_summary}

| 项目 | 数值 |
|------|------|
| 上传原视频时长 | **${video_duration}**（${video_duration_sec} 秒） |
| 目标成片播放时长 | **${target_duration}**（约为原片 50%） |
| 成片最短 | **≥ ${min_duration}** |
| 成片最长（硬上限，不得超过） | **≤ ${max_duration}** |
| 建议片段数量 | **${min_segment_count}–${max_segment_count}** 个 |
| 原片时间轴实际取用 | **${source_timeline_pick_ratio}**（其余段落跳过） |
| 解说段单段跨度 | **${narration_span_range}** 秒 |
| 原声段单段跨度 | **${ost_span_range}** 秒 |

## 核心目标（压缩，不是重播全片）
为《${drama_name}》生成混剪脚本，使**最终成片播放时长**：
- **尽量接近 ${target_duration}**
- **不得超过 ${max_duration}**（超过即失败，必须删段/缩短跨度后重写）
- 禁止把原片几乎从头到尾连续剪进去（那会导致成片≈原片全长）

## 素材

### 剧情概述
<plot>
${plot_analysis}
</plot>

### 原片字幕（只能复制其中时间戳）
<subtitles>
${subtitle_content}
</subtitles>

## 成片时长怎么算（生成前必须自检）
- **原声 OST=1**：播放时长 = 该段 `timestamp` 跨度
- **解说 OST=0**：播放时长 ≈ max(TTS 朗读时长, 该段 `timestamp` 跨度)
- **预计成片** = 以上所有片段播放时长之和
- **目标**：预计成片在 **${min_duration}–${target_duration}** 之间，且 **≤ ${max_duration}**
- **错误示范**：26 段、每段 15–20 秒、时间轴从 00:00 连续排到片尾 → 成片约 6 分钟（过长）

## 选段策略（跳跃压缩，不是线性全覆盖）
- 只保留**推动主线**的关键情节，删去重复对话、过渡、无关细节
- 片头、中段、片尾都要有代表段，但**段与段之间应跳过大量原片时间**（原片时间轴取用 ${source_timeline_pick_ratio}）
- 禁止 90% 片段挤在原片前 1/3；也禁止从 00:00 到片尾几乎连续选段
- timestamp 按时间顺序排列，**不得重叠**

## 片段结构（解说 : 原声 ≈ 4 : 6）
- 全片 **OST=0 与 OST=1 段数比例** 约为 **4:6**（原声占多数）
- 交替节奏参考：**每 2 段解说后接 3 段原声**，循环往复
- **解说 OST=0**：
  - 文案 **20 字以内**（含标点），**精简、准确、不废话**，一句说清一个信息点
  - 禁止超过 20 字；禁止堆砌形容词、禁止复述原片台词、禁止空洞抒情
  - `timestamp` 跨度 **${narration_span_range}** 秒，与文案朗读时长匹配，禁止写过长跨度
- **原声 OST=1**：
  - `narration` 为「播放原片+序号」
  - 跨度 **${ost_span_range}** 秒，只保留最精彩一句对白，禁止整段长对话

## 解说文案要求
- **短**：每段 **≤ 20 字**（含标点），超出必须删改
- **准**：只写字幕/剧情中有的信息，不臆造
- **狠**：直接点明冲突、转折、人物关系，少用「然而」「此时」等套话

## 时间戳规则
- 格式 `HH:MM:SS,mmm-HH:MM:SS,mmm`
- 时间只能从字幕复制，禁止编造
- 范围 `00:00:00,000` ～ **${video_duration}**

## 输出 JSON
{
  "items": [
    {"_id": 1, "timestamp": "...", "picture": "...", "narration": "...", "OST": 0},
    {"_id": 2, "timestamp": "...", "picture": "...", "narration": "播放原片2", "OST": 1}
  ]
}

## 生成前最后检查
1. 预计成片是否在 **${min_duration}–${max_duration}**？（超过 ${max_duration} 必须删段缩短）
2. 是否跳跃选段，而非连续覆盖全片时间轴？
3. 片段数是否在 **${min_segment_count}–${max_segment_count}**？
4. OST=0 与 OST=1 段数比例是否接近 **4:6**？
5. 每段解说文案是否 **≤ 20 字**、精简准确？
6. 是否只输出 JSON？

现在请为《${drama_name}》生成脚本："""
