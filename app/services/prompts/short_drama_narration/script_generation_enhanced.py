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
            version="v1.1",
            description="智能混剪解说：根据原片秒数动态计算成片目标时长与片段数量",
            model_type=ModelType.TEXT,
            output_format=OutputFormat.JSON,
            tags=["智能混剪", "长成片", "动态时长", "解说脚本"],
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
            "你是智能混剪解说导演。你必须根据「时长计划」生成**精简**混剪脚本："
            "成片约为原片一半、片段数受限、严禁超时，并严格输出 JSON，不得包含说明或代码块。"
        )

    def get_template(self) -> str:
        return """# 智能混剪解说脚本（动态长成片模式）

## 时长计划（已根据上传视频自动计算，必须遵守）
${duration_plan_summary}

| 项目 | 数值 |
|------|------|
| 上传原视频时长 | **${video_duration}**（${video_duration_sec} 秒） |
| 成片最短时长（硬底线） | **≥ ${min_duration}** |
| 目标成片时长 | **${target_duration}** |
| 成片硬性上限 | **≤ ${max_duration}**（超过无效） |
| 建议片段数量 | **${min_segment_count}–${max_segment_count}** 个 |
| 解说段单段画面跨度 | **${narration_span_range}** 秒 |
| 原声段单段跨度 | **${ost_span_range}** 秒 |

## 核心目标
为《${drama_name}》生成**精简混剪**脚本（不是原片复述）：
- 最终成片播放时长之和：**约 ${target_duration}**（原片约一半）
- **严禁超过 ${max_duration}**（超过即失败）
- 可略低于 ${min_duration}，但**绝不可**接近原片全长
- **片段数不得超过 ${max_segment_count}**，宁可少选精华，不要堆砌过多片段

## 素材

### 剧情概述
<plot>
${plot_analysis}
</plot>

### 原片字幕（只能复制其中时间戳）
<subtitles>
${subtitle_content}
</subtitles>

## 成片时长计算方式（生成前自检）
- **原声 OST=1**：播放时长 = 该段 `timestamp` 跨度（对齐字幕句界，播完整句）
- **解说 OST=0**：播放时长 ≈ max(TTS 朗读时长, 该段 `timestamp` 跨度)
- **自检**：所有片段播放时长之和 **≈ ${target_duration}**，且 **≤ ${max_duration}**（最重要）
- **自检**：`items` 数量 **≤ ${max_segment_count}**（宁少勿多）
- 单段跨度不得超过上表「解说/原声跨度」上限，禁止用超长 timestamp 凑时长

## 时间轴选段（跳剪精华，非全片堆砌）
- 在片头、片中、片尾**各选若干**关键情节即可，**不要**为覆盖全片而增加片段数
- 允许跳过次要过场；各段 timestamp 跨度之和必须 **≤ ${max_duration}**
- 按时间顺序排列，**不得重叠**

## 片段结构（解说 : 原声 ≈ 1 : 1）
- 每 1–2 段解说后接 1 段原声
- **解说 OST=0**：每段文案 **60–120 字**；`timestamp` 跨度 **${narration_span_range}** 秒
- **原声 OST=1**：`narration` 为「播放原片+序号」；跨度 **${ost_span_range}** 秒；禁止句中截断

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
1. 片段数是否 **≤ ${max_segment_count}**？
2. 预计成片总时长是否在 **${min_duration}–${max_duration}** 之间（以 **${target_duration}** 为首选）？
3. 是否避免为「铺满时间轴」而增加无关片段？
4. 是否只输出 JSON？

现在请为《${drama_name}》生成脚本："""
