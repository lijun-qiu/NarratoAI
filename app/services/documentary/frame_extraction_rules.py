#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""视频解说抽帧 JSON 规则：以资深剪辑师视角规范 scene_segments / frame_observations。"""

from __future__ import annotations

from typing import Any, Optional

from app.services.documentary.documentary_settings import get_documentary_settings

# scene_segments 核心六字段 + 剪辑师扩展字段（可选，供 OST 选材与 picture 写稿）
SCENE_SEGMENT_CORE_FIELDS = (
    "timestamp",
    "scene",
    "observation",
    "action",
    "emotion",
    "key_visual",
)

SCENE_SEGMENT_EDITOR_FIELDS = (
    "shot_scale",
    "lighting_time",
    "edit_role",
    "audio_cue",
    "importance",
)

SCENE_SEGMENT_ALL_FIELDS = SCENE_SEGMENT_CORE_FIELDS + SCENE_SEGMENT_EDITOR_FIELDS

SCENE_SEGMENT_FIELD_COMMENTS: dict[str, str] = {
    "timestamp": "剪辑区间 HH:MM:SS,mmm-HH:MM:SS,mmm，须落在本批次真实时间轴内",
    "scene": "唯一场景地点（如楼顶天台、审讯室）；必填，禁止空",
    "observation": "本段画面一句话：谁+在哪+做什么+氛围（30–80字；不复述对白）",
    "action": "可见动作与站位；人物须带性别，如胡小跃(男)握拳逼近",
    "emotion": "画面情绪张力（压抑、爆发、冷峻、悲怆等）",
    "key_visual": "光线/色调/景别/构图/运镜（如「阴天冷调中景，对称构图，低角度仰拍」）",
    "shot_scale": "景别：远景/全景/中景/近景/特写/大特写",
    "lighting_time": "昼夜与光线：日/夜/晨/昏/室内/室外+暖/冷/高反差等",
    "edit_role": "剪辑用途：定场/对话/反应/动作/过渡/空镜/高潮/线索/闪回",
    "audio_cue": "可保留原声提示：对白高潮/环境音/静默/爆炸等",
    "importance": "剪辑优先级：高（OST=1候选）/中/低",
}

SHOT_SCALE_VALUES = ("远景", "全景", "中景", "近景", "特写", "大特写")
EDIT_ROLE_VALUES = (
    "定场",
    "对话",
    "反应",
    "动作",
    "过渡",
    "空镜",
    "高潮",
    "线索",
    "闪回",
)


def resolve_frame_max_segment_duration_sec(settings: dict | None = None) -> int:
    cfg = settings or get_documentary_settings()
    return max(5, int(cfg.get("frame_max_segment_duration_sec", 30) or 30))


def build_frame_editor_role_preamble() -> str:
    return (
        "你是一位拥有 **30 年经验** 的资深影视剪辑师，同时担任短视频解说项目的 **素材策划**。"
        "你的任务不是写影评，而是为后期剪辑师留下**可裁、可用、可配音**的帧级素材账本："
        "每一帧都要回答——「这一刀切在哪」「画面里有什么」「情绪到哪了」「能不能当原声高潮」。"
    )


def build_frame_reading_workflow_rules(*, frame_count: int) -> str:
    return f"""## 逐帧阅读工作流（硬性 · 先帧后段）

1. **先逐帧扫读**全部 {frame_count} 张图：记录景别变化、人物出入画、光线是否跳变、是否有硬字幕
2. **再归纳 scene_segments**：仅当「地点 / 主场景 / 主事件 / 光线」发生实质变化时才新开一条
3. **同批同景合并**：连续帧同一地点、同一组人物、同一动作链 → **1 条** segment
4. **跨景必拆**：硬切、转场、换场、时间跳跃 → 必须拆成多条，timestamp 不得重叠
5. **声画思维**：想象解说员下一镜需要什么——定场、反应、动作、空镜、高潮各就各位
6. **禁止脑补**：画面未出现的人、地点、事件、航拍、牺牲、追车一律不写"""


def build_frame_scene_segment_spec(*, max_seg_sec: int) -> str:
    fields = "\n".join(
        f"- **{name}**：{SCENE_SEGMENT_FIELD_COMMENTS[name]}"
        for name in SCENE_SEGMENT_ALL_FIELDS
    )
    return f"""## scene_segments 字段规范（每条 segment）

{fields}

**硬性约束**：
- 核心六字段 **必填**；扩展五字段（shot_scale / lighting_time / edit_role / audio_cue / importance）**强烈建议填写**
- 单条 segment 时长 **≤ {max_seg_sec} 秒**；超长须按场景或动作节点拆分
- **scene 禁止留空**；同一批次内 timestamp **不得重叠**
- observation **禁止复述对白**（对白放 subtitle / 硬字幕）；action 写「看得见」的肢体与走位
- 可作 OST=1 的对白/冲突场面：`importance` 标「高」，`edit_role` 标「高潮/对话」，`audio_cue` 写关键词
- key_visual 须含 **至少两项**：光线 + 景别或构图（例：「夜间暖灯近景，浅景深，人物占画面左侧三分线」）"""


def build_frame_observation_spec(*, frame_count: int) -> str:
    return f"""## frame_observations 逐帧规范（必须 {frame_count} 条 · 与输入帧一一对应）

每条对应一帧，按时间顺序输出，**不得遗漏**：
- **timestamp**：该帧时间 `HH:MM:SS,mmm`（取自文件名或批次时间轴）
- **observation**：15–40 字，格式建议「[景别] 地点，可见人物(性别)+动作，光线关键词」
  - 例：「[特写] 审讯室，胡小跃(男)拍桌，顶光硬阴影」
- 若本帧有硬字幕：填写 burned_in_subtitle / has_burned_in_subtitle（见硬字幕规则）
- 逐帧只写**这一帧**可见内容；相邻帧若画面相同，仍须分别描述细微变化（表情、手势、字幕出现）"""


def build_frame_timestamp_rules() -> str:
    return """## timestamp 规则（剪辑师视角 · 硬性）

- 格式：`HH:MM:SS,mmm-HH:MM:SS,mmm`（scene_segments）或 `HH:MM:SS,mmm`（单帧）
- **对齐字幕**：起止须落本批次 SRT/硬字幕对白时段附近，方便 OST=1 精确下刀
- **切点思维**：segment 起点宜在「动作开始前 0.5–1 秒」，终点宜在「反应/台词句号后 0.5 秒」
- **禁止**编造批次外时间；**禁止**同一批次重叠区间"""


def build_frame_extraction_json_skeleton(
    *,
    frame_count: int,
    burned_in_subtitle_example: str = "",
) -> str:
    burned = burned_in_subtitle_example or ""
    return f"""## 输出 JSON 结构（只返回 JSON，不要 markdown 包裹）

```json
{{
  "scene_segments": [
    {{
      "timestamp": "00:00:01,940-00:00:09,940",
      "scene": "楼顶天台",
      "observation": "阴天，两名男子并肩立于天台边缘交谈，气氛严肃压抑",
      "action": "叶天佑(男)与未名人员(男)面向城市远景站立对话",
      "emotion": "严肃、压抑",
      "key_visual": "阴天冷色调中景，对称构图，云层低垂与城市远景",
      "shot_scale": "中景",
      "lighting_time": "日/室外/冷调",
      "edit_role": "对话",
      "audio_cue": "对白交锋",
      "importance": "中"
    }}
  ],
  "frame_observations": [
    {{"timestamp": "00:00:00,000", "observation": "[全景] 楼顶天台，两人入画，阴天冷调{burned}"}}
  ],
  "overall_activity_summary": "本批次：天台双人密谈，由全景推至中近景，气氛持续压抑"
}}
```

- scene_segments：至少 1 条
- frame_observations：**必须** {frame_count} 条
- overall_activity_summary：1–2 句，供批次索引与后期快速定位"""


def build_frame_extraction_prompt_body(
    *,
    frame_count: int,
    burned_in_subtitle_example: str = "",
    settings: dict | None = None,
) -> str:
    """组装视觉模型主 prompt（不含批次字幕摘录等动态补充）。"""
    cfg = settings or get_documentary_settings()
    max_seg_sec = resolve_frame_max_segment_duration_sec(cfg)
    sections = [
        build_frame_editor_role_preamble(),
        build_frame_reading_workflow_rules(frame_count=frame_count),
        build_frame_timestamp_rules(),
        build_frame_scene_segment_spec(max_seg_sec=max_seg_sec),
        build_frame_observation_spec(frame_count=frame_count),
        build_frame_extraction_json_skeleton(
            frame_count=frame_count,
            burned_in_subtitle_example=burned_in_subtitle_example,
        ),
        "请只返回 JSON 字符串，不要附加解释文字。",
    ]
    return "\n\n".join(sections)


def slim_scene_segment_editor_fields(segment: dict[str, Any]) -> dict[str, str]:
    """提取并规范化剪辑师扩展字段。"""
    slim: dict[str, str] = {}
    for key in SCENE_SEGMENT_EDITOR_FIELDS:
        value = str(segment.get(key) or "").strip()
        if not value:
            continue
        if key == "shot_scale" and value not in SHOT_SCALE_VALUES:
            for candidate in SHOT_SCALE_VALUES:
                if candidate in value:
                    value = candidate
                    break
        slim[key] = value
    return slim


def enrich_scene_segment_from_editor_fields(segment: dict[str, Any]) -> dict[str, Any]:
    """落盘前补全：扩展字段与 key_visual 互相补位。"""
    payload = dict(segment)
    editor = slim_scene_segment_editor_fields(payload)
    for key, value in editor.items():
        payload[key] = value

    key_visual = str(payload.get("key_visual") or "").strip()
    lighting = str(payload.get("lighting_time") or "").strip()
    shot = str(payload.get("shot_scale") or "").strip()

    if lighting and lighting not in key_visual:
        key_visual = f"{lighting}，{key_visual}" if key_visual else lighting
    if shot and shot not in key_visual:
        prefix = f"{shot}"
        key_visual = f"{prefix}，{key_visual}" if key_visual else prefix
    if key_visual:
        payload["key_visual"] = key_visual

    if not str(payload.get("importance") or "").strip():
        role = str(payload.get("edit_role") or "")
        emotion = str(payload.get("emotion") or "")
        if role in {"高潮", "动作"} or any(
            word in emotion for word in ("爆发", "激烈", "对峙", "悲怆", "紧张")
        ):
            payload["importance"] = "高"
        else:
            payload["importance"] = "中"
    return payload
