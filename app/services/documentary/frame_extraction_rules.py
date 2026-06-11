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
    "observation": "本段画面一句话：地点+动作+氛围（30–80字；不复述对白；**不写人名**，人名放 characters）",
    "action": "可见动作与站位（不写人名；谁在场见 characters 字段）",
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
3. **同批同景合并**：连续帧**同一地点、同一动作链**（如整段天台对话）→ 可合并为 1 条 segment
3b. **动作段必拆**：地点变化（停车场↔车顶↔地面）、景别从定场切到追逐、人物从车上到地面奔跑 → **必须**拆成多条 segment，禁止整批压成 1 条
4. **跨景必拆**：硬切、转场、换场、时间跳跃 → 必须拆成多条，timestamp 不得重叠
5. **声画思维**：想象解说员下一镜需要什么——定场、反应、动作、空镜、高潮各就各位
6. **禁止脑补**：画面未出现的人、地点、事件、航拍、牺牲、追车一律不写
7. **禁止**输出解说脚本 JSON（`_id` / `picture` / `narration` / `OST` 数组）；必须输出含 `frame_observations` 与 `scene_segments` 的抽帧分析 JSON"""


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
- **characters**（数组）：本段/本帧画面内可见人物的规范姓名（面孔匹配后填写）；**禁止**在 observation/action/key_visual 中写 `姓名(男/女)`
- 可作 OST=1 的对白/冲突场面：`importance` 标「高」，`edit_role` 标「高潮/对话」，`audio_cue` 写关键词
- key_visual 须含 **至少两项**：光线 + 景别或构图（例：「夜间暖灯近景，浅景深，人物占画面左侧三分线」）"""


def build_frame_timeline_narrative_rules() -> str:
    return """## 时间轴叙事（硬性 · 读 JSON 须能还原「发生了什么」）

- **frame_observations** 是逐秒账本：每一帧写清 **[景别] 地点，人物+动作**；读 sequential 列表应能还原事件顺序
- **scene_segments** 是剪辑段落：按地点/动作阶段拆分，每条 timestamp 须落在本批真实时间轴内
- **overall_activity_summary** 须按时间顺序写事件链（用 → 连接），例：
  「本批次：5:00 停车场车辆行驶 → 5:01–5:03 车顶角色A趴伏 → 5:04 车辆甩尾 → 5:07 角色A持枪奔跑」
- 禁止用一条笼统 segment 覆盖整批多种动作（如车顶追逐 + 地面奔跑 + 车辆甩尾）"""


def build_frame_spatial_accuracy_rules() -> str:
    return """## 空间位置描述（硬性 · 车内/车顶易错）

- **车内 / 车顶 / 车外 / 车旁**须据可见结构判断，**禁止**夜间特写默认写「车内」：
  - **车顶**：可见天空/远景、车身顶面钣金、人物趴/伏/立于车顶、仅有金属顶与护栏、无座椅方向盘
  - **车内**：可见座椅、方向盘、A柱内饰、内后视镜、车窗框从**内侧**看出去
  - **车外/车旁**：人物在地面或车体侧面，可见完整车身侧面/轮胎/停车场地面
- 夜间背景全黑时：若人物贴近车身顶面、呈趴伏/抓握姿态、**无内饰元素** → 写「**车顶**」或「车体上方」，勿写车内
- **拿不准**时写「靠近车辆，位置待辨」或「车顶/车内待辨」，勿臆测
- 同批次相邻帧若已有「停车场/车辆行驶/甩尾」，中间人物特写优先判为**车顶或车外**动作镜，与前后景保持一致
- 追逐/追捕中「趴车/跳车/车顶对峙」与「车内对话/驾驶」是不同 scene，**不得**合并成一条 segment"""


def build_frame_observation_spec(*, frame_count: int) -> str:
    return f"""## frame_observations 逐帧规范（必须 {frame_count} 条 · 与输入帧一一对应）

每条对应**一帧**，按时间顺序输出，**不得遗漏**；**每一帧独立分析**，禁止整批复制同一句：
- **timestamp**：该帧时间 `HH:MM:SS,mmm`（**须与输入帧文件名时间码一致**，勿从 00:00:00 重计）
- **characters**（数组）：本帧画面内可见人物的规范姓名（须逐脸对照定妆照匹配）；无匹配则不写该项
- **observation**：15–40 字，格式建议「[景别] 地点，动作/姿态，光线关键词」——**禁止写人名**
  - 例：「[特写] 审讯室，拍桌质问，顶光硬阴影」，characters: ["角色A"]
  - 例：「[特写] 车顶，趴伏抓边，夜/室外/冷调」，characters: ["角色B"]
- 若本帧有硬字幕：另填 JSON 字段 `burned_in_subtitle` / `has_burned_in_subtitle`（**不要**写进 observation 字符串里）
- **硬字幕 ≠ 本帧说话人**：仅复制文字；谁说话须看本帧嘴型/手势，反应镜/聆听镜勿标「开口说话」
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

**禁止**输出解说脚本片段（`_id`、`picture`、`narration`、`OST` 数组）。必须是下方抽帧结构，且 `frame_observations` 条数 = {frame_count}。

```json
{{
  "scene_segments": [
    {{
      "timestamp": "00:05:00,000-00:05:03,000",
      "scene": "废弃停车场车顶",
      "characters": ["角色A"],
      "observation": "夜间，趴伏于行驶中的车顶，神情紧张",
      "action": "抓握车顶边缘，车辆高速行驶",
      "emotion": "紧张",
      "key_visual": "夜/室外/冷调，特写，动态模糊",
      "shot_scale": "特写",
      "lighting_time": "夜/室外/冷调",
      "edit_role": "动作",
      "audio_cue": "风噪与引擎",
      "importance": "高"
    }},
    {{
      "timestamp": "00:05:04,000-00:05:09,000",
      "scene": "废弃停车场",
      "characters": ["角色A"],
      "observation": "车辆甩尾后，持枪在停车场内奔跑追捕",
      "action": "持枪奔跑，后方有人跟随",
      "emotion": "紧迫",
      "key_visual": "夜/室外/冷调，中景，低角度跟拍",
      "shot_scale": "中景",
      "lighting_time": "夜/室外/冷调",
      "edit_role": "动作",
      "importance": "高"
    }}
  ],
  "frame_observations": [
    {{"timestamp": "00:05:00,000", "observation": "[远景] 废弃停车场，汽车行驶，夜间暗调{burned}"}},
    {{"timestamp": "00:05:02,000", "characters": ["角色A"], "observation": "[特写] 车顶，侧脸凝重{burned}"}}
  ],
  "overall_activity_summary": "本批次：5:00 停车场车辆行驶 → 5:01–5:03 车顶角色A趴伏 → 5:04 车辆甩尾 → 5:07 角色A持枪奔跑"
}}
```

- scene_segments：至少 1 条
- frame_observations：**必须** {frame_count} 条
- overall_activity_summary：1–2 句，供批次索引与后期快速定位"""


def build_frame_slim_role_preamble() -> str:
    return (
        "你是一位资深的影视素材分析师。你的任务是为**视频分析阶段**建立每秒视觉索引："
        "对照定妆照**逐帧识别人物**、记录场景与硬字幕 OCR；"
        "人物姓名**仅**来自面孔与定妆照对比，**禁止**凭字幕或剧情猜人。"
    )


def build_frame_slim_workflow_rules(*, frame_count: int) -> str:
    return f"""## 精简抽帧工作流（硬性 · 仅 frame_timeline）

1. **逐帧扫读**全部 {frame_count} 张图：头像对比识别人物、记录场景、复制硬字幕
2. **禁止**输出 scene_segments / frame_observations / 解说脚本 JSON
3. **禁止**凭硬字幕、SRT、剧情或关系表猜规范姓名；`characters` **仅**来自本帧面孔与定妆照对照匹配
4. **禁止**在本阶段写 speaker 或「谁在说」；`visual_cue` 只写本帧可见嘴型/姿态/景别线索，**不得**据字幕推断说话人
5. **禁止脑补**画面未出现的人、地点、事件；**禁止**因剧情推断某参照人物「不应入画」而漏标"""


def build_frame_slim_timeline_spec(*, frame_count: int) -> str:
    return f"""## frame_timeline 逐帧规范（必须 {frame_count} 条 · 与输入帧一一对应）

每条对应**一帧**，按时间顺序输出，**不得遗漏**：
- **timestamp**：`HH:MM:SS,mmm`（须与输入帧文件名时间码一致）
- **title**：4–8 字片段标题（如「天台对峙」「停车场追逐」）；同场景连续帧可相同
- **scene**：15–40 字场景描述（地点+动作/氛围；**不写人名**，人名放 characters）
- **characters**（数组）：本帧可见人物规范姓名，**仅**当面孔与定妆照对照匹配（≥75% 相似）；无匹配则留空或用未名人员
- **burned_in_subtitle**：硬字幕原文；无则空字符串；**与 characters 无关**，不得据字幕填人名
- **visual_cue**：本帧可见线索（如「嘴型张开」「静听」「过肩背对镜头」）；**禁止**写规范姓名，**禁止**据字幕推断谁在说话
- 硬字幕仅 OCR 复制，**不得**用于猜 `characters`"""


def build_frame_slim_json_skeleton(
    *,
    frame_count: int,
    burned_in_subtitle_example: str = "",
) -> str:
    burned = burned_in_subtitle_example or ""
    return f"""## 输出 JSON 结构（只返回 JSON，不要 markdown 包裹）

**禁止**输出 scene_segments / frame_observations / 解说脚本。`frame_timeline` 条数必须 = {frame_count}。

```json
{{
  "frame_timeline": [
    {{
      "timestamp": "00:05:00,000",
      "title": "停车场追逐",
      "scene": "废弃停车场，汽车高速行驶，夜间冷调",
      "characters": ["角色A"],
      "burned_in_subtitle": "",
      "visual_cue": "远景定场"
    }},
    {{
      "timestamp": "00:05:02,000",
      "title": "车顶趴伏",
      "scene": "车顶，趴伏抓边，夜/室外/冷调",
      "characters": ["角色A"],
      "burned_in_subtitle": "你以为你赢了吗？",
      "visual_cue": "特写，嘴型未张开，静听"
    }}
  ],
  "overall_activity_summary": "本批次：5:00 停车场行驶 → 5:02 车顶角色A趴伏{burned}"
}}
```

- frame_timeline：**必须** {frame_count} 条
- overall_activity_summary：1–2 句事件链（可选）"""


def build_frame_slim_extraction_prompt_body(
    *,
    frame_count: int,
    burned_in_subtitle_example: str = "",
    settings: dict | None = None,
) -> str:
    """精简模式：仅输出 frame_timeline（供视频分析推断说话人）。"""
    from app.services.documentary.frame_dialogue_alignment import build_frame_dialogue_speaker_rules

    sections = [
        build_frame_slim_role_preamble(),
        build_frame_slim_workflow_rules(frame_count=frame_count),
        build_frame_timestamp_rules(),
        build_frame_spatial_accuracy_rules(),
        build_frame_dialogue_speaker_rules(),
        build_frame_slim_timeline_spec(frame_count=frame_count),
        build_frame_slim_json_skeleton(
            frame_count=frame_count,
            burned_in_subtitle_example=burned_in_subtitle_example,
        ),
        "请只返回 JSON 字符串，不要附加解释文字。",
    ]
    return "\n\n".join(sections)


def build_frame_extraction_prompt_body(
    *,
    frame_count: int,
    burned_in_subtitle_example: str = "",
    settings: dict | None = None,
) -> str:
    """组装视觉模型主 prompt（不含批次字幕摘录等动态补充）。"""
    cfg = settings or get_documentary_settings()
    if cfg.get("frame_slim_output"):
        return build_frame_slim_extraction_prompt_body(
            frame_count=frame_count,
            burned_in_subtitle_example=burned_in_subtitle_example,
            settings=cfg,
        )
    max_seg_sec = resolve_frame_max_segment_duration_sec(cfg)
    from app.services.documentary.frame_dialogue_alignment import build_frame_dialogue_speaker_rules

    sections = [
        build_frame_editor_role_preamble(),
        build_frame_reading_workflow_rules(frame_count=frame_count),
        build_frame_timestamp_rules(),
        build_frame_timeline_narrative_rules(),
        build_frame_spatial_accuracy_rules(),
        build_frame_dialogue_speaker_rules(),
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
