#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""抽帧人名：仅定妆照/头像面孔精准匹配可写规范姓名，禁止凭字幕猜人。"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from app.services.documentary.documentary_settings import (
    FRAME_UNKNOWN_CHARACTER_FEMALE,
    FRAME_UNKNOWN_CHARACTER_MALE,
)
from app.services.short_drama_drama_knowledge import (
    ObviousCharacterRelation,
    PLOT_BLUEPRINT_NAME_ALIAS_GROUPS,
    correct_name_mistakes_in_text,
    resolve_obvious_character_relations,
)

_GENDER_SUFFIX_RE = re.compile(r"[\(（][男女不明][\)）]$")
_NAMED_WITH_GENDER_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z·]{2,8})([\(（][男女不明][\)）])"
)
_RELATION_LABEL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "师徒": ("师徒", "师傅", "弟子", "爱徒", "门生"),
    "师兄弟": ("师兄弟", "师兄", "师弟", "同门"),
    "父子": ("父子", "父亲", "儿子", "生父", "老爸", "爸"),
    "母子": ("母子", "母亲", "儿子", "亲妈", "老妈"),
    "养母子": ("养母子", "养母", "养子"),
    "母女": ("母女", "母亲", "女儿"),
    "兄妹": ("兄妹", "哥哥", "妹妹", "兄长", "姐姐", "弟弟"),
    "养兄弟": ("养兄弟", "异姓兄弟"),
    "夫妻": ("夫妻", "丈夫", "妻子", "老公", "老婆"),
    "上下级": ("上下级", "上级", "下级", "领导", "下属"),
    "至交": ("至交", "老友", "旧友"),
}
_RELATION_PRIORITY: dict[str, int] = {
    "师徒": 0,
    "师兄弟": 1,
    "父子": 2,
    "母子": 3,
    "养母子": 4,
    "母女": 5,
    "兄妹": 6,
    "养兄弟": 7,
    "夫妻": 8,
    "上下级": 9,
    "至交": 10,
}
# 附定妆照时禁止用作姓名、须改为头像匹配规范名的模糊代称
_GENERIC_FACE_ROLE_LABELS: tuple[str, ...] = (
    "便衣男警察",
    "便衣男",
    "年轻男子",
    "年轻女子",
    "警服男",
    "警服女",
    "男警官",
    "女警官",
    "男警员",
    "女警员",
    "男警察",
    "女警察",
    "年轻警官",
    "年轻警察",
    "男子",
    "女子",
)
# 旧版解说剧本常用名，不在上传头像名单时应一律剔除（模型易幻觉）
_LEGACY_HALLUCINATED_CHARACTER_NAMES: frozenset[str] = frozenset(
    {"伟业", "老叶", "常征", "赵鹏超", "小跃", "小月"}
)


def build_frame_naming_priority_rules(
    *,
    has_drama_knowledge: bool = False,
    has_character_references: bool = False,
    is_carryover_batch: bool = False,
) -> str:
    """抽帧写人名的优先级：仅面孔匹配，禁止字幕猜人。"""
    lines = [
        "## 人名写入规则（硬性 · 仅头像/定妆照面孔匹配）",
        "1. **唯一依据**：本帧/本批关键帧中**脸/侧脸清晰可见**，且与已上传**定妆照/头像**一致 → **必须**写对应规范姓名 `姓名(男/女)`，"
        "**禁止**用便衣男/年轻男子/警员等代称代替；",
        "2. **硬字幕/SRT/subtitle_entries 中的姓名、称呼、关系词**（如「二师兄」「老叶」「秦枫」台词）"
        "**不得**用于推断画面人物身份，仅作对白摘录；",
        "3. **人物关系表** 仅用于已写入姓名的**谐音校正**（秦峰→秦枫），"
        "以及**两人姓名均已由面孔匹配写入后**补写明显关系（师徒/父子）；"
        "**禁止**凭职级词、对白内容、关系表名单猜人；",
        f"4. 脸不可辨、仅有背影/侧背、或无法与任一头像匹配 → "
        f"写「{FRAME_UNKNOWN_CHARACTER_MALE}」「{FRAME_UNKNOWN_CHARACTER_FEMALE}」或带**可见特征**的暂称"
        f"（如「深色夹克便衣男」），禁止臆测姓名；",
        "5. **同一人回溯（谨慎）**：本批后帧某人物已通过头像匹配确认为 A 时，"
        "仅当**前序帧可见同一身形+同一服装/发型**且能合理判定为同一人，才可将该前序帧暂称改为 A(性别)；"
        "**禁止**把本批所有「便衣男」一律改成 A；不同身形/不同服装的便衣须分开保留暂称。",
    ]
    if has_drama_knowledge:
        lines.append(
            "- 关系表**不是**出场名单：未在本批画面中完成头像匹配的人物，**不得**写入 observation/action/characters。"
        )
    if has_character_references:
        lines.append(
            "- 勾选头像**不是**默认全员在场：每一规范姓名须对应本批某帧可见面孔与参照图一致。"
        )
    if is_carryover_batch:
        lines.append(
            "- 本批未重复发送参照图：仍须在本批画面中重新完成面孔匹配才写人名，**禁止**沿用首批姓名或字幕猜人。"
        )
    return "\n".join(lines)


def build_frame_face_match_batch_hint(
    character_references: list[dict[str, str]] | None,
    *,
    frame_count: int = 0,
) -> str:
    """多人同框时强调逐脸对照定妆照（唯一写名途径）。"""
    names = [
        str(item.get("name") or "").strip()
        for item in (character_references or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    if not names:
        return ""
    numbered = "、".join(f"#{index + 1}{name}" for index, name in enumerate(names))
    frame_hint = f"本批 {frame_count} 张关键帧" if frame_count > 0 else "本批每一张关键帧"
    return "\n".join(
        [
            "## 定妆照逐脸对照（硬性 · 写规范姓名的唯一途径）",
            f"参照图从左到右依次为：{numbered}。",
            f"**逐帧独立对照**：{frame_hint}，**每一帧分别**查看可见面孔并对照参照图；"
            "该帧匹配成功 → 在该帧 observation 写 `姓名(男/女)`，**禁止**用整批统一代称敷衍；",
            "本批关键帧中**脸/侧脸清晰可见**时，须**逐脸对照参照图**；匹配成功 → **必须**写 `姓名(男)` 或 `姓名(女)`；",
            "**禁止**在脸已清晰可辨时仍用「便衣男(男)」「年轻男子(男)」「警服警官(男)」「警员(男)」等代称敷衍；",
            "仅当脸不可辨、背对镜头、远景模糊、或确实无法与任一头像匹配时，"
            f"才写带服装特征的暂称或「{FRAME_UNKNOWN_CHARACTER_MALE}」「{FRAME_UNKNOWN_CHARACTER_FEMALE}」；",
            "**禁止**凭硬字幕/SRT 称呼（二师兄、老叶等）猜人名；",
            "**同一人回溯**：后帧面孔匹配为某人后，前序帧仅当**同一身形+同一服装**可确认为同一人时才改规范名；"
            "无法确认是否同一人则保留暂称，**禁止**整批便衣统一替换。",
            "两人姓名均已由面孔匹配（或同一人回溯）写入后，可补明显师徒/父子/上下级等关系词。",
        ]
    )


def _strip_gender_suffix(name: str) -> str:
    return _GENDER_SUFFIX_RE.sub("", (name or "").strip())


def _canonical_for_name(name: str) -> str:
    cleaned = _strip_gender_suffix(name)
    for canonical, aliases in PLOT_BLUEPRINT_NAME_ALIAS_GROUPS:
        if cleaned == canonical or cleaned in aliases:
            return canonical
    return cleaned


def _name_tokens_for_matching(name: str) -> set[str]:
    cleaned = _strip_gender_suffix(name)
    if not cleaned:
        return set()
    tokens = {cleaned}
    for canonical, aliases in PLOT_BLUEPRINT_NAME_ALIAS_GROUPS:
        if cleaned == canonical or cleaned in aliases:
            tokens.add(canonical)
            tokens.update(aliases)
            break
    return {token for token in tokens if len(token) >= 2}


def _min_reliable_face_mentions(frame_count: int) -> int:
    """面孔匹配：本批至少 1 帧出现规范姓名即视为可靠。"""
    return 1 if frame_count >= 1 else 0


def _batch_text_mentions_ref_name(text: str, ref_names: set[str]) -> bool:
    if not text or not ref_names:
        return False
    for name in ref_names:
        if f"{name}(男)" in text or f"{name}(女)" in text:
            return True
        if re.search(rf"(?<![\u4e00-\u9fff]){re.escape(name)}(?![\u4e00-\u9fff])", text):
            return True
    return False


def _batch_text_uses_generic_face_role_labels(text: str) -> bool:
    if not text:
        return False
    return any(label in text for label in _GENERIC_FACE_ROLE_LABELS)


def _frame_excuses_no_canonical_name(text: str) -> bool:
    """本帧明确脸不可辨/无人物时可不写规范姓名。"""
    if not text:
        return True
    excuses = ("背对", "侧背", "不可辨", "背影", "无人物", "空镜", "未入画", "无人", "仅环境", "仅字幕")
    return any(token in text for token in excuses)


def _frame_likely_needs_face_match(text: str) -> bool:
    """本帧画面里很可能有需对照定妆照的人脸/人物。"""
    if not text or _frame_excuses_no_canonical_name(text):
        return False
    if _batch_text_uses_generic_face_role_labels(text):
        return True
    if re.search(r"[\(（][男女][\)）]", text):
        return True
    if any(token in text for token in ("特写", "近景", "中景", "侧脸", "正脸", "面部", "两人", "三人")):
        if any(token in text for token in ("男", "女", "人", "警官", "警员", "警察")):
            return True
    return False


def validate_face_naming_when_references_attached(
    *,
    frame_observations: list[dict[str, Any]],
    scene_segments: list[dict[str, Any]],
    character_references: list[dict[str, str]] | None,
    reference_images_attached: bool,
) -> str:
    """
    本批已附定妆照时：逐帧校验——脸/人物可见的帧须写参照图匹配后的规范姓名。
    """
    if not reference_images_attached:
        return ""

    ref_names = {
        str(item.get("name") or "").strip()
        for item in (character_references or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    if not ref_names:
        return ""

    bad_frames: list[str] = []
    for index, frame in enumerate(frame_observations):
        if not isinstance(frame, dict):
            continue
        obs = str(frame.get("observation") or "").strip()
        if not obs or not _frame_likely_needs_face_match(obs):
            continue
        if _batch_text_mentions_ref_name(obs, ref_names):
            continue
        if FRAME_UNKNOWN_CHARACTER_MALE in obs or FRAME_UNKNOWN_CHARACTER_FEMALE in obs:
            continue
        timestamp = str(frame.get("timestamp") or f"frame_{index}").strip()
        bad_frames.append(timestamp)

    if bad_frames:
        sample = "、".join(bad_frames[:5])
        suffix = f" 等{len(bad_frames)}帧" if len(bad_frames) > 5 else ""
        return (
            "Batch attached character reference photos but per-frame observations lack "
            f"reference-matched 姓名(男/女) at {sample}{suffix}; "
            "each frame with a visible face must independently match reference photos"
        )

    # 整批无人写规范名且出现模糊代称
    parts = [str(f.get("observation") or "") for f in frame_observations if isinstance(f, dict)]
    combined = "\n".join(parts)
    if combined and _batch_text_uses_generic_face_role_labels(combined):
        if not _batch_text_mentions_ref_name(combined, ref_names):
            return (
                "Batch attached character reference photos but all frames use generic role labels "
                "without any canonical name from the reference list"
            )

    return ""


def collect_face_identified_names_from_frames(
    frames: list[dict[str, Any]],
    ref_names: set[str],
) -> set[str]:
    """逐帧 observation 中带性别标记的规范姓名（模型面孔匹配结果）。"""
    found: set[str] = set()
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        obs = str(frame.get("observation") or "")
        for name in ref_names:
            if f"{name}(男)" in obs or f"{name}(女)" in obs:
                found.add(name)
                continue
            if re.search(rf"(?<![\u4e00-\u9fff]){re.escape(name)}(?![\u4e00-\u9fff])", obs):
                found.add(name)
    return found


def count_face_name_mentions_in_frames(name: str, frames: list[dict[str, Any]]) -> int:
    count = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        obs = str(frame.get("observation") or "")
        if f"{name}(男)" in obs or f"{name}(女)" in obs:
            count += obs.count(f"{name}(男)") + obs.count(f"{name}(女)")
        elif re.search(rf"(?<![\u4e00-\u9fff]){re.escape(name)}(?![\u4e00-\u9fff])", obs):
            count += 1
    return count


def collect_reliable_face_identified_names(
    frames: list[dict[str, Any]],
    ref_names: set[str],
    *,
    min_mentions: int | None = None,
) -> set[str]:
    """本批次内多次出现在逐帧观察中的面孔匹配姓名（过滤单次 hallucination）。"""
    valid_frames = [item for item in frames if isinstance(item, dict)]
    if not valid_frames or not ref_names:
        return set()
    threshold = (
        min_mentions
        if min_mentions is not None
        else _min_reliable_face_mentions(len(valid_frames))
    )
    reliable: set[str] = set()
    for name in ref_names:
        if count_face_name_mentions_in_frames(name, valid_frames) >= threshold:
            reliable.add(name)
    return reliable


def is_character_name_face_backed(name: str, reliable_faces: set[str]) -> bool:
    """人名是否在本批逐帧观察中经面孔匹配可靠出现。"""
    if not name or not reliable_faces:
        return False
    canonical = _canonical_for_name(name)
    return canonical in reliable_faces


def is_character_name_evidence_backed(name: str, evidence_text: str) -> bool:
    """（非抽帧写名路径）字幕/硬字幕文本是否含该人名或谐音。"""
    if not name or not evidence_text.strip():
        return False
    for token in _name_tokens_for_matching(name):
        if token in evidence_text:
            return True
    return False


def _default_gender_for_name(name: str) -> str:
    if name in {"文江燕", "文琴", "赵子怡", "彭含章"}:
        return "女"
    return "男"


def strip_unreliable_names_in_text(
    text: str,
    *,
    reliable_faces: set[str],
    ref_names: set[str],
) -> str:
    """将无本批面孔匹配依据的人名替换为未名人员。"""
    if not text:
        return text or ""
    updated = text
    for name in sorted(ref_names, key=len, reverse=True):
        if is_character_name_face_backed(name, reliable_faces):
            continue
        gender = _default_gender_for_name(name)
        updated = updated.replace(f"{name}({gender})", FRAME_UNKNOWN_CHARACTER_MALE)
        updated = re.sub(
            rf"(?<![\u4e00-\u9fff]){re.escape(name)}(?![\u4e00-\u9fff])",
            "未名人员",
            updated,
        )

    if ref_names:
        for legacy in sorted(_LEGACY_HALLUCINATED_CHARACTER_NAMES, key=len, reverse=True):
            if legacy in ref_names or is_character_name_face_backed(legacy, reliable_faces):
                continue
            gender = _default_gender_for_name(legacy)
            updated = updated.replace(f"{legacy}({gender})", FRAME_UNKNOWN_CHARACTER_MALE)
            updated = re.sub(
                rf"(?<![\u4e00-\u9fff]){re.escape(legacy)}",
                "未名人员",
                updated,
            )
    return updated


def collect_subtitle_evidence_text(
    segment: dict[str, Any],
    *,
    batch_observations: list[dict[str, Any]] | None = None,
) -> str:
    parts: list[str] = []
    subtitle = str(segment.get("subtitle") or "").strip()
    if subtitle:
        parts.append(subtitle)
    entries = segment.get("subtitle_entries")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict):
                text = str(entry.get("text") or "").strip()
                if text:
                    parts.append(text)
    for observation in batch_observations or []:
        if not isinstance(observation, dict):
            continue
        if observation.get("has_burned_in_subtitle"):
            burned = str(observation.get("burned_in_subtitle") or "").strip()
            if burned:
                parts.append(burned)
    return "\n".join(parts)


def sanitize_segment_character_names(
    segment: dict[str, Any],
    *,
    reliable_faces: set[str],
    reference_names: set[str] | None = None,
) -> list[str]:
    """过滤 characters 中无本批面孔匹配依据的名字。"""
    if not isinstance(segment, dict):
        return []

    characters = segment.get("characters")
    if isinstance(characters, str):
        char_list = [part.strip() for part in re.split(r"[、,，/]", characters) if part.strip()]
    elif isinstance(characters, list):
        char_list = [str(name).strip() for name in characters if str(name).strip()]
    else:
        return []

    kept: list[str] = []
    removed: list[str] = []
    for name in char_list:
        corrected = correct_name_mistakes_in_text(name)
        if is_character_name_face_backed(corrected, reliable_faces):
            kept.append(corrected)
        else:
            removed.append(name)

    if removed:
        logger.debug(
            f"抽帧 characters 移除无面孔匹配依据人名: {removed}"
            + (f"（参照表含 {sorted(reference_names or [])}，须本批头像匹配）" if reference_names else "")
        )
    segment["characters"] = kept
    return removed


def _process_face_gated_batch(
    frames: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    ref_names: set[str],
) -> int:
    reliable = collect_reliable_face_identified_names(frames, ref_names)
    total_removed = 0

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        removed = sanitize_segment_character_names(
            segment,
            reliable_faces=reliable,
            reference_names=ref_names,
        )
        total_removed += len(removed)
        for key in ("observation", "action", "key_visual"):
            value = str(segment.get(key) or "")
            if not value:
                continue
            updated = strip_unreliable_names_in_text(
                value,
                reliable_faces=reliable,
                ref_names=ref_names,
            )
            if updated != value:
                segment[key] = updated

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        obs = str(frame.get("observation") or "")
        updated = strip_unreliable_names_in_text(
            obs,
            reliable_faces=reliable,
            ref_names=ref_names,
        )
        if updated != obs:
            frame["observation"] = updated

    return total_removed


def apply_face_gated_names_to_artifact(artifact: dict[str, Any]) -> None:
    """整份 artifact：仅保留本批逐帧面孔匹配可靠出现的 characters / 人名。"""
    if not isinstance(artifact, dict):
        return

    ref_names = {
        str(item.get("name") or "").strip()
        for item in (artifact.get("character_references") or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    if not ref_names:
        return

    obs_by_batch: dict[int, list[dict[str, Any]]] = {}
    for observation in artifact.get("frame_observations") or []:
        if not isinstance(observation, dict):
            continue
        batch_index = int(observation.get("batch_index", 0))
        obs_by_batch.setdefault(batch_index, []).append(observation)

    total_removed = 0
    for segment in artifact.get("scene_segments") or []:
        if not isinstance(segment, dict):
            continue
        batch_index = int(segment.get("batch_index", 0))
        frames = obs_by_batch.get(batch_index, [])
        total_removed += _process_face_gated_batch(frames, [segment], ref_names)

    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        frames = [
            item for item in (batch.get("frame_observations") or []) if isinstance(item, dict)
        ]
        segments = [
            item for item in (batch.get("scene_segments") or []) if isinstance(item, dict)
        ]
        total_removed += _process_face_gated_batch(frames, segments, ref_names)
        reliable = collect_reliable_face_identified_names(frames, ref_names)
        for key in ("overall_activity_summary", "fallback_summary"):
            value = str(batch.get(key) or "")
            if not value:
                continue
            updated = strip_unreliable_names_in_text(
                value,
                reliable_faces=reliable,
                ref_names=ref_names,
            )
            if updated != value:
                batch[key] = updated

    for summary in artifact.get("overall_activity_summaries") or []:
        if not isinstance(summary, dict):
            continue
        batch_index = int(summary.get("batch_index", 0))
        frames = obs_by_batch.get(batch_index, [])
        reliable = collect_reliable_face_identified_names(frames, ref_names)
        value = str(summary.get("summary") or "")
        if not value:
            continue
        updated = strip_unreliable_names_in_text(
            value,
            reliable_faces=reliable,
            ref_names=ref_names,
        )
        if updated != value:
            summary["summary"] = updated

    if total_removed:
        logger.info(f"抽帧 artifact：已移除 {total_removed} 条无面孔匹配依据的 characters 人名")


def apply_subtitle_gated_names_to_artifact(artifact: dict[str, Any]) -> None:
    """兼容旧名：抽帧已改为仅面孔匹配，转发至 apply_face_gated_names_to_artifact。"""
    apply_face_gated_names_to_artifact(artifact)


def apply_subtitle_alias_normalization_to_artifact(artifact: dict[str, Any]) -> None:
    """抽帧阶段禁用：不因字幕简称（老叶等）归并人名。"""
    return


def extract_canonical_names_from_text(
    text: str,
    *,
    known_names: set[str] | None = None,
) -> set[str]:
    """从 action/observation 提取已写入的规范姓名。"""
    combined = text or ""
    if not combined.strip():
        return set()

    found: set[str] = set()
    for match in _NAMED_WITH_GENDER_RE.finditer(combined):
        canonical = _canonical_for_name(match.group(1))
        if known_names is None or canonical in known_names:
            found.add(canonical)

    if known_names:
        for name in sorted(known_names, key=len, reverse=True):
            if name in combined:
                found.add(name)
    return found


def _relation_already_mentioned(text: str, label: str) -> bool:
    keywords = _RELATION_LABEL_KEYWORDS.get(label, (label,))
    return any(keyword in (text or "") for keyword in keywords)


def find_obvious_relation_for_pair(
    name_a: str,
    name_b: str,
    *,
    relations: tuple[ObviousCharacterRelation, ...],
    evidence_text: str = "",
) -> ObviousCharacterRelation | None:
    pair = frozenset({name_a, name_b})
    candidates = [item for item in relations if item.pair_key() == pair]
    if not candidates:
        return None

    triggered = [
        item
        for item in candidates
        if item.triggers and any(trigger in evidence_text for trigger in item.triggers)
    ]
    if triggered:
        return min(triggered, key=lambda item: _RELATION_PRIORITY.get(item.label, 99))

    unconditional = [item for item in candidates if not item.triggers]
    if not unconditional:
        return None
    return min(unconditional, key=lambda item: _RELATION_PRIORITY.get(item.label, 99))


def enrich_segment_with_obvious_relationships(
    segment: dict[str, Any],
    *,
    relations: tuple[ObviousCharacterRelation, ...],
    reliable_faces: set[str] | None = None,
) -> bool:
    """两人姓名均已由面孔匹配写入时，补充明显关系。"""
    if not relations or not isinstance(segment, dict):
        return False

    known_names = {item.a for item in relations} | {item.b for item in relations}
    action = str(segment.get("action") or "")
    observation = str(segment.get("observation") or "")
    combined = f"{action}\n{observation}"
    present = extract_canonical_names_from_text(combined, known_names=known_names)
    if reliable_faces:
        present = {name for name in present if is_character_name_face_backed(name, reliable_faces)}
    if len(present) < 2:
        return False

    discovered: list[dict[str, str]] = []
    present_list = sorted(present)
    for index, name_a in enumerate(present_list):
        for name_b in present_list[index + 1 :]:
            relation = find_obvious_relation_for_pair(
                name_a,
                name_b,
                relations=relations,
            )
            if not relation:
                continue
            if _relation_already_mentioned(combined, relation.label):
                continue
            discovered.append({"a": relation.a, "b": relation.b, "type": relation.label})

    if not discovered:
        return False

    existing = segment.get("character_relationships")
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in (existing if isinstance(existing, list) else []) + discovered:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("a") or ""), str(item.get("b") or ""), str(item.get("type") or ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append({"a": key[0], "b": key[1], "type": key[2]})
    segment["character_relationships"] = merged

    notes = "；".join(f"{item['a']}与{item['b']}（{item['type']}）" for item in discovered)
    if notes and f"（{notes}）" not in observation:
        segment["observation"] = observation.rstrip() + f"（{notes}）"
    return True


def apply_obvious_character_relationships_to_artifact(artifact: dict[str, Any]) -> None:
    """整份 artifact：对已同时完成面孔匹配的两人补充明显关系。"""
    if not isinstance(artifact, dict):
        return

    drama_id = str(artifact.get("drama_id") or "").strip()
    relations = resolve_obvious_character_relations(drama_id)
    if not relations:
        return

    ref_names = {
        str(item.get("name") or "").strip()
        for item in (artifact.get("character_references") or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }

    obs_by_batch: dict[int, list[dict[str, Any]]] = {}
    for observation in artifact.get("frame_observations") or []:
        if not isinstance(observation, dict):
            continue
        batch_index = int(observation.get("batch_index", 0))
        obs_by_batch.setdefault(batch_index, []).append(observation)

    enriched = 0
    for segment in artifact.get("scene_segments") or []:
        if not isinstance(segment, dict):
            continue
        batch_index = int(segment.get("batch_index", 0))
        frames = obs_by_batch.get(batch_index, [])
        reliable = collect_reliable_face_identified_names(frames, ref_names)
        if enrich_segment_with_obvious_relationships(
            segment,
            relations=relations,
            reliable_faces=reliable,
        ):
            enriched += 1

    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        batch_index = int(batch.get("batch_index", 0))
        frames = [
            item for item in (batch.get("frame_observations") or []) if isinstance(item, dict)
        ]
        reliable = collect_reliable_face_identified_names(frames, ref_names)
        for segment in batch.get("scene_segments") or []:
            if not isinstance(segment, dict):
                continue
            enrich_segment_with_obvious_relationships(
                segment,
                relations=relations,
                reliable_faces=reliable,
            )

    if enriched:
        logger.info(f"抽帧 artifact：已为 {enriched} 条 segment 补充明显人物关系")
