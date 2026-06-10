#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""抽帧人名：定妆照/头像面孔对照匹配（约70%相似）可写规范姓名，禁止凭字幕猜人。"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from app.services.documentary.documentary_settings import (
    FRAME_FACE_MATCH_SIMILARITY_HINT,
    FRAME_UNKNOWN_CHARACTER_FEMALE,
    FRAME_UNKNOWN_CHARACTER_MALE,
)

_GENDER_SUFFIX_RE = re.compile(r"[\(（][男女不明][\)）]$")
_NAMED_WITH_GENDER_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z·]{2,8})([\(（][男女不明][\)）])"
)
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
def build_frame_naming_priority_rules(
    *,
    has_drama_knowledge: bool = False,
    has_character_references: bool = False,
    is_carryover_batch: bool = False,
) -> str:
    """抽帧写人名的优先级：仅面孔匹配，禁止字幕猜人。"""
    lines = [
        "## 人名写入规则（硬性 · 头像/定妆照面孔对照匹配）",
        f"1. **唯一依据**：本帧/本批关键帧中**脸/侧脸清晰可见**，且与已上传**定妆照/头像**对照匹配（{FRAME_FACE_MATCH_SIMILARITY_HINT}）"
        " → **必须**在 **characters** 字段写规范姓名，"
        "**禁止**在 observation/action 写人名，**禁止**用便衣男/年轻男子/警员等代称代替；",
        "2. **硬字幕/SRT/subtitle_entries 中的姓名、称呼、关系词、对白内容**"
        "**不得**用于推断画面人物身份，仅作对白摘录；**禁止**听声/剧情猜说话人；",
        "3. **人物关系表** 仅用于已写入姓名的**写法校正**，"
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
            f"- 勾选头像**不是**默认全员在场：每一规范姓名须对应本批某帧可见面孔与参照图达到相似度阈值（{FRAME_FACE_MATCH_SIMILARITY_HINT}）。"
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
            "该帧匹配成功 → 在该帧 **characters** 数组写规范姓名，**禁止**在 observation 写人名；",
            "**禁止**把上一帧/下一帧的人物照抄到本帧；",
            f"本批关键帧中**脸/侧脸清晰可见**时，须**逐脸对照参照图**（{FRAME_FACE_MATCH_SIMILARITY_HINT}）；"
            "匹配成功 → **必须**写 `姓名(男)` 或 `姓名(女)`；",
            "**禁止**在脸已清晰可辨时仍用「便衣男(男)」「年轻男子(男)」「警服警官(男)」「警员(男)」等代称敷衍；",
            "仅当脸不可辨、背对镜头、远景模糊、或确实无法与任一头像匹配时，"
            f"才写带服装特征的暂称或「{FRAME_UNKNOWN_CHARACTER_MALE}」「{FRAME_UNKNOWN_CHARACTER_FEMALE}」；",
            "**禁止**凭硬字幕/SRT 称呼猜人名；",
            "**同一人回溯**：后帧面孔匹配为某人后，前序帧仅当**同一身形+同一服装**可确认为同一人时才改规范名；"
            "无法确认是否同一人则保留暂称，**禁止**整批便衣统一替换。",
            "两人姓名均已由面孔匹配（或同一人回溯）写入后，可补明显师徒/父子/上下级等关系词。",
        ]
    )


def _strip_gender_suffix(name: str) -> str:
    return _GENDER_SUFFIX_RE.sub("", (name or "").strip())


def _canonical_for_name(name: str) -> str:
    return _strip_gender_suffix(name)


def _name_tokens_for_matching(name: str) -> set[str]:
    cleaned = _strip_gender_suffix(name)
    if not cleaned:
        return set()
    return {cleaned}


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
        chars = frame.get("characters")
        if isinstance(chars, list):
            for name in chars:
                cleaned = _canonical_for_name(str(name).strip())
                if cleaned and (not ref_names or cleaned in ref_names):
                    found.add(cleaned)
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
    canonical = _canonical_for_name(name)
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        chars = frame.get("characters")
        if isinstance(chars, list):
            for item in chars:
                if _canonical_for_name(str(item).strip()) == canonical:
                    count += 1
                    break
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
        if is_character_name_face_backed(name, reliable_faces):
            kept.append(name)
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
        chars = frame.get("characters")
        if isinstance(chars, list):
            kept = [
                _canonical_for_name(str(name).strip())
                for name in chars
                if str(name).strip()
                and is_character_name_face_backed(_canonical_for_name(str(name).strip()), reliable)
            ]
            if kept:
                frame["characters"] = sorted(set(kept))
            else:
                frame.pop("characters", None)

    return total_removed


_ROLE_LABEL_WITH_GENDER_RE = re.compile(
    r"(?:未名人员|(?:另)?一?名?警员|年轻警员|黄衣男子|男子|女子|"
    + "|".join(re.escape(label) for label in _GENERIC_FACE_ROLE_LABELS)
    + r")[\(（][男女不明][\)）]"
)
_PERSON_IN_SCENE_LABEL_RE = re.compile(r"[\(（][男女][\)）]")


def _clean_descriptive_text_punctuation(text: str) -> str:
    updated = (text or "").strip()
    if not updated:
        return ""
    updated = re.sub(r"[，,]{2,}", "，", updated)
    updated = re.sub(r"，\s*，", "，", updated)
    updated = re.sub(r"^[\s，,；;]+", "", updated)
    updated = re.sub(r"[\s，,；;]+$", "", updated)
    return updated.strip()


def extract_character_names_from_text(
    text: str,
    *,
    ref_names: set[str] | None = None,
) -> list[str]:
    """从带性别标记的文本中提取规范人名（不含未名人员/代称）。"""
    found: list[str] = []
    for match in _NAMED_WITH_GENDER_RE.finditer(text or ""):
        canonical = _canonical_for_name(match.group(1))
        if not canonical or canonical == "未名人员" or "未名" in canonical:
            continue
        if ref_names is not None and canonical not in ref_names:
            continue
        if canonical not in found:
            found.append(canonical)
    return found


def strip_character_tokens_from_descriptive_text(
    text: str,
    *,
    ref_names: set[str] | None = None,
) -> str:
    """从场景/动作描述中移除人名与带性别代称，仅保留环境与动作信息。"""
    updated = (text or "").strip()
    if not updated:
        return ""

    updated = _NAMED_WITH_GENDER_RE.sub("", updated)
    updated = _ROLE_LABEL_WITH_GENDER_RE.sub("", updated)

    if ref_names:
        for name in sorted(ref_names, key=len, reverse=True):
            updated = re.sub(
                rf"(?<![\u4e00-\u9fff]){re.escape(name)}(?![\u4e00-\u9fff])",
                "",
                updated,
            )

    updated = re.sub(
        r"(身着警服的|身穿警服的|身穿棕色夹克的|穿警服的)",
        "",
        updated,
    )
    return _clean_descriptive_text_punctuation(updated)


def populate_frame_characters_from_observation(
    frame: dict[str, Any],
    *,
    ref_names: set[str] | None = None,
) -> None:
    """逐帧：从 observation 提取人名写入 characters，并从描述中剥离人名。"""
    if not isinstance(frame, dict):
        return
    obs = str(frame.get("observation") or "")
    names = extract_character_names_from_text(obs, ref_names=ref_names)
    if names:
        frame["characters"] = names
    stripped = strip_character_tokens_from_descriptive_text(obs, ref_names=ref_names)
    if stripped:
        frame["observation"] = stripped
    elif obs:
        frame.pop("observation", None)


def separate_characters_from_segment_fields(
    segment: dict[str, Any],
    *,
    ref_names: set[str] | None = None,
) -> None:
    """segment：汇总 characters 字段，并从 observation/action/key_visual 剥离人名。"""
    if not isinstance(segment, dict):
        return

    names: list[str] = []
    existing = segment.get("characters")
    if isinstance(existing, list):
        names.extend(str(name).strip() for name in existing if str(name).strip())
    elif isinstance(existing, str) and existing.strip():
        names.extend(
            part.strip()
            for part in re.split(r"[、,，/]", existing)
            if part.strip()
        )

    for key in ("observation", "action", "key_visual"):
        for name in extract_character_names_from_text(
            str(segment.get(key) or ""),
            ref_names=ref_names,
        ):
            if name not in names:
                names.append(name)

    if names:
        segment["characters"] = sorted(set(names))
    else:
        segment.pop("characters", None)

    for key in ("observation", "action", "key_visual"):
        value = str(segment.get(key) or "").strip()
        if not value:
            continue
        stripped = strip_character_tokens_from_descriptive_text(value, ref_names=ref_names)
        if stripped:
            segment[key] = stripped
        else:
            segment.pop(key, None)


def normalize_person_as_scene_label(segment: dict[str, Any]) -> None:
    """scene 误填为人物描述时，改回地点标签。"""
    if not isinstance(segment, dict):
        return
    scene = str(segment.get("scene") or "").strip()
    if not scene or not _PERSON_IN_SCENE_LABEL_RE.search(scene):
        return
    from app.services.documentary.frame_timeline_sampling import infer_scene_label_from_segment

    inferred = infer_scene_label_from_segment(segment)
    if inferred and not _PERSON_IN_SCENE_LABEL_RE.search(inferred):
        segment["scene"] = inferred
    else:
        segment.pop("scene", None)


def apply_character_field_separation_to_artifact(artifact: dict[str, Any]) -> None:
    """整份 artifact：人名进 characters 字段，描述文本不再含人名。"""
    if not isinstance(artifact, dict):
        return

    ref_names = {
        str(item.get("name") or "").strip()
        for item in (artifact.get("character_references") or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    ref_set = ref_names or None

    all_frames: list[dict[str, Any]] = []
    for observation in artifact.get("frame_observations") or []:
        if isinstance(observation, dict):
            all_frames.append(observation)
    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        for observation in batch.get("frame_observations") or []:
            if isinstance(observation, dict):
                all_frames.append(observation)

    for frame in all_frames:
        populate_frame_characters_from_observation(frame, ref_names=ref_set)

    all_segments: list[dict[str, Any]] = []
    for segment in artifact.get("scene_segments") or []:
        if isinstance(segment, dict):
            all_segments.append(segment)
    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        for segment in batch.get("scene_segments") or []:
            if isinstance(segment, dict) and segment not in all_segments:
                all_segments.append(segment)

    for segment in all_segments:
        separate_characters_from_segment_fields(segment, ref_names=ref_set)
        normalize_person_as_scene_label(segment)


def reconcile_segment_observation_action_names(segment: dict[str, Any]) -> None:
    """当 observation 已写出真名而 action 仍用未名人员作主语时，对齐主语。"""
    if not isinstance(segment, dict):
        return
    observation = str(segment.get("observation") or "")
    action = str(segment.get("action") or "")
    if not observation or not action or "未名人员" not in action:
        return

    primary = ""
    gender = "男"
    for match in _NAMED_WITH_GENDER_RE.finditer(observation):
        candidate = _canonical_for_name(match.group(1))
        if not candidate or candidate == "未名人员" or "未名" in candidate:
            continue
        primary = candidate
        gender_match = re.search(r"[\(（]([男女])[\)）]", match.group(2))
        gender = gender_match.group(1) if gender_match else "男"
        break
    if not primary or primary not in observation:
        return
    updated = re.sub(
        r"未名人员\([男女]\)",
        f"{primary}({gender})",
        action,
        count=1,
    )
    if updated != action:
        segment["action"] = updated


def populate_segment_characters_from_batch(
    segments: list[dict[str, Any]],
    *,
    frame_observations: list[dict[str, Any]],
    reference_names: set[str] | None = None,
) -> None:
    """从 observation/action 与可靠面孔匹配结果回填 segment.characters。"""
    ref_names = reference_names or set()
    reliable = (
        collect_reliable_face_identified_names(frame_observations, ref_names)
        if ref_names
        else set()
    )

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        texts = [
            str(segment.get("observation") or ""),
            str(segment.get("action") or ""),
            str(segment.get("key_visual") or ""),
        ]
        combined = "\n".join(texts)
        names = extract_canonical_names_from_text(combined, known_names=ref_names or None)
        if reliable:
            names = {name for name in names if name in reliable}
            names.update(
                name
                for name in reliable
                if any(name in text for text in texts)
            )
        elif names and ref_names:
            names = {name for name in names if name in ref_names}
        if names:
            segment["characters"] = sorted(names)


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

    if total_removed:
        logger.info(f"抽帧 artifact：已移除 {total_removed} 条无面孔匹配依据的 characters 人名")


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


def apply_obvious_character_relationships_to_artifact(artifact: dict[str, Any]) -> None:
    """通用版不注入剧专属人物关系。"""
    return


def apply_segment_character_consistency_to_artifact(artifact: dict[str, Any]) -> None:
    """回填 characters，并将人名从描述文本分离到 characters 字段。"""
    if not isinstance(artifact, dict):
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

    all_segments: list[dict[str, Any]] = []
    for segment in artifact.get("scene_segments") or []:
        if isinstance(segment, dict):
            all_segments.append(segment)

    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        for segment in batch.get("scene_segments") or []:
            if isinstance(segment, dict) and segment not in all_segments:
                all_segments.append(segment)

    for segment in all_segments:
        batch_index = int(segment.get("batch_index", 0))
        frames = obs_by_batch.get(batch_index, [])
        populate_segment_characters_from_batch(
            [segment],
            frame_observations=frames,
            reference_names=ref_names,
        )

    apply_character_field_separation_to_artifact(artifact)
