#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""剧集人物注册：剧名目录、人物关系表、头像参照（headImg）。"""

from __future__ import annotations

import os
import re
import hashlib
from typing import Any

from loguru import logger

from app.data.drama_knowledge.fazu2_upload_roster import (
    roster_for_drama,
    roster_groups_for_drama,
)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_HEAD_IMG_DIRNAME = "headImg"
_SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

_CHARACTER_SECTION_RE = re.compile(
    r"^###\s+(?:\d+\.\s*)?([\u4e00-\u9fffA-Za-z·]{2,8})（",
    re.MULTILINE,
)
_CHARACTER_BOLD_RE = re.compile(
    r"^-\s+\*\*([\u4e00-\u9fffA-Za-z·]{2,8})\*\*",
    re.MULTILINE,
)
_CHARACTER_TABLE_RE = re.compile(
    r"^\|\s*([\u4e00-\u9fffA-Za-z·]{2,8})\s*\|",
    re.MULTILINE,
)
_NON_CHARACTER_HEADERS = frozenset(
    {
        "秦枫下属",
        "其他",
        "金鼎集团",
        "汉洲商会",
        "儒颂集团",
        "人物",
        "易错点",
    }
)

DRAMA_CATALOG: dict[str, dict[str, str]] = {
    "罚罪2": {
        "label": "罚罪2",
        "knowledge_file": "app/data/drama_knowledge/fazu2_relationships.md",
    },
}

DEFAULT_DRAMA_ID = "罚罪2"
RELATIONSHIP_DIAGRAM_STEM = "_relationship"


def project_root() -> str:
    return _PROJECT_ROOT


def head_img_root() -> str:
    return os.path.join(_PROJECT_ROOT, _HEAD_IMG_DIRNAME)


def head_img_dir(drama_id: str) -> str:
    safe_id = _safe_path_segment(drama_id or DEFAULT_DRAMA_ID)
    return os.path.join(head_img_root(), safe_id)


def list_dramas() -> list[dict[str, str]]:
    return [
        {"id": drama_id, "label": meta.get("label") or drama_id}
        for drama_id, meta in DRAMA_CATALOG.items()
    ]


def get_drama(drama_id: str) -> dict[str, str] | None:
    drama_id = (drama_id or "").strip()
    if not drama_id:
        return None
    meta = DRAMA_CATALOG.get(drama_id)
    if not meta:
        return None
    return {"id": drama_id, **meta}


def resolve_knowledge_path_for_drama(drama_id: str) -> str:
    meta = get_drama(drama_id)
    if not meta:
        return ""
    rel_path = str(meta.get("knowledge_file") or "").strip()
    if not rel_path:
        return ""
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(_PROJECT_ROOT, rel_path.replace("/", os.sep))


def _safe_path_segment(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", (value or "").strip())
    return cleaned or DEFAULT_DRAMA_ID


def _safe_character_filename(name: str) -> str:
    return _safe_path_segment(name)


def ensure_head_img_dir(drama_id: str) -> str:
    directory = head_img_dir(drama_id)
    os.makedirs(directory, exist_ok=True)
    return directory


def _load_knowledge_text(drama_id: str) -> str:
    path = resolve_knowledge_path_for_drama(drama_id)
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8") as fp:
            return fp.read()
    except OSError as exc:
        logger.warning(f"读取剧集人物关系表失败 {path}: {exc}")
        return ""


def list_characters_for_drama(drama_id: str) -> list[str]:
    """返回剧集头像上传名单；有定制 roster 时优先于 Markdown 解析。"""
    roster = roster_for_drama(drama_id)
    if roster:
        return [str(item.get("name") or "").strip() for item in roster if item.get("name")]

    content = _load_knowledge_text(drama_id)
    if not content:
        return []

    names: list[str] = []
    seen: set[str] = set()
    for pattern in (_CHARACTER_SECTION_RE, _CHARACTER_BOLD_RE, _CHARACTER_TABLE_RE):
        for match in pattern.finditer(content):
            name = match.group(1).strip()
            if len(name) < 2 or name in _NON_CHARACTER_HEADERS or name in seen:
                continue
            seen.add(name)
            names.append(name)
    return names


def list_character_roster_groups(drama_id: str) -> list[dict[str, Any]]:
    """按频率分级返回上传名单（无 roster 时整表归为 single 组）。"""
    groups = roster_groups_for_drama(drama_id)
    if groups:
        return groups

    names = list_characters_for_drama(drama_id)
    if not names:
        return []
    return [
        {
            "tier": "all",
            "label": "人物表",
            "characters": [{"name": name, "tier": "all", "role_hint": ""} for name in names],
        }
    ]


def find_head_image_path(drama_id: str, character_name: str) -> str:
    directory = head_img_dir(drama_id)
    stem = _safe_character_filename(character_name)
    for ext in _SUPPORTED_IMAGE_EXTENSIONS:
        candidate = os.path.join(directory, f"{stem}{ext}")
        if os.path.isfile(candidate):
            return candidate
    return ""


def list_character_head_slots(drama_id: str) -> list[dict[str, Any]]:
    """按上传名单返回槽位（含 tier、role_hint、已上传路径）。"""
    roster = roster_for_drama(drama_id)
    entries: list[dict[str, str]] = (
        [dict(item) for item in roster]
        if roster
        else [{"name": name, "tier": "", "role_hint": ""} for name in list_characters_for_drama(drama_id)]
    )

    slots: list[dict[str, Any]] = []
    for entry in entries:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        image_path = find_head_image_path(drama_id, name)
        slots.append(
            {
                "name": name,
                "tier": str(entry.get("tier") or ""),
                "role_hint": str(entry.get("role_hint") or ""),
                "image_path": image_path,
                "uploaded": bool(image_path),
            }
        )
    return slots


def list_character_head_slot_groups(drama_id: str) -> list[dict[str, Any]]:
    """按频率分级返回头像上传槽位（供 UI 分组展示）。"""
    groups = list_character_roster_groups(drama_id)
    slot_by_name = {str(s["name"]): s for s in list_character_head_slots(drama_id)}
    if not groups:
        slots = list(slot_by_name.values())
        if not slots:
            return []
        return [{"tier": "all", "label": "人物表", "slots": slots}]

    result: list[dict[str, Any]] = []
    for group in groups:
        tier_slots = [
            slot_by_name[str(item.get("name") or "")]
            for item in group.get("characters") or []
            if str(item.get("name") or "") in slot_by_name
        ]
        if tier_slots:
            result.append(
                {
                    "tier": group.get("tier"),
                    "label": group.get("label"),
                    "slots": tier_slots,
                }
            )
    return result


def save_head_image(
    drama_id: str,
    character_name: str,
    file_bytes: bytes,
    *,
    original_filename: str = "",
) -> str:
    if not file_bytes:
        raise ValueError("头像文件为空")

    directory = ensure_head_img_dir(drama_id)
    stem = _safe_character_filename(character_name)
    ext = os.path.splitext(original_filename or "")[1].lower()
    if ext not in _SUPPORTED_IMAGE_EXTENSIONS:
        ext = ".jpg"

    target_path = os.path.join(directory, f"{stem}{ext}")
    for old_ext in _SUPPORTED_IMAGE_EXTENSIONS:
        old_path = os.path.join(directory, f"{stem}{old_ext}")
        if old_path != target_path and os.path.isfile(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass

    with open(target_path, "wb") as fp:
        fp.write(file_bytes)
    logger.info(f"已保存人物头像: {target_path}")
    return target_path


def resolve_character_references(
    drama_id: str,
    *,
    selected_names: set[str] | None = None,
) -> list[dict[str, str]]:
    """返回已上传且（可选）已勾选的人物参照列表（按人物表顺序）。"""
    references: list[dict[str, str]] = []
    for slot in list_character_head_slots(drama_id):
        name = str(slot.get("name") or "").strip()
        path = str(slot.get("image_path") or "").strip()
        if not name or not path or not os.path.isfile(path):
            continue
        if selected_names is not None and name not in selected_names:
            continue
        references.append({"name": name, "path": path})
    return references


def character_widget_slot_id(drama_id: str, character_name: str) -> str:
    """生成稳定的 ASCII widget key 后缀（避免中文 key 在 Streamlit 中异常）。"""
    raw = f"{(drama_id or '').strip()}|{(character_name or '').strip()}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def head_uploader_session_key(drama_id: str, character_name: str) -> str:
    return f"doc_head_img_{character_widget_slot_id(drama_id, character_name)}"


def head_selection_session_key(drama_id: str, character_name: str) -> str:
    return f"doc_head_sel_{character_widget_slot_id(drama_id, character_name)}"


def head_upload_saved_sig_key(drama_id: str, character_name: str) -> str:
    return f"doc_head_saved_sig_{character_widget_slot_id(drama_id, character_name)}"


def head_pending_select_session_key(drama_id: str, character_name: str) -> str:
    return f"doc_head_pending_sel_{character_widget_slot_id(drama_id, character_name)}"


def list_unrecognized_head_images(drama_id: str) -> list[str]:
    """列出目录中文件名未匹配上传名单的图片（含手动拖入的截图）。"""
    directory = head_img_dir(drama_id)
    if not os.path.isdir(directory):
        return []

    known_stems = {
        _safe_character_filename(name)
        for name in list_characters_for_drama(drama_id)
    }
    known_stems.add(RELATIONSHIP_DIAGRAM_STEM)

    orphans: list[str] = []
    for fname in sorted(os.listdir(directory)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in _SUPPORTED_IMAGE_EXTENSIONS:
            continue
        if stem not in known_stems:
            orphans.append(fname)
    return orphans


def find_relationship_diagram_path(drama_id: str) -> str:
    directory = head_img_dir(drama_id)
    for ext in _SUPPORTED_IMAGE_EXTENSIONS:
        candidate = os.path.join(directory, f"{RELATIONSHIP_DIAGRAM_STEM}{ext}")
        if os.path.isfile(candidate):
            return candidate
    return ""


def save_relationship_diagram(
    drama_id: str,
    file_bytes: bytes,
    *,
    original_filename: str = "",
) -> str:
    if not file_bytes:
        raise ValueError("关系图文件为空")

    directory = ensure_head_img_dir(drama_id)
    ext = os.path.splitext(original_filename or "")[1].lower()
    if ext not in _SUPPORTED_IMAGE_EXTENSIONS:
        ext = ".png"

    target_path = os.path.join(directory, f"{RELATIONSHIP_DIAGRAM_STEM}{ext}")
    for old_ext in _SUPPORTED_IMAGE_EXTENSIONS:
        old_path = os.path.join(directory, f"{RELATIONSHIP_DIAGRAM_STEM}{old_ext}")
        if old_path != target_path and os.path.isfile(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass

    with open(target_path, "wb") as fp:
        fp.write(file_bytes)
    logger.info(f"已保存人物关系图: {target_path}")
    return target_path


def resolve_relationship_diagram_path(drama_id: str) -> str:
    return find_relationship_diagram_path(drama_id)


def resolve_media_path(path: str) -> str:
    cleaned = (path or "").strip()
    if not cleaned:
        return ""
    if not os.path.isabs(cleaned):
        cleaned = os.path.join(project_root(), cleaned.replace("/", os.sep))
    return cleaned if os.path.isfile(cleaned) else ""


def build_batch_vision_reference_prompt_section(
    *,
    relationship_diagram_path: str = "",
    character_references: list[dict[str, str]] | None = None,
    video_frame_count: int,
    drama_label: str = "",
    character_collage: bool = False,
    reference_image_count: int | None = None,
) -> str:
    """构建关系图 + 人物头像参照的 prompt 说明（均排在视频关键帧之前）。"""
    rel_path = resolve_media_path(relationship_diagram_path)
    refs = [item for item in (character_references or []) if isinstance(item, dict)]
    if reference_image_count is not None:
        prefix_count = reference_image_count
    else:
        head_images = 1 if character_collage and len(refs) >= 2 else len(refs)
        prefix_count = (1 if rel_path else 0) + head_images
    if prefix_count <= 0:
        return ""

    work = (drama_label or "本剧").strip()
    lines: list[str] = [
        f"## 视觉参照图（本请求最前 {prefix_count} 张 · **不是视频帧**）",
    ]
    image_index = 1
    if rel_path:
        lines.extend(
            [
                f"**图 #{image_index}**：**{work}** 人物关系图 — 对照身份、亲属、阵营与易错关系；",
                "**禁止**将关系图本身写入 frame_observations 或 scene_segments。",
                "",
            ]
        )
        image_index += 1

    if refs:
        if character_collage and len(refs) >= 2:
            names = "、".join(str(item.get("name") or "").strip() for item in refs if item.get("name"))
            lines.append(
                f"**图 #{image_index}**：人物定妆照拼图（从左到右依次为 **{names}**），用于识别关键帧面孔；"
            )
            image_index += 1
        else:
            lines.append("以下定妆照按顺序对应剧中人物，**用于识别关键帧中的面孔**：")
            for item in refs:
                name = str(item.get("name") or "").strip()
                lines.append(f"- **{name}** — 参照图 #{image_index}")
                image_index += 1
        lines.append("")

    lines.extend(
        [
            f"从第 **{prefix_count + 1}** 张起共 **{video_frame_count}** 张为本批次视频关键帧。",
            "识别到与参照一致且**本批画面可见**的面孔时，才可写规范姓名；",
            "仍须与本批硬字幕/SRT 不冲突；无法匹配时用「未名人员(男/女)」。",
            "**禁止**因关系表/定妆照列表而默认全员在场。",
        ]
    )
    return "\n".join(lines)


def build_character_reference_prompt_section(
    references: list[dict[str, str]],
    *,
    video_frame_count: int,
) -> str:
    return build_batch_vision_reference_prompt_section(
        character_references=references,
        video_frame_count=video_frame_count,
    )


def resolve_active_relationship_diagram_path(
    drama_id: str,
    *,
    enabled: bool,
) -> str:
    """仅勾选启用时返回已上传的关系图路径。"""
    if not enabled:
        return ""
    return find_relationship_diagram_path(drama_id)


def merge_frame_analysis_settings_for_drama(
    settings: dict[str, Any] | None,
    drama_id: str,
    *,
    enable_knowledge_text: bool = False,
) -> dict[str, Any]:
    """选中剧集时写入主题；文字关系表仅勾选后注入。"""
    merged = dict(settings or {})
    drama = get_drama(drama_id)
    if not drama:
        return merged

    merged["enable_frame_analysis_drama_knowledge"] = bool(enable_knowledge_text)
    if enable_knowledge_text:
        knowledge_path = resolve_knowledge_path_for_drama(drama_id)
        if knowledge_path:
            merged["drama_knowledge_file"] = knowledge_path
    merged["selected_drama_id"] = drama_id
    merged.setdefault("default_video_theme", drama.get("label") or drama_id)
    return merged
