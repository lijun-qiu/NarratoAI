#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""抽帧视觉参照图：缩小、拼图、仅首批发送，降低 token 消耗。"""

from __future__ import annotations

import os
from typing import Any

import PIL.Image
from loguru import logger

from app.services.documentary.documentary_settings import (
    FRAME_FACE_MATCH_SIMILARITY_HINT,
    FRAME_UNKNOWN_CHARACTER_FEMALE,
    FRAME_UNKNOWN_CHARACTER_MALE,
)
from app.services.drama_character_registry import resolve_media_path
from app.utils import utils

ATTACH_MODE_EVERY_BATCH = "every_batch"
ATTACH_MODE_FIRST_BATCH = "first_batch"
REFERENCE_COLLAGE_MAX_HEADS_PER_SHEET = 4


def resolve_reference_collage_mode(
    settings: dict[str, Any] | None,
    *,
    head_count: int,
) -> bool:
    """
    是否将多头像合成拼图。
    token_saver 仅在头像数量超过 individual_max_heads 时强制拼图；
    少量头像用分张发送，便于面孔匹配。
    """
    cfg = settings or {}
    head_count = max(0, int(head_count))
    if head_count <= 1:
        return False
    if cfg.get("frame_reference_force_individual_heads"):
        return False

    max_individual = max(1, int(cfg.get("frame_reference_individual_max_heads", REFERENCE_COLLAGE_MAX_HEADS_PER_SHEET) or REFERENCE_COLLAGE_MAX_HEADS_PER_SHEET))
    prefer_collage = bool(cfg.get("frame_reference_use_collage", True))
    token_saver = bool(cfg.get("frame_reference_token_saver", True))

    if head_count <= max_individual:
        return prefer_collage

    if token_saver:
        return True
    return prefer_collage


def resolve_reference_attach_mode(settings: dict[str, Any] | None) -> str:
    cfg = settings or {}
    if cfg.get("frame_reference_token_saver", True):
        return ATTACH_MODE_FIRST_BATCH
    mode = str(cfg.get("frame_reference_attach_mode") or ATTACH_MODE_EVERY_BATCH).strip()
    if mode not in (ATTACH_MODE_EVERY_BATCH, ATTACH_MODE_FIRST_BATCH):
        return ATTACH_MODE_EVERY_BATCH
    return mode


def should_attach_reference_images(
    batch_index: int,
    settings: dict[str, Any] | None,
    *,
    head_count: int = 0,
    use_collage: bool = False,
) -> bool:
    cfg = settings or {}
    mode = resolve_reference_attach_mode(cfg)
    if mode == ATTACH_MODE_EVERY_BATCH:
        return True
    # 拼图仅多 1 张图；少量分张也负担可控 → 每批附上以保证逐脸对照
    if head_count > 0:
        max_individual = max(1, int(cfg.get("frame_reference_individual_max_heads", REFERENCE_COLLAGE_MAX_HEADS_PER_SHEET) or REFERENCE_COLLAGE_MAX_HEADS_PER_SHEET))
        if use_collage or head_count <= max_individual:
            return True
    return batch_index == 0


def collage_max_heads_per_sheet(settings: dict[str, Any] | None) -> int:
    cfg = settings or {}
    return max(
        1,
        int(
            cfg.get(
                "frame_reference_collage_max_heads",
                cfg.get("frame_reference_individual_max_heads", REFERENCE_COLLAGE_MAX_HEADS_PER_SHEET),
            )
            or REFERENCE_COLLAGE_MAX_HEADS_PER_SHEET
        ),
    )


def split_character_references_into_collage_sheets(
    character_references: list[dict[str, str]] | None,
    *,
    max_per_sheet: int = REFERENCE_COLLAGE_MAX_HEADS_PER_SHEET,
) -> list[list[dict[str, str]]]:
    """将头像参照拆成多组，每组最多 max_per_sheet 张（避免整板拼图过小难辨认）。"""
    valid = [
        {"name": str(item.get("name") or "").strip(), "path": str(item.get("path") or "").strip()}
        for item in (character_references or [])
        if isinstance(item, dict) and item.get("name") and item.get("path") and os.path.isfile(str(item.get("path")))
    ]
    if not valid:
        return []
    cap = max(1, int(max_per_sheet))
    return [valid[index : index + cap] for index in range(0, len(valid), cap)]


def build_reference_carryover_prompt(
    *,
    character_references: list[dict[str, str]],
    relationship_diagram_attached: bool,
    drama_label: str = "",
) -> str:
    names = [str(item.get("name") or "").strip() for item in character_references if item.get("name")]
    if not names and not relationship_diagram_attached:
        return ""
    work = (drama_label or "本剧").strip()
    lines = [
        "## 视觉参照沿用（本批不再重复发送参照图）",
        f"首批已提供 **{work}** 人物定妆照；本批写规范姓名仍须**对照该批关键帧可见面孔**与参照匹配（{FRAME_FACE_MATCH_SIMILARITY_HINT}）。",
    ]
    if relationship_diagram_attached:
        lines.append("- 关系图仅作谐音/关系**校正**，不可凭关系图猜人")
    if names:
        lines.append(
            f"- 定妆照人物（{ '、'.join(names) }）须逐脸对照、{FRAME_FACE_MATCH_SIMILARITY_HINT} 后才可写规范名；"
            f"硬字幕/SRT/对白内容（二师兄、老叶等）**不得**猜人"
        )
    lines.append(f"- 无依据时用「{FRAME_UNKNOWN_CHARACTER_MALE}」「{FRAME_UNKNOWN_CHARACTER_FEMALE}」")
    return "\n".join(lines)


def _reference_cache_dir() -> str:
    directory = os.path.join(utils.temp_dir(), "frame_reference_cache")
    os.makedirs(directory, exist_ok=True)
    return directory


def _file_fingerprint(path: str) -> str:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    return f"{path}|{mtime}"


def _load_resized_rgb(path: str, max_edge: int) -> PIL.Image.Image:
    with PIL.Image.open(path) as source:
        image = source.convert("RGB")
        if max(image.size) > max_edge:
            image.thumbnail((max_edge, max_edge), PIL.Image.Resampling.LANCZOS)
        return image.copy()


def _save_jpeg(image: PIL.Image.Image, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    image.save(output_path, format="JPEG", quality=85, optimize=True)
    return output_path


def _build_horizontal_collage(
    images: list[PIL.Image.Image],
    *,
    max_cell_edge: int,
    padding: int = 8,
) -> PIL.Image.Image:
    if not images:
        raise ValueError("collage 需要至少一张图")
    if len(images) == 1:
        return images[0]

    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    cell_w = min(cell_w, max_cell_edge)
    cell_h = min(cell_h, max_cell_edge)

    normalized: list[PIL.Image.Image] = []
    for img in images:
        fitted = img.copy()
        fitted.thumbnail((cell_w, cell_h), PIL.Image.Resampling.LANCZOS)
        canvas = PIL.Image.new("RGB", (cell_w, cell_h), (24, 24, 24))
        offset = ((cell_w - fitted.width) // 2, (cell_h - fitted.height) // 2)
        canvas.paste(fitted, offset)
        normalized.append(canvas)

    width = len(normalized) * cell_w + padding * (len(normalized) + 1)
    height = cell_h + padding * 2
    collage = PIL.Image.new("RGB", (width, height), (16, 16, 16))
    x = padding
    for img in normalized:
        collage.paste(img, (x, padding))
        x += cell_w + padding
    return collage


def _prepare_collage_path(
    *,
    label: str,
    source_paths: list[str],
    max_edge: int,
    cache_key: str,
) -> str:
    if not source_paths:
        return ""
    output_path = os.path.join(_reference_cache_dir(), f"{cache_key}_{label}.jpg")
    if os.path.isfile(output_path):
        return output_path

    images = [_load_resized_rgb(path, max_edge) for path in source_paths]
    if len(images) == 1:
        collage = images[0]
    else:
        collage = _build_horizontal_collage(images, max_cell_edge=max_edge)
    _save_jpeg(collage, output_path)
    logger.debug(f"已生成参照拼图: {output_path}（{len(source_paths)} 张源图）")
    return output_path


def prepare_reference_prefix_images(
    *,
    batch_index: int,
    relationship_diagram_path: str = "",
    character_references: list[dict[str, str]] | None = None,
    settings: dict[str, Any] | None = None,
) -> tuple[list[str], str]:
    """
    返回 (本批应 prepend 的参照图路径, 无图批次的 carryover prompt)。
    """
    cfg = settings or {}
    refs = [
        {"name": str(item.get("name") or "").strip(), "path": str(item.get("path") or "").strip()}
        for item in (character_references or [])
        if isinstance(item, dict) and item.get("name") and item.get("path")
    ]
    rel_path = resolve_media_path(relationship_diagram_path)
    rel_attached = bool(rel_path)
    has_refs = bool(refs) or rel_attached

    if not has_refs:
        return [], ""

    max_edge = max(128, int(cfg.get("frame_reference_max_edge", 384) or 384))
    head_paths = [item["path"] for item in refs if os.path.isfile(item["path"])]
    use_collage = resolve_reference_collage_mode(cfg, head_count=len(head_paths))

    if not should_attach_reference_images(
        batch_index,
        cfg,
        head_count=len(head_paths),
        use_collage=use_collage,
    ):
        carryover = build_reference_carryover_prompt(
            character_references=refs,
            relationship_diagram_attached=rel_attached,
            drama_label=str(cfg.get("default_video_theme") or ""),
        )
        return [], carryover

    prefix_paths: list[str] = []
    fingerprints = [_file_fingerprint(rel_path)] if rel_path else []
    fingerprints.extend(_file_fingerprint(item["path"]) for item in refs)
    cache_key = utils.md5(
        "|".join(
            [
                "frame-ref-v3",
                str(max_edge),
                str(use_collage),
                str(len(head_paths)),
                *fingerprints,
            ]
        )
    )

    if rel_path:
        if use_collage and len(refs) >= 1:
            rel_cache = os.path.join(_reference_cache_dir(), f"{cache_key}_relationship.jpg")
            if os.path.isfile(rel_cache):
                prefix_paths.append(rel_cache)
            else:
                prefix_paths.append(_save_jpeg(_load_resized_rgb(rel_path, max_edge * 2), rel_cache))
        else:
            rel_only = os.path.join(_reference_cache_dir(), f"{cache_key}_relationship_only.jpg")
            if os.path.isfile(rel_only):
                prefix_paths.append(rel_only)
            else:
                prefix_paths.append(_save_jpeg(_load_resized_rgb(rel_path, max_edge * 2), rel_only))

    if refs:
        max_per_sheet = collage_max_heads_per_sheet(cfg)
        head_sheets = split_character_references_into_collage_sheets(refs, max_per_sheet=max_per_sheet)
        head_max_edge = max(max_edge, 512) if use_collage else max(max_edge, 512)

        if use_collage and head_sheets:
            for sheet_index, sheet_refs in enumerate(head_sheets):
                sheet_paths = [item["path"] for item in sheet_refs]
                sheet_names = "_".join(
                    utils.md5(item["name"])[:6] for item in sheet_refs[:max_per_sheet]
                )
                if len(sheet_paths) == 1:
                    single_cache = os.path.join(
                        _reference_cache_dir(),
                        f"{cache_key}_head_s{sheet_index}_{sheet_names}.jpg",
                    )
                    if os.path.isfile(single_cache):
                        prefix_paths.append(single_cache)
                    else:
                        prefix_paths.append(
                            _save_jpeg(_load_resized_rgb(sheet_paths[0], head_max_edge), single_cache)
                        )
                else:
                    collage_path = _prepare_collage_path(
                        label=f"heads_sheet_{sheet_index}_{sheet_names}",
                        source_paths=sheet_paths,
                        max_edge=head_max_edge,
                        cache_key=f"{cache_key}_s{sheet_index}",
                    )
                    if collage_path:
                        prefix_paths.append(collage_path)
        elif head_paths:
            for index, path in enumerate(head_paths):
                single_cache = os.path.join(_reference_cache_dir(), f"{cache_key}_head_{index}.jpg")
                if os.path.isfile(single_cache):
                    prefix_paths.append(single_cache)
                else:
                    prefix_paths.append(_save_jpeg(_load_resized_rgb(path, head_max_edge), single_cache))

    return prefix_paths, ""
