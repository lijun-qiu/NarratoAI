#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""抽帧视觉参照图：缩小、拼图、仅首批发送，降低 token 消耗。"""

from __future__ import annotations

import os
from typing import Any

import PIL.Image
from loguru import logger

from app.services.documentary.documentary_settings import (
    FRAME_UNKNOWN_CHARACTER_FEMALE,
    FRAME_UNKNOWN_CHARACTER_MALE,
)
from app.services.drama_character_registry import resolve_media_path
from app.utils import utils

ATTACH_MODE_EVERY_BATCH = "every_batch"
ATTACH_MODE_FIRST_BATCH = "first_batch"


def resolve_reference_attach_mode(settings: dict[str, Any] | None) -> str:
    cfg = settings or {}
    if cfg.get("frame_reference_token_saver", True):
        return ATTACH_MODE_FIRST_BATCH
    mode = str(cfg.get("frame_reference_attach_mode") or ATTACH_MODE_EVERY_BATCH).strip()
    if mode not in (ATTACH_MODE_EVERY_BATCH, ATTACH_MODE_FIRST_BATCH):
        return ATTACH_MODE_EVERY_BATCH
    return mode


def should_attach_reference_images(batch_index: int, settings: dict[str, Any] | None) -> bool:
    mode = resolve_reference_attach_mode(settings)
    if mode == ATTACH_MODE_FIRST_BATCH:
        return batch_index == 0
    return True


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
        f"首批已提供 **{work}** 人物参照；本批写人名仍须满足：**本批硬字幕/SRT/可见面孔** 之一。",
    ]
    if relationship_diagram_attached:
        lines.append("- 关系图仅作身份/亲属/阵营**校正**，不可凭关系图给未出现人物填名")
    if names:
        lines.append(
            f"- 定妆照人物（{ '、'.join(names) }）仅在本批**可见面孔匹配**且与字幕不冲突时可写规范名"
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

    if not should_attach_reference_images(batch_index, cfg):
        carryover = build_reference_carryover_prompt(
            character_references=refs,
            relationship_diagram_attached=rel_attached,
            drama_label=str(cfg.get("default_video_theme") or ""),
        )
        return [], carryover

    max_edge = max(128, int(cfg.get("frame_reference_max_edge", 384) or 384))
    use_collage = bool(cfg.get("frame_reference_use_collage", True))
    token_saver = bool(cfg.get("frame_reference_token_saver", True))
    if token_saver:
        use_collage = True

    prefix_paths: list[str] = []
    fingerprints = [_file_fingerprint(rel_path)] if rel_path else []
    fingerprints.extend(_file_fingerprint(item["path"]) for item in refs)
    cache_key = utils.md5(
        "|".join(
            [
                "frame-ref-v1",
                str(max_edge),
                str(use_collage),
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
        head_paths = [item["path"] for item in refs if os.path.isfile(item["path"])]
        if head_paths:
            if use_collage and len(head_paths) >= 2:
                collage_path = _prepare_collage_path(
                    label="heads",
                    source_paths=head_paths,
                    max_edge=max_edge,
                    cache_key=cache_key,
                )
                if collage_path:
                    prefix_paths.append(collage_path)
            else:
                for index, path in enumerate(head_paths):
                    single_cache = os.path.join(_reference_cache_dir(), f"{cache_key}_head_{index}.jpg")
                    if os.path.isfile(single_cache):
                        prefix_paths.append(single_cache)
                    else:
                        prefix_paths.append(_save_jpeg(_load_resized_rgb(path, max_edge), single_cache))

    return prefix_paths, ""
