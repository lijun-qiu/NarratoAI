#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""抽帧视觉参照图：缩小、拼图、仅首批发送，降低 token 消耗。"""

from __future__ import annotations

import os
from typing import Any

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
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
    默认每人一张分开发送（识别率更高）；仅 frame_reference_use_collage=true 且未强制分张时拼图。
    """
    cfg = settings or {}
    head_count = max(0, int(head_count))
    if head_count <= 1:
        return False
    if cfg.get("frame_reference_force_individual_heads", True):
        return False
    return bool(cfg.get("frame_reference_use_collage", False))


def resolve_reference_attach_mode(settings: dict[str, Any] | None) -> str:
    cfg = settings or {}
    if cfg.get("frame_reference_token_saver", True):
        return ATTACH_MODE_FIRST_BATCH
    mode = str(cfg.get("frame_reference_attach_mode") or ATTACH_MODE_EVERY_BATCH).strip()
    if mode not in (ATTACH_MODE_EVERY_BATCH, ATTACH_MODE_FIRST_BATCH):
        return ATTACH_MODE_EVERY_BATCH
    return mode


def resolve_head_max_edge(settings: dict[str, Any] | None) -> int:
    """定妆照最长边；默认同 frame_reference_max_edge（不再强制 512）。"""
    cfg = settings or {}
    base = max(128, int(cfg.get("frame_reference_max_edge", 384) or 384))
    head_edge = int(cfg.get("frame_reference_head_max_edge", 0) or 0)
    if head_edge > 0:
        return max(128, head_edge)
    return base


def should_use_labeled_collage(settings: dict[str, Any] | None, *, use_collage: bool) -> bool:
    cfg = settings or {}
    if not use_collage:
        return False
    return bool(cfg.get("frame_reference_labeled_collage", True))


def should_attach_reference_images(
    batch_index: int,
    settings: dict[str, Any] | None,
    *,
    head_count: int = 0,
    use_collage: bool = False,
) -> bool:
    cfg = settings or {}
    if head_count <= 0:
        return False
    force_individual = bool(cfg.get("frame_reference_force_individual_heads", True))
    if force_individual and not use_collage:
        return True
    mode = resolve_reference_attach_mode(cfg)
    if mode == ATTACH_MODE_FIRST_BATCH:
        return batch_index == 0
    interval = max(0, int(cfg.get("frame_reference_reattach_interval", 0) or 0))
    if interval <= 0:
        return True
    return batch_index == 0 or batch_index % interval == 0


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
            f"硬字幕/SRT/对白内容**不得**猜人"
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


def _load_label_font(size: int = 16) -> PIL.ImageFont.FreeTypeFont | PIL.ImageFont.ImageFont:
    size = max(10, int(size))
    for candidate in (
        os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "msyh.ttc"),
        os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "msyhbd.ttc"),
        os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "simhei.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ):
        if candidate and os.path.isfile(candidate):
            try:
                return PIL.ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return PIL.ImageFont.load_default()


def _annotate_reference_label(image: PIL.Image.Image, label: str) -> PIL.Image.Image:
    """在参照图底部标注姓名，拼图模式下便于逐格识脸。"""
    text = (label or "").strip()
    if not text:
        return image
    bar_h = max(22, min(36, image.height // 8))
    canvas = PIL.Image.new("RGB", (image.width, image.height + bar_h), (16, 16, 16))
    canvas.paste(image, (0, 0))
    draw = PIL.ImageDraw.Draw(canvas)
    font = _load_label_font(size=max(12, bar_h - 8))
    draw.text((6, image.height + 2), text, fill=(255, 220, 120), font=font)
    return canvas


def _save_jpeg(image: PIL.Image.Image, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    image.save(output_path, format="JPEG", quality=85, optimize=True)
    return output_path


def _build_horizontal_collage(
    images: list[PIL.Image.Image],
    *,
    max_cell_edge: int,
    padding: int = 8,
    labels: list[str] | None = None,
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
    label_list = labels or []
    for index, img in enumerate(images):
        fitted = img.copy()
        fitted.thumbnail((cell_w, cell_h), PIL.Image.Resampling.LANCZOS)
        canvas = PIL.Image.new("RGB", (cell_w, cell_h), (24, 24, 24))
        offset = ((cell_w - fitted.width) // 2, (cell_h - fitted.height) // 2)
        canvas.paste(fitted, offset)
        if index < len(label_list) and label_list[index]:
            canvas = _annotate_reference_label(canvas, label_list[index])
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
    name_labels: list[str] | None = None,
    labeled_collage: bool = False,
) -> str:
    if not source_paths:
        return ""
    output_path = os.path.join(_reference_cache_dir(), f"{cache_key}_{label}.jpg")
    if os.path.isfile(output_path):
        return output_path

    images = [_load_resized_rgb(path, max_edge) for path in source_paths]
    labels = name_labels if labeled_collage else None
    if len(images) == 1:
        collage = images[0]
        if labeled_collage and labels:
            collage = _annotate_reference_label(collage, labels[0])
    else:
        collage = _build_horizontal_collage(
            images,
            max_cell_edge=max_edge,
            labels=labels,
        )
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
    head_max_edge = resolve_head_max_edge(cfg)
    head_paths = [item["path"] for item in refs if os.path.isfile(item["path"])]
    use_collage = resolve_reference_collage_mode(cfg, head_count=len(head_paths))
    labeled_collage = should_use_labeled_collage(cfg, use_collage=use_collage)

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
                str(head_max_edge),
                str(use_collage),
                str(labeled_collage),
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
        if use_collage and head_sheets:
            for sheet_index, sheet_refs in enumerate(head_sheets):
                sheet_paths = [item["path"] for item in sheet_refs]
                sheet_labels = [str(item.get("name") or "").strip() for item in sheet_refs]
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
                        image = _load_resized_rgb(sheet_paths[0], head_max_edge)
                        if labeled_collage and sheet_labels[0]:
                            image = _annotate_reference_label(image, sheet_labels[0])
                        prefix_paths.append(_save_jpeg(image, single_cache))
                else:
                    collage_path = _prepare_collage_path(
                        label=f"heads_sheet_{sheet_index}_{sheet_names}",
                        source_paths=sheet_paths,
                        max_edge=head_max_edge,
                        cache_key=f"{cache_key}_s{sheet_index}",
                        name_labels=sheet_labels,
                        labeled_collage=labeled_collage,
                    )
                    if collage_path:
                        prefix_paths.append(collage_path)
        elif head_paths:
            for index, path in enumerate(head_paths):
                single_cache = os.path.join(_reference_cache_dir(), f"{cache_key}_head_{index}.jpg")
                if os.path.isfile(single_cache):
                    prefix_paths.append(single_cache)
                else:
                    image = _load_resized_rgb(path, head_max_edge)
                    if index < len(refs) and str(refs[index].get("name") or "").strip():
                        image = _annotate_reference_label(image, str(refs[index]["name"]))
                    prefix_paths.append(_save_jpeg(image, single_cache))

    return prefix_paths, ""
