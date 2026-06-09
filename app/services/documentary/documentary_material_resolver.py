#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""逐帧解说/精剪：成片视频与抽帧·字幕素材来源视频的配对解析。"""

from __future__ import annotations

import os

from app.services.documentary.frame_analysis_pairing import resolve_reusable_analysis_path


def normalize_material_source_video_path(material_source_video_path: str) -> str:
    return (material_source_video_path or "").strip()


def resolve_subtitle_path_for_documentary(
    output_video_path: str,
    *,
    material_source_video_path: str = "",
    explicit_path: str | None = None,
) -> str:
    """解析字幕路径：成片视频 → 显式路径 → 素材来源视频配对。"""
    from app.services.subtitle_video_pairing import (
        find_paired_subtitle_path,
        resolve_subtitle_path_for_video,
    )

    path = resolve_subtitle_path_for_video(
        output_video_path,
        explicit_path=explicit_path,
    )
    if path:
        return path

    source = normalize_material_source_video_path(material_source_video_path)
    if source and os.path.isfile(source):
        if normalize_material_source_video_path(output_video_path) != source:
            paired = find_paired_subtitle_path(source)
            if paired:
                return paired
    return ""


def load_subtitle_content_for_documentary(
    output_video_path: str,
    *,
    material_source_video_path: str = "",
    explicit_path: str | None = None,
    fallback_content: str = "",
) -> str:
    path = resolve_subtitle_path_for_documentary(
        output_video_path,
        material_source_video_path=material_source_video_path,
        explicit_path=explicit_path,
    )
    if path:
        from app.services.subtitle_video_pairing import load_subtitle_content

        content = load_subtitle_content(path).strip()
        if content:
            return content
    return (fallback_content or "").strip()


def resolve_frame_analysis_path_for_documentary(
    output_video_path: str,
    *,
    material_source_video_path: str = "",
    explicit_path: str | None = None,
    reuse: bool = True,
) -> str | None:
    """解析抽帧分析 JSON：显式路径 → 成片配对 → 素材来源视频配对。"""
    resolved = resolve_reusable_analysis_path(
        output_video_path,
        explicit_path=explicit_path,
        reuse=reuse,
    )
    if resolved:
        return resolved

    source = normalize_material_source_video_path(material_source_video_path)
    output = normalize_material_source_video_path(output_video_path)
    if source and os.path.isfile(source) and source != output:
        return resolve_reusable_analysis_path(
            source,
            explicit_path=None,
            reuse=reuse,
        )
    return None


def resolve_video_episode_analysis_path_for_documentary(
    output_video_path: str,
    *,
    material_source_video_path: str = "",
    explicit_path: str | None = None,
) -> str | None:
    """解析整片视频分析 JSON：显式路径 → 默认落盘路径 → 素材来源视频配对。"""
    from app.services.documentary.video_episode_analysis import (
        default_video_episode_analysis_path,
    )

    explicit = (explicit_path or "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit

    output = normalize_material_source_video_path(output_video_path)
    if output and os.path.isfile(output):
        default_path = default_video_episode_analysis_path(output)
        if os.path.isfile(default_path):
            return default_path

    source = normalize_material_source_video_path(material_source_video_path)
    if source and os.path.isfile(source) and source != output:
        source_default = default_video_episode_analysis_path(source)
        if os.path.isfile(source_default):
            return source_default
    return None
