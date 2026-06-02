#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""视频与转录字幕文件的默认配对（同名 stem + _transcribed.srt）。"""

from __future__ import annotations

import os
from typing import Optional

from app.utils import utils


def get_transcription_subtitle_path(video_path: str) -> str:
    """根据视频路径生成默认转录字幕输出路径。"""
    if not video_path:
        return ""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(utils.subtitle_dir(), f"{stem}_transcribed.srt")


def find_paired_subtitle_path(video_path: str) -> str:
    """查找与视频配对的已有字幕（优先转录产物）。"""
    if not video_path or not os.path.isfile(video_path):
        return ""

    stem = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path) or "."
    candidates = [
        get_transcription_subtitle_path(video_path),
        os.path.join(video_dir, f"{stem}_transcribed.srt"),
        os.path.join(utils.subtitle_dir(), f"{stem}.srt"),
        os.path.splitext(video_path)[0] + ".srt",
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    return ""


def load_subtitle_content(subtitle_path: str) -> str:
    if not subtitle_path or not os.path.isfile(subtitle_path):
        return ""
    with open(subtitle_path, "r", encoding="utf-8") as f:
        return f.read()


def resolve_transcription_media_path(
    video_path: str,
    uploaded_media_path: Optional[str] = None,
    *,
    prefer_video: bool = True,
    uploaded_first: bool = True,
) -> str:
    """解析转录媒体路径：有单独上传则优先，否则默认用上方所选视频。"""
    video_path = (video_path or "").strip()
    uploaded_media_path = (uploaded_media_path or "").strip()

    if uploaded_first and uploaded_media_path and os.path.isfile(uploaded_media_path):
        return uploaded_media_path
    if prefer_video and video_path and os.path.isfile(video_path):
        return video_path
    if uploaded_media_path and os.path.isfile(uploaded_media_path):
        return uploaded_media_path
    return ""
