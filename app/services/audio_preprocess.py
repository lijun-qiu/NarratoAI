#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Extract and compress audio before ASR to improve stability."""

from __future__ import annotations

import os
import subprocess
from typing import Optional

from loguru import logger

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm", ".wmv", ".mpeg", ".mpg"}
DEFAULT_MAX_MB = 24.0
DEFAULT_CHUNK_SECONDS = 600


def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0 or not os.path.exists(cmd[-1]):
        raise RuntimeError(f"ffmpeg 处理失败: {result.stderr[-800:]}")


def get_media_duration_seconds(media_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        media_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        return 0.0
    try:
        return max(0.0, float(result.stdout.strip()))
    except ValueError:
        return 0.0


def extract_audio_mp3(
    media_path: str,
    output_path: str,
    *,
    sample_rate: int = 16000,
    bitrate: str = "64k",
) -> str:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        media_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-b:a",
        bitrate,
        "-f",
        "mp3",
        output_path,
    ]
    _run_ffmpeg(cmd)
    return output_path


def extract_audio_chunk(
    media_path: str,
    output_path: str,
    *,
    start_sec: float,
    duration_sec: float,
    sample_rate: int = 16000,
    bitrate: str = "64k",
) -> str:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(max(0.0, start_sec)),
        "-t",
        str(max(0.1, duration_sec)),
        "-i",
        media_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-b:a",
        bitrate,
        "-f",
        "mp3",
        output_path,
    ]
    _run_ffmpeg(cmd)
    return output_path


def prepare_media_for_asr(
    media_path: str,
    work_dir: str,
    *,
    prefix: str = "asr",
    max_upload_mb: float = DEFAULT_MAX_MB,
    force_extract: bool = False,
) -> str:
    """
    Convert video/large audio to compact mono MP3 for ASR upload.
    Returns path to prepared audio (may equal input for small mp3/wav).
    """
    if not media_path or not os.path.exists(media_path):
        raise FileNotFoundError(f"媒体文件不存在: {media_path}")

    os.makedirs(work_dir, exist_ok=True)
    ext = os.path.splitext(media_path)[1].lower()
    size_mb = os.path.getsize(media_path) / (1024 * 1024)

    needs_extract = force_extract or ext in VIDEO_EXTENSIONS or ext not in {".mp3", ".m4a", ".wav"} or size_mb > max_upload_mb
    if not needs_extract:
        return media_path

    output_path = os.path.join(work_dir, f"{prefix}_prepared.mp3")
    logger.info(f"正在提取/压缩音频用于转录: {media_path} ({size_mb:.1f}MB)")
    return extract_audio_mp3(media_path, output_path)


def split_media_for_asr(
    media_path: str,
    work_dir: str,
    *,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    prefix: str = "chunk",
) -> list[tuple[str, float]]:
    """Split long media into chunks. Returns list of (chunk_path, start_offset_sec)."""
    duration = get_media_duration_seconds(media_path)
    if duration <= 0 or duration <= chunk_seconds:
        return [(media_path, 0.0)]

    os.makedirs(work_dir, exist_ok=True)
    chunks: list[tuple[str, float]] = []
    start = 0.0
    index = 0
    while start < duration:
        chunk_duration = min(chunk_seconds, duration - start)
        chunk_path = os.path.join(work_dir, f"{prefix}_{index:03d}.mp3")
        extract_audio_chunk(media_path, chunk_path, start_sec=start, duration_sec=chunk_duration)
        chunks.append((chunk_path, start))
        start += chunk_duration
        index += 1
    logger.info(f"长音频已切分为 {len(chunks)} 段，每段约 {chunk_seconds:.0f}s")
    return chunks
