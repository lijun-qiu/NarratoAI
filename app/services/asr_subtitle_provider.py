#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Unified ASR subtitle provider with multi-engine fallback."""

from __future__ import annotations

import os
import subprocess

from loguru import logger

from app.services.media_transcription import transcribe_media_to_entries as transcribe_file_to_entries
from app.services.srt_utils import SrtEntry


def extract_audio_with_ffmpeg(media_path: str, output_path: str) -> str:
    if not media_path or not os.path.exists(media_path):
        raise FileNotFoundError(f"媒体文件不存在: {media_path}")

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        media_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"ffmpeg 提取音频失败: {result.stderr[-500:]}")
    return output_path


def resolve_asr_provider(provider: str = "auto") -> str:
    from app.services.media_transcription import resolve_provider_chain

    chain = resolve_provider_chain(provider, enable_fallback=True)
    return chain[0] if chain else "none"


def transcribe_media_to_entries(
    media_path: str,
    *,
    task_dir: str,
    segment_id: int | str,
    provider: str = "auto",
    max_chars: int = 18,
    max_duration: float = 4.0,
) -> list[SrtEntry]:
    audio_path = media_path
    temp_audio = ""
    ext = os.path.splitext(media_path)[1].lower()
    if ext in {".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm"}:
        temp_audio = os.path.join(task_dir, f"asr_audio_{segment_id}.wav")
        audio_path = extract_audio_with_ffmpeg(media_path, temp_audio)

    try:
        entries, used_provider = transcribe_file_to_entries(
            audio_path,
            provider=provider,
            enable_fallback=True,
            max_chars=max_chars,
            max_duration=max_duration,
        )
        logger.debug(f"片段 #{segment_id} 使用 {used_provider} 完成 ASR")
        return entries
    finally:
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except OSError:
                logger.debug(f"清理临时音频失败: {temp_audio}")
