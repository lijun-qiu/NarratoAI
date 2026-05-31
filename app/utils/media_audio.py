"""Shared audio extraction helpers for subtitle transcription services."""

from __future__ import annotations

import os
import subprocess
import time

from loguru import logger

from app.utils import utils

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".wmv", ".mpeg", ".mpg"}


class MediaAudioError(RuntimeError):
    """Raised when media audio cannot be prepared for transcription."""


def is_video_file(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() in VIDEO_EXTENSIONS


def _run_subprocess(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )


def has_audio_stream(media_path: str) -> bool:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        media_path,
    ]
    try:
        result = _run_subprocess(cmd)
    except FileNotFoundError as exc:
        raise MediaAudioError("未找到 ffprobe，请确认 ffmpeg 已安装并加入 PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise MediaAudioError(f"ffprobe 检测失败: {stderr[-300:] if stderr else exc}") from exc
    return "audio" in result.stdout.lower()


def extract_audio_with_ffmpeg(media_path: str, audio_path: str, *, compact: bool = False) -> None:
    cmd = ["ffmpeg", "-y", "-i", media_path, "-vn"]
    if compact:
        cmd.extend(["-ac", "1", "-ar", "16000", "-acodec", "libmp3lame", "-b:a", "32k"])
    else:
        cmd.extend(["-acodec", "libmp3lame", "-b:a", "64k"])
    cmd.append(audio_path)
    try:
        _run_subprocess(cmd)
    except FileNotFoundError as exc:
        raise MediaAudioError("未找到 ffmpeg，请先安装 ffmpeg 并加入系统 PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        if "does not contain any stream" in stderr or "Output file does not contain any stream" in stderr:
            raise MediaAudioError(
                "该文件不含音轨（仅有画面），无法转写。请换一个有说话声或背景音的视频/音频文件。"
            ) from exc
        raise MediaAudioError(f"ffmpeg 提取音频失败: {stderr[-500:] if stderr else exc}") from exc
    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        raise MediaAudioError("ffmpeg 提取音频失败：输出文件为空")


def compress_audio_for_upload(audio_path: str, output_dir: str = "") -> tuple[str, bool]:
    """Re-encode audio to mono 16kHz/32k for smaller API payloads."""
    if not output_dir:
        output_dir = os.path.dirname(audio_path) or utils.temp_dir("asr")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    compressed_path = os.path.join(output_dir, f"{base_name}_compact_{int(time.time())}.mp3")
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "libmp3lame",
        "-b:a", "32k",
        compressed_path,
    ]
    try:
        _run_subprocess(cmd)
    except FileNotFoundError as exc:
        raise MediaAudioError("未找到 ffmpeg，请先安装 ffmpeg 并加入系统 PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise MediaAudioError(f"ffmpeg 压缩音频失败: {stderr[-500:] if stderr else exc}") from exc
    if not os.path.isfile(compressed_path) or os.path.getsize(compressed_path) == 0:
        raise MediaAudioError("ffmpeg 压缩音频失败：输出文件为空")
    return compressed_path, True


def extract_audio_from_media(
    media_path: str,
    output_dir: str = "",
    temp_prefix: str = "asr",
    *,
    compact: bool = False,
) -> tuple[str, bool]:
    """Extract audio from video when needed. Returns (audio_path, is_temp)."""
    if not is_video_file(media_path):
        if compact and os.path.isfile(media_path):
            return compress_audio_for_upload(media_path, output_dir or utils.temp_dir(temp_prefix))
        return media_path, False

    if not output_dir:
        output_dir = utils.temp_dir(temp_prefix)
    os.makedirs(output_dir, exist_ok=True)

    if not has_audio_stream(media_path):
        raise MediaAudioError(
            "该文件不含音轨（仅有画面），无法转写。请换一个有说话声或背景音的视频/音频文件。"
        )

    base_name = os.path.splitext(os.path.basename(media_path))[0]
    audio_path = os.path.join(output_dir, f"{base_name}_{int(time.time())}.mp3")
    logger.info(f"从视频提取音频 (ffmpeg): {media_path} -> {audio_path}")
    extract_audio_with_ffmpeg(media_path, audio_path, compact=compact)
    return audio_path, True
