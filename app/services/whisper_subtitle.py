"""OpenAI Whisper API subtitle transcription."""

from __future__ import annotations

import os
import time
from typing import Optional

import requests
from loguru import logger

from app.config import config
from app.utils import utils
from app.utils.media_audio import MediaAudioError, extract_audio_from_media

DEFAULT_MODEL = "whisper-1"
DEFAULT_LANGUAGE = "zh"
MAX_UPLOAD_MB = 25


class WhisperSubtitleError(RuntimeError):
    """Raised for user-actionable Whisper transcription failures."""


def _resolve_api_config(
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    language: str = "",
) -> tuple[str, str, str, str]:
    resolved_key = (
        api_key
        or config.whisper_asr.get("api_key", "")
        or config.app.get("vision_openai_api_key", "")
        or config.app.get("text_openai_api_key", "")
    ).strip()
    if not resolved_key:
        raise WhisperSubtitleError("Whisper API Key 未配置")

    resolved_base = (
        base_url
        or config.whisper_asr.get("base_url", "")
        or config.app.get("vision_openai_base_url", "")
        or config.app.get("text_openai_base_url", "")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    if resolved_base.endswith("/chat/completions"):
        resolved_base = resolved_base[: -len("/chat/completions")]

    resolved_model = (model or config.whisper_asr.get("model", "") or DEFAULT_MODEL).strip()
    resolved_language = (language or config.whisper_asr.get("language", "") or DEFAULT_LANGUAGE).strip()
    return resolved_key, resolved_base, resolved_model, resolved_language


def _build_transcription_url(base_url: str) -> str:
    if base_url.endswith("/audio/transcriptions"):
        return base_url
    return f"{base_url.rstrip('/')}/audio/transcriptions"


def transcribe_with_whisper_api(
    audio_path: str,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    language: str = "",
    timeout: float = 600.0,
) -> str:
    resolved_key, resolved_base, resolved_model, resolved_language = _resolve_api_config(
        api_key, base_url, model, language
    )
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise WhisperSubtitleError(f"文件过大（{size_mb:.1f} MB），Whisper API 建议不超过 {MAX_UPLOAD_MB} MB")

    url = _build_transcription_url(resolved_base)
    headers = {"Authorization": f"Bearer {resolved_key}"}
    data = {"model": resolved_model, "response_format": "srt", "language": resolved_language}
    filename = os.path.basename(audio_path)

    with open(audio_path, "rb") as file_obj:
        files = {"file": (filename, file_obj, "application/octet-stream")}
        response = requests.post(url, headers=headers, data=data, files=files, timeout=timeout)

    if response.status_code != 200:
        raise WhisperSubtitleError(
            f"Whisper API 转写失败: HTTP {response.status_code} - {response.text[:500]}"
        )
    transcript = (response.text or "").strip()
    if not transcript:
        raise WhisperSubtitleError("Whisper API 返回空字幕内容")
    return transcript


def write_srt_file(srt_content: str, subtitle_file: str = "") -> str:
    if not subtitle_file:
        subtitle_file = os.path.join(utils.subtitle_dir(), f"whisper_{int(time.time())}.srt")
    parent = os.path.dirname(subtitle_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(subtitle_file, "w", encoding="utf-8") as file_obj:
        file_obj.write(srt_content if srt_content.endswith("\n") else srt_content + "\n")
    return subtitle_file


def create_with_whisper(
    local_file: str,
    subtitle_file: str = "",
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    language: str = "",
    timeout: Optional[float] = None,
) -> Optional[str]:
    if not os.path.isfile(local_file):
        raise WhisperSubtitleError(f"待转写文件不存在: {local_file}")

    request_timeout = max(float(timeout or config.app.get("llm_text_timeout", 180) or 180), 300)
    temp_audio_path = ""
    try:
        audio_path, is_temp = extract_audio_from_media(local_file, temp_prefix="whisper_asr")
        temp_audio_path = audio_path if is_temp else ""
        transcript = transcribe_with_whisper_api(
            audio_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
            language=language,
            timeout=request_timeout,
        )
        output_file = write_srt_file(transcript, subtitle_file)
        logger.info(f"Whisper 字幕文件已生成: {output_file}")
        return output_file
    except MediaAudioError as exc:
        raise WhisperSubtitleError(str(exc)) from exc
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError:
                logger.warning(f"清理临时音频失败: {temp_audio_path}")
