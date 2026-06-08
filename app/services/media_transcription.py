#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
统一媒体字幕转录：Fun-ASR / Whisper API / Gemini 兼容 API，支持失败自动切换。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Optional

from loguru import logger

from app.config import config
from app.services import fun_asr_subtitle
from app.services.audio_preprocess import (
    get_media_duration_seconds,
    prepare_media_for_asr,
    split_media_for_asr,
)
from app.services.srt_utils import SrtEntry, parse_srt_file, write_srt_file
from app.services.whisper_api_subtitle import (
    WhisperApiUnsupportedError,
    transcribe_audio_to_entries,
)


class MediaTranscriptionError(RuntimeError):
    pass


PROVIDER_FUN_ASR = "fun_asr"
PROVIDER_WHISPER = "whisper_api"
PROVIDER_GEMINI = "gemini_asr"

ALL_PROVIDERS = [PROVIDER_GEMINI, PROVIDER_FUN_ASR, PROVIDER_WHISPER]

PROVIDER_LABELS = {
    PROVIDER_FUN_ASR: "阿里百炼 Fun-ASR",
    PROVIDER_WHISPER: "Whisper API",
    PROVIDER_GEMINI: "Gemini 兼容 API",
}

_UNSUPPORTED_TRANSCRIPTION_HOSTS: set[str] = set()


def get_transcription_settings() -> dict[str, Any]:
    section = config.transcription if hasattr(config, "transcription") else {}
    defaults: dict[str, Any] = {
        "enable_fallback": True,
        "fallback_order": [PROVIDER_GEMINI, PROVIDER_FUN_ASR, PROVIDER_WHISPER],
        "max_chars": 20,
        "max_duration": 3.5,
        "preprocess_audio": True,
        "max_upload_mb": 24.0,
        "chunk_seconds": 600,
        "ssl_verify": True,
    }
    if isinstance(section, dict):
        for key in defaults:
            if key in section and section[key] is not None:
                defaults[key] = section[key]
    order = defaults.get("fallback_order")
    if isinstance(order, list):
        defaults["fallback_order"] = [str(item).strip().lower() for item in order if str(item).strip()]
    else:
        defaults["fallback_order"] = ALL_PROVIDERS.copy()
    return defaults


def _provider_configured(provider: str) -> bool:
    provider = provider.strip().lower()
    if provider == PROVIDER_FUN_ASR:
        return bool((config.fun_asr.get("api_key") or "").strip())
    if provider == PROVIDER_WHISPER:
        return bool((config.whisper_asr.get("api_key") or "").strip())
    if provider == PROVIDER_GEMINI:
        gemini_cfg = config.gemini_asr if hasattr(config, "gemini_asr") else {}
        return bool((gemini_cfg.get("api_key") or "").strip())
    return False


def _provider_host(provider: str) -> str:
    provider = provider.strip().lower()
    if provider == PROVIDER_WHISPER:
        return str(config.whisper_asr.get("base_url", "") or "").strip().lower()
    if provider == PROVIDER_GEMINI:
        gemini_cfg = config.gemini_asr if hasattr(config, "gemini_asr") else {}
        return str(gemini_cfg.get("base_url", "") or "").strip().lower()
    return ""


def _should_skip_provider(provider: str) -> bool:
    if provider not in (PROVIDER_WHISPER, PROVIDER_GEMINI):
        return False
    host = _provider_host(provider)
    return bool(host and host in _UNSUPPORTED_TRANSCRIPTION_HOSTS)


def _mark_provider_unsupported(provider: str) -> None:
    host = _provider_host(provider)
    if host:
        _UNSUPPORTED_TRANSCRIPTION_HOSTS.add(host)
        logger.warning(f"已标记网关不支持语音转写，后续将跳过: {host}")


def resolve_provider_chain(provider: str = "auto", *, enable_fallback: bool = True) -> list[str]:
    settings = get_transcription_settings()
    normalized = (provider or "auto").strip().lower()

    full_chain: list[str] = []
    for name in settings.get("fallback_order", ALL_PROVIDERS):
        if name in ALL_PROVIDERS and _provider_configured(name) and not _should_skip_provider(name):
            full_chain.append(name)

    if normalized == "auto":
        return full_chain

    if normalized not in ALL_PROVIDERS:
        return full_chain

    if not _provider_configured(normalized) or _should_skip_provider(normalized):
        return full_chain

    if enable_fallback:
        return [normalized] + [name for name in full_chain if name != normalized]
    return [normalized]


def _offset_entries(entries: list[SrtEntry], offset_ms: int) -> list[SrtEntry]:
    if offset_ms <= 0:
        return entries
    return [
        SrtEntry(
            start_ms=entry.start_ms + offset_ms,
            end_ms=entry.end_ms + offset_ms,
            text=entry.text,
            label=entry.label,
        )
        for entry in entries
    ]


def _entries_with_provider_raw(
    media_path: str,
    provider: str,
    *,
    max_chars: int,
    max_duration: float,
    subtitle_file: str = "",
) -> list[SrtEntry]:
    provider = provider.strip().lower()

    if provider == PROVIDER_FUN_ASR:
        api_key = config.fun_asr.get("api_key", "")
        generated = fun_asr_subtitle.create_with_fun_asr(
            media_path,
            subtitle_file=subtitle_file or None,
            api_key=api_key,
        )
        if not generated or not os.path.exists(generated):
            raise MediaTranscriptionError("Fun-ASR 未返回有效字幕文件")
        return parse_srt_file(generated)

    if provider == PROVIDER_WHISPER:
        whisper_cfg = config.whisper_asr
        return transcribe_audio_to_entries(
            media_path,
            api_key=whisper_cfg.get("api_key", ""),
            base_url=whisper_cfg.get("base_url", ""),
            model=whisper_cfg.get("model", "whisper-1"),
            language=whisper_cfg.get("language", "zh"),
            max_chars=max_chars,
            max_duration=max_duration,
        )

    if provider == PROVIDER_GEMINI:
        gemini_cfg = config.gemini_asr if hasattr(config, "gemini_asr") else {}
        return transcribe_audio_to_entries(
            media_path,
            api_key=gemini_cfg.get("api_key", ""),
            base_url=gemini_cfg.get("base_url", ""),
            model=gemini_cfg.get("model", "gemini-2.0-flash"),
            language=gemini_cfg.get("language", "zh"),
            max_chars=max_chars,
            max_duration=max_duration,
        )

    raise MediaTranscriptionError(f"不支持的转录方式: {provider}")


def _entries_with_provider(
    media_path: str,
    provider: str,
    *,
    work_dir: str,
    max_chars: int,
    max_duration: float,
    subtitle_file: str = "",
) -> list[SrtEntry]:
    settings = get_transcription_settings()
    prepared_path = media_path
    temp_dir = ""

    if settings.get("preprocess_audio", True):
        prepared_path = prepare_media_for_asr(
            media_path,
            work_dir,
            prefix=f"{provider}_prep",
            max_upload_mb=float(settings.get("max_upload_mb", 24.0)),
        )

    chunk_seconds = float(settings.get("chunk_seconds", 600))
    duration = get_media_duration_seconds(prepared_path)
    chunks = (
        split_media_for_asr(prepared_path, work_dir, chunk_seconds=chunk_seconds, prefix=f"{provider}_chunk")
        if duration > chunk_seconds > 0
        else [(prepared_path, 0.0)]
    )

    merged: list[SrtEntry] = []
    try:
        for chunk_path, offset_sec in chunks:
            chunk_entries = _entries_with_provider_raw(
                chunk_path,
                provider,
                max_chars=max_chars,
                max_duration=max_duration,
                subtitle_file=subtitle_file if len(chunks) == 1 else "",
            )
            merged.extend(_offset_entries(chunk_entries, int(offset_sec * 1000)))
        return merged
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def transcribe_media_to_entries(
    media_path: str,
    *,
    provider: str = "auto",
    enable_fallback: Optional[bool] = None,
    max_chars: Optional[int] = None,
    max_duration: Optional[float] = None,
    subtitle_file: str = "",
    work_dir: str = "",
) -> tuple[list[SrtEntry], str]:
    if not media_path or not os.path.exists(media_path):
        raise MediaTranscriptionError(f"媒体文件不存在: {media_path}")

    settings = get_transcription_settings()
    use_fallback = settings.get("enable_fallback", True) if enable_fallback is None else enable_fallback
    chain = resolve_provider_chain(provider, enable_fallback=use_fallback)
    if not chain:
        raise MediaTranscriptionError(
            "未配置可用转录 API。"
            "请至少配置 gemini_asr（Gemini 兼容 API，推荐）、fun_asr（阿里百炼），"
            "或支持 /audio/transcriptions 的 whisper_asr。"
        )

    chars = int(max_chars if max_chars is not None else settings.get("max_chars", 20))
    duration = float(max_duration if max_duration is not None else settings.get("max_duration", 3.5))

    if not work_dir:
        work_dir = tempfile.mkdtemp(prefix="narrato_transcribe_")

    errors: list[str] = []
    for name in chain:
        label = PROVIDER_LABELS.get(name, name)
        try:
            logger.info(f"尝试使用 {label} 转录: {media_path}")
            entries = _entries_with_provider(
                media_path,
                name,
                work_dir=work_dir,
                max_chars=chars,
                max_duration=duration,
                subtitle_file=subtitle_file if len(chain) == 1 else "",
            )
            if not entries:
                raise MediaTranscriptionError(f"{label} 转写结果为空")
            logger.info(f"{label} 转录成功，共 {len(entries)} 条字幕")
            return entries, name
        except WhisperApiUnsupportedError as exc:
            _mark_provider_unsupported(name)
            if name == PROVIDER_GEMINI:
                _mark_provider_unsupported(PROVIDER_WHISPER)
            message = f"{label}: {exc}"
            logger.warning(message)
            errors.append(message)
        except Exception as exc:
            message = f"{label}: {exc}"
            logger.warning(f"转录失败，{'尝试下一种方式' if use_fallback else '停止'} — {message}")
            errors.append(message)

    raise MediaTranscriptionError("所有转录方式均失败:\n" + "\n".join(errors))


def transcribe_media_to_srt(
    media_path: str,
    subtitle_file: str,
    *,
    provider: str = "auto",
    enable_fallback: Optional[bool] = None,
    max_chars: Optional[int] = None,
    max_duration: Optional[float] = None,
) -> tuple[str, str]:
    work_dir = tempfile.mkdtemp(prefix="narrato_transcribe_")
    try:
        entries, used_provider = transcribe_media_to_entries(
            media_path,
            provider=provider,
            enable_fallback=enable_fallback,
            max_chars=max_chars,
            max_duration=max_duration,
            work_dir=work_dir,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    output_path = write_srt_file(entries, subtitle_file)
    return output_path, used_provider
