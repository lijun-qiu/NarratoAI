#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""OpenAI-compatible Whisper/Gemini transcription API."""

from __future__ import annotations

import os
from typing import Any

import requests
from loguru import logger

from app.services.http_client import create_http_session, request_post
from app.services.srt_utils import SrtEntry, parse_srt, write_srt_file


class WhisperApiError(RuntimeError):
    pass


class WhisperApiUnsupportedError(WhisperApiError):
    """Gateway does not implement /audio/transcriptions."""


_UNSUPPORTED_HINT = (
    "当前 API 网关不支持 /audio/transcriptions 接口。"
    "请优先使用「阿里百炼 Fun-ASR」，或更换支持 Whisper 转写的 API 地址。"
)


def _require_api_key(api_key: str) -> str:
    api_key = (api_key or "").strip()
    if not api_key:
        raise WhisperApiError("Whisper API Key 未配置，请在 config.toml [whisper_asr] 中设置 api_key")
    return api_key


def _normalize_base_url(base_url: str) -> str:
    url = (base_url or "https://api.openai.com/v1").strip().rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def _is_unsupported_response(status_code: int, body: str) -> bool:
    text = (body or "").lower()
    if status_code in (404, 501):
        return True
    if status_code == 500 and ("not implemented" in text or "convert_request_failed" in text):
        return True
    if status_code == 429 and ("model_not_found" in text or "model not found" in text):
        return True
    if "not support" in text and "audio" in text:
        return True
    return False


def _segments_from_response(payload: dict[str, Any]) -> list[dict[str, Any]]:
    segments = payload.get("segments")
    if isinstance(segments, list) and segments:
        return segments

    words = payload.get("words")
    if isinstance(words, list) and words:
        grouped: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for word in words:
            text = str(word.get("word") or word.get("text") or "").strip()
            if not text:
                continue
            start = float(word.get("start", 0.0))
            end = float(word.get("end", start))
            if current is None:
                current = {"start": start, "end": end, "text": text}
                continue
            if start - current["end"] <= 0.35 and len(current["text"]) + len(text) <= 24:
                current["end"] = end
                current["text"] += text
            else:
                grouped.append(current)
                current = {"start": start, "end": end, "text": text}
        if current:
            grouped.append(current)
        return grouped

    text = str(payload.get("text") or "").strip()
    if text:
        return [{"start": 0.0, "end": max(1.0, len(text) * 0.18), "text": text}]
    return []


def _response_to_entries(payload: dict[str, Any], max_chars: int, max_duration: float) -> list[SrtEntry]:
    entries: list[SrtEntry] = []
    for segment in _segments_from_response(payload):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start_ms = int(round(float(segment.get("start", 0.0)) * 1000))
        end_ms = int(round(float(segment.get("end", start_ms / 1000.0 + 0.5)) * 1000))
        end_ms = max(end_ms, start_ms + 200)

        if len(text) <= max_chars and (end_ms - start_ms) <= int(max_duration * 1000):
            entries.append(SrtEntry(start_ms=start_ms, end_ms=end_ms, text=text))
            continue

        duration_ms = max(end_ms - start_ms, 500)
        chunks = []
        buffer = ""
        for char in text:
            buffer += char
            if char in "，。！？；,.!?;" or len(buffer) >= max_chars:
                chunks.append(buffer.strip())
                buffer = ""
        if buffer.strip():
            chunks.append(buffer.strip())

        total_chars = max(1, sum(len(chunk) for chunk in chunks))
        cursor = start_ms
        for index, chunk in enumerate(chunks):
            if index == len(chunks) - 1:
                chunk_end = end_ms
            else:
                chunk_end = cursor + int(duration_ms * (len(chunk) / total_chars))
                chunk_end = max(cursor + 200, chunk_end)
            entries.append(SrtEntry(start_ms=cursor, end_ms=chunk_end, text=chunk))
            cursor = chunk_end
    return entries


def _parse_srt_text(text: str) -> list[SrtEntry]:
    return parse_srt(text)


def _post_transcription(
    endpoint: str,
    api_key: str,
    audio_path: str,
    *,
    model: str,
    language: str,
    response_format: str,
    session=None,
) -> requests.Response:
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(audio_path, "rb") as fp:
        mime = "audio/mpeg" if audio_path.lower().endswith(".mp3") else "application/octet-stream"
        files = {"file": (os.path.basename(audio_path), fp, mime)}
        data = {
            "model": model,
            "response_format": response_format,
            "language": language,
        }
        return request_post(endpoint, headers=headers, files=files, data=data, session=session, timeout=600)


def transcribe_audio_to_entries(
    audio_path: str,
    *,
    api_key: str,
    base_url: str = "",
    model: str = "whisper-1",
    language: str = "zh",
    max_chars: int = 18,
    max_duration: float = 4.0,
    session=None,
) -> list[SrtEntry]:
    if not audio_path or not os.path.exists(audio_path):
        raise WhisperApiError(f"音频文件不存在: {audio_path}")

    api_key = _require_api_key(api_key)
    endpoint = f"{_normalize_base_url(base_url)}/audio/transcriptions"
    http_session = session or create_http_session()

    last_error = ""
    for response_format in ("verbose_json", "json", "srt", "text"):
        try:
            response = _post_transcription(
                endpoint,
                api_key,
                audio_path,
                model=model,
                language=language,
                response_format=response_format,
                session=http_session,
            )
        except Exception as exc:
            error_text = str(exc)
            if "SSLError" in error_text or "EOF occurred" in error_text:
                raise WhisperApiError(
                    f"HTTPS 连接中断（常见于大文件或网关不稳定）。"
                    f"系统已自动压缩音频，若仍失败请改用 Fun-ASR。详情: {error_text[:200]}"
                ) from exc
            raise WhisperApiError(error_text) from exc

        body = response.text or ""
        if _is_unsupported_response(response.status_code, body):
            raise WhisperApiUnsupportedError(f"{_UNSUPPORTED_HINT} ({response.status_code}: {body[:180]})")

        if response.status_code >= 400:
            last_error = f"{response.status_code}: {body[:300]}"
            continue

        if response_format == "srt":
            entries = _parse_srt_text(body)
        elif response_format == "text":
            text = body.strip()
            entries = _response_to_entries({"text": text}, max_chars, max_duration) if text else []
        else:
            try:
                payload = response.json()
            except Exception as exc:
                last_error = f"无效 JSON: {body[:200]}"
                continue
            entries = _response_to_entries(payload, max_chars=max_chars, max_duration=max_duration)

        if entries:
            logger.info(
                f"Whisper 兼容 API 转写完成 ({response_format}): {audio_path}, {len(entries)} 条字幕"
            )
            return entries
        last_error = "转写结果为空"

    raise WhisperApiError(last_error or "Whisper 兼容 API 转写失败")


def transcribe_audio_to_srt(
    audio_path: str,
    subtitle_file: str,
    *,
    api_key: str,
    base_url: str = "",
    model: str = "whisper-1",
    language: str = "zh",
    max_chars: int = 18,
    max_duration: float = 4.0,
    session=None,
) -> str:
    entries = transcribe_audio_to_entries(
        audio_path,
        api_key=api_key,
        base_url=base_url,
        model=model,
        language=language,
        max_chars=max_chars,
        max_duration=max_duration,
        session=session,
    )
    return write_srt_file(entries, subtitle_file)
