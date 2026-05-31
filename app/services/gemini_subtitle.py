"""Gemini audio/video transcription helpers for SRT subtitle generation."""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import time
from typing import Optional

import requests
from loguru import logger

from app.config import config
from app.config.defaults import normalize_openai_compatible_model_name
from app.utils import utils
from app.utils.media_audio import MediaAudioError, compress_audio_for_upload, extract_audio_from_media

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_REST_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
MAX_INLINE_AUDIO_MB = 25
RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
TRANSCRIPTION_PROMPT = (
    "请将这段音频转写为 SRT 字幕。\n"
    "要求：\n"
    "1. 仅输出 SRT 正文，不要解释，不要用 markdown 代码块包裹\n"
    "2. 时间戳格式为 HH:MM:SS,mmm --> HH:MM:SS,mmm\n"
    "3. 按语义分段，每段不宜过长\n"
    "4. 中文音频输出简体中文"
)


class GeminiSubtitleError(RuntimeError):
    """Raised for user-actionable Gemini transcription failures."""


def _require_api_key(api_key: str) -> str:
    key = (api_key or "").strip()
    if not key:
        raise GeminiSubtitleError("Gemini API Key 未配置")
    return key


def _normalize_model_name(model_name: str) -> str:
    model = normalize_openai_compatible_model_name(model_name or DEFAULT_MODEL)
    if model.lower().startswith("gemini/"):
        return model.split("/", 1)[1]
    return model or DEFAULT_MODEL


def _resolve_rest_base_url(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_REST_BASE_URL
    if ":generateContent" in url:
        return url.rsplit("/models/", 1)[0]
    if url.endswith("/v1"):
        return f"{url[:-3]}/v1beta"
    if "/v1/" in url and "/v1beta/" not in url:
        return url.replace("/v1/", "/v1beta/", 1)
    return url


def _build_generate_content_url(base_url: str, model_name: str) -> str:
    base = _resolve_rest_base_url(base_url)
    if base.endswith(":generateContent"):
        return base
    model = _normalize_model_name(model_name)
    return f"{base}/models/{model}:generateContent"


def _guess_mime_type(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        return mime_type
    extension = os.path.splitext(file_path)[1].lower()
    fallback = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".opus": "audio/opus",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }
    return fallback.get(extension, "application/octet-stream")


def _validate_file_size(file_path: str) -> None:
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > MAX_INLINE_AUDIO_MB:
        raise GeminiSubtitleError(
            f"文件过大（{size_mb:.1f} MB），Gemini 内联音频建议不超过 {MAX_INLINE_AUDIO_MB} MB。"
        )


def _strip_markdown_fence(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:srt|text)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_text_from_rest_response(response_data: dict) -> str:
    candidates = response_data.get("candidates") or []
    if not candidates:
        raise GeminiSubtitleError("Gemini 返回无效响应，可能触发了安全过滤")

    candidate = candidates[0]
    finish_reason = str(candidate.get("finishReason") or "").upper()
    if finish_reason == "SAFETY":
        raise GeminiSubtitleError("内容被 Gemini 安全过滤器阻止")

    parts = (candidate.get("content") or {}).get("parts") or []
    text_parts = [part["text"] for part in parts if isinstance(part, dict) and part.get("text")]
    transcript = "".join(text_parts).strip()
    if not transcript:
        raise GeminiSubtitleError("Gemini 返回空转写结果")
    return _strip_markdown_fence(transcript)


def _is_retryable_http_error(status_code: int) -> bool:
    return status_code in RETRYABLE_HTTP_CODES


def _post_json_with_retry(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout: float,
    retries: int = 3,
) -> requests.Response:
    last_response: requests.Response | None = None
    for attempt in range(retries):
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        last_response = response
        if response.status_code == 200:
            return response
        if _is_retryable_http_error(response.status_code) and attempt < retries - 1:
            wait_seconds = 2 ** attempt
            logger.warning(
                f"Gemini REST 请求 HTTP {response.status_code}，{wait_seconds}s 后重试 "
                f"({attempt + 1}/{retries})"
            )
            time.sleep(wait_seconds)
            continue
        return response
    return last_response  # type: ignore[return-value]


def _transcribe_via_rest_api(
    audio_path: str,
    api_key: str,
    model_name: str,
    base_url: str,
    timeout: float,
) -> str:
    mime_type = _guess_mime_type(audio_path)
    with open(audio_path, "rb") as file_obj:
        audio_base64 = base64.b64encode(file_obj.read()).decode("utf-8")

    payload = {
        "contents": [{
            "parts": [
                {"text": TRANSCRIPTION_PROMPT},
                {"inline_data": {"mime_type": mime_type, "data": audio_base64}},
            ]
        }],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }
    url = _build_generate_content_url(base_url, model_name)
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    response = _post_json_with_retry(url, payload, headers, timeout)
    if response.status_code != 200:
        raise GeminiSubtitleError(
            f"Gemini 转写请求失败: HTTP {response.status_code} - {response.text[:500]}"
        )
    return _extract_text_from_rest_response(response.json())


def _transcribe_via_sdk(audio_path: str, api_key: str, model_name: str, timeout: float) -> str:
    try:
        import google.generativeai as genai
        from google.generativeai.types import HarmBlockThreshold, HarmCategory
    except ImportError as exc:
        raise GeminiSubtitleError("缺少 google-generativeai 依赖") from exc

    genai.configure(api_key=api_key, transport="rest")
    model = genai.GenerativeModel(
        model_name=_normalize_model_name(model_name),
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    )
    mime_type = _guess_mime_type(audio_path)
    with open(audio_path, "rb") as file_obj:
        audio_bytes = file_obj.read()
    response = model.generate_content(
        [TRANSCRIPTION_PROMPT, {"mime_type": mime_type, "data": audio_bytes}],
        request_options={"timeout": timeout},
    )
    transcript = _strip_markdown_fence(getattr(response, "text", "") or "")
    if not transcript:
        raise GeminiSubtitleError("Gemini SDK 返回空转写结果")
    return transcript


def _transcribe_via_openai_compatible(
    audio_path: str,
    api_key: str,
    model_name: str,
    base_url: str,
    timeout: float,
) -> str:
    from openai import OpenAI

    mime_type = _guess_mime_type(audio_path)
    audio_format = mime_type.split("/")[-1]
    if audio_format == "mpeg":
        audio_format = "mp3"

    with open(audio_path, "rb") as file_obj:
        audio_base64 = base64.b64encode(file_obj.read()).decode("utf-8")

    client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"), timeout=timeout)
    response = client.chat.completions.create(
        model=_normalize_model_name(model_name),
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": TRANSCRIPTION_PROMPT},
                {"type": "input_audio", "input_audio": {"data": audio_base64, "format": audio_format}},
            ],
        }],
        temperature=0.2,
        max_tokens=8192,
    )
    transcript = _strip_markdown_fence(response.choices[0].message.content or "")
    if not transcript:
        raise GeminiSubtitleError("OpenAI 兼容接口返回空转写结果")
    return transcript


def _resolve_defaults(api_key: str, model_name: str, base_url: str, provider: str) -> tuple[str, str, str, str]:
    resolved_key = _require_api_key(
        api_key
        or config.gemini_asr.get("api_key", "")
        or config.app.get("vision_openai_api_key", "")
    )
    resolved_model = (
        model_name
        or config.gemini_asr.get("model", "")
        or config.app.get("vision_openai_model_name", DEFAULT_MODEL)
    )
    resolved_base_url = (
        base_url
        or config.gemini_asr.get("base_url", "")
        or config.app.get("vision_openai_base_url", "")
    )
    resolved_provider = (provider or config.gemini_asr.get("provider", "") or "auto").strip().lower()
    return resolved_key, _normalize_model_name(resolved_model), resolved_base_url, resolved_provider


def _choose_backend(provider: str, base_url: str, model_name: str = "") -> str:
    if provider in {"rest", "sdk", "openai"}:
        return provider
    model = (model_name or "").lower()
    base = (base_url or "").lower()
    if not base or "generativelanguage.googleapis.com" in base or "/v1beta" in base:
        return "rest"
    if "gemini" in model:
        return "rest"
    if base.endswith("/v1") or "chat/completions" in base:
        return "openai"
    return "rest"


def _should_fallback_to_whisper() -> bool:
    value = config.gemini_asr.get("fallback_whisper", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _fallback_transcribe_with_whisper(audio_path: str, timeout: float) -> str:
    from app.services.whisper_subtitle import WhisperSubtitleError, transcribe_with_whisper_api

    logger.warning("Gemini 转写失败，自动切换 Whisper API")
    try:
        return transcribe_with_whisper_api(audio_path, timeout=timeout)
    except WhisperSubtitleError as exc:
        raise GeminiSubtitleError(f"Gemini 与 Whisper 均失败: {exc}") from exc


def _transcribe_audio(
    audio_path: str,
    api_key: str,
    model_name: str,
    base_url: str,
    provider: str,
    timeout: float,
) -> str:
    primary = _choose_backend(provider, base_url, model_name)
    if provider == "auto":
        backends = []
        for candidate in (primary, "openai", "rest"):
            if candidate not in backends:
                backends.append(candidate)
    else:
        backends = [primary]

    errors: list[str] = []
    for backend in backends:
        try:
            logger.info(f"尝试 Gemini 转写 backend={backend}")
            if backend == "sdk":
                return _transcribe_via_sdk(audio_path, api_key, model_name, timeout)
            if backend == "openai":
                return _transcribe_via_openai_compatible(
                    audio_path, api_key, model_name, base_url, timeout
                )
            return _transcribe_via_rest_api(
                audio_path, api_key, model_name, base_url, timeout
            )
        except GeminiSubtitleError as exc:
            errors.append(f"{backend}: {exc}")
            logger.warning(f"Gemini backend={backend} 失败: {exc}")

    if _should_fallback_to_whisper():
        try:
            return _fallback_transcribe_with_whisper(audio_path, timeout)
        except GeminiSubtitleError as exc:
            errors.append(str(exc))

    raise GeminiSubtitleError(
        "Gemini 转写失败。"
        + (" 代理返回 502 通常表示网关超时或服务不可用，建议稍后重试或直接使用 Whisper。" if any("502" in item for item in errors) else "")
        + f" 详情: {' | '.join(errors[-3:])}"
    )


def write_srt_file(srt_content: str, subtitle_file: str = "") -> str:
    if not subtitle_file:
        subtitle_file = os.path.join(utils.subtitle_dir(), f"gemini_{int(time.time())}.srt")
    parent = os.path.dirname(subtitle_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(subtitle_file, "w", encoding="utf-8") as file_obj:
        file_obj.write(srt_content if srt_content.endswith("\n") else srt_content + "\n")
    return subtitle_file


def create_with_gemini(
    local_file: str,
    subtitle_file: str = "",
    api_key: str = "",
    model_name: str = "",
    base_url: str = "",
    provider: str = "auto",
    timeout: Optional[float] = None,
) -> Optional[str]:
    """Transcribe local media via Gemini and write an SRT file."""
    if not os.path.isfile(local_file):
        raise GeminiSubtitleError(f"待转写文件不存在: {local_file}")

    resolved_api_key, resolved_model, resolved_base_url, resolved_provider = _resolve_defaults(
        api_key, model_name, base_url, provider
    )
    request_timeout = max(float(timeout or config.app.get("llm_vision_timeout", 120) or 120), 300)

    temp_audio_path = ""
    temp_compact_path = ""
    try:
        audio_path, is_temp = extract_audio_from_media(local_file, temp_prefix="gemini_asr", compact=True)
        temp_audio_path = audio_path if is_temp else ""

        if not is_temp:
            compact_path, is_compact_temp = compress_audio_for_upload(
                audio_path, output_dir=utils.temp_dir("gemini_asr")
            )
            if is_compact_temp:
                temp_compact_path = compact_path
            audio_path = compact_path

        _validate_file_size(audio_path)
        logger.info(
            f"Gemini 字幕转写: model={resolved_model}, provider={resolved_provider}, "
            f"audio={os.path.getsize(audio_path) / 1024:.1f}KB, file={local_file}"
        )
        transcript = _transcribe_audio(
            audio_path,
            resolved_api_key,
            resolved_model,
            resolved_base_url,
            resolved_provider,
            request_timeout,
        )

        output_file = write_srt_file(transcript, subtitle_file)
        logger.info(f"Gemini 字幕文件已生成: {output_file}")
        return output_file
    except MediaAudioError as exc:
        raise GeminiSubtitleError(str(exc)) from exc
    except GeminiSubtitleError:
        raise
    except Exception as exc:
        raise GeminiSubtitleError("Gemini 字幕转写失败，请检查文件、网络或 API 配置") from exc
    finally:
        for path in (temp_compact_path, temp_audio_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    logger.warning(f"清理临时音频失败: {path}")
