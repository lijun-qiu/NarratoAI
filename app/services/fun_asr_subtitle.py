"""Aliyun Bailian Fun-ASR subtitle transcription helpers.

This module intentionally uses the REST API because the official Fun-ASR
recorded-file API supports temporary `oss://` resources only through REST.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
from loguru import logger

from app.utils import utils

from app.services.srt_utils import clean_subtitle_dialogue_text

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com"
UPLOAD_POLICY_URL = f"{DASHSCOPE_BASE_URL}/api/v1/uploads"
TRANSCRIPTION_URL = f"{DASHSCOPE_BASE_URL}/api/v1/services/audio/asr/transcription"
TASK_URL_TEMPLATE = f"{DASHSCOPE_BASE_URL}/api/v1/tasks/{{task_id}}"
MODEL_NAME = "fun-asr"
TERMINAL_FAILED_STATUSES = {"FAILED", "CANCELED", "UNKNOWN"}
PUNCTUATION_BREAKS = set("，。！？；,.!?;")


class FunAsrError(RuntimeError):
    """Raised for user-actionable Fun-ASR transcription failures."""


@dataclass
class UploadPolicy:
    upload_host: str
    upload_dir: str
    policy: str
    signature: str
    oss_access_key_id: str
    x_oss_object_acl: str = "private"
    x_oss_forbid_overwrite: str = "true"
    max_file_size_mb: Optional[float] = None


def _auth_headers(api_key: str, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def _raise_for_http(response: requests.Response, action: str) -> None:
    try:
        response.raise_for_status()
    except Exception as exc:  # requests may be mocked with generic exceptions
        raise FunAsrError(f"{action}失败，请检查阿里百炼 API Key、网络或服务状态") from exc


def _json(response: requests.Response, action: str) -> dict[str, Any]:
    _raise_for_http(response, action)
    try:
        data = response.json()
    except Exception as exc:
        raise FunAsrError(f"{action}返回了无效 JSON") from exc
    if not isinstance(data, dict):
        raise FunAsrError(f"{action}返回格式无效")
    return data


def _require_api_key(api_key: str) -> str:
    api_key = (api_key or "").strip()
    if not api_key:
        raise FunAsrError("请先输入阿里百炼 API Key")
    return api_key


def _safe_upload_name(local_file: str) -> str:
    name = os.path.basename(local_file).strip() or f"audio_{int(time.time())}.wav"
    return name.replace("/", "_").replace("\\", "_")


def _session_get(session, url: str, **kwargs):
    return session.get(url, **kwargs)


def _session_post(session, url: str, **kwargs):
    return session.post(url, **kwargs)


def request_upload_policy(api_key: str, model: str = MODEL_NAME, session=requests) -> UploadPolicy:
    """Request Bailian temporary-storage upload policy for the target model."""
    api_key = _require_api_key(api_key)
    response = _session_get(
        session,
        UPLOAD_POLICY_URL,
        params={"action": "getPolicy", "model": model},
        headers=_auth_headers(api_key),
        timeout=30,
    )
    data = _json(response, "获取临时存储上传凭证")
    policy_data = data.get("data") or {}
    required = ["upload_host", "upload_dir", "policy", "signature", "oss_access_key_id"]
    missing = [field for field in required if not policy_data.get(field)]
    if missing:
        raise FunAsrError(f"临时存储上传凭证缺少字段: {', '.join(missing)}")

    return UploadPolicy(
        upload_host=str(policy_data["upload_host"]),
        upload_dir=str(policy_data["upload_dir"]).rstrip("/"),
        policy=str(policy_data["policy"]),
        signature=str(policy_data["signature"]),
        oss_access_key_id=str(policy_data["oss_access_key_id"]),
        x_oss_object_acl=str(policy_data.get("x_oss_object_acl") or "private"),
        x_oss_forbid_overwrite=str(policy_data.get("x_oss_forbid_overwrite") or "true"),
        max_file_size_mb=policy_data.get("max_file_size_mb"),
    )


def _validate_file_size(local_file: str, policy: UploadPolicy) -> None:
    if policy.max_file_size_mb is None:
        return
    max_bytes = float(policy.max_file_size_mb) * 1024 * 1024
    size = os.path.getsize(local_file)
    if size > max_bytes:
        raise FunAsrError(
            f"文件大小超过阿里百炼临时存储限制: {size / 1024 / 1024:.2f}MB > {float(policy.max_file_size_mb):.2f}MB"
        )


def upload_to_temporary_oss(local_file: str, policy: UploadPolicy, session=requests) -> str:
    """Upload local file to temporary OSS and return `oss://...` URL."""
    if not os.path.isfile(local_file):
        raise FunAsrError(f"待转写文件不存在: {local_file}")
    _validate_file_size(local_file, policy)

    key = f"{policy.upload_dir}/{_safe_upload_name(local_file)}"
    data = {
        "OSSAccessKeyId": policy.oss_access_key_id,
        "policy": policy.policy,
        "Signature": policy.signature,
        "key": key,
        "x-oss-object-acl": policy.x_oss_object_acl,
        "x-oss-forbid-overwrite": policy.x_oss_forbid_overwrite,
        "success_action_status": "200",
    }
    with open(local_file, "rb") as file_obj:
        files = {"file": (_safe_upload_name(local_file), file_obj)}
        response = _session_post(session, policy.upload_host, data=data, files=files, timeout=120)
    _raise_for_http(response, "上传文件到阿里百炼临时存储")
    return f"oss://{key}"


def submit_transcription_task(
    api_key: str,
    oss_url: str,
    speaker_count: Optional[int] = None,
    model: str = MODEL_NAME,
    session=requests,
) -> str:
    """Submit async Fun-ASR task and return task_id."""
    api_key = _require_api_key(api_key)
    parameters: dict[str, Any] = {"diarization_enabled": True}
    if speaker_count:
        parameters["speaker_count"] = int(speaker_count)

    payload = {
        "model": model,
        "input": {"file_urls": [oss_url]},
        "parameters": parameters,
    }
    response = _session_post(
        session,
        TRANSCRIPTION_URL,
        headers=_auth_headers(
            api_key,
            {
                "X-DashScope-Async": "enable",
                "X-DashScope-OssResourceResolve": "enable",
            },
        ),
        json=payload,
        timeout=30,
    )
    data = _json(response, "提交 Fun-ASR 转写任务")
    task_id = ((data.get("output") or {}).get("task_id") or "").strip()
    if not task_id:
        raise FunAsrError("提交 Fun-ASR 转写任务失败：未返回 task_id")
    return task_id


def poll_transcription_task(
    api_key: str,
    task_id: str,
    poll_interval: float = 2.0,
    timeout: float = 600.0,
    session=requests,
) -> dict[str, Any]:
    """Poll task until terminal status and return successful result item."""
    api_key = _require_api_key(api_key)
    deadline = time.time() + timeout
    last_status = "PENDING"
    while time.time() < deadline:
        response = _session_post(
            session,
            TASK_URL_TEMPLATE.format(task_id=task_id),
            headers=_auth_headers(api_key),
            timeout=30,
        )
        data = _json(response, "查询 Fun-ASR 转写任务")
        output = data.get("output") or {}
        last_status = str(output.get("task_status") or "").upper()

        if last_status == "SUCCEEDED":
            results = output.get("results") or []
            for result in results:
                subtask_status = str(result.get("subtask_status") or "").upper()
                if subtask_status and subtask_status != "SUCCEEDED":
                    raise FunAsrError(f"Fun-ASR 子任务失败: {subtask_status}")
            if not results:
                raise FunAsrError("Fun-ASR 转写成功但未返回结果")
            return results[0]

        if last_status in TERMINAL_FAILED_STATUSES:
            code = str(output.get("code") or "").strip()
            message = str(output.get("message") or "").strip()
            detail = f"{last_status}"
            if code:
                detail += f" ({code})"
            if message and message != code:
                detail += f": {message}"
            results = output.get("results") or []
            if results:
                first = results[0] if isinstance(results[0], dict) else {}
                sub_code = str(first.get("code") or "").strip()
                sub_msg = str(first.get("message") or "").strip()
                if sub_code and sub_code not in detail:
                    detail += f" | 子任务: {sub_code}"
                if sub_msg and sub_msg not in detail:
                    detail += f" - {sub_msg}"
            if code == "ASR_RESPONSE_HAVE_NO_WORDS":
                detail += "（未识别到语音，请确认文件含清晰对白且非纯音乐/静音）"
            raise FunAsrError(f"Fun-ASR 转写任务失败: {detail}")

        time.sleep(poll_interval)

    raise FunAsrError(f"Fun-ASR 转写任务超时，最后状态: {last_status}")


def download_transcription_result(transcription_url: str, session=requests) -> dict[str, Any]:
    if not transcription_url:
        raise FunAsrError("Fun-ASR 结果缺少 transcription_url")
    response = _session_get(session, transcription_url, timeout=60)
    return _json(response, "下载 Fun-ASR 转写结果")


def _ms_to_srt_time(ms: float) -> str:
    total_ms = max(0, int(round(float(ms))))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    seconds = total_ms // 1_000
    milliseconds = total_ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _srt_block(index: int, start_ms: float, end_ms: float, text: str) -> str:
    if end_ms <= start_ms:
        end_ms = start_ms + 500
    return f"{index}\n{_ms_to_srt_time(start_ms)} --> {_ms_to_srt_time(end_ms)}\n{text.strip()}\n"


def _timestamp_ms(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise FunAsrError(f"Fun-ASR 转写结果时间戳无效: {field_name}={value!r}") from exc


def _include_speaker_label() -> bool:
    try:
        from app.config import config

        section = config.transcription if hasattr(config, "transcription") else {}
        if isinstance(section, dict) and "include_speaker_label" in section:
            return bool(section.get("include_speaker_label"))
    except Exception:
        pass
    return False


def _speaker_prefix(speaker_id: Any) -> str:
    if not _include_speaker_label():
        return ""
    if speaker_id is None or speaker_id == "":
        return ""
    try:
        label = int(speaker_id) + 1
    except (TypeError, ValueError):
        label = str(speaker_id)
    return f"说话人{label}: "


def _iter_sentences(result_json: dict[str, Any]):
    transcripts = result_json.get("transcripts")
    if transcripts is None and "sentences" in result_json:
        transcripts = [{"sentences": result_json.get("sentences") or []}]
    if not transcripts:
        raise FunAsrError("Fun-ASR 转写结果为空：未找到 transcripts")
    for transcript in transcripts:
        for sentence in transcript.get("sentences") or []:
            yield sentence


def _word_text(word: dict[str, Any]) -> str:
    text = str(word.get("text") or word.get("word") or "")
    punctuation = str(word.get("punctuation") or "")
    if punctuation and not text.endswith(punctuation):
        text += punctuation
    return text


def _flush_block(blocks: list[dict[str, Any]], current: dict[str, Any]) -> None:
    text = current.get("text", "").strip()
    if text:
        blocks.append(current.copy())


def _blocks_from_words(sentence: dict[str, Any], max_chars: int, max_duration: float) -> list[dict[str, Any]]:
    words = sentence.get("words") or []
    blocks: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    max_duration_ms = max_duration * 1000
    sentence_speaker = sentence.get("speaker_id")

    for word in words:
        text = _word_text(word)
        if not text:
            continue
        start = word.get("begin_time", word.get("start_time"))
        end = word.get("end_time")
        if start is None or end is None:
            continue
        speaker_id = word.get("speaker_id", sentence_speaker)
        start_ms = _timestamp_ms(start, "word.begin_time")
        end_ms = _timestamp_ms(end, "word.end_time")

        if current is None:
            current = {"start": start_ms, "end": end_ms, "text": text, "speaker_id": speaker_id}
        else:
            should_split_before = (
                speaker_id != current.get("speaker_id")
                or len(current["text"] + text) > max_chars
                or (end_ms - current["start"]) > max_duration_ms
            )
            if should_split_before:
                _flush_block(blocks, current)
                current = {"start": start_ms, "end": end_ms, "text": text, "speaker_id": speaker_id}
            else:
                current["text"] += text
                current["end"] = end_ms

        if current and text[-1:] in PUNCTUATION_BREAKS:
            _flush_block(blocks, current)
            current = None

    if current:
        _flush_block(blocks, current)
    return blocks


def _split_text(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for char in text:
        current += char
        if char in PUNCTUATION_BREAKS or len(current) >= max_chars:
            chunks.append(current.strip())
            current = ""
    if current.strip():
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def _blocks_from_sentence(sentence: dict[str, Any], max_chars: int) -> list[dict[str, Any]]:
    text = str(sentence.get("text") or "").strip()
    if not text:
        return []
    start = sentence.get("begin_time", 0)
    end = sentence.get("end_time")
    start_ms = _timestamp_ms(start, "sentence.begin_time")
    end_ms = _timestamp_ms(end, "sentence.end_time") if end is not None else start_ms + 500
    chunks = _split_text(text, max_chars)
    if not chunks:
        return []
    duration = max(500.0, end_ms - start_ms)
    total_chars = max(1, sum(len(chunk) for chunk in chunks))
    cursor = start_ms
    blocks: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        if i == len(chunks) - 1:
            chunk_end = end_ms
        else:
            chunk_end = cursor + duration * (len(chunk) / total_chars)
        blocks.append(
            {
                "start": cursor,
                "end": max(cursor + 200, chunk_end),
                "text": chunk,
                "speaker_id": sentence.get("speaker_id"),
            }
        )
        cursor = chunk_end
    return blocks


def fun_asr_result_to_srt(result_json: dict[str, Any], max_chars: int = 20, max_duration: float = 3.5) -> str:
    """Convert downloaded Fun-ASR JSON into fine-grained SRT.

    Official downloaded schema is `transcripts[*].sentences[*].words[*]`.
    Fun-ASR timestamps are milliseconds.
    """
    blocks: list[dict[str, Any]] = []
    for sentence in _iter_sentences(result_json):
        sentence_blocks = _blocks_from_words(sentence, max_chars, max_duration)
        if not sentence_blocks:
            sentence_blocks = _blocks_from_sentence(sentence, max_chars)
        blocks.extend(sentence_blocks)

    if not blocks:
        raise FunAsrError("Fun-ASR 转写结果为空：未找到可用字幕内容")

    lines = []
    for index, block in enumerate(blocks, start=1):
        raw_text = f"{_speaker_prefix(block.get('speaker_id'))}{block['text']}"
        text = clean_subtitle_dialogue_text(raw_text)
        if not text:
            continue
        lines.append(_srt_block(index, block["start"], block["end"], text))
    return "\n".join(lines).rstrip() + "\n"


def write_srt_file(srt_content: str, subtitle_file: str = "") -> str:
    if not subtitle_file:
        subtitle_file = os.path.join(utils.subtitle_dir(), f"fun_asr_{int(time.time())}.srt")
    parent = os.path.dirname(subtitle_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return subtitle_file


def create_with_fun_asr(
    local_file: str,
    subtitle_file: str = "",
    api_key: str = "",
    speaker_count: Optional[int] = None,
    poll_interval: float = 2.0,
    timeout: float = 600.0,
    session=requests,
) -> Optional[str]:
    """Upload local media to Bailian temporary storage and create a Fun-ASR SRT file."""
    api_key = _require_api_key(api_key)
    try:
        policy = request_upload_policy(api_key, session=session)
        oss_url = upload_to_temporary_oss(local_file, policy, session=session)
        task_id = submit_transcription_task(api_key, oss_url, speaker_count=speaker_count, session=session)
        task_result = poll_transcription_task(
            api_key,
            task_id,
            poll_interval=poll_interval,
            timeout=timeout,
            session=session,
        )
        transcription_url = task_result.get("transcription_url")
        result_json = download_transcription_result(transcription_url, session=session)
        srt_content = fun_asr_result_to_srt(result_json)
        output_file = write_srt_file(srt_content, subtitle_file)
        logger.info(f"Fun-ASR 字幕文件已生成: {output_file}")
        return output_file
    except FunAsrError:
        raise
    except Exception as exc:
        raise FunAsrError(f"Fun-ASR 字幕转写失败: {exc}") from exc
