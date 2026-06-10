#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""从视频导出互不重复的人声 MP3，供用户自行重命名后做听音辨人。"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from pydub import AudioSegment

from app.services.audio_preprocess import extract_audio_chunk, extract_audio_mp3, get_media_duration_seconds
from app.services.documentary.frame_analysis_pairing import sanitize_video_stem
from app.services.srt_utils import clean_subtitle_dialogue_text, parse_srt_file
from app.utils import utils

VOICE_EXPORT_ARTIFACT_VERSION = "voice-export-v1"

_SILENCE_START_RE = re.compile(r"silence_start:\s*([\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([\d.]+)")

DEFAULT_SILENCE_NOISE_DB = -35.0
DEFAULT_SILENCE_MIN_SECONDS = 0.2
DEFAULT_MIN_SPEECH_SECONDS = 0.8
DEFAULT_MERGE_GAP_SECONDS = 0.25
DEFAULT_MAX_CLIP_SECONDS = 12.0
DEFAULT_PRE_ROLL_SECONDS = 0.05
DEFAULT_POST_ROLL_SECONDS = 0.08
DEFAULT_CANDIDATE_CHUNK_SECONDS = 4.0
DEFAULT_CANDIDATE_STEP_SECONDS = 2.0
DEFAULT_VOICE_MIN_DISTANCE = 0.10
DEFAULT_MAX_DISTINCT_VOICES = 16
DEFAULT_FINGERPRINT_BINS = 24


@dataclass(frozen=True)
class SpeechRange:
    start_seconds: float
    end_seconds: float
    hint_text: str = ""


@dataclass
class VoiceCandidate:
    start_seconds: float
    end_seconds: float
    temp_path: str
    hint_text: str = ""
    fingerprint: list[float] | None = None
    quality_score: float = 0.0


def voice_export_dir(video_path: str) -> str:
    stem = sanitize_video_stem(video_path)
    return os.path.join(utils.storage_dir(), "voice_export", stem)


def default_export_index_path(video_path: str) -> str:
    return os.path.join(voice_export_dir(video_path), "export_index.json")


def _format_range_timestamp(start_seconds: float, end_seconds: float) -> str:
    return f"{utils.seconds_to_time(start_seconds)}-{utils.seconds_to_time(end_seconds)}"


def _run_ffmpeg_silencedetect(
    audio_path: str,
    *,
    noise_db: float = DEFAULT_SILENCE_NOISE_DB,
    min_silence_seconds: float = DEFAULT_SILENCE_MIN_SECONDS,
) -> list[tuple[float, float]]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        audio_path,
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence_seconds}",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=600,
        )
    except (subprocess.SubprocessError, OSError):
        return []

    stderr = result.stderr or ""
    silences: list[tuple[float, float]] = []
    pending_start: float | None = None
    for line in stderr.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            pending_start = float(start_match.group(1))
            continue
        end_match = _SILENCE_END_RE.search(line)
        if end_match and pending_start is not None:
            silences.append((pending_start, float(end_match.group(1))))
            pending_start = None
    return silences


def _silence_ranges_to_speech_ranges(
    silence_ranges: list[tuple[float, float]],
    *,
    duration_seconds: float,
) -> list[tuple[float, float]]:
    if duration_seconds <= 0:
        return []

    sorted_silences = sorted(silence_ranges, key=lambda item: item[0])
    speech: list[tuple[float, float]] = []
    cursor = 0.0
    for silence_start, silence_end in sorted_silences:
        if silence_start > cursor + 0.01:
            speech.append((cursor, silence_start))
        cursor = max(cursor, silence_end)
    if cursor < duration_seconds - 0.01:
        speech.append((cursor, duration_seconds))
    return speech


def _merge_close_ranges(
    ranges: list[tuple[float, float]],
    *,
    gap_seconds: float,
) -> list[tuple[float, float]]:
    if not ranges:
        return []
    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges, key=lambda item: item[0]):
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap_seconds:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _split_long_ranges(
    ranges: list[tuple[float, float]],
    *,
    max_seconds: float,
) -> list[tuple[float, float]]:
    if max_seconds <= 0:
        return ranges
    split: list[tuple[float, float]] = []
    for start, end in ranges:
        cursor = start
        while cursor < end - 0.01:
            seg_end = min(cursor + max_seconds, end)
            if seg_end > cursor + 0.01:
                split.append((cursor, seg_end))
            cursor = seg_end
    return split


def _chunk_ranges_for_candidates(
    ranges: list[SpeechRange],
    *,
    chunk_seconds: float = DEFAULT_CANDIDATE_CHUNK_SECONDS,
    step_seconds: float = DEFAULT_CANDIDATE_STEP_SECONDS,
    min_seconds: float = DEFAULT_MIN_SPEECH_SECONDS,
) -> list[SpeechRange]:
    """将长人声段切成较短滑窗，便于按音色聚类去重。"""
    chunked: list[SpeechRange] = []
    for item in ranges:
        duration = item.end_seconds - item.start_seconds
        if duration <= chunk_seconds + 0.05:
            if duration >= min_seconds:
                chunked.append(item)
            continue
        cursor = item.start_seconds
        while cursor + min_seconds <= item.end_seconds:
            seg_end = min(cursor + chunk_seconds, item.end_seconds)
            if seg_end - cursor >= min_seconds:
                chunked.append(
                    SpeechRange(
                        start_seconds=round(cursor, 3),
                        end_seconds=round(seg_end, 3),
                        hint_text=item.hint_text,
                    )
                )
            cursor += step_seconds
            if item.end_seconds - cursor < min_seconds:
                break
    return chunked


def normalize_speech_ranges(
    ranges: list[tuple[float, float]],
    *,
    duration_seconds: float,
    min_speech_seconds: float = DEFAULT_MIN_SPEECH_SECONDS,
    merge_gap_seconds: float = DEFAULT_MERGE_GAP_SECONDS,
    max_clip_seconds: float = DEFAULT_MAX_CLIP_SECONDS,
    pre_roll_seconds: float = DEFAULT_PRE_ROLL_SECONDS,
    post_roll_seconds: float = DEFAULT_POST_ROLL_SECONDS,
) -> list[SpeechRange]:
    padded: list[tuple[float, float]] = []
    for start, end in ranges:
        padded_start = max(0.0, start - pre_roll_seconds)
        padded_end = min(duration_seconds, end + post_roll_seconds)
        if padded_end - padded_start >= min_speech_seconds:
            padded.append((padded_start, padded_end))

    merged = _merge_close_ranges(padded, gap_seconds=merge_gap_seconds)
    split = _split_long_ranges(merged, max_seconds=max_clip_seconds)
    return [
        SpeechRange(start_seconds=round(start, 3), end_seconds=round(end, 3))
        for start, end in split
        if end - start >= min_speech_seconds
    ]


def detect_speech_ranges_from_audio(
    audio_path: str,
    *,
    noise_db: float = DEFAULT_SILENCE_NOISE_DB,
    min_silence_seconds: float = DEFAULT_SILENCE_MIN_SECONDS,
    min_speech_seconds: float = DEFAULT_MIN_SPEECH_SECONDS,
    merge_gap_seconds: float = DEFAULT_MERGE_GAP_SECONDS,
    max_clip_seconds: float = DEFAULT_MAX_CLIP_SECONDS,
) -> list[SpeechRange]:
    duration = get_media_duration_seconds(audio_path)
    if duration <= 0:
        return []
    silences = _run_ffmpeg_silencedetect(
        audio_path,
        noise_db=noise_db,
        min_silence_seconds=min_silence_seconds,
    )
    raw_ranges = _silence_ranges_to_speech_ranges(silences, duration_seconds=duration)
    return normalize_speech_ranges(
        raw_ranges,
        duration_seconds=duration,
        min_speech_seconds=min_speech_seconds,
        merge_gap_seconds=merge_gap_seconds,
        max_clip_seconds=max_clip_seconds,
    )


def speech_ranges_from_subtitle(subtitle_path: str) -> list[SpeechRange]:
    entries = parse_srt_file(subtitle_path)
    ranges: list[SpeechRange] = []
    for entry in entries:
        start = entry.start_ms / 1000.0
        end = entry.end_ms / 1000.0
        if end <= start:
            continue
        ranges.append(
            SpeechRange(
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                hint_text=clean_subtitle_dialogue_text(entry.text),
            )
        )
    return ranges


def _audio_to_mono_array(audio_path: str, *, target_rate: int = 8000) -> np.ndarray:
    segment = AudioSegment.from_file(audio_path).set_channels(1).set_frame_rate(target_rate)
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if samples.size == 0:
        return samples
    peak = float(segment.max_possible_amplitude or 1)
    return samples / peak


def compute_voice_fingerprint(
    audio_path: str,
    *,
    bins: int = DEFAULT_FINGERPRINT_BINS,
) -> list[float]:
    """轻量音色指纹：对数频带能量，用于粗聚类去重。"""
    samples = _audio_to_mono_array(audio_path)
    if samples.size < bins * 8:
        return [0.0] * bins

    # 取中间较稳定的一段，减少首尾静音干扰
    if samples.size > bins * 64:
        trim = int(samples.size * 0.15)
        samples = samples[trim : samples.size - trim]

    spectrum = np.abs(np.fft.rfft(samples))
    if spectrum.size <= 1:
        return [0.0] * bins

    freq_bins = np.array_split(spectrum[1:], bins)
    energies = [float(np.mean(chunk) + 1e-9) for chunk in freq_bins if chunk.size]
    if len(energies) < bins:
        energies.extend([1e-9] * (bins - len(energies)))
    log_energies = np.log(np.array(energies[:bins], dtype=np.float64))
    normalized = log_energies - np.mean(log_energies)
    norm = float(np.linalg.norm(normalized))
    if norm <= 1e-9:
        return [0.0] * bins
    return (normalized / norm).tolist()


def fingerprint_distance(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 1.0
    return float(np.mean(np.abs(np.array(left) - np.array(right))))


def _clip_quality_score(audio_path: str) -> float:
    segment = AudioSegment.from_file(audio_path)
    duration = len(segment) / 1000.0
    if duration <= 0:
        return 0.0
    rms = segment.rms / float(segment.max_possible_amplitude or 1)
    if duration < 1.5:
        duration_score = 0.35
    elif duration <= 8.0:
        duration_score = 1.0
    elif duration <= 12.0:
        duration_score = 0.75
    else:
        duration_score = 0.5
    return duration_score * 0.45 + rms * 0.55


def _prepare_candidates(candidates: list[VoiceCandidate]) -> list[VoiceCandidate]:
    prepared: list[VoiceCandidate] = []
    for candidate in candidates:
        if not candidate.fingerprint:
            candidate.fingerprint = compute_voice_fingerprint(candidate.temp_path)
        if candidate.quality_score <= 0 and os.path.isfile(candidate.temp_path):
            candidate.quality_score = _clip_quality_score(candidate.temp_path)
        elif candidate.quality_score <= 0:
            duration = max(0.0, candidate.end_seconds - candidate.start_seconds)
            candidate.quality_score = min(1.0, duration / 6.0)
        prepared.append(candidate)
    return prepared


def select_distinct_voice_representatives(
    candidates: list[VoiceCandidate],
    *,
    min_distance: float = DEFAULT_VOICE_MIN_DISTANCE,
    max_voices: int = DEFAULT_MAX_DISTINCT_VOICES,
) -> list[VoiceCandidate]:
    """
    最远点采样：优先保留音质好、且彼此音色差异足够大的片段。
    避免把不同说话人误合并成一条。
    """
    if not candidates:
        return []

    prepared = _prepare_candidates(candidates)
    ordered = sorted(prepared, key=lambda item: item.quality_score, reverse=True)
    picks: list[VoiceCandidate] = [ordered[0]]

    while len(picks) < min(max_voices, len(ordered)):
        best_candidate: VoiceCandidate | None = None
        best_spacing = -1.0
        for candidate in ordered:
            if candidate in picks:
                continue
            spacing = min(
                fingerprint_distance(candidate.fingerprint or [], pick.fingerprint or [])
                for pick in picks
            )
            if spacing > best_spacing:
                best_spacing = spacing
                best_candidate = candidate
        if best_candidate is None or best_spacing < min_distance:
            break
        picks.append(best_candidate)
    return picks


def _count_similar_candidates(candidate: VoiceCandidate, candidates: list[VoiceCandidate], *, min_distance: float) -> int:
    count = 0
    for other in candidates:
        if fingerprint_distance(candidate.fingerprint or [], other.fingerprint or []) < min_distance:
            count += 1
    return count


def load_export_index(path: str) -> dict[str, Any] | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def save_export_index(path: str, payload: dict[str, Any]) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return path


def export_distinct_voice_mp3s(
    video_path: str,
    *,
    output_dir: str = "",
    subtitle_path: str = "",
    drama_id: str = "",
    min_voice_distance: float = DEFAULT_VOICE_MIN_DISTANCE,
    max_distinct_voices: int = DEFAULT_MAX_DISTINCT_VOICES,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """
    导出视频中互不重复的人声 MP3 到目录：voice_001.mp3, voice_002.mp3, ...

    用户可自行将文件重命名为角色名（如 秦枫.mp3）以建立听音辨人样本库。
    """
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f"视频不存在: {video_path}")

    export_dir = (output_dir or voice_export_dir(video_path)).strip()
    work_dir = os.path.join(export_dir, "_work")
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    if progress_callback:
        progress_callback(5, "正在提取音轨...")

    prepared_audio = os.path.join(work_dir, "source_audio.mp3")
    extract_audio_mp3(video_path, prepared_audio)
    duration_seconds = get_media_duration_seconds(prepared_audio)

    if progress_callback:
        progress_callback(15, "正在定位人声片段...")

    if subtitle_path and os.path.isfile(subtitle_path):
        source_ranges = speech_ranges_from_subtitle(subtitle_path)
        source_label = "subtitle"
    else:
        source_ranges = detect_speech_ranges_from_audio(prepared_audio)
        source_ranges = _chunk_ranges_for_candidates(source_ranges)
        source_label = "vad"

    source_ranges = [
        item for item in source_ranges if item.end_seconds - item.start_seconds >= DEFAULT_MIN_SPEECH_SECONDS
    ]
    if not source_ranges:
        raise RuntimeError("未检测到人声片段，请先确认视频含对白，或先完成字幕转录。")

    if progress_callback:
        progress_callback(30, f"已定位 {len(source_ranges)} 段候选人声，正在分析音色...")

    candidates: list[VoiceCandidate] = []
    for index, speech_range in enumerate(source_ranges, start=1):
        temp_path = os.path.join(work_dir, f"candidate_{index:04d}.mp3")
        extract_audio_chunk(
            prepared_audio,
            temp_path,
            start_sec=speech_range.start_seconds,
            duration_sec=speech_range.end_seconds - speech_range.start_seconds,
        )
        candidates.append(
            VoiceCandidate(
                start_seconds=speech_range.start_seconds,
                end_seconds=speech_range.end_seconds,
                temp_path=temp_path,
                hint_text=speech_range.hint_text,
            )
        )

    representatives = select_distinct_voice_representatives(
        candidates,
        min_distance=min_voice_distance,
        max_voices=max_distinct_voices,
    )

    if progress_callback:
        progress_callback(70, f"识别到 {len(representatives)} 种不同人声，正在导出 MP3...")

    # 清理旧导出，避免编号混乱
    for name in os.listdir(export_dir):
        if name.startswith("voice_") and name.lower().endswith(".mp3"):
            try:
                os.remove(os.path.join(export_dir, name))
            except OSError:
                pass

    exports: list[dict[str, Any]] = []
    for index, rep in enumerate(representatives, start=1):
        output_name = f"voice_{index:03d}.mp3"
        output_path = os.path.join(export_dir, output_name)
        shutil.copy2(rep.temp_path, output_path)
        similar_count = _count_similar_candidates(rep, candidates, min_distance=min_voice_distance)
        exports.append(
            {
                "file_name": output_name,
                "file_path": output_path,
                "timestamp": _format_range_timestamp(rep.start_seconds, rep.end_seconds),
                "start_seconds": rep.start_seconds,
                "end_seconds": rep.end_seconds,
                "duration_seconds": round(rep.end_seconds - rep.start_seconds, 3),
                "hint_text": rep.hint_text,
                "similar_segment_count": similar_count,
                "rename_hint": "请自行重命名为角色名，例如 秦枫.mp3",
            }
        )

    payload: dict[str, Any] = {
        "artifact_version": VOICE_EXPORT_ARTIFACT_VERSION,
        "video_path": os.path.abspath(video_path),
        "video_stem": sanitize_video_stem(video_path),
        "drama_id": drama_id,
        "duration_seconds": round(duration_seconds, 3),
        "source_mode": source_label,
        "subtitle_path": subtitle_path or "",
        "export_dir": os.path.abspath(export_dir),
        "candidate_count": len(candidates),
        "distinct_voice_count": len(exports),
        "min_voice_distance": min_voice_distance,
        "max_distinct_voices": max_distinct_voices,
        "exports": exports,
    }
    index_path = os.path.join(export_dir, "export_index.json")
    save_export_index(index_path, payload)

    try:
        shutil.rmtree(work_dir)
    except OSError:
        pass

    logger.info(
        f"人声音频导出：{sanitize_video_stem(video_path)} · {len(exports)} 个不重复人声 → {export_dir}"
    )
    if progress_callback:
        progress_callback(100, f"完成，导出 {len(exports)} 个 MP3")
    payload["index_path"] = index_path
    return payload


# 兼容旧调用名
def voice_calibration_dir(video_path: str) -> str:
    return voice_export_dir(video_path)


def default_manifest_path(video_path: str) -> str:
    return default_export_index_path(video_path)


def load_manifest(path: str) -> dict[str, Any] | None:
    return load_export_index(path)


def save_manifest(path: str, payload: dict[str, Any]) -> str:
    return save_export_index(path, payload)


def extract_voice_samples(
    video_path: str,
    *,
    mode: str = "vad",
    subtitle_path: str = "",
    drama_id: str = "",
    progress_callback: Any | None = None,
    **_: Any,
) -> dict[str, Any]:
    return export_distinct_voice_mp3s(
        video_path,
        subtitle_path=subtitle_path if (mode or "").strip().lower() == "subtitle" else "",
        drama_id=drama_id,
        progress_callback=progress_callback,
    )
