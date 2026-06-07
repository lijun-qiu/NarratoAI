#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
影视解说：字幕 + 关键帧视觉模型增强。

1. 剧情分析前：每 30 秒抽帧 + 对照字幕 → 视觉观察并入一体化剧情分析
2. 脚本生成后：按各片段时间点抽帧 → 优化 OST=0 解说词与 OST=1 picture 旁白
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from app.config import config
from app.services.film_tv_script_optimizer import AUTO_NARRATION_MARKER, parse_timestamp_range
from app.services.film_tv_settings import get_film_tv_settings
from app.services.llm.migration_adapter import VisionAnalyzerAdapter, _run_async_safely
from app.services.srt_utils import parse_srt
from app.utils.video_processor import VideoProcessor


def compute_sample_timestamps(
    duration_sec: float,
    *,
    interval_sec: float = 30.0,
    max_samples: int = 80,
) -> List[float]:
    """在原片时长内均匀采样时间点（秒）。"""
    duration = max(float(duration_sec or 0), 0.0)
    if duration <= 0:
        return []

    interval = max(float(interval_sec or 0), 1.0)
    cap = max(int(max_samples or 0), 1)
    timestamps: List[float] = []

    ts = min(interval * 0.5, duration * 0.02)
    while ts < duration and len(timestamps) < cap:
        timestamps.append(round(ts, 3))
        ts += interval

    if not timestamps:
        timestamps.append(round(min(1.0, duration * 0.5), 3))
    return timestamps[:cap]


def _format_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _parse_json_payload(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return None

    candidates = [text]
    block = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if block:
        candidates.insert(0, block.group(1).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def resolve_vision_credentials() -> Optional[Dict[str, Any]]:
    from app.config.llm_gateway_router import resolve_llm_credentials

    provider = (config.app.get("vision_llm_provider") or "openai").lower()
    model = config.app.get(f"vision_{provider}_model_name") or ""
    api_key, base_url = resolve_llm_credentials(model, role="vision")
    if not api_key or not model:
        return None
    return {
        "provider": provider,
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
        "batch_size": max(int(config.frames.get("vision_batch_size", 5) or 5), 1),
        "max_concurrency": max(int(config.frames.get("vision_max_concurrency", 2) or 2), 1),
    }


def _extract_frames_at_timestamps(
    video_path: str,
    timestamps: List[float],
    work_dir: str,
) -> List[Tuple[float, str]]:
    if not timestamps:
        return []
    os.makedirs(work_dir, exist_ok=True)
    processor = VideoProcessor(video_path)
    extracted: List[Tuple[float, str]] = []
    for index, ts in enumerate(timestamps):
        output_path = os.path.join(work_dir, f"ftv_frame_{index:04d}_{int(ts * 1000)}.jpg")
        if processor._extract_frame_ultra_compatible(ts, output_path):
            extracted.append((ts, output_path))
        else:
            logger.warning(f"抽帧失败: {ts:.2f}s")
    return extracted


def _analyze_images(
    image_paths: List[str],
    prompt: str,
    vision_cfg: Dict[str, Any],
) -> str:
    if not image_paths:
        return ""
    analyzer = VisionAnalyzerAdapter(
        provider=vision_cfg["provider"],
        api_key=vision_cfg["api_key"],
        model=vision_cfg["model"],
        base_url=vision_cfg.get("base_url") or None,
    )
    batches = _run_async_safely(
        analyzer.analyze_images,
        images=image_paths,
        prompt=prompt,
        batch_size=vision_cfg["batch_size"],
        max_concurrency=vision_cfg["max_concurrency"],
    )
    parts = [str(batch.get("response") or "").strip() for batch in batches if batch.get("response")]
    return "\n".join(part for part in parts if part)


def _observations_from_scene_response(response_text: str, frame_times: List[float]) -> List[Dict[str, Any]]:
    parsed = _parse_json_payload(response_text)
    observations: List[Dict[str, Any]] = []

    if isinstance(parsed, dict):
        raw_items = parsed.get("observations") or parsed.get("frame_observations") or []
        if isinstance(raw_items, list):
            for index, item in enumerate(raw_items):
                if isinstance(item, dict):
                    scene = str(item.get("scene") or item.get("observation") or "").strip()
                    time_sec = item.get("time_sec")
                    if time_sec is None and index < len(frame_times):
                        time_sec = frame_times[index]
                    if scene:
                        row = {"time_sec": float(time_sec or 0), "scene": scene}
                        hint = str(item.get("subtitle_hint") or "").strip()
                        if hint:
                            row["subtitle_hint"] = hint
                        observations.append(row)
                elif isinstance(item, str) and item.strip():
                    observations.append(
                        {"time_sec": frame_times[index] if index < len(frame_times) else 0.0, "scene": item.strip()}
                    )

    if observations:
        return observations

    for index, line in enumerate(response_text.splitlines()):
        line = line.strip().lstrip("-*0123456789. ")
        if len(line) >= 4:
            observations.append(
                {
                    "time_sec": frame_times[index] if index < len(frame_times) else 0.0,
                    "scene": line[:80],
                }
            )
    return observations


def format_vision_scene_notes(observations: List[Dict[str, Any]]) -> str:
    if not observations:
        return ""
    lines = [
        "**视觉拉片时间轴（约每 30 秒一帧，供与字幕交叉验证）**",
        "说明：下列为画面观察，对白时间戳请对照下方字幕区块。",
    ]
    for item in observations:
        ts = _format_hms(float(item.get("time_sec") or 0))
        scene = str(item.get("scene") or "").strip()
        subtitle_hint = str(item.get("subtitle_hint") or "").strip()
        if scene:
            line = f"* [{ts}] 画面：{scene}"
            if subtitle_hint:
                line += f"｜附近对白：{subtitle_hint}"
            lines.append(line)
    return "\n".join(lines)


def _subtitle_hint_for_timestamp(
    subtitle_content: str,
    time_sec: float,
    *,
    window_sec: float = 20.0,
    max_lines: int = 2,
) -> str:
    """取时间点附近字幕对白摘要。"""
    entries = parse_srt(subtitle_content or "")
    if not entries:
        return ""

    center_ms = int(max(0.0, time_sec) * 1000)
    window_ms = int(max(window_sec, 1.0) * 1000)
    start_ms = center_ms - window_ms
    end_ms = center_ms + window_ms

    matched: List[str] = []
    for entry in entries:
        if entry.end_ms < start_ms or entry.start_ms > end_ms:
            continue
        text = (entry.text or "").strip().replace("\n", " ")
        if text:
            matched.append(text)
        if len(matched) >= max_lines:
            break

    if not matched:
        return ""
    hint = " / ".join(matched)
    return hint[:120] + ("…" if len(hint) > 120 else "")


def _build_scene_analysis_prompt(
    film_name: str,
    batch_times: List[float],
    subtitle_content: str,
) -> str:
    lines = [
        f"你是专业影视拉片师，正在为《{film_name}》做「字幕 + 画面」联合分析。",
        f"以下 {len(batch_times)} 张图按时间顺序从原片抽取（约每 30 秒一帧）。",
        "每张图附有该时间点附近字幕，请对照画面描述人物、表情、动作、场景与悬疑/冲突氛围。",
        "",
    ]
    for index, time_sec in enumerate(batch_times, start=1):
        hint = _subtitle_hint_for_timestamp(subtitle_content, time_sec)
        lines.append(f"【帧 {index}】约 {_format_hms(time_sec)}（{time_sec:.1f}s）")
        lines.append(f"附近字幕：{hint or '（该时段无字幕）'}")
        lines.append("")

    lines.extend(
        [
            "只输出 JSON：",
            '{"observations":[{"time_sec":123.4,"scene":"25字内画面描述","subtitle_hint":"可选"}, ...]}',
            f"observations 长度必须为 {len(batch_times)}，time_sec 约为：{batch_times}",
            "不要 markdown，不要复述整段对白。",
        ]
    )
    return "\n".join(lines)


def collect_vision_scene_notes(
    *,
    video_path: str,
    film_name: str,
    subtitle_content: str = "",
    source_duration_sec: float,
    settings: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """均匀抽帧并对照字幕，生成供剧情分析使用的视觉观察文本。"""
    cfg = settings or {}
    if not cfg.get("enable_vision_enrichment", True):
        return ""

    vision_cfg = resolve_vision_credentials()
    if not vision_cfg:
        logger.warning("未配置视觉模型 API，跳过视觉拉片")
        return ""

    interval = float(cfg.get("vision_scene_interval_sec") or 30)
    max_samples = int(cfg.get("vision_max_scene_samples") or 80)
    timestamps = compute_sample_timestamps(
        source_duration_sec,
        interval_sec=interval,
        max_samples=max_samples,
    )
    if not timestamps:
        return ""

    if progress_callback:
        progress_callback(f"剧情拉片：每 {int(interval)} 秒 1 帧，共 {len(timestamps)} 帧（对照字幕）...")

    work_dir = tempfile.mkdtemp(prefix="ftv_vision_plot_")
    all_observations: List[Dict[str, Any]] = []
    batch_size = min(vision_cfg["batch_size"], 10)

    try:
        frames = _extract_frames_at_timestamps(video_path, timestamps, work_dir)
        if not frames:
            logger.warning("未能抽取任何关键帧，跳过视觉拉片")
            return ""

        frame_map = {round(ts, 3): path for ts, path in frames}
        ordered_times = [ts for ts in timestamps if round(ts, 3) in frame_map]
        ordered_paths = [frame_map[round(ts, 3)] for ts in ordered_times]

        for offset in range(0, len(ordered_times), batch_size):
            batch_times = ordered_times[offset : offset + batch_size]
            batch_paths = ordered_paths[offset : offset + batch_size]
            prompt = _build_scene_analysis_prompt(film_name, batch_times, subtitle_content)
            if progress_callback:
                progress_callback(
                    f"视觉分析第 {offset // batch_size + 1} 批（{len(batch_paths)} 帧）..."
                )
            response_text = _analyze_images(batch_paths, prompt, vision_cfg)
            batch_obs = _observations_from_scene_response(response_text, batch_times)
            for obs, ts in zip(batch_obs, batch_times):
                if not obs.get("subtitle_hint"):
                    obs["subtitle_hint"] = _subtitle_hint_for_timestamp(subtitle_content, ts)
                all_observations.append(obs)

        notes = format_vision_scene_notes(all_observations)
        if notes:
            logger.info(f"视觉拉片完成：{len(all_observations)} 帧，已对齐字幕")
        return notes
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


_PLAYBACK_ORIGINAL_RE = re.compile(r"^播放原片\s*\d*", re.IGNORECASE)


def _is_playback_original(text: str) -> bool:
    return bool(_PLAYBACK_ORIGINAL_RE.match((text or "").strip()))


def _cjk_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _clamp_cjk_text(text: str, min_chars: int, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", "", (text or "").strip())
    if not cleaned:
        return ""
    if _cjk_length(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars]


def _resolve_narration_char_limits(
    item: Dict[str, Any],
    *,
    is_opening: bool,
    settings: Dict[str, Any],
) -> Tuple[int, int]:
    chars_min = int(settings.get("narration_chars_min") or 48)
    chars_max = int(settings.get("narration_chars_max") or 72)
    if is_opening:
        chars_max = max(chars_max, int(settings.get("opening_chars_max") or 110))
    return chars_min, chars_max


def _build_segment_targets(
    items: List[Dict[str, Any]],
    settings: Dict[str, Any],
    *,
    enrich_narration: bool,
    enrich_picture: bool,
    max_items: int,
) -> List[Dict[str, Any]]:
    first_ost0_index = next(
        (idx for idx, item in enumerate(items) if int(item.get("OST") or 0) == 0),
        None,
    )
    targets: List[Dict[str, Any]] = []

    for index, item in enumerate(items):
        ost = int(item.get("OST") or 0)
        timestamp = str(item.get("timestamp") or "")
        midpoint = _segment_midpoint_sec(timestamp)
        if midpoint is None:
            continue

        narration = str(item.get("narration") or "").strip()
        picture = str(item.get("picture") or "").strip()
        need_narration = (
            enrich_narration
            and ost == 0
            and narration
            and narration != AUTO_NARRATION_MARKER
            and not _is_playback_original(narration)
        )
        need_picture = enrich_picture and ost == 1
        if not need_narration and not need_picture:
            continue

        chars_min, chars_max = _resolve_narration_char_limits(
            item,
            is_opening=(index == first_ost0_index),
            settings=settings,
        )
        targets.append(
            {
                "item_index": index,
                "time_sec": midpoint,
                "timestamp": timestamp,
                "ost": ost,
                "draft_narration": narration,
                "draft_picture": picture,
                "need_narration": need_narration,
                "need_picture": need_picture,
                "chars_min": chars_min,
                "chars_max": chars_max,
            }
        )

    return targets[: max(int(max_items or 0), 1)]


def _parse_segment_refinement_response(
    response_text: str,
    expected_count: int,
) -> List[Dict[str, str]]:
    parsed = _parse_json_payload(response_text)
    rows: List[Any] = []
    if isinstance(parsed, list):
        rows = parsed
    elif isinstance(parsed, dict):
        rows = parsed.get("segments") or parsed.get("items") or parsed.get("results") or []

    results: List[Dict[str, str]] = []
    for entry in rows[:expected_count]:
        if not isinstance(entry, dict):
            results.append({})
            continue
        results.append(
            {
                "narration": str(entry.get("narration") or entry.get("optimized_narration") or "").strip(),
                "picture": str(entry.get("picture") or entry.get("optimized_picture") or "").strip(),
            }
        )
    while len(results) < expected_count:
        results.append({})
    return results


def _build_refinement_prompt(
    film_name: str,
    batch: List[Dict[str, Any]],
    *,
    picture_chars_max: int = 12,
) -> str:
    lines = [
        f"你是拥有二十年经验的《{film_name}》影视解说剪辑大师，擅长悬疑节奏与情绪化解说。",
        f"以下 {len(batch)} 张图按顺序对应解说脚本片段的代表画面。",
        "请对照画面，修正初稿中的错误或空泛描述，使旁白更贴合视频、更准确、更有感染力。",
        "",
        "硬性要求：",
        "- OST=0 解说：重写/优化 narration，字数必须在给定范围内；承上启下，有观点有情绪",
        f"- OST=1 原声段：优化 picture（{picture_chars_max} 字以内），精简承上启下，描述画面/神情/动作/氛围，禁止复述对白、禁止长句",
        "- 禁止输出「播放原片」、禁止 markdown、禁止引号包裹",
        "- 若初稿已基本准确，仅做微调，不要完全改题",
        "",
        "片段列表（与图片顺序一致）：",
    ]
    for seq, seg in enumerate(batch, start=1):
        ost = seg["ost"]
        if ost == 0:
            lines.append(
                f"{seq}. [OST=0 解说] 时间 {seg['timestamp']} "
                f"字数 {seg['chars_min']}-{seg['chars_max']} 初稿：{seg['draft_narration']}"
            )
        else:
            draft = seg["draft_picture"] or "（空）"
            lines.append(
                f"{seq}. [OST=1 原声旁白] 时间 {seg['timestamp']} picture 初稿：{draft}"
            )

    lines.extend(
        [
            "",
            f"只输出 JSON 数组，长度必须为 {len(batch)}：",
            '[{"narration":"..."}, {"picture":"..."}]',
            "OST=0 条目只填 narration；OST=1 条目只填 picture。",
        ]
    )
    return "\n".join(lines)


def enrich_script_with_vision(
    *,
    video_path: str,
    film_name: str,
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """抽帧并用视觉模型优化 OST=0 解说词与 OST=1 picture 旁白。"""
    cfg = get_film_tv_settings(settings)
    if not cfg.get("enable_vision_enrichment", True):
        return items

    enrich_narration = bool(cfg.get("vision_enrich_narration", True))
    enrich_picture = bool(cfg.get("vision_enrich_picture", True))
    if not enrich_narration and not enrich_picture:
        return items

    vision_cfg = resolve_vision_credentials()
    if not vision_cfg:
        logger.warning("未配置视觉模型 API，跳过脚本旁白视觉优化")
        return items

    picture_chars_max = max(int(cfg.get("picture_chars_max") or 12), 4)
    max_total = int(cfg.get("max_total_segments") or 36)
    vision_cap = max(int(cfg.get("vision_segment_max_items") or cfg.get("vision_picture_max_items") or 30), 1)
    max_items = min(vision_cap, max_total) if max_total > 0 else vision_cap
    targets = _build_segment_targets(
        items,
        cfg,
        enrich_narration=enrich_narration,
        enrich_picture=enrich_picture,
        max_items=max_items,
    )
    if not targets:
        return items

    if progress_callback:
        progress_callback(f"正在为 {len(targets)} 个片段优化旁白（对照画面）...")

    work_dir = tempfile.mkdtemp(prefix="ftv_vision_refine_")
    updated = [dict(item) for item in items]
    narration_updated = 0
    picture_updated = 0

    try:
        batch_size = vision_cfg["batch_size"]
        for offset in range(0, len(targets), batch_size):
            batch = targets[offset : offset + batch_size]
            timestamps = [seg["time_sec"] for seg in batch]
            frames = _extract_frames_at_timestamps(video_path, timestamps, work_dir)
            if len(frames) < len(batch):
                logger.warning(f"旁白优化批次抽帧不足: {len(frames)}/{len(batch)}")
                continue

            image_paths = [path for _, path in frames]
            prompt = _build_refinement_prompt(
                film_name, batch, picture_chars_max=picture_chars_max
            )
            response_text = _analyze_images(image_paths, prompt, vision_cfg)
            refinements = _parse_segment_refinement_response(response_text, len(batch))

            for seg, ref in zip(batch, refinements):
                item_index = seg["item_index"]
                if seg["need_narration"]:
                    new_narration = ref.get("narration") or ""
                    new_narration = _clamp_cjk_text(
                        new_narration,
                        seg["chars_min"],
                        seg["chars_max"],
                    )
                    if (
                        new_narration
                        and _cjk_length(new_narration) >= max(seg["chars_min"] - 12, 20)
                        and not _is_playback_original(new_narration)
                    ):
                        updated[item_index]["narration"] = new_narration
                        narration_updated += 1

                if seg["need_picture"]:
                    new_picture = (ref.get("picture") or "").strip()
                    new_picture = _clamp_cjk_text(new_picture, 4, picture_chars_max)
                    if _cjk_length(new_picture) >= 4:
                        updated[item_index]["picture"] = new_picture
                        picture_updated += 1

        logger.info(
            f"视觉旁白优化完成: 解说 {narration_updated} 段, picture {picture_updated} 段 "
            f"(共处理 {len(targets)} 个片段)"
        )
        return updated
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _segment_midpoint_sec(timestamp: str) -> Optional[float]:
    try:
        start, end = parse_timestamp_range(timestamp)
        if end <= start:
            return start
        return round(start + (end - start) * 0.35, 3)
    except Exception:
        return None


def enrich_plot_analysis_with_vision(
    *,
    video_path: str,
    film_name: str,
    plot_analysis: str,
    source_duration_sec: float,
    subtitle_content: str = "",
    settings: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """在已有剧情分析后追加视觉观察（兼容旧流程）。"""
    notes = collect_vision_scene_notes(
        video_path=video_path,
        film_name=film_name,
        subtitle_content=subtitle_content,
        source_duration_sec=source_duration_sec,
        settings=settings,
        progress_callback=progress_callback,
    )
    if not notes:
        return plot_analysis
    return f"{plot_analysis.rstrip()}\n\n{notes}"


def enrich_script_pictures_with_vision(
    *,
    video_path: str,
    film_name: str,
    items: List[Dict[str, Any]],
    settings: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """兼容旧接口：仅优化 picture 时委托给 enrich_script_with_vision。"""
    merged_settings = dict(settings or {})
    merged_settings.setdefault("vision_enrich_picture", True)
    merged_settings.setdefault("vision_enrich_narration", False)
    return enrich_script_with_vision(
        video_path=video_path,
        film_name=film_name,
        items=items,
        settings=merged_settings,
        progress_callback=progress_callback,
    )
