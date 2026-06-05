import asyncio
import json
import os
import re
from datetime import datetime
from typing import Any, Callable

from loguru import logger

from app.config import config
from app.services.documentary.frame_analysis_models import FrameBatchResult
from app.services.documentary.frame_analysis_pairing import (
    analysis_artifact_dir as pairing_analysis_artifact_dir,
    default_analysis_path_for_video as pairing_default_analysis_path_for_video,
    is_valid_analysis_artifact as pairing_is_valid_analysis_artifact,
    load_analysis_artifact as pairing_load_analysis_artifact,
    resolve_reusable_analysis_path as pairing_resolve_reusable_analysis_path,
)
from app.services.documentary.documentary_settings import (
    build_frame_highlight_hint,
    get_documentary_settings,
)
from app.services.documentary.documentary_subtitle_enrichment import subtitle_excerpt_for_time_range
from app.services.llm.migration_adapter import create_vision_analyzer
from app.utils import utils, video_processor


class DocumentaryFrameExtractionService:
    PROMPT_TEMPLATE = """
我提供了 {frame_count} 张视频帧，它们按时间顺序排列，代表一个连续的视频片段。
首先，请详细描述每一帧的关键视觉信息（包含：主要内容、人物、动作、表情、肢体细节和场景氛围）。
然后，基于所有帧的分析，请用简洁的语言总结整个视频片段中发生的主要活动或事件流程。
请务必使用 JSON 格式输出。
JSON 必须包含以下键：
- frame_observations: 数组，且长度必须为 {frame_count}
- overall_activity_summary: 字符串，描述整个批次主要活动
示例结构：
{{
  "frame_observations": [
    {{"timestamp": "00:00:00,000", "observation": "画面描述（含动作/表情修饰）"{burned_in_subtitle_example}}}
  ],
  "overall_activity_summary": "本批次主要活动总结"
}}
请务必不要遗漏视频帧，我提供了 {frame_count} 张视频帧，frame_observations 必须包含 {frame_count} 个元素
请只返回 JSON 字符串，不要附加解释文字。
""".strip()

    BURNED_IN_SUBTITLE_PROMPT_SUFFIX = """
同时，请识别每帧画面**底部烧录硬字幕**（烧录在画面上的对白文字，非画面描述）：
- burned_in_subtitle: 硬字幕原文；若无硬字幕、仅水印/logo 或无法辨认，则为空字符串
- has_burned_in_subtitle: 布尔值，画面底部是否清晰可见硬字幕
不要猜测听不清的内容；多行字幕合并为一行。该字段用于对照原字幕文件修正错别字，请尽量逐字准确。
""".strip()

    BURNED_IN_SUBTITLE_JSON_EXAMPLE = (
        ', "burned_in_subtitle": "硬字幕原文", "has_burned_in_subtitle": true'
    )

    async def analyze_video(
        self,
        *,
        video_path: str,
        video_theme: str = "",
        custom_prompt: str = "",
        frame_interval_input: int | float | None = None,
        vision_batch_size: int | None = None,
        vision_llm_provider: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        vision_api_key: str | None = None,
        vision_model_name: str | None = None,
        vision_base_url: str | None = None,
        max_concurrency: int | None = None,
        documentary_settings: dict | None = None,
        subtitle_content: str = "",
    ) -> dict[str, Any]:
        progress = progress_callback or (lambda _p, _m: None)
        doc_settings = documentary_settings or get_documentary_settings()

        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        frame_interval_seconds = self._resolve_frame_interval(frame_interval_input)
        batch_size = self._resolve_batch_size(vision_batch_size)
        concurrency = self._resolve_max_concurrency(max_concurrency)
        provider = (vision_llm_provider or config.app.get("vision_llm_provider", "openai")).lower()

        api_key = vision_api_key if vision_api_key is not None else config.app.get(f"vision_{provider}_api_key")
        model_name = (
            vision_model_name if vision_model_name is not None else config.app.get(f"vision_{provider}_model_name")
        )
        base_url = vision_base_url if vision_base_url is not None else config.app.get(f"vision_{provider}_base_url", "")
        if not api_key or not model_name:
            raise ValueError(
                f"未配置 {provider} 的 API Key 或模型名称。"
                f"请在设置中配置 vision_{provider}_api_key 和 vision_{provider}_model_name"
            )

        progress(10, "正在提取关键帧...")
        keyframe_files = self._load_or_extract_keyframes(video_path, frame_interval_seconds)
        progress(25, f"关键帧准备完成，共 {len(keyframe_files)} 帧")

        progress(30, "正在初始化视觉分析器...")
        analyzer = create_vision_analyzer(
            provider=provider,
            api_key=api_key,
            model=model_name,
            base_url=base_url,
        )

        batches = self._chunk_keyframes(keyframe_files, batch_size=batch_size)
        if not batches:
            raise RuntimeError("未能构建任何关键帧批次")

        progress(40, f"正在分析关键帧，共 {len(batches)} 个批次...")
        batch_results = await self._analyze_batches(
            analyzer=analyzer,
            batches=batches,
            custom_prompt=custom_prompt,
            video_theme=video_theme,
            max_concurrency=concurrency,
            progress_callback=progress,
            documentary_settings=doc_settings,
            subtitle_content=subtitle_content,
        )

        progress(65, "正在整理分析结果...")
        sorted_batches = self._sort_batch_results(batch_results)
        artifact = self._build_analysis_artifact(
            sorted_batches,
            video_path=video_path,
            frame_interval_seconds=frame_interval_seconds,
            vision_batch_size=batch_size,
            vision_llm_provider=provider,
            vision_model_name=model_name,
            max_concurrency=concurrency,
        )
        analysis_json_path = self._save_analysis_artifact(artifact, video_path=video_path)
        video_clip_json = self._build_video_clip_json(sorted_batches, doc_settings)

        progress(75, "逐帧分析完成")
        return {
            "analysis_json_path": analysis_json_path,
            "analysis_artifact": artifact,
            "video_clip_json": video_clip_json,
            "keyframe_files": keyframe_files,
        }

    def _resolve_frame_interval(self, frame_interval_input: int | float | None) -> float:
        interval = frame_interval_input
        if interval in (None, ""):
            interval = config.frames.get("frame_interval_input", 3)
        try:
            value = float(interval)
        except (TypeError, ValueError):
            value = 3.0
        if value <= 0:
            raise ValueError("frame_interval_input must be > 0")
        return value

    def _resolve_batch_size(self, vision_batch_size: int | None) -> int:
        size = vision_batch_size or config.frames.get("vision_batch_size", 10)
        try:
            value = int(size)
        except (TypeError, ValueError):
            value = 10
        if value <= 0:
            raise ValueError("vision_batch_size must be > 0")
        return value

    def _resolve_max_concurrency(self, max_concurrency: int | None) -> int:
        value = max_concurrency if max_concurrency is not None else config.frames.get("vision_max_concurrency", 2)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 1
        return max(1, parsed)

    def _load_or_extract_keyframes(self, video_path: str, frame_interval_seconds: float) -> list[str]:
        keyframes_root = os.path.join(utils.temp_dir(), "keyframes")
        os.makedirs(keyframes_root, exist_ok=True)
        cache_key = self._build_keyframe_cache_key(video_path, frame_interval_seconds)
        cache_dir = os.path.join(keyframes_root, cache_key)
        os.makedirs(cache_dir, exist_ok=True)

        cached_files = self._collect_keyframe_paths(cache_dir)
        if cached_files:
            logger.info(f"使用已缓存关键帧: {cache_dir}, 共 {len(cached_files)} 帧")
            return cached_files

        processor = video_processor.VideoProcessor(video_path)
        extracted = processor.extract_frames_by_interval_with_fallback(
            output_dir=cache_dir,
            interval_seconds=frame_interval_seconds,
        )
        keyframe_files = sorted(str(path) for path in extracted if str(path).endswith(".jpg"))
        if not keyframe_files:
            keyframe_files = self._collect_keyframe_paths(cache_dir)
        if not keyframe_files:
            raise RuntimeError("未提取到任何关键帧")

        logger.info(f"关键帧提取完成: {cache_dir}, 共 {len(keyframe_files)} 帧")
        return keyframe_files

    def _build_keyframe_cache_key(self, video_path: str, frame_interval_seconds: float) -> str:
        try:
            video_mtime = os.path.getmtime(video_path)
        except OSError:
            video_mtime = 0

        legacy_prefix = utils.md5(f"{video_path}{video_mtime}")
        payload = "|".join(
            [
                str(video_path),
                str(video_mtime),
                str(frame_interval_seconds),
                "documentary-keyframes-v2",
            ]
        )
        return f"{legacy_prefix}_{utils.md5(payload)}"

    @staticmethod
    def _collect_keyframe_paths(cache_dir: str) -> list[str]:
        if not os.path.exists(cache_dir):
            return []
        return sorted(
            os.path.join(cache_dir, name)
            for name in os.listdir(cache_dir)
            if re.fullmatch(r"keyframe_\d{6}_\d{9}\.jpg", name)
        )

    @staticmethod
    def _chunk_keyframes(keyframe_files: list[str], batch_size: int) -> list[list[str]]:
        return [keyframe_files[index : index + batch_size] for index in range(0, len(keyframe_files), batch_size)]

    async def _analyze_batches(
        self,
        *,
        analyzer: Any,
        batches: list[list[str]],
        custom_prompt: str,
        video_theme: str,
        max_concurrency: int,
        progress_callback: Callable[[float, str], None],
        documentary_settings: dict | None = None,
        subtitle_content: str = "",
    ) -> list[FrameBatchResult]:
        doc_settings = documentary_settings or get_documentary_settings()
        semaphore = asyncio.Semaphore(max(1, max_concurrency))
        total = len(batches)
        done = 0
        done_lock = asyncio.Lock()

        batch_time_ranges: list[str] = []
        previous_batch_files: list[str] | None = None
        for batch_files in batches:
            _, _, time_range = self._get_batch_timestamps(batch_files, previous_batch_files)
            batch_time_ranges.append(time_range)
            previous_batch_files = batch_files

        async def run_single(batch_index: int, frame_paths: list[str], time_range: str) -> FrameBatchResult:
            nonlocal done
            prompt = self._build_batch_prompt(
                frame_count=len(frame_paths),
                video_theme=video_theme,
                custom_prompt=custom_prompt,
                documentary_settings=doc_settings,
                time_range=time_range,
                subtitle_content=subtitle_content,
            )
            try:
                async with semaphore:
                    raw_results = await analyzer.analyze_images(
                        images=frame_paths,
                        prompt=prompt,
                        batch_size=max(1, len(frame_paths)),
                        max_concurrency=1,
                    )
                raw_response, error_message = self._extract_batch_response(raw_results)
                if error_message:
                    return self._build_failed_batch_result(
                        batch_index=batch_index,
                        raw_response=raw_response,
                        error_message=error_message,
                        frame_paths=frame_paths,
                        time_range=time_range,
                    )
                return self._parse_batch_response(
                    batch_index=batch_index,
                    raw_response=raw_response,
                    frame_paths=frame_paths,
                    time_range=time_range,
                )
            except Exception as exc:
                return self._build_failed_batch_result(
                    batch_index=batch_index,
                    raw_response="",
                    error_message=str(exc),
                    frame_paths=frame_paths,
                    time_range=time_range,
                )
            finally:
                async with done_lock:
                    done += 1
                    progress = 40 + (done / max(1, total)) * 25
                    progress_callback(progress, f"正在分析关键帧批次 ({done}/{total})...")

        tasks = [
            run_single(batch_index=index, frame_paths=batch_files, time_range=batch_time_ranges[index])
            for index, batch_files in enumerate(batches)
        ]
        return await asyncio.gather(*tasks)

    def _build_batch_prompt(
        self,
        *,
        frame_count: int,
        video_theme: str,
        custom_prompt: str,
        documentary_settings: dict | None = None,
        time_range: str = "",
        subtitle_content: str = "",
    ) -> str:
        prompt = self._build_analysis_prompt(
            frame_count=frame_count,
            include_burned_in_subtitle=bool(cfg.get("enable_hard_subtitle_ocr", True)),
        )
        cfg = documentary_settings or get_documentary_settings()
        extra_lines: list[str] = []
        highlight_hint = build_frame_highlight_hint(cfg)
        if highlight_hint:
            extra_lines.append(highlight_hint)
        if cfg.get("enable_subtitle_enrichment", True) and (subtitle_content or "").strip() and time_range:
            pad_sec = int(cfg.get("subtitle_batch_pad_sec", 5))
            dialogue = subtitle_excerpt_for_time_range(
                subtitle_content,
                time_range,
                pad_ms=pad_sec * 1000,
            )
            if dialogue:
                extra_lines.append(
                    f"本批次时间范围 {time_range} 附近字幕对白（分析画面时请对照，不要虚构台词）：{dialogue}"
                )
        if (video_theme or "").strip():
            extra_lines.append(f"视频主题：{video_theme.strip()}")
        if (custom_prompt or "").strip():
            extra_lines.append(custom_prompt.strip())
        if not extra_lines:
            return prompt

        extras = "\n".join(f"- {line}" for line in extra_lines)
        return f"{prompt}\n\n补充分析要求：\n{extras}"

    def _extract_batch_response(self, raw_results: Any) -> tuple[str, str]:
        if not raw_results:
            return "", "Batch response is empty"

        first_result = raw_results[0] if isinstance(raw_results, list) else raw_results
        if isinstance(first_result, dict):
            raw_response = str(first_result.get("response", "") or "")
            error_message = str(first_result.get("error", "") or "")
            if error_message:
                if not raw_response:
                    raw_response = error_message
                return raw_response, error_message
            if not raw_response.strip():
                return raw_response, "Batch response is empty"
            return raw_response, ""

        raw_response = str(first_result or "")
        if not raw_response.strip():
            return raw_response, "Batch response is empty"
        return raw_response, ""

    def _sort_batch_results(self, batch_results: list[FrameBatchResult]) -> list[FrameBatchResult]:
        return sorted(batch_results, key=lambda item: (self._time_range_sort_key(item.time_range), item.batch_index))

    def _build_analysis_artifact(
        self,
        batch_results: list[FrameBatchResult],
        *,
        video_path: str,
        frame_interval_seconds: float,
        vision_batch_size: int,
        vision_llm_provider: str,
        vision_model_name: str,
        max_concurrency: int,
    ) -> dict[str, Any]:
        sorted_batches = self._sort_batch_results(batch_results)

        batch_dicts: list[dict[str, Any]] = []
        frame_observations: list[dict[str, Any]] = []
        overall_activity_summaries: list[dict[str, Any]] = []

        for batch in sorted_batches:
            batch_payload = {
                "batch_index": batch.batch_index,
                "status": batch.status,
                "time_range": batch.time_range,
                "raw_response": batch.raw_response,
                "frame_paths": list(batch.frame_paths),
                "frame_observations": list(batch.frame_observations),
                "overall_activity_summary": batch.overall_activity_summary,
                "fallback_summary": batch.fallback_summary,
                "error_message": batch.error_message,
            }
            batch_dicts.append(batch_payload)

            for observation in batch.frame_observations:
                observation_payload = dict(observation)
                observation_payload["batch_index"] = batch.batch_index
                observation_payload["time_range"] = batch.time_range
                frame_observations.append(observation_payload)

            summary_text = (batch.overall_activity_summary or batch.fallback_summary or "").strip()
            if summary_text:
                overall_activity_summaries.append(
                    {
                        "batch_index": batch.batch_index,
                        "time_range": batch.time_range,
                        "summary": summary_text,
                    }
                )

        return {
            "artifact_version": "documentary-frame-analysis-v2",
            "generated_at": datetime.now().isoformat(),
            "video_path": video_path,
            "frame_interval_seconds": frame_interval_seconds,
            "vision_batch_size": vision_batch_size,
            "vision_llm_provider": vision_llm_provider,
            "vision_model_name": vision_model_name,
            "vision_max_concurrency": max_concurrency,
            "batches": batch_dicts,
            # 向后兼容旧解析器结构
            "frame_observations": frame_observations,
            "overall_activity_summaries": overall_activity_summaries,
        }

    @staticmethod
    def analysis_artifact_dir() -> str:
        return pairing_analysis_artifact_dir()

    @classmethod
    def default_analysis_path_for_video(cls, video_path: str) -> str:
        return pairing_default_analysis_path_for_video(video_path)

    @classmethod
    def _is_valid_analysis_artifact(cls, payload: Any) -> bool:
        return pairing_is_valid_analysis_artifact(payload)

    @classmethod
    def load_analysis_artifact(cls, analysis_json_path: str) -> dict[str, Any]:
        return pairing_load_analysis_artifact(analysis_json_path)

    @classmethod
    def resolve_reusable_analysis_path(
        cls,
        video_path: str,
        *,
        explicit_path: str | None = None,
        reuse: bool = True,
    ) -> str | None:
        return pairing_resolve_reusable_analysis_path(
            video_path,
            explicit_path=explicit_path,
            reuse=reuse,
        )

    def _save_analysis_artifact(
        self,
        artifact: dict[str, Any],
        *,
        video_path: str = "",
    ) -> str:
        analysis_dir = self.analysis_artifact_dir()
        os.makedirs(analysis_dir, exist_ok=True)

        if video_path:
            file_path = self.default_analysis_path_for_video(video_path)
        else:
            filename = f"frame_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path = os.path.join(analysis_dir, filename)
            suffix = 1
            while os.path.exists(file_path):
                filename = (
                    f"frame_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix:02d}.json"
                )
                file_path = os.path.join(analysis_dir, filename)
                suffix += 1

        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(artifact, fp, ensure_ascii=False, indent=2)
        logger.info(f"分析结果已保存到: {file_path}")
        return file_path

    def _build_video_clip_json(
        self,
        batch_results: list[FrameBatchResult],
        documentary_settings: dict | None = None,
    ) -> list[dict]:
        cfg = documentary_settings or get_documentary_settings()
        default_ost = int(cfg.get("default_narration_ost", 2))
        if default_ost not in (0, 1, 2):
            default_ost = 2
        clips: list[dict] = []
        for batch in self._sort_batch_results(batch_results):
            picture = self._build_batch_picture(batch)
            clips.append(
                {
                    "timestamp": batch.time_range,
                    "picture": picture,
                    "narration": "",
                    "OST": default_ost,
                }
            )
        return clips

    @staticmethod
    def _get_video_duration_sec(video_path: str) -> float:
        try:
            processor = video_processor.VideoProcessor(video_path)
            return float(processor.duration or 0)
        except Exception as exc:
            logger.warning(f"无法读取视频时长: {video_path} ({exc})")
            return 0.0

    def _build_batch_picture(self, batch: FrameBatchResult) -> str:
        summary = (batch.overall_activity_summary or "").strip()
        if summary:
            return summary

        fallback = (batch.fallback_summary or "").strip()
        if fallback:
            return fallback

        observation_lines = []
        for frame in batch.frame_observations:
            timestamp = str(frame.get("timestamp", "") or "").strip()
            observation = str(frame.get("observation", "") or "").strip()
            if timestamp and observation:
                observation_lines.append(f"{timestamp}: {observation}")
            elif observation:
                observation_lines.append(observation)
        if observation_lines:
            return " ".join(observation_lines)

        raw_response = (batch.raw_response or "").strip()
        if raw_response:
            return raw_response[:200]
        return "该批次分析失败，未返回可用描述。"

    def _time_range_sort_key(self, time_range: str) -> tuple[int, str]:
        start = (time_range or "").split("-", 1)[0].strip()
        return self._timestamp_to_milliseconds(start), time_range

    @staticmethod
    def _timestamp_to_milliseconds(timestamp: str) -> int:
        text = (timestamp or "").strip()
        try:
            if "," in text:
                time_part, ms_part = text.split(",", 1)
                milliseconds = int(ms_part)
            else:
                time_part = text
                milliseconds = 0

            parts = [int(part) for part in time_part.split(":") if part]
            while len(parts) < 3:
                parts.insert(0, 0)
            hours, minutes, seconds = parts[-3], parts[-2], parts[-1]
            return ((hours * 3600 + minutes * 60 + seconds) * 1000) + milliseconds
        except Exception:
            return 0

    def _get_batch_timestamps(
        self,
        batch_files: list[str],
        prev_batch_files: list[str] | None = None,
    ) -> tuple[str, str, str]:
        if not batch_files:
            return "00:00:00,000", "00:00:00,000", "00:00:00,000-00:00:00,000"

        if len(batch_files) == 1 and prev_batch_files:
            first_frame = os.path.basename(prev_batch_files[-1])
            last_frame = os.path.basename(batch_files[0])
        else:
            first_frame = os.path.basename(batch_files[0])
            last_frame = os.path.basename(batch_files[-1])

        first_timestamp = self._timestamp_from_keyframe_name(first_frame)
        last_timestamp = self._timestamp_from_keyframe_name(last_frame)
        return first_timestamp, last_timestamp, f"{first_timestamp}-{last_timestamp}"

    def _timestamp_from_keyframe_name(self, filename: str) -> str:
        match = re.search(r"keyframe_\d{6}_(\d{9})\.jpg$", filename)
        if not match:
            return "00:00:00,000"
        token = match.group(1)
        hours = int(token[0:2])
        minutes = int(token[2:4])
        seconds = int(token[4:6])
        milliseconds = int(token[6:9])
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _build_analysis_prompt(self, frame_count: int, *, include_burned_in_subtitle: bool = False) -> str:
        example_suffix = (
            self.BURNED_IN_SUBTITLE_JSON_EXAMPLE if include_burned_in_subtitle else ""
        )
        prompt = self.PROMPT_TEMPLATE.format(
            frame_count=frame_count,
            burned_in_subtitle_example=example_suffix,
        )
        if include_burned_in_subtitle:
            prompt = f"{prompt}\n\n{self.BURNED_IN_SUBTITLE_PROMPT_SUFFIX}"
        return prompt

    def _build_failed_batch_result(
        self,
        *,
        batch_index: int,
        raw_response: str,
        error_message: str,
        frame_paths: list[str],
        time_range: str,
    ) -> FrameBatchResult:
        fallback_summary = (raw_response or "").strip()[:200]
        if not fallback_summary:
            fallback_summary = f"Batch {batch_index} analysis failed: {error_message or 'unknown error'}"

        return FrameBatchResult(
            batch_index=batch_index,
            status="failed",
            time_range=time_range,
            raw_response=raw_response,
            frame_paths=list(frame_paths),
            fallback_summary=fallback_summary,
            error_message=error_message,
        )

    def _build_cache_key(
        self,
        video_path: str,
        interval_seconds: float,
        prompt_version: str,
        model_name: str,
        batch_size: int,
        max_concurrency: int,
    ) -> str:
        try:
            video_mtime = os.path.getmtime(video_path)
        except OSError:
            video_mtime = 0

        legacy_prefix = utils.md5(f"{video_path}{video_mtime}")

        payload = "|".join(
            [
                str(video_path),
                str(video_mtime),
                str(interval_seconds),
                str(prompt_version),
                str(model_name),
                str(batch_size),
                str(max_concurrency),
                "documentary-frame-analysis-v2",
            ]
        )
        return f"{legacy_prefix}_{utils.md5(payload)}"

    def _strip_code_fence(self, response_text: str) -> str:
        cleaned = (response_text or "").strip()
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def _parse_batch_response(
        self,
        *,
        batch_index: int,
        raw_response: str,
        frame_paths: list[str],
        time_range: str,
    ) -> FrameBatchResult:
        try:
            payload = json.loads(self._strip_code_fence(raw_response))
        except Exception as exc:
            return self._build_failed_batch_result(
                batch_index=batch_index,
                raw_response=raw_response,
                error_message=str(exc),
                frame_paths=frame_paths,
                time_range=time_range,
            )

        validation_error = self._validate_batch_payload_contract(payload, expected_frame_count=len(frame_paths))
        if validation_error:
            return self._build_failed_batch_result(
                batch_index=batch_index,
                raw_response=raw_response,
                error_message=validation_error,
                frame_paths=frame_paths,
                time_range=time_range,
            )

        raw_observations = payload["frame_observations"]

        frame_observations: list[dict] = []
        for index, frame_path in enumerate(frame_paths):
            entry = raw_observations[index] if index < len(raw_observations) else {}
            if isinstance(entry, dict):
                observation = str(entry.get("observation", "") or "")
                timestamp = str(entry.get("timestamp", "") or "")
                burned_in_subtitle = str(
                    entry.get("burned_in_subtitle")
                    or entry.get("subtitle_text")
                    or entry.get("on_screen_subtitle")
                    or ""
                ).strip()
                has_burned_raw = entry.get("has_burned_in_subtitle")
                if has_burned_raw is None:
                    has_burned_raw = entry.get("has_subtitle")
                if has_burned_raw is None:
                    has_burned_in_subtitle = bool(burned_in_subtitle)
                else:
                    has_burned_in_subtitle = bool(has_burned_raw) and bool(burned_in_subtitle)
            else:
                observation = str(entry or "")
                timestamp = ""
                burned_in_subtitle = ""
                has_burned_in_subtitle = False
            if not timestamp:
                timestamp = self._timestamp_from_keyframe_name(frame_path)
            frame_observations.append(
                {
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                    "observation": observation,
                    "burned_in_subtitle": burned_in_subtitle if has_burned_in_subtitle else "",
                    "has_burned_in_subtitle": has_burned_in_subtitle,
                }
            )

        raw_summary = payload.get("overall_activity_summary", "")
        if isinstance(raw_summary, str):
            summary = raw_summary
        elif raw_summary is None:
            summary = ""
        else:
            summary = str(raw_summary)

        return FrameBatchResult(
            batch_index=batch_index,
            status="success",
            time_range=time_range,
            raw_response=raw_response,
            frame_paths=list(frame_paths),
            frame_observations=frame_observations,
            overall_activity_summary=summary,
        )

    def _validate_batch_payload_contract(self, payload: object, *, expected_frame_count: int) -> str:
        if not isinstance(payload, dict):
            return "Batch response JSON payload must be an object"

        if "frame_observations" not in payload or not isinstance(payload["frame_observations"], list):
            return "Batch response must include frame_observations as a list"

        if len(payload["frame_observations"]) < expected_frame_count:
            return (
                "Batch response frame_observations length is shorter than provided frame_paths: "
                f"{len(payload['frame_observations'])} < {expected_frame_count}"
            )

        return ""
