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
    build_frame_character_naming_hint,
    build_frame_gender_hint,
    build_frame_highlight_hint,
    get_documentary_settings,
    warn_frame_analysis_gender_mismatch,
)
from app.services.documentary.frame_analysis_compact import slim_scene_segment_for_artifact
from app.services.documentary.documentary_subtitle_enrichment import (
    attach_subtitles_to_frame_analysis_artifact,
    subtitle_excerpt_for_time_range,
)
from app.config.llm_gateway_router import describe_llm_route, resolve_llm_credentials
from app.services.documentary.vision_model_rotation import (
    VisionModelRotation,
    resolve_vision_model_chain,
)
from app.utils import utils, video_processor


class DocumentaryFrameExtractionService:
    PROMPT_TEMPLATE = """
我提供了 {frame_count} 张视频帧，它们按时间顺序排列，代表一个连续的视频片段。
请综合这些帧，识别其中的**独立场景片段**（同一地点、同一组人物、连续动作可合并为一条），输出结构化 JSON。

**timestamp 规则（硬性）**：
- 格式必须为 `HH:MM:SS,mmm-HH:MM:SS,mmm`（起止时间，逗号分隔毫秒）
- 必须与本批次提供的字幕对白时间尽量对齐，用于后期剪辑定位
- 每条 segment 的 timestamp 应覆盖该场景在批次内的实际时间，不要编造批次外的时间
- 同一批次内各 segment 的 timestamp 不要重叠

**scene_segments 每条必填字段（仅输出以下 6 项，不要额外键）**：
- timestamp: 时间范围字符串
- scene: 场景名称（如「办公室」「楼顶天台」）
- observation: 本场景一句话画面观察（人物+动作+氛围，30–80 字；勿复述对白）
- action: 人物在做什么（如「老叶与伟业并肩站在天台边缘交谈」），须含可见性别
- emotion: 画面情绪（紧张、愤怒、悲伤、绝望等）
- key_visual: 光线/色调/构图等特殊视觉（如「阴天冷色调，云层低垂，城市远景」）

人物姓名与性别请写入 action / observation，不要单独输出 characters 数组。

JSON 必须包含以下键：
- scene_segments: 数组，至少 1 条，结构见上
- frame_observations: 数组，长度必须为 {frame_count}（每帧一条，用于逐帧 OCR 对照）
- overall_activity_summary: 字符串，描述整个批次主要活动

示例结构：
{{
  "scene_segments": [
    {{
      "timestamp": "00:00:01,940-00:00:09,940",
      "scene": "楼顶天台",
      "observation": "阴天楼顶，老叶与伟业并肩对峙，气氛压抑",
      "action": "老叶与伟业并肩站在天台边缘交谈",
      "emotion": "严肃、压抑",
      "key_visual": "阴天冷色调，城市建筑远景，云层低垂"
    }}
  ],
  "frame_observations": [
    {{"timestamp": "00:00:00,000", "observation": "楼顶天台，老叶与伟业并肩，阴天冷色调"{burned_in_subtitle_example}}}
  ],
  "overall_activity_summary": "本批次主要活动总结"
}}

请务必不要遗漏视频帧：frame_observations 必须包含 {frame_count} 个元素。
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

        model_name = (
            vision_model_name if vision_model_name is not None else config.app.get(f"vision_{provider}_model_name")
        )
        if not model_name:
            raise ValueError(
                f"未配置视觉模型名称。请在设置中配置 vision_{provider}_model_name"
            )
        api_key, base_url = resolve_llm_credentials(model_name, role="vision")
        if not api_key:
            raise ValueError(
                f"未配置模型 {model_name} 对应的 API Key。"
                f"Qwen 系列请配置 llm_dashscope_api_key，其他模型请配置 llm_alt_api_key"
            )
        logger.info(f"视觉模型 {model_name} → {describe_llm_route(model_name, role='vision')}")

        progress(10, "正在提取关键帧...")
        keyframe_files = self._load_or_extract_keyframes(video_path, frame_interval_seconds)
        progress(25, f"关键帧准备完成，共 {len(keyframe_files)} 帧")

        progress(30, "正在初始化视觉分析器...")
        model_chain = resolve_vision_model_chain(
            model_name,
            config.frames.get("vision_fallback_model_names"),
        )
        rotation = self._build_vision_model_rotation(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model_names=model_chain,
        )
        if len(model_chain) > 1:
            progress(30, f"视觉模型链: {' → '.join(model_chain)}（额度用尽时自动切换）")

        batches = self._chunk_keyframes(keyframe_files, batch_size=batch_size)
        if not batches:
            raise RuntimeError("未能构建任何关键帧批次")

        progress(40, f"正在分析关键帧，共 {len(batches)} 个批次...")
        batch_results = await self._analyze_batches(
            rotation=rotation,
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
            vision_fallback_model_names=config.frames.get("vision_fallback_model_names", ""),
            vision_models_used=sorted(rotation.models_used),
            max_concurrency=concurrency,
            subtitle_content=subtitle_content,
            documentary_settings=doc_settings,
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

    async def retry_failed_batches(
        self,
        *,
        analysis_json_path: str,
        video_path: str = "",
        video_theme: str = "",
        custom_prompt: str = "",
        vision_llm_provider: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        vision_api_key: str | None = None,
        vision_model_name: str | None = None,
        vision_base_url: str | None = None,
        max_concurrency: int | None = None,
        documentary_settings: dict | None = None,
        subtitle_content: str = "",
    ) -> dict[str, Any]:
        """仅重跑 artifact 中 status=failed 的批次，并写回原 JSON。"""
        progress = progress_callback or (lambda _p, _m: None)
        if not analysis_json_path or not os.path.isfile(analysis_json_path):
            raise FileNotFoundError(f"抽帧分析文件不存在: {analysis_json_path}")

        artifact = pairing_load_analysis_artifact(analysis_json_path)
        batch_dicts = [b for b in (artifact.get("batches") or []) if isinstance(b, dict)]
        if not batch_dicts:
            raise ValueError("抽帧分析缺少 batches，无法重跑失败批次")

        failed_batches = [b for b in batch_dicts if b.get("status") != "success"]
        if not failed_batches:
            progress(100, "没有失败批次需要重跑")
            return {
                "analysis_json_path": analysis_json_path,
                "retried": 0,
                "recovered": 0,
                "still_failed": 0,
            }

        resolved_video = (video_path or str(artifact.get("video_path") or "")).strip()
        frame_interval = float(artifact.get("frame_interval_seconds") or config.frames.get("frame_interval_input", 3))
        batch_size = int(artifact.get("vision_batch_size") or config.frames.get("vision_batch_size", 10))
        concurrency = self._resolve_max_concurrency(
            max_concurrency if max_concurrency is not None else artifact.get("vision_max_concurrency")
        )
        provider = (
            vision_llm_provider
            or str(artifact.get("vision_llm_provider") or "")
            or config.app.get("vision_llm_provider", "openai")
        ).lower()

        model_name = (
            vision_model_name
            if vision_model_name is not None
            else str(artifact.get("vision_model_name") or "")
            or config.app.get(f"vision_{provider}_model_name")
        )
        api_key, base_url = resolve_llm_credentials(model_name, role="vision")
        if not api_key or not model_name:
            raise ValueError(
                f"未配置视觉模型 {model_name} 的 API Key。"
                f"Qwen 系列请配置 llm_dashscope_api_key，其他模型请配置 llm_alt_api_key"
            )
        logger.info(f"重跑视觉模型 {model_name} → {describe_llm_route(model_name, role='vision')}")

        progress(10, f"准备重跑 {len(failed_batches)} 个失败批次...")
        fallback_raw = (
            config.frames.get("vision_fallback_model_names")
            or artifact.get("vision_fallback_model_names")
        )
        model_chain = resolve_vision_model_chain(model_name, fallback_raw)
        rotation = self._build_vision_model_rotation(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model_names=model_chain,
        )

        retry_items: list[tuple[int, list[str], str]] = []
        for batch in failed_batches:
            batch_index = int(batch.get("batch_index", 0))
            time_range = str(batch.get("time_range") or "").strip()
            frame_paths = self._resolve_batch_frame_paths(
                batch,
                video_path=resolved_video,
                frame_interval_seconds=frame_interval,
                batch_size=batch_size,
            )
            if not frame_paths:
                logger.warning(f"批次 {batch_index} 无法定位关键帧，跳过重跑")
                continue
            if not time_range:
                _, _, time_range = self._get_batch_timestamps(frame_paths, None)
            retry_items.append((batch_index, frame_paths, time_range))

        if not retry_items:
            raise ValueError(
                "失败批次的关键帧文件已丢失。请确认原视频仍可访问后重新「抽帧并分析」，"
                "或确保 artifact 中 frame_paths 指向的缓存目录仍存在。"
            )

        doc_settings = documentary_settings or get_documentary_settings()
        progress(20, f"正在重跑 {len(retry_items)} 个批次...")
        retry_results = await self._analyze_batch_items(
            rotation=rotation,
            items=retry_items,
            custom_prompt=custom_prompt,
            video_theme=video_theme,
            max_concurrency=concurrency,
            progress_callback=progress,
            documentary_settings=doc_settings,
            subtitle_content=subtitle_content,
        )

        merged_by_index: dict[int, FrameBatchResult] = {
            self._batch_dict_to_result(batch).batch_index: self._batch_dict_to_result(batch)
            for batch in batch_dicts
        }
        recovered = 0
        still_failed = 0
        for result in retry_results:
            merged_by_index[result.batch_index] = result
            if result.status == "success":
                recovered += 1
            else:
                still_failed += 1

        merged_results = [merged_by_index[index] for index in sorted(merged_by_index.keys())]
        doc_settings = documentary_settings or get_documentary_settings()
        progress(85, "正在合并并重写分析 JSON...")
        new_artifact = self._build_analysis_artifact(
            merged_results,
            video_path=resolved_video or str(artifact.get("video_path") or ""),
            frame_interval_seconds=frame_interval,
            vision_batch_size=batch_size,
            vision_llm_provider=provider,
            vision_model_name=model_name,
            vision_fallback_model_names=fallback_raw or "",
            vision_models_used=sorted(rotation.models_used),
            max_concurrency=concurrency,
            subtitle_content=subtitle_content,
            documentary_settings=doc_settings,
        )
        new_artifact["generated_at"] = datetime.now().isoformat()
        new_artifact["last_retry_at"] = new_artifact["generated_at"]
        new_artifact["last_retry_recovered"] = recovered
        new_artifact["last_retry_still_failed"] = still_failed

        saved_path = self._save_analysis_artifact(
            new_artifact,
            output_path=analysis_json_path,
        )
        progress(100, f"重跑完成：成功 {recovered}，仍失败 {still_failed}")
        logger.info(
            f"失败批次重跑完成: {saved_path}，重试 {len(retry_results)}，"
            f"成功 {recovered}，仍失败 {still_failed}"
        )
        return {
            "analysis_json_path": saved_path,
            "analysis_artifact": new_artifact,
            "retried": len(retry_results),
            "recovered": recovered,
            "still_failed": still_failed,
        }

    @staticmethod
    def count_failed_batches(artifact: dict[str, Any]) -> int:
        batches = artifact.get("batches") or []
        return sum(
            1 for batch in batches
            if isinstance(batch, dict) and batch.get("status") != "success"
        )

    @staticmethod
    def list_failed_batch_details(artifact: dict[str, Any]) -> list[dict[str, Any]]:
        """列出失败批次的诊断信息，供 WebUI 展示。"""
        details: list[dict[str, Any]] = []
        for batch in artifact.get("batches") or []:
            if not isinstance(batch, dict) or batch.get("status") == "success":
                continue

            frame_paths = [str(path) for path in (batch.get("frame_paths") or []) if str(path).strip()]
            missing_frames = sum(1 for path in frame_paths if not os.path.isfile(path))
            error_message = str(batch.get("error_message") or "").strip()
            if not error_message:
                error_message = str(batch.get("fallback_summary") or "未知错误").strip()

            raw_response = str(batch.get("raw_response") or "")
            details.append(
                {
                    "batch_index": int(batch.get("batch_index", 0)),
                    "time_range": str(batch.get("time_range") or ""),
                    "error_message": error_message,
                    "frame_count": len(frame_paths),
                    "frames_on_disk": len(frame_paths) - missing_frames,
                    "frames_missing": missing_frames,
                    "has_scene_segments": bool(batch.get("scene_segments")),
                    "has_frame_observations": bool(batch.get("frame_observations")),
                    "raw_response_preview": raw_response[:500],
                    "raw_response_chars": len(raw_response),
                }
            )

        return sorted(details, key=lambda item: item["batch_index"])

    @staticmethod
    def enrich_analysis_artifact_subtitles(
        artifact: dict[str, Any],
        subtitle_content: str,
        *,
        documentary_settings: dict | None = None,
    ) -> dict[str, Any]:
        """为已有抽帧分析 JSON 注入 SRT 字幕字段（无需重跑视觉模型）。"""
        return attach_subtitles_to_frame_analysis_artifact(
            artifact,
            subtitle_content,
            settings=documentary_settings or get_documentary_settings(),
        )

    @staticmethod
    def compact_analysis_artifact(
        artifact: dict[str, Any],
        *,
        include_frame_observations: bool = True,
        include_summaries: bool = True,
        include_batch_index: bool = True,
        keep_batch_meta: bool = True,
        source_path: str = "",
    ) -> dict[str, Any]:
        from app.services.documentary.frame_analysis_compact import compact_analysis_artifact

        return compact_analysis_artifact(
            artifact,
            include_frame_observations=include_frame_observations,
            include_summaries=include_summaries,
            include_batch_index=include_batch_index,
            keep_batch_meta=keep_batch_meta,
            source_path=source_path,
        )

    @staticmethod
    def save_compact_analysis_artifact(source_path: str, *, output_path: str = "", **compact_options: Any) -> dict[str, Any]:
        from app.services.documentary.frame_analysis_compact import save_compact_analysis_artifact

        return save_compact_analysis_artifact(source_path, output_path=output_path, **compact_options)

    @staticmethod
    def save_split_analysis_artifacts(
        source_path: str,
        part_count: int,
        *,
        artifact: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from app.services.documentary.material_output_split import save_split_frame_analysis_artifacts

        return save_split_frame_analysis_artifacts(source_path, part_count, artifact=artifact)

    @staticmethod
    def estimate_compact_analysis_sizes(artifact: dict[str, Any], *, source_bytes: int | None = None) -> dict[str, int]:
        from app.services.documentary.frame_analysis_compact import estimate_compact_sizes

        return estimate_compact_sizes(artifact, source_bytes=source_bytes)

    @staticmethod
    def format_failed_batches_report(artifact: dict[str, Any]) -> str:
        """纯文本失败批次报告（日志 / 提示用）。"""
        details = DocumentaryFrameExtractionService.list_failed_batch_details(artifact)
        if not details:
            return "无失败批次"

        lines = [f"共 {len(details)} 个失败批次："]
        for item in details:
            lines.append(
                f"- 批次 #{item['batch_index']} · {item['time_range']} · "
                f"关键帧 {item['frames_on_disk']}/{item['frame_count']} · "
                f"{item['error_message']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _batch_dict_to_result(batch: dict[str, Any]) -> FrameBatchResult:
        return FrameBatchResult(
            batch_index=int(batch.get("batch_index", 0)),
            status=str(batch.get("status") or "failed"),
            time_range=str(batch.get("time_range") or ""),
            raw_response=str(batch.get("raw_response") or ""),
            frame_paths=[str(path) for path in (batch.get("frame_paths") or []) if str(path).strip()],
            frame_observations=list(batch.get("frame_observations") or []),
            scene_segments=list(batch.get("scene_segments") or []),
            overall_activity_summary=str(batch.get("overall_activity_summary") or ""),
            fallback_summary=str(batch.get("fallback_summary") or ""),
            error_message=str(batch.get("error_message") or ""),
            vision_model_used=str(batch.get("vision_model_used") or ""),
        )

    def _build_vision_model_rotation(
        self,
        *,
        provider: str,
        api_key: str,
        base_url: str,
        model_names: list[str],
    ) -> VisionModelRotation:
        return VisionModelRotation(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model_names=model_names,
            extract_response=self._extract_batch_response,
        )

    def _resolve_batch_frame_paths(
        self,
        batch: dict[str, Any],
        *,
        video_path: str,
        frame_interval_seconds: float,
        batch_size: int,
    ) -> list[str]:
        stored = [str(path) for path in (batch.get("frame_paths") or []) if str(path).strip()]
        existing = [path for path in stored if os.path.isfile(path)]
        if existing:
            return existing

        if not video_path or not os.path.isfile(video_path):
            return []

        keyframe_files = self._load_or_extract_keyframes(video_path, frame_interval_seconds)
        chunks = self._chunk_keyframes(keyframe_files, batch_size=max(1, batch_size))
        batch_index = int(batch.get("batch_index", 0))
        if 0 <= batch_index < len(chunks):
            return chunks[batch_index]
        return []

    async def _analyze_batch_items(
        self,
        *,
        rotation: VisionModelRotation,
        items: list[tuple[int, list[str], str]],
        custom_prompt: str,
        video_theme: str,
        max_concurrency: int,
        progress_callback: Callable[[float, str], None],
        documentary_settings: dict | None = None,
        subtitle_content: str = "",
    ) -> list[FrameBatchResult]:
        doc_settings = documentary_settings or get_documentary_settings()
        semaphore = asyncio.Semaphore(max(1, max_concurrency))
        total = len(items)
        done = 0
        done_lock = asyncio.Lock()

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
                    raw_results, model_used, error_message = await rotation.analyze_images(
                        images=frame_paths,
                        prompt=prompt,
                        batch_size=max(1, len(frame_paths)),
                        max_concurrency=1,
                    )
                if rotation.pending_switch_message:
                    progress_callback(20, rotation.pending_switch_message)
                    rotation.pending_switch_message = ""
                if error_message or raw_results is None:
                    return self._build_failed_batch_result(
                        batch_index=batch_index,
                        raw_response="",
                        error_message=error_message or "视觉模型分析失败",
                        frame_paths=frame_paths,
                        time_range=time_range,
                        vision_model_used=model_used,
                    )
                raw_response, parse_error = self._extract_batch_response(raw_results)
                if parse_error:
                    return self._build_failed_batch_result(
                        batch_index=batch_index,
                        raw_response=raw_response,
                        error_message=parse_error,
                        frame_paths=frame_paths,
                        time_range=time_range,
                        vision_model_used=model_used,
                    )
                return self._parse_batch_response(
                    batch_index=batch_index,
                    raw_response=raw_response,
                    frame_paths=frame_paths,
                    time_range=time_range,
                    vision_model_used=model_used,
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
                    progress = 20 + (done / max(1, total)) * 60
                    switch_hint = ""
                    if rotation.pending_switch_message:
                        switch_hint = f" · {rotation.pending_switch_message}"
                    progress_callback(
                        progress,
                        f"正在重跑失败批次 ({done}/{total})：{time_range}{switch_hint}...",
                    )

        tasks = [
            run_single(batch_index=batch_index, frame_paths=frame_paths, time_range=time_range)
            for batch_index, frame_paths, time_range in items
        ]
        return await asyncio.gather(*tasks)

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
        rotation: VisionModelRotation,
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
                    raw_results, model_used, error_message = await rotation.analyze_images(
                        images=frame_paths,
                        prompt=prompt,
                        batch_size=max(1, len(frame_paths)),
                        max_concurrency=1,
                    )
                if rotation.pending_switch_message:
                    progress_callback(40, rotation.pending_switch_message)
                    rotation.pending_switch_message = ""
                if error_message or raw_results is None:
                    return self._build_failed_batch_result(
                        batch_index=batch_index,
                        raw_response="",
                        error_message=error_message or "视觉模型分析失败",
                        frame_paths=frame_paths,
                        time_range=time_range,
                        vision_model_used=model_used,
                    )
                raw_response, parse_error = self._extract_batch_response(raw_results)
                if parse_error:
                    return self._build_failed_batch_result(
                        batch_index=batch_index,
                        raw_response=raw_response,
                        error_message=parse_error,
                        frame_paths=frame_paths,
                        time_range=time_range,
                        vision_model_used=model_used,
                    )
                return self._parse_batch_response(
                    batch_index=batch_index,
                    raw_response=raw_response,
                    frame_paths=frame_paths,
                    time_range=time_range,
                    vision_model_used=model_used,
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
                    switch_hint = ""
                    if rotation.pending_switch_message:
                        switch_hint = f" · {rotation.pending_switch_message}"
                    progress_callback(
                        progress,
                        f"正在分析关键帧批次 ({done}/{total}){switch_hint}...",
                    )

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
        cfg = documentary_settings or get_documentary_settings()
        prompt = self._build_analysis_prompt(
            frame_count=frame_count,
            include_burned_in_subtitle=bool(cfg.get("enable_hard_subtitle_ocr", True)),
        )
        extra_lines: list[str] = []
        naming_hint = build_frame_character_naming_hint(cfg)
        if naming_hint:
            extra_lines.append(naming_hint)
        gender_hint = build_frame_gender_hint(cfg)
        if gender_hint:
            extra_lines.append(gender_hint)
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

    @staticmethod
    def _looks_like_vision_api_failure(response_text: str) -> bool:
        text = (response_text or "").strip().lower()
        if not text:
            return False
        if text.startswith("批次处理失败"):
            return True
        markers = (
            "[api_call_error]",
            "api 错误",
            "api调用失败",
            "error code:",
            "model_not_found",
            "does not exist",
            "invalid_request_error",
        )
        return any(marker in text for marker in markers)

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
            if self._looks_like_vision_api_failure(raw_response):
                return raw_response, raw_response
            return raw_response, ""

        raw_response = str(first_result or "")
        if not raw_response.strip():
            return raw_response, "Batch response is empty"
        if self._looks_like_vision_api_failure(raw_response):
            return raw_response, raw_response
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
        subtitle_content: str = "",
        documentary_settings: dict | None = None,
        vision_fallback_model_names: str = "",
        vision_models_used: list[str] | None = None,
    ) -> dict[str, Any]:
        sorted_batches = self._sort_batch_results(batch_results)

        batch_dicts: list[dict[str, Any]] = []
        frame_observations: list[dict[str, Any]] = []
        scene_segments: list[dict[str, Any]] = []
        overall_activity_summaries: list[dict[str, Any]] = []

        for batch in sorted_batches:
            slim_batch_segments = [
                slim_scene_segment_for_artifact(segment)
                for segment in batch.scene_segments
                if isinstance(segment, dict)
            ]
            batch_payload = {
                "batch_index": batch.batch_index,
                "status": batch.status,
                "time_range": batch.time_range,
                "raw_response": batch.raw_response,
                "frame_paths": list(batch.frame_paths),
                "scene_segments": slim_batch_segments,
                "frame_observations": list(batch.frame_observations),
                "overall_activity_summary": batch.overall_activity_summary,
                "fallback_summary": batch.fallback_summary,
                "error_message": batch.error_message,
                "vision_model_used": batch.vision_model_used,
            }
            batch_dicts.append(batch_payload)
            scene_segments.extend(slim_batch_segments)

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

        artifact = {
            "artifact_version": "documentary-frame-analysis-v3",
            "generated_at": datetime.now().isoformat(),
            "video_path": video_path,
            "frame_interval_seconds": frame_interval_seconds,
            "vision_batch_size": vision_batch_size,
            "vision_llm_provider": vision_llm_provider,
            "vision_model_name": vision_model_name,
            "vision_fallback_model_names": (vision_fallback_model_names or "").strip(),
            "vision_models_used": list(vision_models_used or []),
            "vision_max_concurrency": max_concurrency,
            "scene_segments": scene_segments,
            "batches": batch_dicts,
            # 向后兼容旧解析器结构
            "frame_observations": frame_observations,
            "overall_activity_summaries": overall_activity_summaries,
        }
        if (subtitle_content or "").strip():
            attach_subtitles_to_frame_analysis_artifact(
                artifact,
                subtitle_content,
                settings=documentary_settings or get_documentary_settings(),
            )
        self._finalize_scene_segments_in_artifact(artifact)
        return artifact

    @staticmethod
    def _finalize_scene_segments_in_artifact(artifact: dict[str, Any]) -> None:
        """统一 scene_segments 为六核心字段（+ 可选字幕对位字段）。"""
        segments = artifact.get("scene_segments") or []
        if isinstance(segments, list):
            artifact["scene_segments"] = [
                slim_scene_segment_for_artifact(segment)
                for segment in segments
                if isinstance(segment, dict)
            ]
        for batch in artifact.get("batches") or []:
            if not isinstance(batch, dict):
                continue
            batch_segments = batch.get("scene_segments") or []
            if isinstance(batch_segments, list):
                batch["scene_segments"] = [
                    slim_scene_segment_for_artifact(segment)
                    for segment in batch_segments
                    if isinstance(segment, dict)
                ]

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
        output_path: str | None = None,
    ) -> str:
        analysis_dir = self.analysis_artifact_dir()
        os.makedirs(analysis_dir, exist_ok=True)

        if output_path:
            file_path = output_path
        elif video_path:
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
            if batch.scene_segments:
                for segment in batch.scene_segments:
                    clips.append(
                        {
                            "timestamp": segment.get("timestamp") or batch.time_range,
                            "picture": self._format_scene_segment_picture(segment),
                            "narration": "",
                            "OST": default_ost,
                        }
                    )
                continue
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
    def _format_scene_segment_picture(segment: dict[str, Any]) -> str:
        observation = str(segment.get("observation") or "").strip()
        if observation:
            return observation
        parts: list[str] = []
        scene = str(segment.get("scene") or "").strip()
        if scene:
            parts.append(scene)
        characters = segment.get("characters") or []
        if isinstance(characters, list):
            names = [str(name).strip() for name in characters if str(name).strip()]
            if names:
                parts.append("、".join(names))
        for key in ("action", "key_visual", "emotion"):
            text = str(segment.get(key) or "").strip()
            if text:
                parts.append(text)
        return "，".join(parts) or str(segment.get("action") or "").strip() or "场景片段"

    @staticmethod
    def _get_video_duration_sec(video_path: str) -> float:
        try:
            processor = video_processor.VideoProcessor(video_path)
            return float(processor.duration or 0)
        except Exception as exc:
            logger.warning(f"无法读取视频时长: {video_path} ({exc})")
            return 0.0

    def _build_batch_picture(self, batch: FrameBatchResult) -> str:
        if batch.scene_segments:
            return " ".join(
                self._format_scene_segment_picture(segment)
                for segment in batch.scene_segments
                if isinstance(segment, dict)
            ).strip()

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
        vision_model_used: str = "",
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
            vision_model_used=vision_model_used,
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
                "documentary-frame-analysis-v3",
            ]
        )
        return f"{legacy_prefix}_{utils.md5(payload)}"

    @staticmethod
    def _normalize_scene_segment(entry: dict[str, Any]) -> dict[str, Any]:
        characters = entry.get("characters")
        if isinstance(characters, str):
            char_list = [part.strip() for part in re.split(r"[、,，/]", characters) if part.strip()]
        elif isinstance(characters, list):
            char_list = [str(name).strip() for name in characters if str(name).strip()]
        else:
            char_list = []

        return {
            "timestamp": str(entry.get("timestamp") or "").strip(),
            "scene": str(entry.get("scene") or "").strip(),
            "observation": str(entry.get("observation") or "").strip(),
            "characters": char_list,
            "action": str(entry.get("action") or "").strip(),
            "emotion": str(entry.get("emotion") or "").strip(),
            "key_visual": str(entry.get("key_visual") or "").strip(),
            "audio_cue": str(entry.get("audio_cue") or "").strip(),
            "importance": str(entry.get("importance") or "").strip(),
        }

    def _parse_scene_segments(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        raw_segments = payload.get("scene_segments")
        if not isinstance(raw_segments, list):
            return []
        segments: list[dict[str, Any]] = []
        for item in raw_segments:
            if isinstance(item, dict):
                normalized = self._normalize_scene_segment(item)
                if any(normalized.values()):
                    segments.append(normalized)
        return segments

    def _synthesize_scene_segments_from_frames(
        self,
        frame_observations: list[dict[str, Any]],
        *,
        time_range: str,
        overall_summary: str,
    ) -> list[dict[str, Any]]:
        if not frame_observations:
            return []

        observations = [
            str(frame.get("observation") or "").strip()
            for frame in frame_observations
            if str(frame.get("observation") or "").strip()
        ]
        action_text = overall_summary.strip() or "；".join(observations[:3])
        if len(observations) > 3:
            action_text = f"{action_text}…"

        return [
            self._normalize_scene_segment(
                {
                    "timestamp": time_range,
                    "scene": "",
                    "observation": observations[0] if observations else action_text,
                    "characters": [],
                    "action": action_text,
                    "emotion": "",
                    "key_visual": observations[0] if observations else "",
                    "audio_cue": "",
                    "importance": "中",
                }
            )
        ]

    def _synthesize_frame_observations_from_scenes(
        self,
        scene_segments: list[dict[str, Any]],
        frame_paths: list[str],
    ) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        fallback_summary = self._format_scene_segment_picture(scene_segments[0]) if scene_segments else ""

        for frame_path in frame_paths:
            timestamp = self._timestamp_from_keyframe_name(frame_path)
            matched = scene_segments[0] if scene_segments else {}
            for segment in scene_segments:
                segment_range = str(segment.get("timestamp") or "").strip()
                if not segment_range or "-" not in segment_range:
                    continue
                start_text, end_text = segment_range.split("-", 1)
                start_ms = self._timestamp_to_milliseconds(start_text.strip())
                end_ms = self._timestamp_to_milliseconds(end_text.strip())
                ts_ms = self._timestamp_to_milliseconds(timestamp)
                if start_ms <= ts_ms <= end_ms or (start_ms == 0 and end_ms == 0):
                    matched = segment
                    break

            observation = self._format_scene_segment_picture(matched) if matched else fallback_summary
            observations.append(
                {
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                    "observation": observation,
                    "burned_in_subtitle": "",
                    "has_burned_in_subtitle": False,
                }
            )
        return observations

    def _strip_code_fence(self, response_text: str) -> str:
        cleaned = (response_text or "").strip()
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def _sanitize_json_text(self, response_text: str) -> str:
        """去掉 code fence 与非法控制字符，便于解析视觉模型 JSON。"""
        cleaned = self._strip_code_fence(response_text)
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)

    def _load_batch_payload_json(self, raw_response: str) -> Any:
        cleaned = self._sanitize_json_text(raw_response)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            for opener, closer in (("[", "]"), ("{", "}")):
                start = cleaned.find(opener)
                end = cleaned.rfind(closer)
                if start >= 0 and end > start:
                    try:
                        return json.loads(cleaned[start : end + 1])
                    except json.JSONDecodeError:
                        continue
            raise

    def _coerce_batch_payload(
        self,
        payload: Any,
        *,
        time_range: str,
    ) -> dict[str, Any]:
        """将视觉模型多种 JSON 形态规范为 scene_segments + frame_observations 结构。"""
        if isinstance(payload, dict):
            if payload.get("scene_segments") or payload.get("frame_observations"):
                return payload
            if any(payload.get(key) for key in ("timestamp", "action", "scene", "picture", "narration")):
                segment = self._normalize_scene_segment(payload)
                if not segment.get("timestamp"):
                    segment["timestamp"] = time_range
                return {
                    "scene_segments": [segment],
                    "frame_observations": [],
                    "overall_activity_summary": segment.get("action") or "",
                }
            return payload

        if not isinstance(payload, list):
            return {}

        segments: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            if any(item.get(key) for key in ("scene", "action", "key_visual", "characters")):
                segment = self._normalize_scene_segment(item)
                if not segment.get("timestamp"):
                    segment["timestamp"] = time_range
                segments.append(segment)
                continue
            picture = str(item.get("picture") or "").strip()
            narration = str(item.get("narration") or "").strip()
            if picture or narration or item.get("timestamp"):
                segments.append(
                    self._normalize_scene_segment(
                        {
                            "timestamp": item.get("timestamp") or time_range,
                            "scene": "",
                            "characters": [],
                            "action": picture or narration[:200],
                            "emotion": "",
                            "key_visual": picture,
                            "audio_cue": "原声对白" if item.get("OST") == 1 else "",
                            "importance": "中",
                        }
                    )
                )

        if not segments:
            return {}

        summary_parts = [segment.get("action") or "" for segment in segments[:3] if segment.get("action")]
        return {
            "scene_segments": segments,
            "frame_observations": [],
            "overall_activity_summary": "；".join(summary_parts)[:300],
        }

    def _parse_batch_response(
        self,
        *,
        batch_index: int,
        raw_response: str,
        frame_paths: list[str],
        time_range: str,
        vision_model_used: str = "",
    ) -> FrameBatchResult:
        try:
            payload_raw = self._load_batch_payload_json(raw_response)
            payload = self._coerce_batch_payload(payload_raw, time_range=time_range)
            if not isinstance(payload, dict) or not payload:
                return self._build_failed_batch_result(
                    batch_index=batch_index,
                    raw_response=raw_response,
                    error_message="Batch response JSON payload must be an object with scene_segments or frame_observations",
                    frame_paths=frame_paths,
                    time_range=time_range,
                )
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

        scene_segments = self._parse_scene_segments(payload)
        raw_observations = payload.get("frame_observations")
        if not isinstance(raw_observations, list):
            raw_observations = []

        frame_observations: list[dict] = []
        if len(raw_observations) >= len(frame_paths):
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
        elif scene_segments:
            frame_observations = self._synthesize_frame_observations_from_scenes(
                scene_segments,
                frame_paths,
            )

        raw_summary = payload.get("overall_activity_summary", "")
        if isinstance(raw_summary, str):
            summary = raw_summary
        elif raw_summary is None:
            summary = ""
        else:
            summary = str(raw_summary)

        if not scene_segments and frame_observations:
            scene_segments = self._synthesize_scene_segments_from_frames(
                frame_observations,
                time_range=time_range,
                overall_summary=summary,
            )

        warn_frame_analysis_gender_mismatch(
            scene_segments=scene_segments,
            frame_observations=frame_observations,
            batch_index=batch_index,
            time_range=time_range,
        )

        return FrameBatchResult(
            batch_index=batch_index,
            status="success",
            time_range=time_range,
            raw_response=raw_response,
            frame_paths=list(frame_paths),
            frame_observations=frame_observations,
            scene_segments=scene_segments,
            overall_activity_summary=summary,
            vision_model_used=vision_model_used,
        )

    def _validate_batch_payload_contract(self, payload: object, *, expected_frame_count: int) -> str:
        if not isinstance(payload, dict):
            return "Batch response JSON payload must be an object"

        scene_segments = payload.get("scene_segments")
        has_scenes = isinstance(scene_segments, list) and len(scene_segments) > 0

        frame_observations = payload.get("frame_observations")
        has_frames = isinstance(frame_observations, list) and len(frame_observations) >= expected_frame_count

        if has_scenes or has_frames:
            return ""

        if isinstance(frame_observations, list) and frame_observations:
            return (
                "Batch response frame_observations length is shorter than provided frame_paths: "
                f"{len(frame_observations)} < {expected_frame_count}"
            )

        return "Batch response must include scene_segments or frame_observations"
