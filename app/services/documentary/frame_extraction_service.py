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
    FRAME_ANALYSIS_ARTIFACT_VERSION,
    analysis_artifact_dir as pairing_analysis_artifact_dir,
    default_analysis_path_for_video as pairing_default_analysis_path_for_video,
    default_test_analysis_path_for_video as pairing_default_test_analysis_path_for_video,
    is_valid_analysis_artifact as pairing_is_valid_analysis_artifact,
    load_analysis_artifact as pairing_load_analysis_artifact,
    resolve_reusable_analysis_path as pairing_resolve_reusable_analysis_path,
)
from app.services.documentary.documentary_settings import (
    build_frame_character_naming_hint,
    build_frame_gender_hint,
    build_frame_highlight_hint,
    build_frame_visible_content_hint,
    get_documentary_settings,
    warn_frame_analysis_gender_mismatch,
)
from app.services.documentary.frame_analysis_compact import (
    keyframe_basename,
    slim_scene_segment_for_artifact,
)
from app.services.documentary.frame_extraction_rules import (
    build_frame_extraction_prompt_body,
    enrich_scene_segment_from_editor_fields,
)
from app.services.documentary.frame_timeline_refinement import (
    build_scene_segments_from_frame_observations,
    refine_batch_from_frame_observations,
)
from app.services.documentary.frame_timeline_sampling import (
    infer_scene_label_from_segment,
    normalize_scene_segments,
    resolve_frame_max_segment_duration_ms,
)
from app.services.documentary.documentary_subtitle_enrichment import (
    attach_subtitles_to_frame_analysis_artifact,
    subtitle_excerpt_for_time_range,
)
from app.services.documentary.frame_dialogue_alignment import (
    apply_dialogue_alignment_to_artifact,
    build_frame_dialogue_speaker_rules,
)
from app.services.documentary.frame_character_naming import (
    apply_face_gated_names_to_artifact,
    apply_obvious_character_relationships_to_artifact,
    build_frame_face_match_batch_hint,
    build_frame_naming_priority_rules,
    validate_face_naming_when_references_attached,
)
from app.services.drama_character_registry import (
    build_batch_vision_reference_prompt_section,
    merge_frame_analysis_settings_for_drama,
    project_root,
    resolve_media_path,
)
from app.services.documentary.plot_reference import build_plot_reference_prompt_section
from app.services.short_drama_drama_knowledge import (
    apply_name_corrections_to_frame_analysis_artifact,
    build_frame_analysis_drama_knowledge_section,
    build_frame_obvious_relationship_hint,
)
from app.config.llm_gateway_router import describe_llm_route, resolve_llm_credentials
from app.services.documentary.vision_model_rotation import (
    VisionModelRotation,
    resolve_vision_model_chain,
)
from app.utils import utils, video_processor


class DocumentaryFrameExtractionService:
    PROMPT_TEMPLATE = (
        "我提供了 {frame_count} 张视频帧，按时间顺序排列，代表一个连续的视频片段。\n"
        "{editor_prompt_body}"
    ).strip()

    BURNED_IN_SUBTITLE_PROMPT_SUFFIX = """
同时，请识别每帧画面**底部烧录硬字幕**（烧录在画面上的对白文字，非画面描述）：
- burned_in_subtitle: **逐字原样**复制硬字幕原文；若无硬字幕、仅水印/logo 或无法辨认，则为空字符串
- has_burned_in_subtitle: 布尔值，画面底部是否清晰可见硬字幕
不要猜测听不清的内容；不要改写、补全或「纠正」错别字；多行字幕合并为一行。该字段用于对照字幕与构思蓝图，请尽量准确复刻画面文字。
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
        plot_reference: str = "",
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
        drama_id: str = "",
        character_references: list[dict[str, str]] | None = None,
        relationship_diagram_path: str = "",
        frame_drama_knowledge_text_enabled: bool = False,
        frame_relationship_diagram_enabled: bool = False,
        max_duration_seconds: float | None = None,
        start_time_seconds: float | None = None,
        test_mode: bool = False,
    ) -> dict[str, Any]:
        progress = progress_callback or (lambda _p, _m: None)
        resolved_max_duration = self._resolve_max_duration_seconds(max_duration_seconds, test_mode=test_mode)
        resolved_start_time = self._resolve_start_time_seconds(start_time_seconds, test_mode=test_mode)
        resolved_relationship = (
            resolve_media_path(relationship_diagram_path)
            if frame_relationship_diagram_enabled
            else ""
        )
        doc_settings = merge_frame_analysis_settings_for_drama(
            documentary_settings or get_documentary_settings(),
            drama_id,
            enable_knowledge_text=frame_drama_knowledge_text_enabled,
        )
        character_references = character_references if character_references is not None else []
        relationship_diagram_path = resolved_relationship

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
        keyframe_files = self._load_or_extract_keyframes(
            video_path,
            frame_interval_seconds,
            max_duration_seconds=resolved_max_duration,
            start_time_seconds=resolved_start_time,
        )
        if resolved_max_duration:
            if resolved_start_time > 0:
                progress(
                    25,
                    f"关键帧准备完成，共 {len(keyframe_files)} 帧"
                    f"（测试：{resolved_start_time:g}s–{resolved_start_time + resolved_max_duration:g}s）",
                )
            else:
                progress(25, f"关键帧准备完成，共 {len(keyframe_files)} 帧（测试：前 {resolved_max_duration:g} 秒）")
        else:
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
            plot_reference=plot_reference,
            video_theme=video_theme,
            max_concurrency=concurrency,
            progress_callback=progress,
            documentary_settings=doc_settings,
            subtitle_content=subtitle_content,
            character_references=character_references,
            relationship_diagram_path=relationship_diagram_path,
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
            drama_id=drama_id,
            character_references=character_references,
            relationship_diagram_path=relationship_diagram_path,
            frame_drama_knowledge_text_enabled=frame_drama_knowledge_text_enabled,
            frame_relationship_diagram_enabled=frame_relationship_diagram_enabled,
            plot_reference=plot_reference,
            test_mode=test_mode,
            test_max_duration_seconds=resolved_max_duration,
            test_start_time_seconds=resolved_start_time,
        )
        output_path = None
        if test_mode and resolved_max_duration:
            output_path = self.default_test_analysis_path_for_video(
                video_path,
                max_duration_seconds=resolved_max_duration,
                start_time_seconds=resolved_start_time,
            )
        analysis_json_path = self._save_analysis_artifact(
            artifact,
            video_path=video_path,
            output_path=output_path,
        )
        video_clip_json = self._build_video_clip_json(sorted_batches, doc_settings)

        overview = artifact.get("video_segment_overview") or {}
        segment_count = int(overview.get("segment_count") or len(artifact.get("scene_segments") or []))
        progress(75, f"逐帧分析完成，全片共 {segment_count} 个场景片段")
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
        plot_reference: str = "",
        vision_llm_provider: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        vision_api_key: str | None = None,
        vision_model_name: str | None = None,
        vision_base_url: str | None = None,
        max_concurrency: int | None = None,
        documentary_settings: dict | None = None,
        subtitle_content: str = "",
        drama_id: str = "",
        character_references: list[dict[str, str]] | None = None,
        relationship_diagram_path: str = "",
        frame_drama_knowledge_text_enabled: bool = False,
        frame_relationship_diagram_enabled: bool = False,
    ) -> dict[str, Any]:
        """仅重跑 artifact 中 status=failed 的批次，并写回原 JSON。"""
        progress = progress_callback or (lambda _p, _m: None)
        if not analysis_json_path or not os.path.isfile(analysis_json_path):
            raise FileNotFoundError(f"抽帧分析文件不存在: {analysis_json_path}")

        artifact = pairing_load_analysis_artifact(analysis_json_path)
        resolved_plot_reference = (plot_reference or str(artifact.get("plot_reference") or "")).strip()
        resolved_drama_id = (drama_id or str(artifact.get("drama_id") or "")).strip()
        enable_knowledge_text = frame_drama_knowledge_text_enabled
        rel_enabled = frame_relationship_diagram_enabled
        resolved_relationship = ""
        if rel_enabled:
            resolved_relationship = (
                resolve_media_path(relationship_diagram_path)
                or resolve_media_path(str(artifact.get("relationship_diagram_path") or ""))
            )
        doc_settings = merge_frame_analysis_settings_for_drama(
            documentary_settings or get_documentary_settings(),
            resolved_drama_id,
            enable_knowledge_text=enable_knowledge_text,
        )
        if character_references is None:
            stored_refs = artifact.get("character_references")
            character_references = (
                [item for item in stored_refs if isinstance(item, dict)]
                if isinstance(stored_refs, list)
                else []
            )
        else:
            character_references = list(character_references)
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
                artifact=artifact,
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
                "或确保 artifact 中 keyframe_cache_key / frame_files 可定位关键帧缓存目录。"
            )

        progress(20, f"正在重跑 {len(retry_items)} 个批次...")
        retry_results = await self._analyze_batch_items(
            rotation=rotation,
            items=retry_items,
            custom_prompt=custom_prompt,
            plot_reference=resolved_plot_reference,
            video_theme=video_theme,
            max_concurrency=concurrency,
            progress_callback=progress,
            documentary_settings=doc_settings,
            subtitle_content=subtitle_content,
            character_references=character_references,
            relationship_diagram_path=resolved_relationship,
        )

        merged_by_index: dict[int, FrameBatchResult] = {
            self._batch_dict_to_result(batch, artifact=artifact).batch_index: self._batch_dict_to_result(
                batch, artifact=artifact
            )
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
        progress(85, "正在合并并重写分析 JSON...")
        test_mode, test_max_duration, test_start_time = self._resolve_test_window_from_artifact(artifact)
        existing_cache_key = str(artifact.get("keyframe_cache_key") or "").strip()
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
            drama_id=resolved_drama_id,
            character_references=character_references,
            relationship_diagram_path=resolved_relationship,
            frame_drama_knowledge_text_enabled=enable_knowledge_text,
            frame_relationship_diagram_enabled=rel_enabled,
            plot_reference=resolved_plot_reference,
            test_mode=test_mode,
            test_max_duration_seconds=test_max_duration,
            test_start_time_seconds=test_start_time,
        )
        if existing_cache_key:
            new_artifact["keyframe_cache_key"] = existing_cache_key
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

            from app.services.documentary.frame_analysis_compact import resolve_batch_frame_files

            frame_paths = resolve_batch_frame_files(artifact, batch)
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
        subtitle_content: str = "",
        *,
        documentary_settings: dict | None = None,
    ) -> dict[str, Any]:
        """刷新抽帧 JSON 内 scene 字幕（仅 burned_in，无需重抽帧、无需 SRT）。"""
        cfg = documentary_settings or get_documentary_settings()
        from app.services.documentary.documentary_subtitle_enrichment import (
            attach_burned_in_subtitles_to_artifact,
        )
        from app.services.documentary.frame_analysis_compact import (
            compress_analysis_artifact,
            normalize_analysis_artifact_storage,
        )

        attach_burned_in_subtitles_to_artifact(artifact, settings=cfg)
        normalize_analysis_artifact_storage(artifact, settings=cfg)
        if cfg.get("compress_frame_analysis_on_save", False):
            compress_analysis_artifact(artifact, settings=cfg, strip_debug=True)
        return artifact

    @staticmethod
    def refresh_subtitles_in_analysis_file(
        analysis_json_path: str,
        *,
        documentary_settings: dict | None = None,
    ) -> str:
        """已有抽帧 JSON：仅按硬字幕重算 segment.subtitle，不重新抽帧。"""
        artifact = pairing_load_analysis_artifact(analysis_json_path)
        cfg = documentary_settings or get_documentary_settings()
        DocumentaryFrameExtractionService.enrich_analysis_artifact_subtitles(
            artifact,
            documentary_settings=cfg,
        )
        with open(analysis_json_path, "w", encoding="utf-8") as fp:
            if cfg.get("compact_analysis_json"):
                json.dump(artifact, fp, ensure_ascii=False, separators=(",", ":"))
            else:
                json.dump(artifact, fp, ensure_ascii=False, indent=2)
        logger.info(f"已刷新抽帧硬字幕字段: {analysis_json_path}")
        return analysis_json_path

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
    def _batch_dict_to_result(
        batch: dict[str, Any],
        *,
        artifact: dict[str, Any] | None = None,
    ) -> FrameBatchResult:
        from app.services.documentary.frame_analysis_compact import resolve_batch_frame_files

        frame_paths = resolve_batch_frame_files(artifact, batch) if artifact else []
        batch_index = int(batch.get("batch_index", 0))
        frame_observations = list(batch.get("frame_observations") or [])
        if not frame_observations and artifact:
            frame_observations = [
                item
                for item in (artifact.get("frame_observations") or [])
                if isinstance(item, dict) and int(item.get("batch_index", 0)) == batch_index
            ]
        return FrameBatchResult(
            batch_index=int(batch.get("batch_index", 0)),
            status=str(batch.get("status") or "failed"),
            time_range=str(batch.get("time_range") or ""),
            raw_response=str(batch.get("raw_response") or ""),
            frame_paths=frame_paths,
            frame_observations=frame_observations,
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
        artifact: dict[str, Any] | None = None,
    ) -> list[str]:
        if artifact:
            from app.services.documentary.frame_analysis_compact import resolve_batch_frame_files

            resolved = resolve_batch_frame_files(artifact, batch)
            if resolved:
                return resolved

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
        plot_reference: str = "",
        video_theme: str = "",
        max_concurrency: int,
        progress_callback: Callable[[float, str], None],
        documentary_settings: dict | None = None,
        subtitle_content: str = "",
        character_references: list[dict[str, str]] | None = None,
        relationship_diagram_path: str = "",
    ) -> list[FrameBatchResult]:
        doc_settings = documentary_settings or get_documentary_settings()
        resolved_refs = self._resolve_character_references(character_references)
        resolved_relationship = resolve_media_path(relationship_diagram_path)
        semaphore = asyncio.Semaphore(max(1, max_concurrency))
        total = len(items)
        done = 0
        done_lock = asyncio.Lock()

        async def run_single(batch_index: int, frame_paths: list[str], time_range: str) -> FrameBatchResult:
            nonlocal done
            vision_images, active_refs, carryover, ref_image_count = self._compose_batch_vision_inputs(
                frame_paths,
                batch_index=batch_index,
                relationship_diagram_path=resolved_relationship,
                character_references=resolved_refs,
                documentary_settings=doc_settings,
            )
            prompt = self._build_batch_prompt(
                frame_count=len(frame_paths),
                video_theme=video_theme,
                custom_prompt=custom_prompt,
                plot_reference=plot_reference,
                documentary_settings=doc_settings,
                time_range=time_range,
                subtitle_content=subtitle_content,
                character_references=active_refs,
                relationship_diagram_path=resolved_relationship,
                drama_label=str(doc_settings.get("default_video_theme") or video_theme or ""),
                reference_carryover_prompt=carryover,
                reference_image_count=ref_image_count,
            )
            try:
                async with semaphore:
                    raw_results, model_used, error_message = await rotation.analyze_images(
                        images=vision_images,
                        prompt=prompt,
                        batch_size=max(1, len(vision_images)),
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
                    character_references=active_refs,
                    reference_images_attached=ref_image_count > 0,
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

    @staticmethod
    def _resolve_max_duration_seconds(
        max_duration_seconds: float | int | None,
        *,
        test_mode: bool = False,
    ) -> float | None:
        if not test_mode:
            return None
        try:
            parsed = float(max_duration_seconds if max_duration_seconds is not None else 5.0)
        except (TypeError, ValueError):
            parsed = 5.0
        return max(1.0, parsed)

    @staticmethod
    def _resolve_start_time_seconds(
        start_time_seconds: float | int | None,
        *,
        test_mode: bool = False,
    ) -> float:
        if not test_mode:
            return 0.0
        try:
            parsed = float(start_time_seconds if start_time_seconds is not None else 0.0)
        except (TypeError, ValueError):
            parsed = 0.0
        return max(0.0, parsed)

    def _load_or_extract_keyframes(
        self,
        video_path: str,
        frame_interval_seconds: float,
        *,
        max_duration_seconds: float | None = None,
        start_time_seconds: float = 0.0,
    ) -> list[str]:
        keyframes_root = os.path.join(utils.temp_dir(), "keyframes")
        os.makedirs(keyframes_root, exist_ok=True)
        cache_key = self._build_keyframe_cache_key(
            video_path,
            frame_interval_seconds,
            max_duration_seconds=max_duration_seconds,
            start_time_seconds=start_time_seconds,
        )
        cache_dir = os.path.join(keyframes_root, cache_key)
        os.makedirs(cache_dir, exist_ok=True)

        cached_files = self._collect_keyframe_paths(cache_dir)
        if cached_files:
            cached_files = self._filter_keyframes_by_window(
                cached_files,
                start_time_seconds=start_time_seconds,
                max_duration_seconds=max_duration_seconds,
            )
            logger.info(f"使用已缓存关键帧: {cache_dir}, 共 {len(cached_files)} 帧")
            return cached_files

        processor = video_processor.VideoProcessor(video_path)
        extracted = processor.extract_frames_by_interval_with_fallback(
            output_dir=cache_dir,
            interval_seconds=frame_interval_seconds,
            max_duration_seconds=max_duration_seconds,
            start_time_seconds=start_time_seconds,
        )
        keyframe_files = sorted(str(path) for path in extracted if str(path).endswith(".jpg"))
        if not keyframe_files:
            keyframe_files = self._collect_keyframe_paths(cache_dir)
        keyframe_files = self._filter_keyframes_by_window(
            keyframe_files,
            start_time_seconds=start_time_seconds,
            max_duration_seconds=max_duration_seconds,
        )
        if not keyframe_files:
            raise RuntimeError("未提取到任何关键帧")

        logger.info(f"关键帧提取完成: {cache_dir}, 共 {len(keyframe_files)} 帧")
        return keyframe_files

    @staticmethod
    def _resolve_test_window_from_artifact(
        artifact: dict[str, Any],
    ) -> tuple[bool, float | None, float]:
        """从 artifact 读取测试抽帧窗口（重跑 / OCR 还原缓存目录时用）。"""
        test_mode = bool(artifact.get("test_mode"))
        max_duration: float | None = None
        start_time = 0.0
        if not test_mode:
            return False, None, 0.0
        try:
            raw_max = artifact.get("test_max_duration_seconds")
            if raw_max not in (None, ""):
                max_duration = float(raw_max)
        except (TypeError, ValueError):
            max_duration = None
        try:
            start_time = max(0.0, float(artifact.get("test_start_time_seconds") or 0))
        except (TypeError, ValueError):
            start_time = 0.0
        return True, max_duration, start_time

    def _build_keyframe_cache_key(
        self,
        video_path: str,
        frame_interval_seconds: float,
        *,
        max_duration_seconds: float | None = None,
        start_time_seconds: float = 0.0,
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
                str(frame_interval_seconds),
                str(start_time_seconds or ""),
                str(max_duration_seconds or ""),
                "documentary-keyframes-v3",
            ]
        )
        return f"{legacy_prefix}_{utils.md5(payload)}"

    @staticmethod
    def _keyframe_timestamp_seconds(path: str) -> float | None:
        match = re.search(r"keyframe_\d{6}_(\d{9})\.jpg$", os.path.basename(path))
        if not match:
            return None
        token = match.group(1)
        hours = int(token[0:2])
        minutes = int(token[2:4])
        seconds = int(token[4:6])
        milliseconds = int(token[6:9])
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    @classmethod
    def _filter_keyframes_by_window(
        cls,
        keyframe_files: list[str],
        *,
        start_time_seconds: float | None = None,
        max_duration_seconds: float | None = None,
    ) -> list[str]:
        start = max(0.0, float(start_time_seconds or 0))
        end: float | None
        if max_duration_seconds and max_duration_seconds > 0:
            end = start + float(max_duration_seconds)
        else:
            end = None

        filtered: list[str] = []
        for path in keyframe_files:
            timestamp = cls._keyframe_timestamp_seconds(path)
            if timestamp is None:
                continue
            if timestamp < start:
                continue
            if end is not None and timestamp >= end:
                continue
            filtered.append(path)
        return filtered

    @classmethod
    def _filter_keyframes_by_max_duration(
        cls,
        keyframe_files: list[str],
        max_duration_seconds: float | None,
    ) -> list[str]:
        return cls._filter_keyframes_by_window(
            keyframe_files,
            start_time_seconds=0.0,
            max_duration_seconds=max_duration_seconds,
        )

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

    @staticmethod
    def _resolve_character_references(
        character_references: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        resolved: list[dict[str, str]] = []
        root = project_root()
        for item in character_references or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            path = str(item.get("path") or "").strip()
            if path and not os.path.isabs(path):
                path = os.path.join(root, path.replace("/", os.sep))
            if name and path and os.path.isfile(path):
                resolved.append({"name": name, "path": path})
        return resolved

    @staticmethod
    def _compose_batch_vision_inputs(
        frame_paths: list[str],
        *,
        batch_index: int = 0,
        relationship_diagram_path: str = "",
        character_references: list[dict[str, str]] | None = None,
        documentary_settings: dict | None = None,
    ) -> tuple[list[str], list[dict[str, str]], str, int]:
        from app.services.documentary.frame_reference_images import prepare_reference_prefix_images

        refs = DocumentaryFrameExtractionService._resolve_character_references(character_references)
        prefix_paths, carryover = prepare_reference_prefix_images(
            batch_index=batch_index,
            relationship_diagram_path=relationship_diagram_path,
            character_references=refs,
            settings=documentary_settings,
        )
        if not prefix_paths:
            return list(frame_paths), refs, carryover, 0
        return prefix_paths + list(frame_paths), refs, carryover, len(prefix_paths)

    async def _analyze_batches(
        self,
        *,
        rotation: VisionModelRotation,
        batches: list[list[str]],
        custom_prompt: str,
        plot_reference: str = "",
        video_theme: str,
        max_concurrency: int,
        progress_callback: Callable[[float, str], None],
        documentary_settings: dict | None = None,
        subtitle_content: str = "",
        character_references: list[dict[str, str]] | None = None,
        relationship_diagram_path: str = "",
    ) -> list[FrameBatchResult]:
        doc_settings = documentary_settings or get_documentary_settings()
        resolved_refs = self._resolve_character_references(character_references)
        resolved_relationship = resolve_media_path(relationship_diagram_path)
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
            vision_images, active_refs, carryover, ref_image_count = self._compose_batch_vision_inputs(
                frame_paths,
                batch_index=batch_index,
                relationship_diagram_path=resolved_relationship,
                character_references=resolved_refs,
                documentary_settings=doc_settings,
            )
            prompt = self._build_batch_prompt(
                frame_count=len(frame_paths),
                video_theme=video_theme,
                custom_prompt=custom_prompt,
                plot_reference=plot_reference,
                documentary_settings=doc_settings,
                time_range=time_range,
                subtitle_content=subtitle_content,
                character_references=active_refs,
                relationship_diagram_path=resolved_relationship,
                drama_label=str(doc_settings.get("default_video_theme") or video_theme or ""),
                reference_carryover_prompt=carryover,
                reference_image_count=ref_image_count,
            )
            try:
                async with semaphore:
                    raw_results, model_used, error_message = await rotation.analyze_images(
                        images=vision_images,
                        prompt=prompt,
                        batch_size=max(1, len(vision_images)),
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
                    character_references=active_refs,
                    reference_images_attached=ref_image_count > 0,
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
        plot_reference: str = "",
        documentary_settings: dict | None = None,
        time_range: str = "",
        subtitle_content: str = "",
        character_references: list[dict[str, str]] | None = None,
        relationship_diagram_path: str = "",
        drama_label: str = "",
        reference_carryover_prompt: str = "",
        reference_image_count: int = 0,
    ) -> str:
        cfg = documentary_settings or get_documentary_settings()
        prompt = self._build_analysis_prompt(
            frame_count=frame_count,
            include_burned_in_subtitle=bool(cfg.get("enable_hard_subtitle_ocr", True)),
            documentary_settings=cfg,
        )
        extra_lines: list[str] = []
        from app.services.documentary.frame_reference_images import (
            collage_max_heads_per_sheet,
            resolve_reference_collage_mode,
            split_character_references_into_collage_sheets,
        )

        resolved_refs = character_references or []
        use_collage = resolve_reference_collage_mode(cfg, head_count=len(resolved_refs))
        collage_sheets = split_character_references_into_collage_sheets(
            resolved_refs,
            max_per_sheet=collage_max_heads_per_sheet(cfg),
        )
        ref_section = ""
        if reference_carryover_prompt.strip():
            extra_lines.append(reference_carryover_prompt.strip())
        else:
            ref_section = build_batch_vision_reference_prompt_section(
                relationship_diagram_path=relationship_diagram_path,
                character_references=character_references or [],
                video_frame_count=frame_count,
                drama_label=drama_label or (video_theme or cfg.get("default_video_theme") or "").strip(),
                character_collage=use_collage,
                reference_image_count=reference_image_count or None,
                collage_sheets=collage_sheets if use_collage else None,
            )
            if ref_section.strip():
                extra_lines.append(ref_section.strip())
                extra_lines.append(
                    "定妆照仅在本批可见面孔匹配时可写规范姓名；关系图仅作谐音/关系校正，不可猜人。"
                )
        has_drama = bool(cfg.get("enable_frame_analysis_drama_knowledge"))
        has_refs = bool(character_references) or bool(resolve_media_path(relationship_diagram_path))
        priority_rules = build_frame_naming_priority_rules(
            has_drama_knowledge=has_drama,
            has_character_references=has_refs,
            is_carryover_batch=bool(reference_carryover_prompt.strip()),
        )
        extra_lines.append(priority_rules)
        dialogue_hint = build_frame_dialogue_speaker_rules()
        if dialogue_hint.strip():
            extra_lines.append(dialogue_hint.strip())
        face_match_hint = build_frame_face_match_batch_hint(character_references, frame_count=frame_count)
        if face_match_hint.strip():
            extra_lines.append(face_match_hint.strip())
        visible_hint = build_frame_visible_content_hint(cfg, frame_count=frame_count)
        if visible_hint:
            extra_lines.append(visible_hint)
        theme_text = (video_theme or cfg.get("default_video_theme") or "").strip()
        drama_block, _ = build_frame_analysis_drama_knowledge_section(theme_text, cfg)
        if drama_block.strip():
            extra_lines.append(drama_block.strip())
        relationship_hint = build_frame_obvious_relationship_hint(
            (drama_label or video_theme or "").strip()
        )
        if relationship_hint.strip():
            extra_lines.append(relationship_hint.strip())
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
                    f"本批次时间范围 {time_range} 附近字幕对白（分析画面时请对照，不要虚构台词；"
                    f"出现的小名/昵称/关系称呼须原样使用）：{dialogue}"
                )
        extra_lines.append(
            "scene_segments 若含 subtitle_entries（每项 start/end/text），其中 text 为**原片字幕逐条原文**；"
            "characters 中的人名须由本批可见面孔与定妆照匹配；observation/action 禁止写人名；"
            "硬字幕对白与面孔匹配的人名可同时成立；勿改写 subtitle_entries 原文。"
        )
        plot_section = build_plot_reference_prompt_section(plot_reference)
        if plot_section.strip():
            extra_lines.append(plot_section.strip())
        if (video_theme or "").strip():
            extra_lines.append(f"视频主题：{video_theme.strip()}")
        if (custom_prompt or "").strip():
            extra_lines.append(custom_prompt.strip())
        if not extra_lines:
            return prompt

        extras = "\n\n".join(extra_lines)
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
        drama_id: str = "",
        character_references: list[dict[str, str]] | None = None,
        relationship_diagram_path: str = "",
        frame_drama_knowledge_text_enabled: bool = False,
        frame_relationship_diagram_enabled: bool = False,
        plot_reference: str = "",
        test_mode: bool = False,
        test_max_duration_seconds: float | None = None,
        test_start_time_seconds: float = 0.0,
    ) -> dict[str, Any]:
        sorted_batches = self._sort_batch_results(batch_results)

        batch_dicts: list[dict[str, Any]] = []
        frame_observations: list[dict[str, Any]] = []
        scene_segments: list[dict[str, Any]] = []

        for batch in sorted_batches:
            is_success = str(batch.status or "").lower() == "success"
            slim_batch_segments = []
            if is_success:
                slim_batch_segments = [
                    slim_scene_segment_for_artifact(segment)
                    for segment in batch.scene_segments
                    if isinstance(segment, dict)
                ]
                scene_segments.extend(slim_batch_segments)

            batch_payload = {
                "batch_index": batch.batch_index,
                "status": batch.status,
                "time_range": batch.time_range,
                "raw_response": batch.raw_response,
                "frame_files": [
                    name
                    for name in (keyframe_basename(path) for path in batch.frame_paths)
                    if name
                ],
                "scene_segments": slim_batch_segments,
                "frame_observations": list(batch.frame_observations) if is_success else [],
                "overall_activity_summary": batch.overall_activity_summary if is_success else "",
                "fallback_summary": batch.fallback_summary if not is_success else "",
                "error_message": batch.error_message,
                "vision_model_used": batch.vision_model_used,
            }
            batch_dicts.append(batch_payload)

            if not is_success:
                continue

            for observation in batch.frame_observations:
                observation_payload = dict(observation)
                observation_payload["batch_index"] = batch.batch_index
                observation_payload["time_range"] = batch.time_range
                frame_observations.append(observation_payload)

        artifact = {
            "artifact_version": FRAME_ANALYSIS_ARTIFACT_VERSION,
            "generated_at": datetime.now().isoformat(),
            "video_path": video_path,
            "keyframe_cache_key": self._build_keyframe_cache_key(
                video_path,
                frame_interval_seconds,
                max_duration_seconds=test_max_duration_seconds if test_mode else None,
                start_time_seconds=test_start_time_seconds if test_mode else 0.0,
            ),
            "frame_interval_seconds": frame_interval_seconds,
            "vision_batch_size": vision_batch_size,
            "vision_llm_provider": vision_llm_provider,
            "vision_model_name": vision_model_name,
            "vision_fallback_model_names": (vision_fallback_model_names or "").strip(),
            "vision_models_used": list(vision_models_used or []),
            "vision_max_concurrency": max_concurrency,
            "scene_segments": scene_segments,
            "batches": batch_dicts,
            "frame_observations": frame_observations,
        }
        if drama_id:
            artifact["drama_id"] = drama_id
        if (plot_reference or "").strip():
            artifact["plot_reference"] = (plot_reference or "").strip()
        artifact["frame_drama_knowledge_text_enabled"] = bool(frame_drama_knowledge_text_enabled)
        artifact["frame_relationship_diagram_enabled"] = bool(frame_relationship_diagram_enabled)
        cfg = documentary_settings or get_documentary_settings()
        artifact["frame_reference_token_saver"] = bool(cfg.get("frame_reference_token_saver", True))
        if test_mode:
            artifact["test_mode"] = True
            if test_max_duration_seconds:
                artifact["test_max_duration_seconds"] = float(test_max_duration_seconds)
            if test_start_time_seconds > 0:
                artifact["test_start_time_seconds"] = float(test_start_time_seconds)
        resolved_relationship = resolve_media_path(relationship_diagram_path)
        if resolved_relationship:
            artifact["relationship_diagram_path"] = os.path.relpath(
                resolved_relationship,
                start=project_root(),
            ).replace("\\", "/")
        resolved_refs = self._resolve_character_references(character_references)
        if resolved_refs:
            artifact["character_references"] = [
                {
                    "name": item["name"],
                    "path": os.path.relpath(item["path"], start=project_root()).replace("\\", "/")
                    if os.path.isabs(item["path"])
                    else item["path"],
                }
                for item in resolved_refs
            ]
        self._finalize_scene_segments_in_artifact(artifact)
        cfg = documentary_settings or get_documentary_settings()
        attach_subtitles_to_frame_analysis_artifact(
            artifact,
            subtitle_content or "",
            settings=cfg,
        )
        from app.services.documentary.frame_analysis_compact import (
            compress_analysis_artifact,
            normalize_analysis_artifact_storage,
        )

        normalize_analysis_artifact_storage(artifact, settings=cfg)
        if cfg.get("compress_frame_analysis_on_save", False):
            compress_analysis_artifact(artifact, settings=cfg, strip_debug=True)
        return artifact

    @staticmethod
    def _finalize_scene_segments_in_artifact(artifact: dict[str, Any]) -> None:
        """统一 scene_segments 为六核心字段（+ 可选字幕对位字段），并去重重叠片段。"""
        from app.services.documentary.documentary_settings import get_documentary_settings

        cfg = get_documentary_settings()
        strict_scene_rules = bool(cfg.get("enable_frame_strict_scene_rules", True))
        cross_scene_overlap_prune_ratio = float(
            cfg.get("frame_cross_scene_overlap_prune_ratio", 0.5) or 0.5
        )
        segments = artifact.get("scene_segments") or []
        if isinstance(segments, list) and segments:
            slimmed = [
                slim_scene_segment_for_artifact(
                    enrich_scene_segment_from_editor_fields(segment)
                )
                for segment in segments
                if isinstance(segment, dict)
            ]
            normalized = normalize_scene_segments(
                slimmed,
                strict_scene_rules=strict_scene_rules,
                cross_scene_overlap_prune_ratio=cross_scene_overlap_prune_ratio,
                max_duration_ms=resolve_frame_max_segment_duration_ms(cfg),
                settings=cfg,
            )
            artifact["scene_segments"] = normalized

            segments_by_batch: dict[int, list[dict[str, Any]]] = {}
            for segment in normalized:
                if "batch_index" not in segment:
                    continue
                batch_index = int(segment.get("batch_index", 0))
                segments_by_batch.setdefault(batch_index, []).append(segment)

            for batch in artifact.get("batches") or []:
                if not isinstance(batch, dict):
                    continue
                batch_index = int(batch.get("batch_index", 0))
                if str(batch.get("status") or "").lower() != "success":
                    batch["scene_segments"] = []
                    batch["frame_observations"] = []
                    batch.pop("raw_response", None)
                    continue
                batch["scene_segments"] = segments_by_batch.get(batch_index, [])
        apply_name_corrections_to_frame_analysis_artifact(artifact)
        from app.services.documentary.frame_character_naming import (
            apply_segment_character_consistency_to_artifact,
        )

        apply_segment_character_consistency_to_artifact(artifact)
        apply_face_gated_names_to_artifact(artifact)
        apply_dialogue_alignment_to_artifact(artifact)
        apply_obvious_character_relationships_to_artifact(artifact)

    @staticmethod
    def analysis_artifact_dir() -> str:
        return pairing_analysis_artifact_dir()

    @classmethod
    def default_analysis_path_for_video(cls, video_path: str) -> str:
        return pairing_default_analysis_path_for_video(video_path)

    @classmethod
    def default_test_analysis_path_for_video(
        cls,
        video_path: str,
        *,
        max_duration_seconds: float = 5.0,
        start_time_seconds: float = 0.0,
    ) -> str:
        return pairing_default_test_analysis_path_for_video(
            video_path,
            max_duration_seconds=max_duration_seconds,
            start_time_seconds=start_time_seconds,
        )

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
            from app.services.documentary.documentary_settings import get_documentary_settings

            cfg = get_documentary_settings()
            if cfg.get("compact_analysis_json"):
                json.dump(artifact, fp, ensure_ascii=False, separators=(",", ":"))
            else:
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
    def _format_scene_segment_picture(
        segment: dict[str, Any],
        *,
        env_context: dict[str, str] | None = None,
    ) -> str:
        from app.services.documentary.frame_timeline_sampling import (
            resolve_segment_display_fields,
            update_segment_environment_context,
        )

        display = resolve_segment_display_fields(segment, env_context)
        if env_context is not None:
            update_segment_environment_context(env_context, segment)

        observation = str(display.get("observation") or "").strip()
        if observation:
            return observation
        parts: list[str] = []
        scene = str(display.get("scene") or "").strip()
        if scene:
            parts.append(scene)
        characters = segment.get("characters") or []
        if isinstance(characters, list):
            names = [str(name).strip() for name in characters if str(name).strip()]
            if names:
                parts.append("、".join(names))
        for key in ("action", "key_visual", "emotion", "shot_scale", "lighting_time"):
            text = str(display.get(key) or segment.get(key) or "").strip()
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

    def _build_analysis_prompt(
        self,
        frame_count: int,
        *,
        include_burned_in_subtitle: bool = False,
        documentary_settings: dict | None = None,
    ) -> str:
        cfg = documentary_settings or get_documentary_settings()
        example_suffix = (
            self.BURNED_IN_SUBTITLE_JSON_EXAMPLE if include_burned_in_subtitle else ""
        )
        editor_body = build_frame_extraction_prompt_body(
            frame_count=frame_count,
            burned_in_subtitle_example=example_suffix,
            settings=cfg,
        )
        prompt = self.PROMPT_TEMPLATE.format(
            frame_count=frame_count,
            editor_prompt_body=editor_body,
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
            "shot_scale": str(entry.get("shot_scale") or "").strip(),
            "lighting_time": str(entry.get("lighting_time") or "").strip(),
            "edit_role": str(entry.get("edit_role") or "").strip(),
            "audio_cue": str(entry.get("audio_cue") or "").strip(),
            "importance": str(entry.get("importance") or "").strip(),
        }

    @classmethod
    def _ensure_scene_label(cls, entry: dict[str, Any]) -> dict[str, Any]:
        payload = dict(entry)
        if not str(payload.get("scene") or "").strip():
            inferred = infer_scene_label_from_segment(payload)
            if inferred:
                payload["scene"] = inferred
        return payload

    def _parse_scene_segments(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        raw_segments = payload.get("scene_segments")
        if not isinstance(raw_segments, list):
            return []
        segments: list[dict[str, Any]] = []
        for item in raw_segments:
            if isinstance(item, dict):
                normalized = self._ensure_scene_label(self._normalize_scene_segment(item))
                if any(normalized.values()):
                    segments.append(normalized)
        return segments

    def _synthesize_scene_segments_from_frames(
        self,
        frame_observations: list[dict[str, Any]],
        *,
        time_range: str,
        overall_summary: str,
        batch_index: int = 0,
    ) -> list[dict[str, Any]]:
        if not frame_observations:
            return []

        built = build_scene_segments_from_frame_observations(
            frame_observations,
            batch_index=batch_index,
            time_range=time_range,
        )
        if built:
            return [self._normalize_scene_segment(item) for item in built]

        observations = [
            str(frame.get("observation") or "").strip()
            for frame in frame_observations
            if str(frame.get("observation") or "").strip()
        ]
        action_text = overall_summary.strip() or "；".join(observations[:3])
        if len(observations) > 3:
            action_text = f"{action_text}…"

        observation = observations[0] if observations else action_text
        inferred_scene = infer_scene_label_from_segment(
            {
                "observation": observation,
                "action": action_text,
                "key_visual": observation,
            }
        )

        return [
            self._normalize_scene_segment(
                {
                    "timestamp": time_range,
                    "scene": inferred_scene,
                    "observation": observation,
                    "characters": [],
                    "action": action_text,
                    "emotion": "",
                    "key_visual": observation,
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

    @staticmethod
    def _is_narration_script_clip_payload(payload: Any) -> bool:
        """视觉模型误返回解说脚本 JSON（非抽帧 frame_observations）。"""

        def _looks_like_script_clip(item: dict[str, Any]) -> bool:
            if item.get("frame_observations") or item.get("scene_segments"):
                return False
            has_picture = bool(str(item.get("picture") or "").strip())
            has_narration = bool(str(item.get("narration") or "").strip())
            has_script_keys = any(key in item for key in ("_id", "OST", "original_line"))
            return has_picture and has_narration and has_script_keys

        if isinstance(payload, dict):
            return _looks_like_script_clip(payload)
        if isinstance(payload, list):
            clips = [item for item in payload if isinstance(item, dict)]
            return bool(clips) and all(_looks_like_script_clip(item) for item in clips)
        return False

    def _coerce_batch_payload(
        self,
        payload: Any,
        *,
        time_range: str,
    ) -> dict[str, Any]:
        """将视觉模型 JSON 规范为 scene_segments + frame_observations 结构。"""
        if self._is_narration_script_clip_payload(payload):
            return {}

        if isinstance(payload, dict):
            if payload.get("scene_segments") or payload.get("frame_observations"):
                return payload
            if any(payload.get(key) for key in ("timestamp", "action", "scene")):
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
            if any(item.get(key) for key in ("scene", "action", "key_visual", "characters", "observation")):
                segment = self._normalize_scene_segment(item)
                if not segment.get("timestamp"):
                    segment["timestamp"] = time_range
                segments.append(segment)

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
        character_references: list[dict[str, str]] | None = None,
        reference_images_attached: bool = False,
    ) -> FrameBatchResult:
        try:
            payload_raw = self._load_batch_payload_json(raw_response)
            if self._is_narration_script_clip_payload(payload_raw):
                return self._build_failed_batch_result(
                    batch_index=batch_index,
                    raw_response=raw_response,
                    error_message=(
                        "Batch response is narration script JSON (_id/picture/narration/OST), "
                        "not frame analysis (frame_observations + scene_segments required)"
                    ),
                    frame_paths=frame_paths,
                    time_range=time_range,
                )
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
                # 时间码以关键帧文件名为准，避免模型在测试片段中从 00:00:00 重计
                timestamp = self._timestamp_from_keyframe_name(frame_path)
                frame_observations.append(
                    {
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
                batch_index=batch_index,
            )

        scene_segments, frame_observations, summary = refine_batch_from_frame_observations(
            scene_segments,
            frame_observations,
            batch_index=batch_index,
            time_range=time_range,
            overall_summary=summary,
        )

        warn_frame_analysis_gender_mismatch(
            scene_segments=scene_segments,
            frame_observations=frame_observations,
            batch_index=batch_index,
            time_range=time_range,
        )

        naming_error = validate_face_naming_when_references_attached(
            frame_observations=frame_observations,
            scene_segments=scene_segments,
            character_references=character_references,
            reference_images_attached=reference_images_attached,
        )
        if naming_error:
            return self._build_failed_batch_result(
                batch_index=batch_index,
                raw_response=raw_response,
                error_message=naming_error,
                frame_paths=frame_paths,
                time_range=time_range,
                vision_model_used=vision_model_used,
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
