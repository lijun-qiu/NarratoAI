import asyncio
import json
import os
import re
from datetime import datetime
from typing import Any, Callable

from loguru import logger

from app.config import config
from app.services.documentary.documentary_narration_chunker import (
    build_chunk_coverage_override,
    merge_narration_items,
    plan_narration_chunks,
    reduce_markdown_to_summaries,
    should_force_narration_chunking,
)
from app.services.documentary.documentary_coverage_fill import fill_timeline_coverage_gaps
from app.services.documentary.documentary_script_optimizer import finalize_documentary_script_items
from app.services.documentary.documentary_settings import (
    build_compact_coverage_instructions,
    build_coverage_instructions,
    build_frame_highlight_hint,
    build_narration_style_instructions,
    build_ost_instructions,
    compute_compact_segment_bounds,
    get_documentary_settings,
    is_compact_documentary_settings,
    is_fazu2_compact_settings,
    resolve_documentary_custom_prompt,
)
from app.services.documentary.documentary_subtitle_enrichment import (
    analyze_subtitle_with_frames,
    build_subtitle_narration_sections,
    subtitle_excerpt_for_time_range,
)
from app.services.documentary.frame_analysis_models import FrameBatchResult
from app.services.generate_narration_script import generate_narration, parse_frame_analysis_to_markdown
from app.services.llm.migration_adapter import create_vision_analyzer
from app.utils import utils, video_processor


class DocumentaryFrameAnalysisService:
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
    {{"timestamp": "00:00:00,000", "observation": "画面描述（含动作/表情修饰）"}}
  ],
  "overall_activity_summary": "本批次主要活动总结"
}}
请务必不要遗漏视频帧，我提供了 {frame_count} 张视频帧，frame_observations 必须包含 {frame_count} 个元素
请只返回 JSON 字符串，不要附加解释文字。
""".strip()

    async def generate_documentary_script(
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
    ) -> list[dict]:
        progress = progress_callback or (lambda _p, _m: None)
        doc_settings = documentary_settings or get_documentary_settings()
        analysis_result = await self.analyze_video(
            video_path=video_path,
            video_theme=video_theme,
            custom_prompt=custom_prompt,
            frame_interval_input=frame_interval_input,
            vision_batch_size=vision_batch_size,
            vision_llm_provider=vision_llm_provider,
            progress_callback=progress_callback,
            vision_api_key=vision_api_key,
            vision_model_name=vision_model_name,
            vision_base_url=vision_base_url,
            max_concurrency=max_concurrency,
            documentary_settings=doc_settings,
            subtitle_content=subtitle_content,
        )
        analysis_json_path = analysis_result["analysis_json_path"]

        progress(78, "正在整理抽帧结果...")
        markdown_output = self._prepare_frame_markdown(
            analysis_json_path,
            documentary_settings=doc_settings,
        )
        source_duration_sec = self._get_video_duration_sec(video_path)

        subtitle_analysis = ""
        if (
            doc_settings.get("enable_subtitle_enrichment", True)
            and (subtitle_content or "").strip()
        ):
            progress(79, "正在分析字幕并对照抽帧...")
            subtitle_analysis = analyze_subtitle_with_frames(
                subtitle_content=subtitle_content,
                frame_markdown=markdown_output,
                video_theme=video_theme,
                progress_callback=lambda msg: progress(79, msg),
                documentary_settings=doc_settings,
            )

        progress(80, "正在生成解说文案...")
        text_provider = config.app.get("text_llm_provider", "openai").lower()
        text_api_key = config.app.get(f"text_{text_provider}_api_key")
        text_model = config.app.get(f"text_{text_provider}_model_name")
        text_base_url = config.app.get(f"text_{text_provider}_base_url")
        if not text_api_key or not text_model:
            raise ValueError(
                f"未配置 {text_provider} 的文本模型参数。"
                f"请在设置中配置 text_{text_provider}_api_key 和 text_{text_provider}_model_name"
            )

        narration_items = self._generate_documentary_narration_items(
            markdown_output=markdown_output,
            video_theme=video_theme,
            custom_prompt=custom_prompt,
            documentary_settings=doc_settings,
            source_duration_sec=source_duration_sec,
            subtitle_content=subtitle_content,
            subtitle_analysis=subtitle_analysis,
            text_api_key=text_api_key,
            text_base_url=text_base_url,
            text_model=text_model,
            progress_callback=progress,
        )
        final_script = finalize_documentary_script_items(narration_items, doc_settings)
        progress(100, "脚本生成完成")
        return final_script

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
        analysis_json_path = self._save_analysis_artifact(artifact)
        video_clip_json = self._build_video_clip_json(sorted_batches, doc_settings)

        progress(75, "逐帧分析完成")
        return {
            "analysis_json_path": analysis_json_path,
            "analysis_artifact": artifact,
            "video_clip_json": video_clip_json,
            "keyframe_files": keyframe_files,
        }

    def _prepare_frame_markdown(
        self,
        analysis_json_path: str,
        *,
        documentary_settings: dict | None = None,
        force_compact: bool = False,
    ) -> str:
        cfg = documentary_settings or get_documentary_settings()
        markdown_output = parse_frame_analysis_to_markdown(analysis_json_path, detail_level="full")
        compact_threshold = int(cfg.get("narration_compact_markdown_chars", 120000))
        if force_compact or len(markdown_output) > compact_threshold:
            compact_markdown = parse_frame_analysis_to_markdown(
                analysis_json_path,
                detail_level="compact",
            )
            logger.info(
                f"抽帧 Markdown 过长（{len(markdown_output)} 字），"
                f"已切换紧凑模式（{len(compact_markdown)} 字）"
            )
            return compact_markdown
        return markdown_output

    def _generate_documentary_narration_items(
        self,
        *,
        markdown_output: str,
        video_theme: str,
        custom_prompt: str,
        documentary_settings: dict,
        source_duration_sec: float | None,
        subtitle_content: str,
        subtitle_analysis: str,
        text_api_key: str,
        text_base_url: str | None,
        text_model: str,
        progress_callback: Callable[[float, str], None],
    ) -> list[dict[str, Any]]:
        segment_min = 0
        segment_target = 0
        if is_compact_documentary_settings(documentary_settings):
            segment_min, segment_target, _segment_max = compute_compact_segment_bounds(
                documentary_settings,
                source_duration_sec,
            )

        probe_input = self._build_narration_input(
            markdown_output=markdown_output,
            video_theme=video_theme,
            custom_prompt=custom_prompt,
            documentary_settings=documentary_settings,
            source_duration_sec=source_duration_sec,
            subtitle_content=subtitle_content,
            subtitle_analysis=subtitle_analysis,
        )
        if not should_force_narration_chunking(
            probe_input,
            settings=documentary_settings,
            total_segment_min=segment_min,
        ):
            try:
                items = self._generate_single_pass_narration_items(
                    narration_input=probe_input,
                    text_api_key=text_api_key,
                    text_base_url=text_base_url,
                    text_model=text_model,
                    documentary_settings=documentary_settings,
                    source_duration_sec=source_duration_sec,
                )
                return self._apply_compact_timeline_fill(
                    items,
                    markdown_output=markdown_output,
                    source_duration_sec=source_duration_sec,
                    documentary_settings=documentary_settings,
                    text_api_key=text_api_key,
                    text_base_url=text_base_url,
                    text_model=text_model,
                )
            except (ValueError, RuntimeError) as exc:
                if not self._is_context_length_error(exc) and not (
                    is_compact_documentary_settings(documentary_settings)
                    and segment_min > 0
                    and "不足最少" in str(exc)
                ):
                    raise
                logger.warning(f"单次解说生成失败，自动切换分块模式: {exc}")

        chunk_plans = plan_narration_chunks(
            markdown_output,
            settings=documentary_settings,
            total_segment_min=segment_min,
            total_segment_target=segment_target,
        )
        logger.info(f"解说输入过长，将分 {len(chunk_plans)} 块生成")
        chunk_items: list[list[dict[str, Any]]] = []
        total_chunks = len(chunk_plans)

        for plan in chunk_plans:
            progress_callback(
                80 + (plan.chunk_index / max(1, total_chunks)) * 15,
                f"正在生成分块解说 ({plan.chunk_index}/{total_chunks})...",
            )
            coverage_override = build_chunk_coverage_override(
                settings=documentary_settings,
                time_range=plan.time_range,
                segment_min=plan.segment_min,
                segment_max=plan.segment_max,
                chunk_index=plan.chunk_index,
                chunk_total=plan.chunk_total,
                core_theme=video_theme,
            )
            chunk_subtitle_analysis = subtitle_analysis if plan.chunk_index == 1 else ""
            if subtitle_content and plan.time_range:
                excerpt = subtitle_excerpt_for_time_range(
                    subtitle_content,
                    plan.time_range,
                    max_lines=20,
                )
                if excerpt:
                    chunk_subtitle_analysis = (
                        f"{chunk_subtitle_analysis}\n\n"
                        f"### 本块 {plan.time_range} 字幕对白（OST=1 时间戳与台词以此为准）\n"
                        f"{excerpt}"
                    ).strip()
            chunk_input = self._build_narration_input(
                markdown_output=plan.markdown,
                video_theme=video_theme,
                custom_prompt=custom_prompt,
                documentary_settings=documentary_settings,
                source_duration_sec=source_duration_sec,
                subtitle_content=subtitle_content if plan.chunk_index == 1 else "",
                subtitle_analysis=chunk_subtitle_analysis,
                coverage_override=coverage_override,
            )
            try:
                chunk_bounds = (plan.segment_min, plan.segment_max, plan.segment_max)
                items = self._generate_single_pass_narration_items(
                    narration_input=chunk_input,
                    text_api_key=text_api_key,
                    text_base_url=text_base_url,
                    text_model=text_model,
                    documentary_settings=documentary_settings,
                    source_duration_sec=source_duration_sec,
                    enforce_compact_min=True,
                    segment_bounds_override=chunk_bounds,
                )
            except Exception as exc:
                if not self._is_context_length_error(exc):
                    raise
                logger.warning(
                    f"分块 {plan.chunk_index} 仍超出上下文，改用摘要模式重试: {exc}"
                )
                reduced_markdown = reduce_markdown_to_summaries(plan.markdown)
                retry_input = self._build_narration_input(
                    markdown_output=reduced_markdown,
                    video_theme=video_theme,
                    custom_prompt=custom_prompt,
                    documentary_settings=documentary_settings,
                    source_duration_sec=source_duration_sec,
                    subtitle_content="",
                    subtitle_analysis="",
                    coverage_override=coverage_override,
                )
                items = self._generate_single_pass_narration_items(
                    narration_input=retry_input,
                    text_api_key=text_api_key,
                    text_base_url=text_base_url,
                    text_model=text_model,
                    documentary_settings=documentary_settings,
                    source_duration_sec=source_duration_sec,
                    enforce_compact_min=True,
                    segment_bounds_override=chunk_bounds,
                )
            chunk_items.append(items)

        merged_items = merge_narration_items(
            chunk_items, settings=documentary_settings
        )
        if is_compact_documentary_settings(documentary_settings):
            min_segments, target_segments, max_segments = compute_compact_segment_bounds(
                documentary_settings,
                source_duration_sec,
            )
            if len(merged_items) < min_segments:
                logger.warning(
                    f"补段后精剪段数 {len(merged_items)} 仍低于最少 {min_segments} 段"
                    f"（目标约 {target_segments} 段），请检查文本模型或抽帧结果"
                )
            if len(merged_items) > max_segments:
                logger.warning(
                    f"分块合并后精剪段数 {len(merged_items)} 超过上限 {max_segments}，将在后处理裁剪"
                )
        return self._apply_compact_timeline_fill(
            merged_items,
            markdown_output=markdown_output,
            source_duration_sec=source_duration_sec,
            documentary_settings=documentary_settings,
            text_api_key=text_api_key,
            text_base_url=text_base_url,
            text_model=text_model,
        )

    def _apply_compact_timeline_fill(
        self,
        items: list[dict[str, Any]],
        *,
        markdown_output: str,
        source_duration_sec: float | None,
        documentary_settings: dict,
        text_api_key: str,
        text_base_url: str | None,
        text_model: str,
    ) -> list[dict[str, Any]]:
        if not is_compact_documentary_settings(documentary_settings):
            return items
        if not documentary_settings.get("enable_full_timeline_coverage", True):
            return items
        if not source_duration_sec or source_duration_sec <= 0:
            return items

        filled = fill_timeline_coverage_gaps(
            items,
            frame_markdown=markdown_output,
            source_duration_sec=source_duration_sec,
            settings=documentary_settings,
            generate_fn=lambda prompt: generate_narration(
                prompt,
                text_api_key,
                base_url=text_base_url,
                model=text_model,
            ),
        )
        min_segments, target_segments, _max_segments = compute_compact_segment_bounds(
            documentary_settings,
            source_duration_sec,
        )
        if len(filled) < min_segments:
            logger.warning(
                f"补段后共 {len(filled)} 段，仍低于目标 {min_segments} 段"
                f"（理想 {target_segments} 段）；将继续后处理"
            )
        else:
            logger.info(
                f"时间线覆盖补段完成，共 {len(filled)} 段（目标约 {target_segments} 段）"
            )
        return filled

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        messages = [str(exc)]
        cause = getattr(exc, "__cause__", None)
        if cause:
            messages.append(str(cause))
        combined = " ".join(messages).lower()
        return (
            "context_length_exceeded" in combined
            or "maximum context length" in combined
            or ("context length" in combined and "reduce the length" in combined)
        )

    def _generate_single_pass_narration_items(
        self,
        *,
        narration_input: str,
        text_api_key: str,
        text_base_url: str | None,
        text_model: str,
        documentary_settings: dict,
        source_duration_sec: float | None,
        enforce_compact_min: bool = True,
        segment_bounds_override: tuple[int, int, int] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            narration_raw = generate_narration(
                narration_input,
                text_api_key,
                base_url=text_base_url,
                model=text_model,
            )
        except Exception as exc:
            if self._is_context_length_error(exc):
                raise ValueError(
                    "文本模型上下文超限，将尝试分块或压缩输入"
                ) from exc
            raise
        return self._generate_narration_items_with_retries(
            narration_input=narration_input,
            narration_raw=narration_raw,
            text_api_key=text_api_key,
            text_base_url=text_base_url,
            text_model=text_model,
            documentary_settings=documentary_settings,
            source_duration_sec=source_duration_sec,
            enforce_compact_min=enforce_compact_min,
            segment_bounds_override=segment_bounds_override,
        )

    def _parse_narration_items(self, narration_raw: str) -> list[dict[str, Any]]:
        raw_text = (narration_raw or "").strip()
        if raw_text.startswith("调用API生成解说文案时出错") or raw_text.startswith("生成解说文案失败"):
            raise ValueError(f"文本模型调用失败: {raw_text[:300]}")

        parsed = self._repair_narration_payload(raw_text)
        items = self._extract_items_from_payload(parsed)

        if not items:
            snippet = raw_text[:800].replace("\n", "\\n")
            logger.error(f"解说 JSON 解析失败，模型返回开头: {snippet}")
            raise ValueError(
                "解说文案格式错误，无法解析 JSON 或缺少 items 字段。"
                f"请检查文本模型是否支持 JSON 输出。返回开头: {raw_text[:200]!r}"
            )

        return items

    def _generate_narration_items_with_retries(
        self,
        *,
        narration_input: str,
        narration_raw: str,
        text_api_key: str,
        text_base_url: str | None,
        text_model: str,
        documentary_settings: dict,
        source_duration_sec: float | None,
        max_segment_retries: int = 2,
        enforce_compact_min: bool = True,
        segment_bounds_override: tuple[int, int, int] | None = None,
    ) -> list[dict[str, Any]]:
        current_input = narration_input
        current_raw = narration_raw

        for attempt in range(max_segment_retries + 2):
            try:
                items = self._parse_narration_items(current_raw)
            except ValueError as parse_error:
                if attempt >= max_segment_retries + 1:
                    raise
                logger.warning(f"解说 JSON 解析失败，正在重试生成 ({attempt + 1}): {parse_error}")
                current_input = (
                    current_input
                    + "\n\n## 重试要求\n"
                    "上次输出无法解析。请严格只输出 JSON 对象，格式为 "
                    '{"items":[{"_id":1,"timestamp":"...","picture":"...","narration":"...","OST":0}]}，'
                    "不要任何其他文字。"
                )
                current_raw = generate_narration(
                    current_input,
                    text_api_key,
                    base_url=text_base_url,
                    model=text_model,
                )
                continue

            if not is_compact_documentary_settings(documentary_settings) or not enforce_compact_min:
                return items

            if segment_bounds_override:
                min_segments, target_segments, max_segments = segment_bounds_override
            else:
                min_segments, target_segments, max_segments = compute_compact_segment_bounds(
                    documentary_settings, source_duration_sec
                )
            if len(items) > max_segments:
                if attempt >= max_segment_retries + 1:
                    logger.warning(
                        f"精剪段数 {len(items)} 超过上限 {max_segments}，将在后处理阶段裁剪"
                    )
                    return items
                logger.warning(
                    f"精剪段数 {len(items)} 超过上限 {max_segments}（目标 {target_segments}），"
                    f"正在重试生成 ({attempt + 1})..."
                )
                trim_hint = (
                    "请合并明显重复、合并同一时刻的冗余段，"
                    if documentary_settings.get("enable_full_timeline_coverage", True)
                    else "请合并冗余、跳过过渡，"
                )
                current_input = (
                    narration_input
                    + f"\n\n## 段数过多重试\n"
                    f"上次输出 {len(items)} 个 items，无效。"
                    f"精剪模式 items 必须在 **{min_segments}–{max_segments} 段** 之间，"
                    f"目标约 **{target_segments} 段**。{trim_hint}"
                    f"仍须满足全片每 {int(documentary_settings.get('coverage_interval_sec', 30))} 秒至少 1 段，"
                    f"严格只输出 JSON，不要解释文字。"
                )
                current_raw = generate_narration(
                    current_input,
                    text_api_key,
                    base_url=text_base_url,
                    model=text_model,
                )
                continue

            if len(items) >= min_segments:
                if len(items) < target_segments:
                    logger.warning(
                        f"精剪段数 {len(items)} 低于目标 {target_segments}，但已达最少 {min_segments} 段"
                    )
                return items

            if attempt >= max_segment_retries + 1:
                raise ValueError(
                    f"精剪脚本段数 {len(items)} 不足最少 {min_segments} 段，"
                    f"目标约 {target_segments} 段。请缩短抽帧间隔或检查文本模型输出。"
                )

            logger.warning(
                f"精剪段数 {len(items)} 不足最少 {min_segments} 段（目标 {target_segments}），"
                f"正在重试生成 ({attempt + 1})..."
            )
            current_input = (
                narration_input
                + f"\n\n## 段数不足重试\n"
                f"上次仅输出 {len(items)} 个 items，无效。"
                f"精剪模式 **必须至少 {min_segments} 段**，目标约 **{target_segments} 段**。"
                f"请按 `<video_frame_description>` 时间线拆成足够数量的精华解说段，"
                f"严格只输出 JSON，不要解释文字。"
            )
            current_raw = generate_narration(
                current_input,
                text_api_key,
                base_url=text_base_url,
                model=text_model,
            )

        raise ValueError("精剪解说生成失败：段数校验未通过")

    @staticmethod
    def _extract_items_from_payload(parsed: Any) -> list[dict[str, Any]]:
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]

        if not isinstance(parsed, dict):
            return []

        raw_items = parsed.get("items")
        if isinstance(raw_items, list):
            return [item for item in raw_items if isinstance(item, dict)]

        if all(key in parsed for key in ("timestamp", "picture", "narration")):
            return [parsed]

        return []

    def _build_narration_input(
        self,
        *,
        markdown_output: str,
        video_theme: str,
        custom_prompt: str,
        documentary_settings: dict | None = None,
        source_duration_sec: float | None = None,
        subtitle_content: str = "",
        subtitle_analysis: str = "",
        coverage_override: str = "",
    ) -> str:
        cfg = documentary_settings or get_documentary_settings()
        sections = [f"## 抽帧画面分析\n{markdown_output.rstrip()}"]

        subtitle_sections = build_subtitle_narration_sections(
            subtitle_content=subtitle_content,
            subtitle_analysis=subtitle_analysis,
            settings=cfg,
        )
        sections.extend(subtitle_sections)

        context_lines: list[str] = []
        if (video_theme or "").strip():
            context_lines.append(f"视频主题：{video_theme.strip()}")
        merged_prompt = resolve_documentary_custom_prompt(custom_prompt, cfg)
        if merged_prompt:
            context_lines.append(f"补充创作要求：{merged_prompt}")
        if context_lines:
            context_block = "\n".join(f"- {line}" for line in context_lines)
            sections.append(f"## 创作上下文\n{context_block}")

        coverage_override_text = (coverage_override or "").strip()
        if coverage_override_text:
            sections.append(coverage_override_text)
        elif is_compact_documentary_settings(cfg):
            coverage_block = build_compact_coverage_instructions(
                cfg, source_duration_sec
            ).strip()
            if coverage_block:
                sections.append(coverage_block)
        elif cfg.get("enable_full_timeline_coverage", True):
            coverage_block = build_coverage_instructions(cfg).strip()
            if coverage_block:
                sections.append(coverage_block)

        ost_block = build_ost_instructions(cfg).strip()
        if ost_block:
            sections.append(ost_block)

        core_theme = (video_theme or "").strip()
        style_block = build_narration_style_instructions(
            cfg, core_theme=core_theme
        ).strip()
        if style_block:
            sections.append(style_block)

        final_lines = [
            "## 最终输出要求",
            "只输出合法 JSON（含 `items` 数组），不要 markdown 代码块、不要解释文字。",
            "每个 item 必须包含 `_id`、`timestamp`、`picture`、`narration`、`OST`。",
        ]
        if is_fazu2_compact_settings(cfg):
            final_lines.append(
                "故事讲述型精剪：严格仿照「输出 JSON 参考模板」与「故事讲述型脚本规则」；"
                "narration/picture 必须用具体人名，禁止警员1/说话人1；"
                "OST=0 讲故事并嵌入对白（30–100 字/段）；OST=1 仅 3–5 个金句原声。"
            )
        sections.append("\n".join(final_lines))

        return "\n\n".join(sections) + "\n"

    def _repair_narration_payload(self, narration_raw: str) -> dict[str, Any] | list[Any] | None:
        def load_json_candidate(payload: str) -> dict[str, Any] | list[Any] | None:
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, (dict, list)):
                    return parsed
                return None
            except Exception:
                return None

        cleaned = (narration_raw or "").strip()
        if not cleaned:
            return None

        candidates: list[str] = [cleaned]
        candidates.append(cleaned.replace("{{", "{").replace("}}", "}"))

        for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE):
            block = match.group(1).strip()
            if block:
                candidates.append(block)

        for match in re.finditer(r"\{[\s\S]*\}", cleaned):
            candidates.append(match.group(0))

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            candidates.append(cleaned[start : end + 1])

        list_start = cleaned.find("[")
        list_end = cleaned.rfind("]")
        if list_start >= 0 and list_end > list_start:
            candidates.append(cleaned[list_start : list_end + 1])

        seen: set[str] = set()
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            parsed = load_json_candidate(candidate)
            if parsed is not None:
                return parsed

        fixed = cleaned.replace("{{", "{").replace("}}", "}")
        fixed_start = fixed.find("{")
        fixed_end = fixed.rfind("}")
        if fixed_start >= 0 and fixed_end > fixed_start:
            fixed = fixed[fixed_start : fixed_end + 1]

        fixed = re.sub(r"^\s*#.*$", "", fixed, flags=re.MULTILINE)
        fixed = re.sub(r"^\s*//.*$", "", fixed, flags=re.MULTILINE)
        fixed = re.sub(r",\s*}", "}", fixed)
        fixed = re.sub(r",\s*]", "]", fixed)
        fixed = re.sub(r"'([^']*)'\s*:", r'"\1":', fixed)
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
        fixed = re.sub(r'([{\[,]\s*)([A-Za-z_][\w\u4e00-\u9fff]*)(\s*:)', r'\1"\2"\3', fixed)
        fixed = re.sub(r'""([^"]*?)""', r'"\1"', fixed)

        return load_json_candidate(fixed)

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
        prompt = self._build_analysis_prompt(frame_count=frame_count)
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

    def _save_analysis_artifact(self, artifact: dict[str, Any]) -> str:
        analysis_dir = os.path.join(utils.storage_dir(), "temp", "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        filename = f"frame_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = os.path.join(analysis_dir, filename)
        suffix = 1
        while os.path.exists(file_path):
            filename = f"frame_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix:02d}.json"
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

    def _build_analysis_prompt(self, frame_count: int) -> str:
        return self.PROMPT_TEMPLATE.format(frame_count=frame_count)

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
            else:
                observation = str(entry or "")
                timestamp = ""
            frame_observations.append(
                {
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                    "observation": observation,
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
