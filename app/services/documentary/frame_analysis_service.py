import json
import re
from typing import Any, Callable

from loguru import logger

from app.config import config
from app.services.documentary.documentary_narration_chunker import reduce_markdown_to_summaries
from app.services.documentary.documentary_coverage_fill import fill_timeline_coverage_gaps
from app.services.documentary.documentary_script_optimizer import finalize_documentary_script_items
from app.services.documentary.documentary_settings import (
    build_append_requirements_section,
    build_compact_coverage_instructions,
    build_coverage_instructions,
    build_effective_documentary_prompt,
    build_narration_style_instructions,
    build_ost_instructions,
    compute_compact_segment_bounds,
    compute_ost1_segment_bounds,
    get_documentary_settings,
    is_compact_documentary_settings,
    is_fazu2_compact_settings,
    resolve_append_custom_prompt,
)
from app.services.documentary.documentary_material_resolver import (
    resolve_frame_analysis_path_for_documentary,
)
from app.services.documentary.documentary_subtitle_enrichment import (
    analyze_subtitle_with_frames,
    build_subtitle_narration_sections,
)
from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService
from app.services.generate_narration_script import generate_narration, parse_frame_analysis_to_markdown


class DocumentaryFrameAnalysisService(DocumentaryFrameExtractionService):
    async def generate_documentary_script(
        self,
        *,
        video_path: str,
        video_theme: str = "",
        custom_prompt: str = "",
        append_custom_prompt: str = "",
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
        analysis_json_path: str | None = None,
        material_source_video_path: str = "",
        reuse_frame_analysis: bool = True,
        plot_blueprint: str | None = None,
    ) -> list[dict]:
        progress = progress_callback or (lambda _p, _m: None)
        doc_settings = documentary_settings or get_documentary_settings()
        blueprint_only = bool((plot_blueprint or "").strip())

        resolved_analysis_path = resolve_frame_analysis_path_for_documentary(
            video_path,
            material_source_video_path=material_source_video_path,
            explicit_path=analysis_json_path,
            reuse=reuse_frame_analysis,
        )
        analysis_json_path = resolved_analysis_path or ""

        if blueprint_only:
            if analysis_json_path:
                progress(70, "可选：复用抽帧分析辅助生成...")
                logger.info(f"构思方案模式：附加抽帧分析 {analysis_json_path}")
            else:
                progress(70, "依据构思方案生成脚本（无需抽帧分析）...")
                logger.info("已有剧情构思方案，跳过抽帧分析 JSON 依赖")
        elif not analysis_json_path:
            raise ValueError(
                "未找到可用的抽帧分析 JSON。请先在「抽帧分析」中点击「抽帧并分析」，"
                "或上传/复用已有分析文件后再生成脚本。"
                "若成片为无字幕版本，请在 WebUI 指定「抽帧/字幕来源视频」。"
            )
        else:
            progress(70, "复用已有抽帧分析，跳过视觉模型...")
            logger.info(f"复用抽帧分析: {analysis_json_path}")
            source = (material_source_video_path or "").strip()
            if source and source != (video_path or "").strip():
                logger.info(
                    f"成片视频与素材来源不同，抽帧分析来自: {source}"
                )

        markdown_output = ""
        analysis_markdown = ""
        if analysis_json_path:
            progress(78, "正在整理抽帧结果...")
            analysis_markdown = self._prepare_frame_markdown(
                analysis_json_path,
                documentary_settings=doc_settings,
                force_compact=True,
            )
            markdown_output = self._prepare_frame_markdown(
                analysis_json_path,
                documentary_settings=doc_settings,
            )
            if not (markdown_output or "").strip() and not blueprint_only:
                raise ValueError(
                    "抽帧分析结果为空。请重新执行「抽帧并分析」，确保视觉模型正常输出。"
                )
        elif not blueprint_only:
            raise ValueError(
                "未找到可用的抽帧分析 JSON。请先在「抽帧分析」中点击「抽帧并分析」，"
                "或上传/复用已有分析文件后再生成脚本。"
            )
        else:
            progress(78, "跳过抽帧 Markdown，以构思方案为主依据...")
        source_duration_sec = self._get_video_duration_sec(video_path)

        if (
            is_fazu2_compact_settings(doc_settings)
            and doc_settings.get("require_subtitle_for_script", True)
            and not blueprint_only
            and not (subtitle_content or "").strip()
        ):
            raise ValueError(
                "逐帧精剪生成脚本需要字幕文件。请先上传/转录 SRT，"
                "建议使用 OCR 校准后的 *_ocr_refined.srt 或 *_refined.srt。"
            )

        subtitle_analysis = ""
        if plot_blueprint is None:
            if (
                doc_settings.get("enable_subtitle_enrichment", True)
                and (subtitle_content or "").strip()
            ):
                progress(79, "正在充分分析字幕并对照抽帧（脚本蓝图）...")
                append_for_analysis = resolve_append_custom_prompt(
                    append_custom_prompt, doc_settings
                )
                if append_for_analysis:
                    logger.info("追加提示词已置于字幕×抽帧蓝图 prompt 首位")
                subtitle_analysis = analyze_subtitle_with_frames(
                    subtitle_content=subtitle_content,
                    frame_markdown=analysis_markdown,
                    video_theme=video_theme,
                    append_custom_prompt=append_custom_prompt,
                    progress_callback=lambda msg: progress(79, msg),
                    documentary_settings=doc_settings,
                    frame_json_path=analysis_json_path or None,
                )
                min_analysis_chars = max(
                    200,
                    int(doc_settings.get("subtitle_analysis_min_chars", 500) or 500),
                )
                if (
                    is_fazu2_compact_settings(doc_settings)
                    and doc_settings.get("require_subtitle_for_script", True)
                    and len(subtitle_analysis.strip()) < min_analysis_chars
                ):
                    raise ValueError(
                        f"字幕×抽帧对照分析过短（{len(subtitle_analysis.strip())} 字），"
                        f"至少需要 {min_analysis_chars} 字。"
                        "请确认已「确认使用」完整 SRT（非空文件），并检查文本模型 API；"
                        "若字幕/抽帧摘要输入过短，请重新上传素材后重试。"
                    )
        else:
            subtitle_analysis = (plot_blueprint or "").strip()
            if subtitle_analysis:
                progress(79, "复用已确认的剧情构思方案，正在生成脚本...")
            min_analysis_chars = max(
                200,
                int(doc_settings.get("subtitle_analysis_min_chars", 500) or 500),
            )
            if (
                is_fazu2_compact_settings(doc_settings)
                and len(subtitle_analysis) < min_analysis_chars
            ):
                raise ValueError(
                    f"剧情构思方案过短（{len(subtitle_analysis)} 字），"
                    f"至少需要 {min_analysis_chars} 字。请完善构思方案。"
                )

        progress(80, "正在依据分析结果生成 JSON 脚本...")
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
            append_custom_prompt=append_custom_prompt,
            documentary_settings=doc_settings,
            source_duration_sec=source_duration_sec,
            subtitle_content=subtitle_content,
            subtitle_analysis=subtitle_analysis,
            text_api_key=text_api_key,
            text_base_url=text_base_url,
            text_model=text_model,
            progress_callback=progress,
        )
        final_script = finalize_documentary_script_items(
            narration_items,
            doc_settings,
            work_name=video_theme,
            subtitle_content=subtitle_content,
            subtitle_frame_analysis=subtitle_analysis,
            frame_analysis_path=analysis_json_path,
        )
        progress(100, "脚本生成完成")
        return final_script

    async def generate_plot_blueprint(
        self,
        *,
        video_path: str,
        video_theme: str = "",
        append_custom_prompt: str = "",
        progress_callback: Callable[[float, str], None] | None = None,
        documentary_settings: dict | None = None,
        subtitle_content: str = "",
        analysis_json_path: str | None = None,
        material_source_video_path: str = "",
        reuse_frame_analysis: bool = True,
        character_relationship: str = "",
    ) -> str:
        """抽帧画面 + SRT 字幕联合分析，产出供写脚本的「完美剧情构思方案」Markdown。"""
        progress = progress_callback or (lambda _p, _m: None)
        doc_settings = documentary_settings or get_documentary_settings()

        resolved_analysis_path = resolve_frame_analysis_path_for_documentary(
            video_path,
            material_source_video_path=material_source_video_path,
            explicit_path=analysis_json_path,
            reuse=reuse_frame_analysis,
        )
        if not resolved_analysis_path:
            raise ValueError(
                "未找到可用的抽帧分析 JSON。请先在「抽帧分析」中点击「抽帧并分析」，"
                "或上传/复用已有分析文件。"
            )
        analysis_json_path = resolved_analysis_path

        progress(20, "正在整理抽帧结果...")
        analysis_markdown = self._prepare_frame_markdown(
            analysis_json_path,
            documentary_settings=doc_settings,
            force_compact=True,
        )
        if not (analysis_markdown or "").strip():
            raise ValueError("抽帧分析结果为空，请重新执行「抽帧并分析」。")

        progress(40, "正在分析字幕并对照抽帧构思剧情方案...")
        append_for_analysis = resolve_append_custom_prompt(
            append_custom_prompt, doc_settings
        )
        if append_for_analysis:
            logger.info("追加提示词已置于抽帧×剧情蓝图 prompt 首位")
        subtitle_analysis = analyze_subtitle_with_frames(
            subtitle_content=subtitle_content,
            frame_markdown=analysis_markdown,
            video_theme=video_theme,
            append_custom_prompt=append_custom_prompt,
            progress_callback=lambda msg: progress(60, msg),
            documentary_settings=doc_settings,
            frame_json_path=analysis_json_path,
            for_plot_blueprint=True,
            source_duration_sec=self._get_video_duration_sec(video_path),
            character_relationship=character_relationship,
        )
        if len((subtitle_analysis or "").strip()) < 200:
            raise ValueError(
                "字幕×抽帧×剧情构思为空或过短，请检查文本模型 API 与素材。"
            )
        progress(100, "剧情构思方案生成完成")
        return subtitle_analysis.strip()

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
        append_custom_prompt: str = "",
        documentary_settings: dict,
        source_duration_sec: float | None,
        subtitle_content: str,
        subtitle_analysis: str,
        text_api_key: str,
        text_base_url: str | None,
        text_model: str,
        progress_callback: Callable[[float, str], None],
    ) -> list[dict[str, Any]]:
        progress_callback(80, "正在生成解说文案...")
        narration_input = self._build_narration_input(
            markdown_output=markdown_output,
            video_theme=video_theme,
            custom_prompt=custom_prompt,
            append_custom_prompt=append_custom_prompt,
            documentary_settings=documentary_settings,
            source_duration_sec=source_duration_sec,
            subtitle_content=subtitle_content,
            subtitle_analysis=subtitle_analysis,
        )
        try:
            items = self._generate_single_pass_narration_items(
                narration_input=narration_input,
                text_api_key=text_api_key,
                text_base_url=text_base_url,
                text_model=text_model,
                documentary_settings=documentary_settings,
                source_duration_sec=source_duration_sec,
            )
        except (ValueError, RuntimeError) as exc:
            if not self._is_context_length_error(exc):
                raise
            logger.warning(f"单次解说生成上下文超限，改用摘要模式重试: {exc}")
            reduced_markdown = reduce_markdown_to_summaries(markdown_output)
            retry_input = self._build_narration_input(
                markdown_output=reduced_markdown,
                video_theme=video_theme,
                custom_prompt=custom_prompt,
                append_custom_prompt=append_custom_prompt,
                documentary_settings=documentary_settings,
                source_duration_sec=source_duration_sec,
                subtitle_content="",
                subtitle_analysis="",
            )
            items = self._generate_single_pass_narration_items(
                narration_input=retry_input,
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
        if not (markdown_output or "").strip():
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
                    "文本模型上下文超限，将尝试压缩输入"
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
                "解说文案格式错误，无法解析 JSON 数组或 items 字段。"
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
        max_segment_retries: int | None = None,
        enforce_compact_min: bool = True,
        segment_bounds_override: tuple[int, int, int] | None = None,
    ) -> list[dict[str, Any]]:
        if max_segment_retries is None:
            max_segment_retries = int(
                documentary_settings.get("narration_segment_max_retries", 3) or 3
            )
        current_input = narration_input
        current_raw = narration_raw

        for attempt in range(max_segment_retries + 1):
            try:
                items = self._parse_narration_items(current_raw)
            except ValueError as parse_error:
                if attempt >= max_segment_retries:
                    raise
                logger.warning(f"解说 JSON 解析失败，正在重试生成 ({attempt + 1}): {parse_error}")
                current_input = (
                    current_input
                    + "\n\n## 重试要求\n"
                    "上次输出无法解析。请严格只输出 JSON 数组，"
                    'OST=1: {"narration":"播放原片","original_line":"「台词」","OST":1}；'
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

            ost1_ok = True
            ost1_count = 0
            min_ost1 = 0
            max_ost1 = 0
            if (
                is_fazu2_compact_settings(documentary_settings)
                and documentary_settings.get("enable_original_audio_highlights", True)
            ):
                min_ost1, max_ost1 = compute_ost1_segment_bounds(
                    len(items), documentary_settings
                )
                ost1_count = sum(1 for item in items if int(item.get("OST", 0)) == 1)
                ost1_ok = min_ost1 <= ost1_count <= max_ost1

            if min_segments <= len(items) <= max_segments and ost1_ok:
                if len(items) < target_segments:
                    logger.warning(
                        f"精剪段数 {len(items)} 低于目标 {target_segments}，"
                        f"但已在 {min_segments}–{max_segments} 段范围内"
                    )
                return items

            if attempt >= max_segment_retries:
                if len(items) > max_segments:
                    raise ValueError(
                        f"精剪脚本段数 {len(items)} 超过上限 {max_segments} 段，"
                        f"目标约 {target_segments} 段。请检查文本模型输出。"
                    )
                if len(items) < min_segments:
                    raise ValueError(
                        f"精剪脚本段数 {len(items)} 不足最少 {min_segments} 段，"
                        f"目标约 {target_segments} 段。请缩短抽帧间隔或检查文本模型输出。"
                    )
                if not ost1_ok:
                    raise ValueError(
                        f"精剪脚本原声 OST=1 为 {ost1_count} 段，"
                        f"必须在 {min_ost1}–{max_ost1} 段之间。请检查文本模型输出。"
                    )

            if not ost1_ok:
                logger.warning(
                    f"原声 OST=1 为 {ost1_count} 段，不在 {min_ost1}–{max_ost1} 段范围内，"
                    f"正在重试生成 ({attempt + 1})..."
                )
                if ost1_count < min_ost1:
                    ost1_fix = (
                        f"上次仅 {ost1_count} 段 OST=1，无效。"
                        f"须 **{min_ost1}–{max_ost1} 段** OST=1（约 50%）；"
                        f"每段原声 **8–18 秒**，须覆盖字幕整句；说完/播完再接 OST=0 解说，禁止半句截断。"
                    )
                else:
                    ost1_fix = (
                        f"上次有 {ost1_count} 段 OST=1，超过上限 {max_ost1}。"
                        f"请保留 **{min_ost1}–{max_ost1} 段** 原声，其余改为 OST=0 解说；"
                        f"禁止解说夹在两个原声之间，须等原声播完再接解说。"
                    )
                current_input = (
                    narration_input
                    + f"\n\n## 原声段数重试\n"
                    f"{ost1_fix}"
                    f"总段数仍须 **{min_segments}–{max_segments} 段**，严格只输出 JSON。"
                )
            elif len(items) > max_segments:
                logger.warning(
                    f"精剪段数 {len(items)} 超过上限 {max_segments} 段（目标 {target_segments}），"
                    f"正在重试生成 ({attempt + 1})..."
                )
                current_input = (
                    narration_input
                    + f"\n\n## 段数过多重试\n"
                    f"上次输出 {len(items)} 个 items，无效。"
                    f"精剪模式 **必须 {min_segments}–{max_segments} 段**，目标约 **{target_segments} 段**。"
                    f"请合并次要情节点、减少分段数量，严格只输出 JSON，不要解释文字。"
                )
            else:
                logger.warning(
                    f"精剪段数 {len(items)} 不足最少 {min_segments} 段（目标 {target_segments}），"
                    f"正在重试生成 ({attempt + 1})..."
                )
                current_input = (
                    narration_input
                    + f"\n\n## 段数不足重试\n"
                    f"上次仅输出 {len(items)} 个 items，无效。"
                    f"精剪模式 **必须 {min_segments}–{max_segments} 段**，目标约 **{target_segments} 段**。"
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
        append_custom_prompt: str = "",
        documentary_settings: dict | None = None,
        source_duration_sec: float | None = None,
        subtitle_content: str = "",
        subtitle_analysis: str = "",
        coverage_override: str = "",
    ) -> str:
        cfg = documentary_settings or get_documentary_settings()
        sections: list[str] = []
        append_block = build_append_requirements_section(append_custom_prompt, cfg).strip()
        if append_block:
            logger.info("追加提示词已置于脚本生成 prompt 首位")
            sections.append(append_block)

        if is_fazu2_compact_settings(cfg):
            from app.services.documentary.documentary_settings import (
                build_compact_pre_script_workflow_instructions,
            )

            workflow = build_compact_pre_script_workflow_instructions(cfg).strip()
            if workflow:
                sections.append(workflow)

        subtitle_sections = build_subtitle_narration_sections(
            subtitle_content=subtitle_content,
            subtitle_analysis=subtitle_analysis,
            settings=cfg,
        )
        sections.extend(subtitle_sections)

        if (markdown_output or "").strip():
            sections.append(
                "## 抽帧画面分析（画面参考 · 截取时间对齐参考；timestamp 仍以字幕/构思方案为准）\n"
                f"{markdown_output.rstrip()}"
            )
        elif (subtitle_analysis or "").strip():
            sections.append(
                "## 说明\n"
                "本次脚本以「剧情构思方案」为主依据生成，未附加抽帧 Markdown。"
                "timestamp、人名与台词以构思方案中的时间线为准。"
            )

        context_lines: list[str] = []
        if (video_theme or "").strip():
            context_lines.append(f"视频主题：{video_theme.strip()}")
        merged_prompt = build_effective_documentary_prompt(
            custom_prompt,
            settings=cfg,
        )
        if merged_prompt:
            context_lines.append(f"补充创作要求（规则模板）：{merged_prompt}")
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
            "只输出合法 JSON 数组 `[...]`，不要 markdown 代码块、不要解释文字、不要输出策划蓝图。",
            "每个 item 必须包含 `_id`、`timestamp`、`picture`、`narration`、`OST`。",
        ]
        if is_fazu2_compact_settings(cfg):
            from app.services.documentary.documentary_settings import (
                resolve_fazu2_opening_climax_hint,
            )

            opening_default = resolve_fazu2_opening_climax_hint(cfg)
            opening_hint = (
                f"第 1 段 OST=1 纯原声（跳楼 sacrifice，{opening_default}）；"
                "第 2 段以「宝子们」开头转场正叙；性别/职级须与抽帧画面对照，勿写错；"
            )
            if append_block:
                opening_hint = (
                    "第 1 段 OST=1 必须落实「本集追加要求」指定的开头高潮，"
                    "纯原声（narration=播放原片+original_line），禁止旁白；"
                )
            final_lines.append(
                "你必须已阅读「原始字幕」「字幕×抽帧 对照分析」「抽帧画面分析」后再写 JSON；"
                "以字幕为主：timestamp、narration、original_line、人名均取自字幕；"
                "抽帧仅用于 picture 与截取时间对齐参考。"
                f"罚罪2 V2：仿照参考模板；{opening_hint}"
                "对照分析中的 OST=1 清单须全部落实；人名用胡小跃/秦枫/伟业/罗博等；"
                "picture 环境描写须与抽帧一致，但不得覆盖字幕剧情。"
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

