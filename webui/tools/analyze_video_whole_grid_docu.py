# 整片视频网格快扫（固定时间格，简化 JSON）
import asyncio
import os
import time
import traceback

import streamlit as st
from loguru import logger

from app.config import config
from app.services.documentary.video_whole_grid_analysis import (
    VideoWholeGridAnalysisService,
    default_video_whole_grid_analysis_path,
)
from app.services.drama_character_registry import (
    get_drama,
    resolve_active_relationship_diagram_path,
    resolve_character_references,
)


def _normalize_progress_value(progress: float | int) -> int:
    try:
        value = float(progress)
    except (TypeError, ValueError):
        return 0
    if 0.0 <= value <= 1.0:
        value *= 100
    return max(0, min(100, int(round(value))))


def analyze_video_whole_grid_docu(
    params,
    *,
    vision_model_name: str = "",
    grid_interval_seconds: int = 20,
    force_one_shot: bool = False,
    output_path: str = "",
) -> None:
    progress_bar = st.progress(0)
    percent_text = st.empty()
    status_text = st.empty()
    log_text = st.empty()
    recent_logs: list[str] = []

    def update_progress(progress: float, message: str = ""):
        normalized_progress = _normalize_progress_value(progress)
        progress_bar.progress(normalized_progress)
        percent_text.markdown(f"**进度：{normalized_progress}%**")
        if message:
            status_text.markdown(f"🎬 {message}")
            recent_logs.append(message)
            if len(recent_logs) > 10:
                recent_logs.pop(0)
            log_text.markdown(
                "**步骤记录**\n\n" + "\n".join(f"- {item}" for item in recent_logs)
            )
        else:
            status_text.markdown(f"📊 处理中... ({normalized_progress}%)")
        time.sleep(0.05)

    try:
        video_path = (params.video_origin_path or "").strip()
        if not video_path or not os.path.isfile(video_path):
            st.error("请先选择有效的视频文件")
            return

        drama_id = str(st.session_state.get("doc_frame_drama_id") or "").strip()
        if not drama_id:
            st.error("请先在素材预处理区域选择作品名称")
            return

        drama_meta = get_drama(drama_id)
        drama_title = str((drama_meta or {}).get("label") or drama_id).strip()
        selected_names = set(st.session_state.get("doc_frame_selected_character_names") or [])
        character_references = (
            st.session_state.get("doc_frame_character_references")
            or resolve_character_references(drama_id, selected_names=selected_names)
        )
        relationship_diagram_path = resolve_active_relationship_diagram_path(
            drama_id,
            enabled=bool(st.session_state.get("doc_frame_enable_relationship_diagram")),
        )
        ref_hint = (
            f" · {len(character_references)} 张头像参照"
            if character_references
            else " · 未附头像（不确定时标注剧中未明确交代）"
        )

        model_name = (vision_model_name or "").strip() or (
            config.app.get("vision_openai_model_name") or ""
        )
        update_progress(
            0,
            f"网格快扫《{drama_title}》· {os.path.basename(video_path)} · "
            f"格距 {grid_interval_seconds}s · 模型 {model_name}{ref_hint}",
        )

        service = VideoWholeGridAnalysisService()
        artifact = asyncio.run(
            service.analyze_episode(
                video_path=video_path,
                drama_title=drama_title,
                drama_id=drama_id,
                character_references=character_references,
                relationship_diagram_path=relationship_diagram_path,
                vision_model_name=model_name,
                grid_interval_seconds=grid_interval_seconds,
                force_one_shot=force_one_shot,
                progress_callback=update_progress,
                output_path=(output_path or "").strip() or None,
                plot_reference=st.session_state.get("doc_plot_reference", ""),
            )
        )

        result_path = artifact.get("output_path") or default_video_whole_grid_analysis_path(video_path)
        st.session_state["video_whole_grid_analysis_json_path"] = result_path
        update_progress(100, "网格快扫完成")
        st.success(f"✅ 整片网格快扫已保存: {result_path}")
        api_mode = (
            "单次生成 · API 1 次"
            if artifact.get("one_shot")
            else f"API {artifact.get('api_call_count', 1)} 批"
        )
        if artifact.get("grid_interval_auto_adjusted"):
            interval_note = (
                f"格距 {artifact.get('grid_interval_requested')}s→"
                f"{artifact.get('grid_interval_seconds')}s（自动加粗）"
            )
        else:
            interval_note = f"格距 {artifact.get('grid_interval_seconds')}s"
        st.caption(
            f"模型: {artifact.get('vision_model_name', '')} · "
            f"{interval_note} · "
            f"网格: {artifact.get('grid_segment_count', 0)} 格 · "
            f"整片上传 {artifact.get('upload_size_mb', '?')}MB · "
            f"{api_mode}"
        )
        if artifact.get("upload_warning"):
            st.warning(str(artifact.get("upload_warning")))
        coverage_warnings = artifact.get("coverage_warnings") or []
        if coverage_warnings:
            with st.expander(f"时间轴告警 ({len(coverage_warnings)})", expanded=True):
                for warning in coverage_warnings:
                    st.caption(warning)
        if artifact.get("plot_reference_truncated"):
            st.caption("剧情参考过长，已截断注入 prompt；完整内容仍保存在 JSON 的 plot_reference 字段。")
        with st.expander("预览 overall_summary", expanded=True):
            st.write(artifact.get("overall_summary") or "")
            if artifact.get("key_conflict"):
                st.caption(f"核心冲突: {artifact.get('key_conflict')}")
        with st.expander("预览 grid_segments（前 10 格）", expanded=False):
            for item in (artifact.get("grid_segments") or [])[:10]:
                st.caption(
                    f"`{item.get('time_range')}` · "
                    f"{', '.join(item.get('characters') or []) or '—'} · "
                    f"{item.get('description')}"
                )
        with st.expander("预览 JSON", expanded=False):
            st.json(
                {
                    key: artifact.get(key)
                    for key in (
                        "overall_summary",
                        "key_conflict",
                        "grid_segments",
                        "grid_interval_seconds",
                        "single_upload",
                    )
                }
            )
    except Exception as err:
        st.error(f"❌ 整片网格快扫失败: {err}")
        logger.exception(f"整片网格快扫失败\n{traceback.format_exc()}")
