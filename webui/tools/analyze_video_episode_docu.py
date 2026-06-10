# 整片视频单集剧情分析（直接传视频，非抽帧）
import asyncio
import os
import time
import traceback

import streamlit as st
from loguru import logger

from app.services.documentary.video_episode_analysis import (
    VideoEpisodeAnalysisService,
    default_video_episode_analysis_path,
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


def analyze_video_episode_docu(params, *, resume: bool = True, output_path: str = "") -> None:
    """整片视频分析，输出单集剧情 JSON；支持从检查点续跑补全。"""
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
            else " · 未上传头像（人名不确定时将标注剧中未明确交代）"
        )
        update_progress(
            0,
            f"{'续跑补全' if resume else '开始分析'}《{drama_title}》· "
            f"{os.path.basename(video_path)}{ref_hint}",
        )

        service = VideoEpisodeAnalysisService()
        artifact = asyncio.run(
            service.analyze_episode(
                video_path=video_path,
                drama_title=drama_title,
                drama_id=drama_id,
                character_references=character_references,
                relationship_diagram_path=relationship_diagram_path,
                progress_callback=update_progress,
                output_path=(output_path or "").strip() or None,
                resume=resume,
                plot_reference=st.session_state.get("doc_plot_reference", ""),
            )
        )

        output_path = artifact.get("output_path") or default_video_episode_analysis_path(video_path)
        st.session_state["video_episode_analysis_json_path"] = output_path
        analysis_status = str(artifact.get("analysis_status") or "complete").strip()
        failed_chunks = artifact.get("failed_chunk_indices") or []
        if analysis_status == "incomplete" and failed_chunks:
            update_progress(100, "部分完成，可继续补全")
            st.warning(
                f"⚠️ 整片视频分析部分完成: {output_path} · "
                f"已完成 {artifact.get('completed_chunk_count', 0)}/{artifact.get('chunk_count', 1)} 段 · "
                f"失败分段 {failed_chunks}。请点击「补全未完成分析」继续。"
            )
        else:
            update_progress(100, "分析完成")
            st.success(f"✅ 整片视频分析已保存: {output_path}")
        warning_count = len(artifact.get("coverage_warnings") or [])
        st.caption(
            f"模型: {artifact.get('vision_model_name', '')} · "
            f"上传分段: {artifact.get('chunk_count', 1)} · "
            f"情节片段: {artifact.get('episodic_segment_count', 0)} · "
            f"台词: {len(artifact.get('important_dialogues') or [])}"
            + (f" · 告警: {warning_count}" if warning_count else "")
            + (
                f" · 状态: {analysis_status}"
                if analysis_status != "complete"
                else ""
            )
        )
        with st.expander("预览 overall_summary", expanded=True):
            st.write(artifact.get("overall_summary") or "")
            if artifact.get("key_conflict"):
                st.caption(f"核心冲突: {artifact.get('key_conflict')}")
        if warning_count:
            with st.expander(f"片段约束告警 ({warning_count})", expanded=False):
                for warning in artifact.get("coverage_warnings") or []:
                    st.caption(warning)
        with st.expander("预览 JSON", expanded=False):
            st.json(
                {
                    key: artifact.get(key)
                    for key in (
                        "overall_summary",
                        "key_conflict",
                        "episodic_segments",
                        "important_dialogues",
                        "cliffhangers_or_foreshadowing",
                        "coverage_warnings",
                    )
                }
            )
    except Exception as err:
        st.error(f"❌ 整片视频分析失败: {err}")
        logger.exception(f"整片视频分析失败\n{traceback.format_exc()}")
