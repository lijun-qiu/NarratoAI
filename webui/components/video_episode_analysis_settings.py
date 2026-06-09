"""整片视频单集剧情分析 UI（直接传视频）。"""

from __future__ import annotations

import json
import os

import streamlit as st

from app.services.documentary.documentary_material_resolver import (
    resolve_video_episode_analysis_path_for_documentary,
)
from app.services.documentary.video_episode_analysis import (
    VideoEpisodeAnalysisService,
    default_checkpoint_path,
    default_video_episode_analysis_path,
    load_video_episode_analysis_artifact,
    load_video_episode_checkpoint,
    parse_video_episode_analysis_payload,
    summarize_checkpoint_progress,
)
from webui.tools.analyze_video_episode_docu import analyze_video_episode_docu


def sync_video_episode_analysis_with_video(video_path: str) -> None:
    """视频切换时自动配对已有整片视频分析 JSON（含素材来源视频回退）。"""
    video_path = (video_path or "").strip()
    if not video_path:
        return
    if st.session_state.get("_video_episode_analysis_synced_video_path") == video_path:
        return

    material_source = (st.session_state.get("doc_material_source_video_path") or "").strip()
    override = (st.session_state.get("video_episode_analysis_json_path") or "").strip()
    if override and os.path.isfile(override):
        try:
            load_video_episode_analysis_artifact(override)
            st.session_state["_video_episode_analysis_synced_video_path"] = video_path
            return
        except Exception:
            pass

    paired = resolve_video_episode_analysis_path_for_documentary(
        video_path,
        material_source_video_path=material_source,
        explicit_path=None,
    )
    if paired:
        st.session_state["video_episode_analysis_json_path"] = paired
    else:
        st.session_state["video_episode_analysis_json_path"] = None
    st.session_state["_video_episode_analysis_synced_video_path"] = video_path


def _active_video_episode_analysis_path(video_path: str) -> str:
    explicit = (st.session_state.get("video_episode_analysis_json_path") or "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    if not video_path:
        return ""
    material_source = (st.session_state.get("doc_material_source_video_path") or "").strip()
    return (
        resolve_video_episode_analysis_path_for_documentary(
            video_path,
            material_source_video_path=material_source,
            explicit_path=None,
        )
        or ""
    )


def render_video_episode_analysis_panel(tr, params) -> None:
    st.markdown("### 整片视频分析")
    st.caption(
        "直接将整集 mp4 传给视觉模型，输出 overall_summary / episodic_segments（含旁白 narration、环境描述 environment_description）"
        "/ important_dialogues 等 JSON。"
        "适合快速把握剧情；精细剪辑时间轴仍建议用「抽帧分析」。"
        "分析时会显示压缩、分段上传、模型调用与合并保存等步骤进度。"
        "人物命名会复用「抽帧分析」中上传的头像参照（请在抽帧分析面板勾选人物并上传头像）。"
    )

    video_path = (params.video_origin_path or "").strip()
    if not video_path:
        st.info("请先在左侧选择视频文件。")
        return
    if not os.path.isfile(video_path):
        st.warning(f"视频文件不存在: {video_path}")
        return

    sync_video_episode_analysis_with_video(video_path)
    default_path = default_video_episode_analysis_path(video_path)
    active_path = _active_video_episode_analysis_path(video_path) or default_path
    if active_path and os.path.isfile(active_path):
        st.session_state["video_episode_analysis_json_path"] = active_path
    checkpoint_path = default_checkpoint_path(active_path if active_path else default_path)
    checkpoint = load_video_episode_checkpoint(checkpoint_path)
    if checkpoint:
        summary = summarize_checkpoint_progress(
            checkpoint,
            int(checkpoint.get("total_chunks") or 0),
        )
        incomplete = summary["failed"] + summary["pending"]
        if incomplete > 0:
            st.warning(
                f"存在未完成的整片分析进度：已完成 {summary['completed']}/{summary['total']} 段，"
                f"失败 {summary['failed']} 段，待处理 {summary['pending']} 段。"
                " 可直接补全，无需从头重新生成。"
            )

    if os.path.isfile(active_path):
        st.success(f"已有分析结果: {active_path}")
        try:
            with open(active_path, encoding="utf-8") as fp:
                payload = json.load(fp)
            parsed = parse_video_episode_analysis_payload(payload)
            st.write(parsed.get("overall_summary") or "")
            status = str(payload.get("analysis_status") or "").strip()
            failed_chunks = payload.get("failed_chunk_indices") or []
            if status == "incomplete" and failed_chunks:
                st.caption(f"分析未完全完成，失败分段索引: {failed_chunks}")
        except Exception as err:
            st.caption(f"无法预览: {err}")
    else:
        st.caption(f"默认输出路径: {default_path}")

    col_analyze, col_resume = st.columns(2)
    with col_analyze:
        if st.button("分析整片视频", key="doc_analyze_video_episode_btn", use_container_width=True):
            analyze_video_episode_docu(params, resume=True)
    with col_resume:
        resume_disabled = not checkpoint or VideoEpisodeAnalysisService.count_incomplete_chunks(
            checkpoint,
            int(checkpoint.get("total_chunks") or 0) if checkpoint else 0,
        ) <= 0
        if st.button(
            "补全未完成分析",
            key="doc_resume_video_episode_btn",
            use_container_width=True,
            disabled=resume_disabled,
        ):
            analyze_video_episode_docu(params, resume=True)
