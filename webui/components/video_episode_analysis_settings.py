"""整片视频单集剧情分析 UI（直接传视频）。"""

from __future__ import annotations

import json
import os

import streamlit as st

from app.services.documentary.documentary_material_resolver import (
    resolve_video_episode_analysis_path_for_documentary,
)
from app.config import config
from app.config.llm_model_presets import CUSTOM_MODEL_OPTION, VISION_MODEL_PRESETS, match_preset_index
from app.services.documentary.video_episode_analysis import (
    checkpoint_needs_resume,
    default_checkpoint_path,
    default_video_episode_analysis_path,
    load_video_episode_analysis_artifact,
    load_video_episode_checkpoint,
    parse_video_episode_analysis_payload,
    summarize_checkpoint_progress,
)
from app.services.documentary.video_episode_analysis import _probe_duration_seconds
from app.services.documentary.video_whole_grid_analysis import (
    WHOLE_GRID_DEFAULT_BATCH_COUNT,
    WHOLE_GRID_DEFAULT_INTERVAL,
    WHOLE_GRID_MAX_INTERVAL,
    WHOLE_GRID_MIN_INTERVAL,
    default_video_whole_grid_analysis_path,
    estimate_grid_run_plan,
    get_video_whole_grid_settings,
    load_video_whole_grid_artifact,
)
from webui.tools.analyze_video_episode_docu import analyze_video_episode_docu
from webui.tools.analyze_video_whole_grid_docu import analyze_video_whole_grid_docu


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
        "分析前先将原片压缩为 **720p** 母版，再按切镜逐镜截取上传并调用视觉模型。"
        "可在 config.toml `[video_episode_analysis]` 调整 `max_upload_mb`、`upload_transcode_profile`。"
        "人物命名复用上方「作品名称 / 头像参照」中勾选的人物头像。"
        "剧情参考亦在上方填写，抽帧与整片分析共用。"
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
    if checkpoint and checkpoint_needs_resume(checkpoint):
        summary = summarize_checkpoint_progress(
            checkpoint,
            int(checkpoint.get("total_chunks") or 0)
            or max(len(checkpoint.get("chunks_meta") or []), 1),
        )
        compressed = len(checkpoint.get("chunks_meta") or [])
        total = int(checkpoint.get("total_chunks") or 0) or summary["total"]
        st.warning(
            f"存在未完成的整片分析进度：已压缩 {compressed}/{total} 段，"
            f"已分析 {summary['completed']}/{total} 段，"
            f"失败 {summary['failed']} 段，待处理 {summary['pending']} 段。"
            " 请点击「补全未完成分析」续跑；「分析整片视频」将清除进度从头开始。"
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
            analyze_video_episode_docu(
                params,
                resume=False,
                output_path=active_path if active_path else default_path,
            )
    with col_resume:
        resume_disabled = not checkpoint_needs_resume(checkpoint)
        if st.button(
            "补全未完成分析",
            key="doc_resume_video_episode_btn",
            use_container_width=True,
            disabled=resume_disabled,
        ):
            analyze_video_episode_docu(
                params,
                resume=True,
                output_path=active_path if active_path else default_path,
            )

    st.divider()
    st.markdown("### 整片网格快扫（实验）")
    st.caption(
        "将整集视频压缩为 **单个文件一次上传**，默认 **20 秒/格、分 2 段 API** 解析全片。"
        "也可勾选「单次生成」改为 API 1 次。"
        "人物命名复用上方「作品名称 / 头像参照」勾选的人物头像（拼图对照识脸）。"
        "剧情参考建议只写本集前情 3–5 句，勿粘贴整份关系网 JSON。"
        "格距可选 5–30 秒；单次生成建议加大格距 + gemini-3-flash-preview。"
        "适合快速浏览时间轴；精细剧情仍建议用上方「分镜分析」或「抽帧分析」。"
    )

    grid_default_path = default_video_whole_grid_analysis_path(video_path)
    grid_active_path = (st.session_state.get("video_whole_grid_analysis_json_path") or "").strip()
    if not grid_active_path or not os.path.isfile(grid_active_path):
        grid_active_path = grid_default_path if os.path.isfile(grid_default_path) else ""

    if grid_active_path and os.path.isfile(grid_active_path):
        st.success(f"已有网格快扫结果: {grid_active_path}")
        try:
            grid_payload = load_video_whole_grid_artifact(grid_active_path)
            st.write(grid_payload.get("overall_summary") or "")
            st.caption(
                f"格距 {grid_payload.get('grid_interval_seconds')}s · "
                f"共 {grid_payload.get('grid_segment_count', len(grid_payload.get('grid_segments') or []))} 格 · "
                f"模型 {grid_payload.get('vision_model_name', '')}"
            )
        except Exception as err:
            st.caption(f"无法预览: {err}")
    else:
        st.caption(f"默认输出路径: {grid_default_path}")

    default_model = config.app.get("vision_openai_model_name") or ""
    preset_labels = [label for label, _ in VISION_MODEL_PRESETS]
    preset_models = [model_id for _, model_id in VISION_MODEL_PRESETS]
    preset_index = match_preset_index(VISION_MODEL_PRESETS, default_model)
    grid_model_label = st.selectbox(
        "网格快扫视觉模型",
        options=preset_labels,
        index=preset_index,
        key="doc_whole_grid_model_preset",
    )
    grid_model_index = preset_labels.index(grid_model_label)
    grid_model_id = preset_models[grid_model_index]
    if grid_model_id == CUSTOM_MODEL_OPTION:
        grid_model_id = st.text_input(
            "自定义视觉模型",
            value=default_model,
            key="doc_whole_grid_custom_model",
        ).strip()

    grid_interval = st.slider(
        "网格间隔（秒）",
        min_value=WHOLE_GRID_MIN_INTERVAL,
        max_value=WHOLE_GRID_MAX_INTERVAL,
        value=WHOLE_GRID_DEFAULT_INTERVAL,
        key="doc_whole_grid_interval",
    )

    grid_cfg = get_video_whole_grid_settings()
    one_shot_key = "doc_whole_grid_force_single_api"
    if one_shot_key not in st.session_state:
        st.session_state[one_shot_key] = bool(grid_cfg.get("force_one_shot", False))
    force_one_shot = st.checkbox(
        "单次生成（API 1 次）",
        key=one_shot_key,
        help=(
            f"勾选后整片一次 API 返回全部 grid_segments；"
            f"默认不勾选，按时间轴均分 {int(grid_cfg.get('batch_count') or WHOLE_GRID_DEFAULT_BATCH_COUNT)} 段"
        ),
    )

    try:
        video_duration = _probe_duration_seconds(video_path)
        run_plan = estimate_grid_run_plan(
            video_duration,
            grid_interval_seconds=grid_interval,
            model_name=grid_model_id or default_model,
            force_one_shot=force_one_shot,
        )
        duration_min = max(1, int(round(video_duration / 60)))
        interval_hint = f"{run_plan['grid_interval_effective']}s"
        if run_plan.get("grid_interval_auto_adjusted"):
            interval_hint = (
                f"{run_plan['grid_interval_requested']}s→{run_plan['grid_interval_effective']}s（自动加粗）"
            )
        if run_plan.get("one_shot"):
            api_hint = "API 1 次"
        else:
            batch_n = int(run_plan["api_call_count"])
            api_hint = f"API {batch_n} 批（均分 {batch_n} 段）"
        st.caption(
            f"预估：约 {duration_min} 分钟 → {run_plan['grid_segment_count']} 格 · "
            f"格距 {interval_hint} · {api_hint}"
        )
    except Exception:
        pass

    if st.button("整片网格快扫", key="doc_analyze_video_whole_grid_btn", use_container_width=True):
        analyze_video_whole_grid_docu(
            params,
            vision_model_name=grid_model_id,
            grid_interval_seconds=grid_interval,
            force_one_shot=force_one_shot,
            output_path=grid_active_path if grid_active_path else grid_default_path,
        )
