"""整片视频单集剧情分析 UI（直接传视频）。"""

from __future__ import annotations

import json
import os
import time

import streamlit as st

from app.config import config
from app.config.defaults import DEFAULT_VISION_OPENAI_MODEL_NAME
from app.config.llm_gateway_router import describe_llm_route
from app.config.llm_model_presets import VISION_MODEL_PRESETS, VISION_PRESET_MODEL_IDS
from app.services.documentary.frame_analysis_pairing import analysis_artifact_dir
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
<<<<<<< HEAD
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
=======
from app.services.documentary.video_episode_constants import (
    SEGMENT_POLICY_SCENE_CUT,
    SEGMENT_POLICY_TIME_CHUNK,
    resolve_segment_split_policy,
    resolve_upload_chunk_seconds,
)
from app.services.subtitle_video_pairing import resolve_subtitle_path_for_video
from webui.components.basic_settings import (
    _render_model_picker,
    normalize_openai_compatible_model_name,
)
from webui.components.subtitle_transcription_settings import render_documentary_subtitle_file_picker
from webui.tools.analyze_video_episode_docu import analyze_video_episode_docu
from webui.utils.documentary_file_picker import (
    apply_video_episode_analysis_path,
    clear_subtitle_path,
    clear_video_episode_analysis_path,
    queue_picker_paths,
    render_saved_file_picker,
)
>>>>>>> 5ff28823f54ba6da58cf214b2e2d57a09a1015df


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


def _import_video_episode_analysis_file(
    tr,
    analysis_file,
    *,
    path_input_key: str,
    pick_key: str,
) -> None:
    try:
        payload = json.loads(analysis_file.getvalue().decode("utf-8"))
        if not isinstance(payload, dict):
            st.error("无效的整片视频分析 JSON：根节点须为对象")
            st.stop()

        safe_filename = os.path.basename(analysis_file.name)
        analysis_dir = analysis_artifact_dir()
        os.makedirs(analysis_dir, exist_ok=True)
        target_path = os.path.join(analysis_dir, safe_filename)
        if os.path.exists(target_path):
            timestamp = time.strftime("%Y%m%d%H%M%S")
            name, ext = os.path.splitext(safe_filename)
            target_path = os.path.join(analysis_dir, f"{name}_{timestamp}{ext}")

        with open(target_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

        load_video_episode_analysis_artifact(target_path)
        apply_video_episode_analysis_path(target_path)
        queue_picker_paths(path_input_key, pick_key, target_path)
        st.success(f"整片视频分析已导入: {os.path.basename(target_path)}")
        st.rerun()
    except json.JSONDecodeError:
        st.error("无法解析 JSON 文件，请检查格式")
    except Exception as exc:
        st.error(f"{tr('Upload failed')}: {str(exc)}")


def render_documentary_video_episode_analysis_file_picker(
    tr,
    video_path: str = "",
    *,
    path_input_key: str = "doc_video_episode_path_input",
    pick_key: str = "doc_video_episode_saved_pick",
    confirm_button_key: str = "doc_confirm_video_episode_path",
    clear_button_key: str = "doc_clear_video_episode",
    import_key: str = "doc_video_episode_uploader",
) -> None:
    """从默认分析目录选用或导入整片视频分析 JSON。"""
    video_path = (video_path or st.session_state.get("video_origin_path") or "").strip()
    if "doc_video_episode_analysis_file_processed" not in st.session_state:
        st.session_state["doc_video_episode_analysis_file_processed"] = False

    analysis_dir = analysis_artifact_dir()
    material_source = (st.session_state.get("doc_material_source_video_path") or "").strip()
    paired_path = ""
    if video_path:
        paired_path = (
            resolve_video_episode_analysis_path_for_documentary(
                video_path,
                material_source_video_path=material_source,
                explicit_path=None,
            )
            or default_video_episode_analysis_path(video_path)
        )
        if paired_path and not os.path.isfile(paired_path):
            paired_path = ""

    active_path = (st.session_state.get("video_episode_analysis_json_path") or "").strip()
    if active_path and os.path.isfile(active_path):
        upload_hint = (
            "（已确认）" if st.session_state.get("doc_video_episode_analysis_file_processed") else ""
        )
        st.info(f"当前分析 JSON: {os.path.basename(active_path)}{upload_hint}")

    render_saved_file_picker(
        label="整片视频分析 JSON",
        directory=analysis_dir,
        glob_pattern="*_video_episode_analysis*.json",
        path_input_key=path_input_key,
        pick_key=pick_key,
        confirm_button_key=confirm_button_key,
        clear_button_key=clear_button_key,
        active_path=active_path,
        paired_path=paired_path,
        on_confirm=apply_video_episode_analysis_path,
        on_clear=lambda: clear_video_episode_analysis_path(
            path_input_key=path_input_key,
            pick_key=pick_key,
        ),
        import_label=tr("导入整片视频分析 JSON 到分析目录"),
        import_types=["json"],
        import_key=import_key,
        on_import=lambda uploaded: _import_video_episode_analysis_file(
            tr,
            uploaded,
            path_input_key=path_input_key,
            pick_key=pick_key,
        ),
    )


def render_video_episode_analysis_panel(tr, params) -> None:
    st.markdown("### 整片视频分析")
    st.caption(
        "直接将整集 mp4 传给视觉模型，输出 overall_summary / episodic_segments（含旁白 narration、环境描述 environment_description）"
        "/ important_dialogues 等 JSON。"
        "适合快速把握剧情；精细剪辑时间轴仍建议用「抽帧分析」。"
        "分析前先将原片压缩为 **720p** 母版，再按切镜逐镜截取上传并调用视觉模型。"
        "可在 config.toml `[video_episode_analysis]` 调整 `max_upload_mb`、`upload_transcode_profile`。"
        "人物命名复用上方「作品名称 / 头像参照」中勾选的人物头像"
        "（默认同抽帧：**2 张合 1 张**拼图 + 姓名标注，可在 config `frame_reference_collage_max_heads` 调整）。"
        "人物关系、剧情参考在上方分别填写；分析按切镜上传后，镜内再按 **5–10 秒** 窗口输出情节段。"
        "不再注入抽帧 timeline 参考。"
        "**不会**在此转录字幕；请自行准备 SRT 并在下方选用，分析完成后可选用来校正台词。"
    )

    video_path = (params.video_origin_path or "").strip()
    if not video_path:
        st.info("请先在左侧选择视频文件。")
        return
    if not os.path.isfile(video_path):
        st.warning(f"视频文件不存在: {video_path}")
        return

    from webui.components.script_settings import _sync_subtitle_with_video

    _sync_subtitle_with_video(video_path)
    sync_video_episode_analysis_with_video(video_path)

    st.markdown("#### 字幕 SRT（可选 · 自行准备）")
    st.caption(
        "选用你**事先转录好**的 SRT 文件即可；本步骤只读字幕、**不会调用 ASR 转写**。"
        "勾选下方选项后，分析完成会**参照 SRT 对齐时间并去重**，"
        "但 `important_dialogues` 的**字词以视频画面读到的为准**（不用 SRT 替换台词）。"
    )
    render_documentary_subtitle_file_picker(
        tr,
        path_input_key="doc_video_episode_subtitle_path_input",
        pick_key="doc_video_episode_subtitle_saved_pick",
        confirm_button_key="doc_confirm_video_episode_subtitle_path",
        clear_button_key="doc_clear_video_episode_subtitle",
        import_key="doc_video_episode_subtitle_uploader",
        on_clear=lambda: clear_subtitle_path(
            path_input_key="doc_video_episode_subtitle_path_input",
            pick_key="doc_video_episode_subtitle_saved_pick",
        ),
    )
    active_subtitle = (st.session_state.get("subtitle_path") or "").strip()
    paired_subtitle = (
        resolve_subtitle_path_for_video(video_path, explicit_path=None) if video_path else ""
    )
    has_subtitle = bool(
        (active_subtitle and os.path.isfile(active_subtitle))
        or (paired_subtitle and os.path.isfile(paired_subtitle))
    )
    if active_subtitle and os.path.isfile(active_subtitle):
        st.caption(f"已选用: `{active_subtitle}`")
    elif paired_subtitle:
        st.warning(
            f"检测到配对字幕 `{os.path.basename(paired_subtitle)}`，请在上方点 **确认选用**。"
        )
    else:
        st.caption("未选用字幕；可直接分析（台词来自视觉模型）。")

    if "doc_video_episode_align_dialogues_with_srt" not in st.session_state:
        st.session_state["doc_video_episode_align_dialogues_with_srt"] = has_subtitle
    st.checkbox(
        "分析完成后参照 SRT 对齐台词时间（字词仍以视频为准）",
        key="doc_video_episode_align_dialogues_with_srt",
        disabled=not has_subtitle,
        help="SRT 仅作时间轴参照与去重；quote 保留视觉模型从画面读到的原话。",
    )

    st.markdown("---")
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

    st.markdown("#### 上传分段")
    _split_labels = ("按时间切段（默认）", "按分镜/场景切")
    _split_values = (SEGMENT_POLICY_TIME_CHUNK, SEGMENT_POLICY_SCENE_CUT)
    default_policy = resolve_segment_split_policy()
    current_policy = st.session_state.get("doc_video_episode_split_policy", default_policy)
    try:
        split_index = _split_values.index(current_policy)
    except ValueError:
        split_index = 0
    split_label = st.radio(
        "上传给模型的视频如何切段",
        _split_labels,
        index=split_index,
        key="doc_video_episode_split_policy_label",
        horizontal=True,
        help="上传切段与 JSON 内 5–10 秒情节窗不同：前者决定每次上传多长的 mp4。",
    )
    st.session_state["doc_video_episode_split_policy"] = _split_values[
        _split_labels.index(split_label)
    ]
    default_chunk_minutes = resolve_upload_chunk_seconds() / 60.0
    chunk_minutes = st.number_input(
        "按时间切段：每段时长（分钟）",
        min_value=1.0,
        max_value=120.0,
        value=float(
            st.session_state.get("doc_video_episode_chunk_minutes", default_chunk_minutes)
        ),
        step=1.0,
        key="doc_video_episode_chunk_minutes",
        disabled=st.session_state["doc_video_episode_split_policy"] != SEGMENT_POLICY_TIME_CHUNK,
    )
    if st.session_state["doc_video_episode_split_policy"] == SEGMENT_POLICY_TIME_CHUNK:
        st.caption(
            f"将按 **每 {chunk_minutes:g} 分钟** 截一段上传；段内仍按 **5–10 秒** 输出 `episodic_segments`。"
        )
    else:
        st.caption("将检测场景/切镜后再上传；段内仍按 **5–10 秒** 输出情节窗。")

    default_vision_model = (
        config.app.get("vision_openai_model_name") or DEFAULT_VISION_OPENAI_MODEL_NAME
    )
    current_vision_model = (
        st.session_state.get("doc_video_episode_vision_model") or default_vision_model
    )
    vision_model_input = _render_model_picker(
        presets=VISION_MODEL_PRESETS,
        preset_model_ids=VISION_PRESET_MODEL_IDS,
        current_model=current_vision_model,
        default_model=default_vision_model,
        key_prefix="doc_video_episode",
        model_type_label="整片分析",
    )
    vision_model_name = normalize_openai_compatible_model_name(vision_model_input)
    if vision_model_name:
        st.session_state["doc_video_episode_vision_model"] = vision_model_name
        st.caption(f"实际连接：{describe_llm_route(vision_model_name, role='vision')}")

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

<<<<<<< HEAD
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
=======
    st.markdown("---")
    st.markdown("#### 剧情解剖")
    st.caption(
        "视频分析完成后，可选用「人物关系」「剧情参考」对照校正人名，"
        f"输出完善 JSON（模型：**deepseek-v4-pro**）。"
    )
    dis_cols = st.columns(2)
    with dis_cols[0]:
        use_rel = st.checkbox(
            "使用人物关系",
            value=True,
            key="doc_dissection_use_character_relationship",
        )
    with dis_cols[1]:
        use_plot = st.checkbox(
            "使用剧情参考",
            value=False,
            key="doc_dissection_use_plot_reference",
        )
    if st.button(
        "运行剧情解剖",
        key="doc_run_plot_dissection_btn",
        use_container_width=True,
        disabled=not os.path.isfile(active_path),
    ):
        from webui.tools.run_plot_dissection_docu import run_plot_dissection_docu

        run_plot_dissection_docu(
            params,
            analysis_path=active_path,
            use_character_relationship=use_rel,
            use_plot_reference=use_plot,
>>>>>>> 5ff28823f54ba6da58cf214b2e2d57a09a1015df
        )
