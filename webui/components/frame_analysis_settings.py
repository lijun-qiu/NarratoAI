"""逐帧解说 / 精剪：抽帧分析 UI（独立于脚本生成，对标字幕转录）。"""

from __future__ import annotations

import json
import os
import time

import streamlit as st

from app.config import config
from app.services.documentary.documentary_settings import get_documentary_compact_settings, get_documentary_settings
from app.services.documentary.documentary_material_resolver import (
    resolve_frame_analysis_path_for_documentary,
    resolve_subtitle_path_for_documentary,
)
from app.services.documentary.frame_analysis_pairing import (
    default_analysis_path_for_video,
    is_valid_analysis_artifact,
    load_analysis_artifact,
)
from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService
from app.services.documentary.vision_model_rotation import DEFAULT_VISION_FALLBACK_MODEL_NAMES
from app.services.documentary.subtitle_calibration_pipeline import (
    requires_subtitle_before_frame_analysis,
)
from app.services.subtitle_video_pairing import resolve_subtitle_path_for_video
from webui.tools.extract_frame_analysis_docu import extract_frame_analysis_docu
from webui.tools.retry_failed_frame_analysis_docu import retry_failed_frame_analysis_docu
from webui.components.documentary_output_split import render_output_split_control
from webui.utils.documentary_file_picker import (
    apply_frame_analysis_path,
    clear_frame_analysis_path,
    consume_pending_reuse_frame_analysis,
    queue_picker_paths,
    render_saved_file_picker,
)
from app.services.drama_character_registry import (
    DEFAULT_DRAMA_ID,
    ensure_head_img_dir,
    find_relationship_diagram_path,
    head_img_dir,
    find_head_image_path,
    head_pending_select_session_key,
    head_selection_session_key,
    head_upload_saved_sig_key,
    head_uploader_session_key,
    list_character_head_slot_groups,
    list_character_head_slots,
    list_dramas,
    list_unrecognized_head_images,
    resolve_character_references,
    resolve_relationship_diagram_path,
    resolve_active_relationship_diagram_path,
    head_selection_session_key,
    save_head_image,
    save_relationship_diagram,
)


def sync_frame_analysis_with_video(video_path: str) -> None:
    """视频切换时自动配对已有抽帧分析 JSON（含素材来源视频回退）。"""
    video_path = (video_path or "").strip()
    if not video_path:
        return
    if st.session_state.get("_frame_analysis_synced_video_path") == video_path:
        return

    material_source = (st.session_state.get("doc_material_source_video_path") or "").strip()
    override = (st.session_state.get("frame_analysis_json_path") or "").strip() or None
    upload_explicit = bool(st.session_state.get("doc_frame_analysis_upload_explicit"))
    if override and os.path.isfile(override):
        try:
            load_analysis_artifact(override)
            st.session_state["_frame_analysis_synced_video_path"] = video_path
            return
        except Exception:
            if upload_explicit:
                st.session_state["_frame_analysis_synced_video_path"] = video_path
                return

    if upload_explicit:
        st.session_state["_frame_analysis_synced_video_path"] = video_path
        return

    last_video = st.session_state.get("_frame_analysis_synced_video_path")
    if last_video and last_video != video_path:
        for key in (
            "doc_frame_analysis_path_input",
            "doc_frame_analysis_saved_pick",
            "doc_full_frame_analysis_path_input",
            "doc_full_frame_analysis_saved_pick",
            "doc_compact_frame_analysis_path_input",
            "doc_compact_frame_analysis_saved_pick",
            "sd_summary_frame_analysis_path_input",
            "sd_summary_frame_analysis_saved_pick",
        ):
            st.session_state.pop(key, None)
            st.session_state.pop(f"__pending__{key}", None)

    paired = resolve_frame_analysis_path_for_documentary(
        video_path,
        material_source_video_path=material_source,
        explicit_path=None,
        reuse=True,
    )
    if paired:
        st.session_state["frame_analysis_json_path"] = paired
    else:
        st.session_state["frame_analysis_json_path"] = None
    st.session_state["_frame_analysis_synced_video_path"] = video_path
    st.session_state["doc_frame_analysis_file_processed"] = False


def _session_subtitle_path(video_path: str) -> str:
    material_source = (st.session_state.get("doc_material_source_video_path") or "").strip()
    return resolve_subtitle_path_for_documentary(
        video_path,
        material_source_video_path=material_source,
        explicit_path=st.session_state.get("subtitle_path"),
    )


def _active_frame_analysis_path(video_path: str) -> str:
    explicit = (st.session_state.get("frame_analysis_json_path") or "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    if not video_path:
        return ""
    material_source = (st.session_state.get("doc_material_source_video_path") or "").strip()
    return resolve_frame_analysis_path_for_documentary(
        video_path,
        material_source_video_path=material_source,
        explicit_path=None,
        reuse=True,
    ) or ""


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / 1024 / 1024:.2f} MB"
    return f"{num_bytes / 1024:.1f} KB"


def _render_compact_export_section(active_path: str) -> None:
    """导出体积更小的精简版抽帧分析 JSON。"""
    if not active_path or not os.path.isfile(active_path):
        return

    try:
        artifact = load_analysis_artifact(active_path)
    except Exception:
        return

    original_bytes = os.path.getsize(active_path)
    with st.expander("导出精简 JSON（减小体积）", expanded=False):
        st.caption(
            f"当前文件 {_format_bytes(original_bytes)}。"
            "精简版会去掉 raw_response、关键帧路径、批次内重复数据，保留脚本生成与字幕校准所需内容。"
        )
        preset_labels = {
            "minimal_scene": "极精简（六核心字段 + subtitle_entries + subtitle，无 batches）",
            "minimal": "最小（scene_segments 全字段，适合脚本生成）",
            "script": "脚本（场景片段 + 批次摘要 + 批次索引）",
            "calibration": "校准（含逐帧观察，适合字幕校准）",
        }
        preset_options = {
            "minimal_scene": {"minimal_scene_only": True},
            "minimal": {
                "include_frame_observations": False,
                "include_summaries": False,
                "include_batch_index": False,
                "keep_batch_meta": False,
            },
            "script": {
                "include_frame_observations": False,
                "include_summaries": True,
                "include_batch_index": True,
                "keep_batch_meta": True,
            },
            "calibration": {
                "include_frame_observations": True,
                "include_summaries": True,
                "include_batch_index": True,
                "keep_batch_meta": True,
            },
        }
        try:
            size_map = DocumentaryFrameExtractionService.estimate_compact_analysis_sizes(
                artifact,
                source_bytes=original_bytes,
            )
        except Exception:
            size_map = {"original": original_bytes}

        preset = st.radio(
            "精简档位",
            options=list(preset_labels.keys()),
            index=0,
            format_func=lambda key: (
                f"{preset_labels[key]} · 约 {_format_bytes(size_map.get(key, 0))}"
                f"（减少 {round(100 * (1 - size_map.get(key, original_bytes) / original_bytes), 1)}%）"
                if original_bytes
                else preset_labels[key]
            ),
            key="doc_frame_analysis_compact_preset",
        )
        st.caption(
            "精简版另存为新文件，不会替换当前使用的完整 JSON。"
            "完整版保留重跑失败批次、硬字幕 OCR 等能力。"
        )
        st.caption("精简版不支持「重跑失败批次」与硬字幕 OCR（需原文件中的关键帧路径）。")
        if st.button("生成精简 JSON", key="doc_export_compact_frame_analysis_btn", use_container_width=True):
            try:
                result = DocumentaryFrameExtractionService.save_compact_analysis_artifact(
                    active_path,
                    **preset_options[preset],
                )
                output_path = result["output_path"]
                st.success(
                    f"已另存精简版 `{os.path.basename(output_path)}`："
                    f"{_format_bytes(result['original_bytes'])} → "
                    f"{_format_bytes(result['compact_bytes'])}（减少 {result['reduction_percent']}%）。"
                    f"当前仍使用原文件 `{os.path.basename(active_path)}`。"
                )
            except Exception as exc:
                st.error(f"导出精简 JSON 失败: {exc}")


def _render_failed_batches_detail(artifact: dict) -> int:
    """展开显示失败批次明细。"""
    failed = DocumentaryFrameExtractionService.list_failed_batch_details(artifact)
    if not failed:
        return 0

    with st.expander(f"失败批次详情（{len(failed)} 个）", expanded=True):
        st.caption("以下为视觉模型未成功解析的批次；可用「重跑失败批次」单独补跑。")
        summary_rows = [
            {
                "批次": item["batch_index"],
                "时间范围": item["time_range"],
                "关键帧": f"{item['frames_on_disk']}/{item['frame_count']}",
                "缺失帧": item["frames_missing"],
                "已有场景": "是" if item["has_scene_segments"] else "否",
                "已有逐帧": "是" if item["has_frame_observations"] else "否",
            }
            for item in failed
        ]
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)

        for item in failed:
            st.markdown(f"#### 批次 #{item['batch_index']} · `{item['time_range']}`")
            st.markdown(f"**错误原因：** {item['error_message']}")
            meta_cols = st.columns(3)
            with meta_cols[0]:
                st.caption(f"关键帧: {item['frames_on_disk']}/{item['frame_count']} 在磁盘")
            with meta_cols[1]:
                st.caption(f"原始响应: {item['raw_response_chars']} 字符")
            with meta_cols[2]:
                st.caption(
                    "结构化数据: "
                    f"scene={item['has_scene_segments']} / frame_obs={item['has_frame_observations']}"
                )
            if item["raw_response_preview"]:
                with st.expander("原始响应片段", expanded=False):
                    st.code(item["raw_response_preview"], language="json")
            st.divider()

    return len(failed)


def _render_frame_analysis_status(video_path: str) -> None:
    """展示当前将用于脚本生成的抽帧分析 JSON。"""
    active_path = _active_frame_analysis_path(video_path)
    upload_explicit = bool(st.session_state.get("doc_frame_analysis_upload_explicit"))

    if not active_path:
        st.warning("抽帧分析：请上传 JSON，或在下方展开面板执行「抽帧并分析」。")
        return

    label = os.path.basename(active_path)
    detail = ""
    failed_count = 0
    artifact = None
    try:
        artifact = load_analysis_artifact(active_path)
        batch_count = len(artifact.get("batches") or [])
        scene_count = len(artifact.get("scene_segments") or [])
        frame_count = len(artifact.get("frame_observations") or [])
        failed_count = DocumentaryFrameExtractionService.count_failed_batches(artifact)
        success_count = batch_count - failed_count
        models_used = artifact.get("vision_models_used") or []
        model_hint = ""
        if models_used:
            model_hint = f" · 模型: {', '.join(str(m) for m in models_used)}"
        if scene_count:
            detail = f"（{scene_count} 场景片段 / {batch_count} 批次，成功 {success_count}"
            if failed_count:
                detail += f"，失败 {failed_count}"
            detail += f"{model_hint}）"
        else:
            detail = f"（{batch_count} 批次 / {frame_count} 帧，成功 {success_count}"
            if failed_count:
                detail += f"，失败 {failed_count}"
            detail += f"{model_hint}）"
    except Exception:
        artifact = None

    source_hint = ""
    if upload_explicit:
        source_hint = "（已上传，生成脚本优先使用此文件）"
    else:
        material_source = (st.session_state.get("doc_material_source_video_path") or "").strip()
        if material_source and material_source != video_path:
            source_hint = f"（素材来源: {os.path.basename(material_source)}）"
    st.success(f"将用于脚本生成: **{label}**{detail}{source_hint}")
    if artifact and failed_count > 0:
        st.warning(f"有 **{failed_count}** 个批次分析失败，可点击下方「重跑失败批次」补全（无需整片重跑）")
        _render_failed_batches_detail(artifact)
    elif artifact:
        last_retry = str(artifact.get("last_retry_at") or "").strip()
        if last_retry:
            recovered = artifact.get("last_retry_recovered")
            still = artifact.get("last_retry_still_failed")
            st.caption(
                f"最近一次重跑: {last_retry}"
                + (f"（成功 {recovered}，仍失败 {still}）" if recovered is not None else "")
            )
    if active_path:
        _render_compact_export_section(active_path)


def _import_frame_analysis_file(
    tr,
    analysis_file,
    *,
    path_input_key: str = "doc_frame_analysis_path_input",
    pick_key: str = "doc_frame_analysis_saved_pick",
) -> None:
    try:
        payload = json.loads(analysis_file.getvalue().decode("utf-8"))
        if not is_valid_analysis_artifact(payload):
            st.error("无效的抽帧分析 JSON：缺少 batches 或 frame_observations 字段")
            st.stop()

        safe_filename = os.path.basename(analysis_file.name)
        analysis_dir = DocumentaryFrameExtractionService.analysis_artifact_dir()
        os.makedirs(analysis_dir, exist_ok=True)
        target_path = os.path.join(analysis_dir, safe_filename)
        if os.path.exists(target_path):
            timestamp = time.strftime("%Y%m%d%H%M%S")
            name, ext = os.path.splitext(safe_filename)
            target_path = os.path.join(analysis_dir, f"{name}_{timestamp}{ext}")

        with open(target_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

        apply_frame_analysis_path(target_path)
        queue_picker_paths(path_input_key, pick_key, target_path)
        st.success(f"抽帧分析已导入: {os.path.basename(target_path)}")
        st.rerun()
    except json.JSONDecodeError:
        st.error("无法解析 JSON 文件，请检查格式")
    except Exception as exc:
        st.error(f"{tr('Upload failed')}: {str(exc)}")


def render_documentary_frame_analysis_file_picker(
    tr,
    video_path: str = "",
    *,
    path_input_key: str = "doc_frame_analysis_path_input",
    pick_key: str = "doc_frame_analysis_saved_pick",
    confirm_button_key: str = "doc_confirm_frame_analysis_path",
    clear_button_key: str = "doc_clear_frame_analysis",
    import_key: str = "docu_frame_analysis_uploader",
) -> None:
    """从默认分析目录选用或导入抽帧 JSON。"""
    video_path = (video_path or st.session_state.get("video_origin_path") or "").strip()
    if "doc_frame_analysis_file_processed" not in st.session_state:
        st.session_state["doc_frame_analysis_file_processed"] = False

    analysis_dir = DocumentaryFrameExtractionService.analysis_artifact_dir()
    material_source = (st.session_state.get("doc_material_source_video_path") or "").strip()
    paired_path = ""
    if video_path:
        paired_path = default_analysis_path_for_video(video_path)
        if not os.path.isfile(paired_path):
            paired_path = resolve_frame_analysis_path_for_documentary(
                video_path,
                material_source_video_path=material_source,
                explicit_path=None,
                reuse=True,
            ) or ""
    active_path = (st.session_state.get("frame_analysis_json_path") or "").strip()

    render_saved_file_picker(
        label="抽帧分析 JSON",
        directory=analysis_dir,
        glob_pattern="*_frame_analysis*.json",
        path_input_key=path_input_key,
        pick_key=pick_key,
        confirm_button_key=confirm_button_key,
        clear_button_key=clear_button_key,
        active_path=active_path,
        paired_path=paired_path,
        on_confirm=apply_frame_analysis_path,
        on_clear=lambda: clear_frame_analysis_path(
            path_input_key=path_input_key,
            pick_key=pick_key,
        ),
        import_label="导入新 JSON 到分析目录",
        import_types=["json"],
        import_key=import_key,
        on_import=lambda uploaded: _import_frame_analysis_file(
            tr,
            uploaded,
            path_input_key=path_input_key,
            pick_key=pick_key,
        ),
    )


def _render_frame_analysis_upload(tr, video_path: str) -> None:
    render_documentary_frame_analysis_file_picker(tr, video_path)


def _render_single_head_upload_slot(
    drama_id: str,
    slot: dict,
    selected_names: list[str],
) -> None:
    name = str(slot["name"])
    select_key = head_selection_session_key(drama_id, name)
    uploader_key = head_uploader_session_key(drama_id, name)
    saved_sig_key = head_upload_saved_sig_key(drama_id, name)
    pending_select_key = head_pending_select_session_key(drama_id, name)
    role_hint = str(slot.get("role_hint") or "").strip()
    if role_hint:
        st.markdown(f"**{name}** · _{role_hint}_")
    else:
        st.markdown(f"**{name}**")
    uploaded_file = st.file_uploader(
        "上传头像",
        type=["jpg", "jpeg", "png", "webp"],
        key=uploader_key,
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        upload_sig = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get(saved_sig_key) != upload_sig:
            try:
                file_bytes = uploaded_file.getvalue()
                if not file_bytes:
                    st.warning("读取图片失败，请重新选择文件")
                else:
                    saved_path = save_head_image(
                        drama_id,
                        name,
                        file_bytes,
                        original_filename=uploaded_file.name,
                    )
                    st.session_state[saved_sig_key] = upload_sig
                    st.session_state[pending_select_key] = True
                    st.success(f"已保存: {os.path.basename(saved_path)}")
                    st.rerun()
            except Exception as exc:
                st.error(f"保存头像失败: {exc}")

    image_path = find_head_image_path(drama_id, name)
    has_image = bool(image_path and os.path.isfile(image_path))
    if has_image:
        st.image(image_path, width=96)
        if pending_select_key in st.session_state:
            st.session_state[select_key] = True
            del st.session_state[pending_select_key]
        elif select_key not in st.session_state:
            st.session_state[select_key] = True
        if st.checkbox("用于抽帧", key=select_key):
            selected_names.append(name)
    else:
        st.caption("未上传")


def _render_head_upload_slot_grid(
    drama_id: str,
    slots: list[dict],
    selected_names: list[str],
    *,
    columns_per_row: int = 3,
) -> None:
    for row_start in range(0, len(slots), columns_per_row):
        cols = st.columns(columns_per_row)
        for col_index, slot in enumerate(slots[row_start : row_start + columns_per_row]):
            with cols[col_index]:
                _render_single_head_upload_slot(drama_id, slot, selected_names)


def _render_drama_character_settings() -> str:
    """剧名选择与人物头像上传（供抽帧人名匹配）。"""
    dramas = list_dramas()
    drama_ids = [item["id"] for item in dramas] or [DEFAULT_DRAMA_ID]
    default_index = drama_ids.index(DEFAULT_DRAMA_ID) if DEFAULT_DRAMA_ID in drama_ids else 0

    if "doc_frame_drama_id" not in st.session_state:
        st.session_state["doc_frame_drama_id"] = DEFAULT_DRAMA_ID

    drama_id = st.selectbox(
        "剧名",
        options=drama_ids,
        index=drama_ids.index(st.session_state["doc_frame_drama_id"])
        if st.session_state.get("doc_frame_drama_id") in drama_ids
        else default_index,
        format_func=lambda value: next(
            (item["label"] for item in dramas if item["id"] == value),
            value,
        ),
        key="doc_frame_drama_id",
        help="选择剧名后可上传关系图/头像；抽帧时是否使用由下方勾选决定",
    )
    st.session_state["video_theme"] = drama_id

    opt_cols = st.columns(2)
    with opt_cols[0]:
        st.checkbox(
            "抽帧时使用文字关系表",
            value=False,
            key="doc_frame_enable_drama_knowledge_text",
            help="勾选后每批注入该剧 Markdown 人物关系对照（约 2700 字）",
        )
    with opt_cols[1]:
        st.checkbox(
            "抽帧时使用关系图",
            value=False,
            key="doc_frame_enable_relationship_diagram",
            help="勾选且已上传关系图时，每批将关系图作为图 #1 发送给视觉模型",
        )

    st.checkbox(
        "参照图省 token（推荐：仅首批发送 + 缩小 + 多头像合成一张）",
        value=True,
        key="doc_frame_reference_token_saver",
        help="关闭后每批都会重复发送全部参照图（5 张头像 × 20 批 = 100 次传图，token 更高）",
    )

    relationship_path = find_relationship_diagram_path(drama_id)
    st.markdown("**① 人物关系图**")
    rel_cols = st.columns([1, 2])
    with rel_cols[0]:
        if relationship_path and os.path.isfile(relationship_path):
            st.image(relationship_path, caption="当前关系图", use_container_width=True)
        else:
            st.caption("未上传 · 勾选「使用关系图」且上传后才会参与抽帧")
    with rel_cols[1]:
        st.caption(
            f"保存为 `{os.path.join(head_img_dir(drama_id), '_relationship.png/jpg')}` · "
            "仅上传不勾选不会消耗 token"
        )
        rel_uploaded = st.file_uploader(
            "上传人物关系图",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"doc_relationship_diagram_{drama_id}",
            label_visibility="collapsed",
        )
        if rel_uploaded is not None:
            saved_rel = save_relationship_diagram(
                drama_id,
                rel_uploaded.getvalue(),
                original_filename=rel_uploaded.name,
            )
            st.success(f"关系图已保存: {os.path.basename(saved_rel)}")
            st.rerun()

    st.markdown("**② 人物头像（按频率分级名单）**")
    slots = list_character_head_slots(drama_id)
    slot_groups = list_character_head_slot_groups(drama_id)
    uploaded_count = sum(1 for slot in slots if slot.get("uploaded"))
    head_dir = ensure_head_img_dir(drama_id)
    selected_names: list[str] = []
    st.caption(
        f"人物头像目录：`{head_dir}`（已上传 {uploaded_count}/{len(slots)} 人）"
        " · 请用下方上传框保存，文件会自动命名为「人物名.jpg/png」"
        " · 已上传默认勾选用于抽帧"
    )
    orphan_files = list_unrecognized_head_images(drama_id)
    if orphan_files:
        preview = "、".join(orphan_files[:5])
        suffix = f" 等 {len(orphan_files)} 个" if len(orphan_files) > 5 else ""
        st.warning(
            f"目录中有 **{len(orphan_files)}** 个未识别文件（{preview}{suffix}）。"
            " 这些文件名不是人物名，界面不会显示为已上传；请用对应人物的上传框重新上传。"
        )

    with st.expander("人物头像上传", expanded=uploaded_count == 0):
        st.caption(
            "上传正面/半身照后默认勾选参与抽帧；取消勾选则该人物头像不发送给视觉模型。"
            " 建议先完成「高频」分组，再按需补充中等/低频角色。"
            " 请勿手动把截图丢进文件夹（需按人物名命名才能被识别）。"
        )
        if not slots:
            st.info("当前剧名暂无上传名单，无法展示槽位。")
            st.session_state["doc_frame_selected_character_names"] = []
            return drama_id

        for group in slot_groups:
            label = str(group.get("label") or "人物")
            group_slots = list(group.get("slots") or [])
            group_uploaded = sum(1 for slot in group_slots if slot.get("uploaded"))
            st.markdown(f"**{label}**（已上传 {group_uploaded}/{len(group_slots)}）")
            _render_head_upload_slot_grid(drama_id, group_slots, selected_names)
            st.markdown("")

    st.session_state["doc_frame_selected_character_names"] = selected_names
    return drama_id


def _render_frame_analysis_controls(
    tr,
    params,
    *,
    compact: bool,
    video_path: str,
    require_subtitle: bool,
    subtitle_path: str,
    subtitle_ready: bool,
    show_output_split: bool = True,
) -> None:
    """抽帧参数与执行按钮（独立面板或 expander 内共用）。"""
    interval_key = "frame_interval_input_compact" if compact else "frame_interval_input_full"
    default_interval = config.frames.get("frame_interval_input", 3)
    if interval_key not in st.session_state:
        st.session_state[interval_key] = default_interval

    if require_subtitle:
        st.caption(
            "已开启抽帧后自动校准（config: auto_subtitle_calibration_on_frame_analysis）。"
            "OCR / LLM 校正请在「校准字幕」Tab 手动执行。"
        )
    else:
        st.caption("抽帧并调用视觉模型写出分析 JSON；生成脚本时将复用该文件。")

    drama_id = _render_drama_character_settings()
    selected_names = set(st.session_state.get("doc_frame_selected_character_names") or [])
    st.session_state["doc_frame_character_references"] = resolve_character_references(
        drama_id,
        selected_names=selected_names,
    )
    st.session_state["doc_frame_relationship_diagram_path"] = resolve_relationship_diagram_path(drama_id)
    st.session_state["doc_frame_active_relationship_diagram_path"] = resolve_active_relationship_diagram_path(
        drama_id,
        enabled=bool(st.session_state.get("doc_frame_enable_relationship_diagram")),
    )

    if require_subtitle:
        if subtitle_ready:
            st.success(f"字幕已就绪: **{os.path.basename(subtitle_path)}**，可开始抽帧")
        else:
            st.warning("请先在「字幕转录」Tab 完成转写或上传字幕，再点击「抽帧并分析」")

    input_cols = st.columns(2)
    with input_cols[0]:
        st.number_input(
            tr("Frame Interval (seconds)"),
            min_value=0.0,
            value=float(st.session_state.get(interval_key, default_interval)),
            help=tr("Frame Interval (seconds) (More keyframes consume more tokens)"),
            key=interval_key,
        )
    with input_cols[1]:
        st.number_input(
            tr("Batch Size"),
            min_value=0,
            value=st.session_state.get("vision_batch_size", config.frames.get("vision_batch_size", 10)),
            help=tr("Batch Size (More keyframes consume more tokens)"),
            key="vision_batch_size",
        )

    fallback_default = (
        config.frames.get("vision_fallback_model_names")
        or DEFAULT_VISION_FALLBACK_MODEL_NAMES
    )
    st.text_input(
        "备用视觉模型（额度用尽时自动切换）",
        value=st.session_state.get("vision_fallback_model_names", fallback_default),
        key="vision_fallback_model_names",
        help="逗号分隔；主模型额度/限流不可用时按顺序切换，无需手动重跑",
    )
    config.frames["vision_fallback_model_names"] = (
        st.session_state.get("vision_fallback_model_names") or fallback_default
    ).strip()

    st.session_state["frame_interval_input"] = st.session_state.get(
        interval_key,
        default_interval,
    )

    if "doc_material_source_video_path" not in st.session_state:
        st.session_state["doc_material_source_video_path"] = video_path or ""
    st.text_input(
        "抽帧/字幕来源视频（可选）",
        key="doc_material_source_video_path",
        help="对有硬字幕的素材抽帧、OCR 后，成片可换无字幕视频；填抽帧时用的视频完整路径",
    )

    consume_pending_reuse_frame_analysis()
    st.checkbox(
        "复用已有抽帧分析（生成脚本时跳过视觉模型）",
        value=bool(st.session_state.get("doc_reuse_frame_analysis", True)),
        key="doc_reuse_frame_analysis",
        help="已上传 JSON 时始终优先使用上传文件；未上传时按视频名或素材来源配对",
    )

    default_path = ""
    if video_path:
        default_path = default_analysis_path_for_video(video_path)
        st.caption(f"未上传时默认可复用路径: `{default_path}`")

    active_analysis_path = _active_frame_analysis_path(video_path)
    failed_batch_count = 0
    if active_analysis_path and os.path.isfile(active_analysis_path):
        try:
            failed_batch_count = DocumentaryFrameExtractionService.count_failed_batches(
                load_analysis_artifact(active_analysis_path)
            )
        except Exception:
            failed_batch_count = 0

    can_extract = bool(
        video_path
        and os.path.isfile(video_path)
        and (not require_subtitle or subtitle_ready)
    )
    if show_output_split:
        render_output_split_control(key="doc_output_split_parts")

    test_duration_key = "doc_frame_test_duration_seconds"
    if test_duration_key not in st.session_state:
        st.session_state[test_duration_key] = 5.0
    test_cols = st.columns([2, 1])
    with test_cols[0]:
        st.number_input(
            "测试时长（秒）",
            min_value=1.0,
            max_value=120.0,
            step=1.0,
            key=test_duration_key,
            help="测试抽帧仅处理片头指定秒数，用于快速验证参数/头像/关系表，节省 token",
        )
    with test_cols[1]:
        st.caption(" ")
        st.caption(" ")
        if st.button(
            "测试抽帧（仅前 N 秒）",
            key="doc_extract_frame_analysis_test_btn",
            use_container_width=True,
            disabled=not can_extract,
            help="结果保存为 *_frame_analysis_test_5s.json，不覆盖完整版 JSON",
        ):
            extract_frame_analysis_docu(
                params,
                compact=compact,
                test_mode=True,
                test_duration_seconds=float(st.session_state.get(test_duration_key, 5.0)),
            )

    action_cols = st.columns(2)
    with action_cols[0]:
        if video_path and os.path.isfile(video_path):
            if st.button(
                "抽帧并分析",
                key="doc_extract_frame_analysis_btn",
                use_container_width=True,
                disabled=not can_extract,
            ):
                extract_frame_analysis_docu(params, compact=compact)
        else:
            st.warning("请先在上方选择或上传视频文件，再进行抽帧分析")
    with action_cols[1]:
        can_retry = bool(active_analysis_path and failed_batch_count > 0)
        if st.button(
            f"重跑失败批次（{failed_batch_count}）" if failed_batch_count else "重跑失败批次",
            key="doc_retry_failed_frame_batches_btn",
            use_container_width=True,
            disabled=not can_retry,
            help="仅对 JSON 中 status=failed 的批次重新调用视觉模型，结果写回原文件",
        ):
            retry_failed_frame_analysis_docu(
                params,
                compact=compact,
                analysis_json_path=active_analysis_path,
            )


def render_frame_analysis_panel(
    tr,
    params,
    *,
    compact: bool = False,
    standalone: bool = False,
    show_output_split: bool = True,
) -> None:
    """抽帧分析：复用/上传/独立抽帧按钮。standalone=True 时作为独立 Tab 展示。"""
    video_path = (st.session_state.get("video_origin_path") or "").strip()
    if video_path:
        sync_frame_analysis_with_video(video_path)

    doc_settings = get_documentary_compact_settings() if compact else get_documentary_settings()
    require_subtitle = requires_subtitle_before_frame_analysis(doc_settings)
    subtitle_path = _session_subtitle_path(video_path) if video_path else ""
    subtitle_ready = bool(subtitle_path)

    _render_frame_analysis_status(video_path)
    render_documentary_frame_analysis_file_picker(tr, video_path)

    if standalone:
        st.divider()
        _render_frame_analysis_controls(
            tr,
            params,
            compact=compact,
            video_path=video_path,
            require_subtitle=require_subtitle,
            subtitle_path=subtitle_path,
            subtitle_ready=subtitle_ready,
            show_output_split=show_output_split,
        )
    else:
        with st.expander("抽帧分析（ffmpeg 抽帧 + 视觉模型）", expanded=False):
            _render_frame_analysis_controls(
                tr,
                params,
                compact=compact,
                video_path=video_path,
                require_subtitle=require_subtitle,
                subtitle_path=subtitle_path,
                subtitle_ready=subtitle_ready,
                show_output_split=show_output_split,
            )
