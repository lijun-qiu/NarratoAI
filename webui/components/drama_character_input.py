"""作品名称与人物头像参照（抽帧分析 / 整片视频分析共用）。"""

from __future__ import annotations

import os

import streamlit as st

from app.services.drama_character_registry import (
    DEFAULT_DRAMA_ID,
    ensure_head_img_dir,
    find_head_image_path,
    find_relationship_diagram_path,
    head_img_dir,
    head_pending_select_session_key,
    head_selection_session_key,
    head_upload_saved_sig_key,
    head_uploader_session_key,
    list_character_head_slot_groups,
    list_character_head_slots,
    list_drama_select_options,
    list_unrecognized_head_images,
    resolve_active_relationship_diagram_path,
    resolve_character_references,
    resolve_relationship_diagram_path,
    save_head_image,
    save_relationship_diagram,
)

DRAMA_ID_SESSION_KEY = "doc_frame_drama_id"


def get_drama_id() -> str:
    return str(st.session_state.get(DRAMA_ID_SESSION_KEY) or "").strip()


def sync_drama_character_session_state(drama_id: str | None = None) -> str:
    """根据当前勾选状态解析头像参照路径，供各分析工具读取。"""
    resolved_id = (drama_id if drama_id is not None else get_drama_id()).strip()
    selected_names = set(st.session_state.get("doc_frame_selected_character_names") or [])
    st.session_state["doc_frame_character_references"] = resolve_character_references(
        resolved_id,
        selected_names=selected_names,
    )
    st.session_state["doc_frame_relationship_diagram_path"] = resolve_relationship_diagram_path(resolved_id)
    st.session_state["doc_frame_active_relationship_diagram_path"] = resolve_active_relationship_diagram_path(
        resolved_id,
        enabled=bool(st.session_state.get("doc_frame_enable_relationship_diagram")),
    )
    return resolved_id


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
        if st.checkbox("用于分析", key=select_key, help="勾选后头像将发送给抽帧与整片视频分析的视觉模型"):
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


def _drama_option_label(drama_id: str, labels: dict[str, str]) -> str:
    return labels.get(drama_id, drama_id) if drama_id else labels.get("", "（请选择作品）")


def render_drama_character_input() -> str:
    """渲染作品名称、关系图与人物头像，返回当前作品名。"""
    if DRAMA_ID_SESSION_KEY not in st.session_state:
        st.session_state[DRAMA_ID_SESSION_KEY] = DEFAULT_DRAMA_ID

    select_options = list_drama_select_options()
    option_ids = [str(item["id"]) for item in select_options]
    option_labels = {str(item["id"]): str(item["label"]) for item in select_options}
    if st.session_state.get(DRAMA_ID_SESSION_KEY) not in option_ids:
        st.session_state[DRAMA_ID_SESSION_KEY] = DEFAULT_DRAMA_ID

    drama_id = st.selectbox(
        "作品名称",
        options=option_ids,
        format_func=lambda value: _drama_option_label(value, option_labels),
        key=DRAMA_ID_SESSION_KEY,
        help="用于 headImg 子目录与视频主题；抽帧与整片视频分析共用",
    ).strip()
    st.session_state["video_theme"] = drama_id

    if not drama_id:
        st.info("请先选择作品，再上传关系图与人物头像。")
        st.session_state["doc_frame_selected_character_names"] = []
        return sync_drama_character_session_state("")

    opt_cols = st.columns(2)
    with opt_cols[0]:
        st.checkbox(
            "抽帧时使用文字关系表",
            value=False,
            key="doc_frame_enable_drama_knowledge_text",
            help="勾选后每批注入作品目录下 relationships.md 人物关系对照（须自行放置）",
        )
    with opt_cols[1]:
        st.checkbox(
            "分析时使用关系图",
            value=False,
            key="doc_frame_enable_relationship_diagram",
            help="勾选且已上传关系图时，每批将关系图作为图 #1 发送给视觉模型",
        )

    st.checkbox(
        "参照图省 token（推荐：仅首批发送 + 缩小 + 多头像合成一张）",
        value=True,
        key="doc_frame_reference_token_saver",
        help="关闭后每批都会重复发送全部参照图，token 消耗更高",
    )

    relationship_path = find_relationship_diagram_path(drama_id)
    st.markdown("**① 人物关系图**")
    rel_cols = st.columns([1, 2])
    with rel_cols[0]:
        if relationship_path and os.path.isfile(relationship_path):
            st.image(relationship_path, caption="当前关系图", use_container_width=True)
        else:
            st.caption("未上传 · 勾选「使用关系图」且上传后才会参与分析")
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

    st.markdown("**② 人物头像**")
    slots = list_character_head_slots(drama_id)
    slot_groups = list_character_head_slot_groups(drama_id)
    uploaded_count = sum(1 for slot in slots if slot.get("uploaded"))
    head_dir = ensure_head_img_dir(drama_id)
    selected_names: list[str] = []
    st.caption(
        f"人物头像目录：`{head_dir}`（已上传 {uploaded_count}/{len(slots)} 人）"
        " · 请用下方上传框保存，文件会自动命名为「人物名.jpg/png」"
        " · 已上传默认勾选用于分析"
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
            "上传正面/半身照后默认勾选参与分析；取消勾选则该人物头像不发送给视觉模型。"
            " 建议先完成「高频」分组，再按需补充中等/低频角色。"
            " 请勿手动把截图丢进文件夹（需按人物名命名才能被识别）。"
        )
        if not slots:
            st.info(
                "当前作品暂无人物槽位。可将头像命名为「人物名.jpg」放入上方目录，"
                "或在下方输入人物姓名后上传。"
            )
            new_name = st.text_input(
                "添加人物姓名",
                key=f"doc_frame_new_character_{drama_id}",
                placeholder="例如：张三",
            ).strip()
            if new_name:
                slots = [{"name": new_name, "tier": "", "role_hint": "", "image_path": "", "uploaded": False}]
                _render_head_upload_slot_grid(drama_id, slots, selected_names)
            st.session_state["doc_frame_selected_character_names"] = selected_names
            return sync_drama_character_session_state(drama_id)

        for group in slot_groups:
            label = str(group.get("label") or "人物")
            group_slots = list(group.get("slots") or [])
            group_uploaded = sum(1 for slot in group_slots if slot.get("uploaded"))
            st.markdown(f"**{label}**（已上传 {group_uploaded}/{len(group_slots)}）")
            _render_head_upload_slot_grid(drama_id, group_slots, selected_names)
            st.markdown("")

    st.session_state["doc_frame_selected_character_names"] = selected_names
    return sync_drama_character_session_state(drama_id)
