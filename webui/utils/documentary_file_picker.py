"""逐帧解说/精剪：从默认保存目录选用字幕与抽帧分析文件。"""

from __future__ import annotations

import glob
import os
from typing import Callable

import streamlit as st

from app.services.subtitle_video_pairing import load_subtitle_content


def list_saved_files(directory: str, glob_pattern: str) -> list[str]:
    if not directory or not os.path.isdir(directory):
        return []
    pattern = os.path.join(directory, glob_pattern)
    files = [path for path in glob.glob(pattern) if os.path.isfile(path)]
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def _pending_path_key(path_input_key: str) -> str:
    return f"__pending__{path_input_key}"


def _pending_pick_key(pick_key: str) -> str:
    return f"__pending__{pick_key}"


_PENDING_REUSE_FRAME_ANALYSIS_KEY = "__pending__doc_reuse_frame_analysis"


def queue_reuse_frame_analysis(value: bool = True) -> None:
    """在 doc_reuse_frame_analysis checkbox 渲染前写入复用开关（确认/导入后通过 rerun 生效）。"""
    st.session_state[_PENDING_REUSE_FRAME_ANALYSIS_KEY] = value


def consume_pending_reuse_frame_analysis() -> None:
    """在实例化 doc_reuse_frame_analysis checkbox 之前消费 pending 值。"""
    pending = st.session_state.pop(_PENDING_REUSE_FRAME_ANALYSIS_KEY, None)
    if pending is not None:
        st.session_state["doc_reuse_frame_analysis"] = bool(pending)


def queue_picker_paths(path_input_key: str, pick_key: str, path: str) -> None:
    """在 widget 渲染前写入路径（导入/确认后通过 rerun 生效）。"""
    st.session_state[_pending_path_key(path_input_key)] = path
    st.session_state[_pending_pick_key(pick_key)] = os.path.basename(path)


def consume_pending_picker_paths(path_input_key: str, pick_key: str) -> None:
    """在实例化 text_input / selectbox 之前消费 pending 路径。"""
    pending_path = st.session_state.pop(_pending_path_key(path_input_key), None)
    pending_pick = st.session_state.pop(_pending_pick_key(pick_key), None)
    if pending_path:
        st.session_state[path_input_key] = pending_path
    if pending_pick:
        st.session_state[pick_key] = pending_pick


def _pick_default_path(
    files: list[str],
    *,
    active_path: str = "",
    paired_path: str = "",
) -> str:
    active = (active_path or "").strip()
    if active and active in files:
        return active
    paired = (paired_path or "").strip()
    if paired and paired in files:
        return paired
    if files:
        return files[0]
    return active or paired


def render_saved_file_picker(
    *,
    label: str,
    directory: str,
    glob_pattern: str,
    path_input_key: str,
    pick_key: str,
    confirm_button_key: str,
    clear_button_key: str,
    active_path: str = "",
    paired_path: str = "",
    on_confirm: Callable[[str], None],
    on_clear: Callable[[], None],
    import_label: str = "",
    import_types: list[str] | None = None,
    import_key: str = "",
    on_import=None,
) -> None:
    """从保存目录选择文件；路径输入框默认指向推荐文件。"""
    consume_pending_picker_paths(path_input_key, pick_key)
    st.caption(f"保存目录: `{directory}`")

    files = list_saved_files(directory, glob_pattern)
    path_by_label: dict[str, str] = {}
    for file_path in files:
        basename = os.path.basename(file_path)
        if basename not in path_by_label:
            path_by_label[basename] = file_path

    default_path = _pick_default_path(
        files,
        active_path=active_path,
        paired_path=paired_path,
    )

    if pick_key not in st.session_state:
        st.session_state[pick_key] = os.path.basename(default_path) if default_path else ""
    if path_input_key not in st.session_state:
        current_pick = (st.session_state.get(pick_key) or "").strip()
        if current_pick in path_by_label:
            st.session_state[path_input_key] = path_by_label[current_pick]
        else:
            st.session_state[path_input_key] = default_path

    def on_pick_change() -> None:
        picked_label = (st.session_state.get(pick_key) or "").strip()
        picked_path = path_by_label.get(picked_label, "")
        if picked_path:
            st.session_state[path_input_key] = picked_path

    if path_by_label:
        labels = list(path_by_label.keys())
        current_label = (st.session_state.get(pick_key) or "").strip()
        if current_label not in labels:
            current_label = os.path.basename(default_path) if default_path in files else labels[0]
            st.session_state[pick_key] = current_label
        st.selectbox(
            f"{label}（目录内已有）",
            options=labels,
            key=pick_key,
            on_change=on_pick_change,
        )
    else:
        st.caption(f"目录内暂无匹配 `{glob_pattern}` 的文件，可填写路径或导入新文件。")

    st.text_input(
        f"{label}路径",
        key=path_input_key,
        help="可直接填写保存目录中的完整路径，或从上方列表选择后点确认",
    )

    action_cols = st.columns([1, 1, 2])
    with action_cols[0]:
        if st.button("确认使用", key=confirm_button_key, use_container_width=True):
            selected_path = (st.session_state.get(path_input_key) or "").strip()
            if not selected_path:
                st.error("请填写或选择文件路径")
            elif not os.path.isfile(selected_path):
                st.error(f"文件不存在: {selected_path}")
            else:
                try:
                    on_confirm(selected_path)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.rerun()
    with action_cols[1]:
        if st.button("清除", key=clear_button_key, use_container_width=True):
            on_clear()
            st.rerun()

    if import_key and on_import is not None:
        import_processed_key = f"{import_key}_processed"
        if import_processed_key not in st.session_state:
            st.session_state[import_processed_key] = False

        imported = st.file_uploader(
            import_label or f"导入新{label}",
            type=import_types or [],
            accept_multiple_files=False,
            key=import_key,
        )
        if imported is None:
            st.session_state[import_processed_key] = False
        elif not st.session_state.get(import_processed_key):
            on_import(imported)
            st.session_state[import_processed_key] = True


def apply_subtitle_path(path: str) -> None:
    content = load_subtitle_content(path).strip()
    if not content:
        raise ValueError("字幕文件为空或无法读取")
    st.session_state["subtitle_path"] = path
    st.session_state["subtitle_content"] = content
    st.session_state["doc_subtitle_file_processed"] = True


def clear_subtitle_path(
    *,
    path_input_key: str = "doc_subtitle_path_input",
    pick_key: str = "doc_subtitle_saved_pick",
) -> None:
    st.session_state["subtitle_path"] = None
    st.session_state["subtitle_content"] = None
    st.session_state["doc_subtitle_file_processed"] = False
    st.session_state.pop(path_input_key, None)
    st.session_state.pop(pick_key, None)


def apply_frame_analysis_path(path: str) -> None:
    st.session_state["frame_analysis_json_path"] = path
    st.session_state["doc_frame_analysis_file_processed"] = True
    st.session_state["doc_frame_analysis_upload_explicit"] = True
    queue_reuse_frame_analysis(True)


def clear_frame_analysis_path(
    *,
    path_input_key: str = "doc_frame_analysis_path_input",
    pick_key: str = "doc_frame_analysis_saved_pick",
) -> None:
    st.session_state["frame_analysis_json_path"] = None
    st.session_state["doc_frame_analysis_file_processed"] = False
    st.session_state["doc_frame_analysis_upload_explicit"] = False
    st.session_state.pop(path_input_key, None)
    st.session_state.pop(pick_key, None)
