"""两步脚本生成：先产出并展示「完美剧情构思方案」，再生成 JSON 脚本。"""

from __future__ import annotations

import hashlib
from typing import Any

import streamlit as st

PLOT_BLUEPRINT_MODES = frozenset({"auto_compact", "summary"})
# 仅用于 text_area widget；勿在 widget 渲染后写入此键
PLOT_BLUEPRINT_EDITOR_KEY = "plot_blueprint_editor"
PLOT_BLUEPRINT_SYNC_EDITOR_FLAG = "_plot_blueprint_sync_editor"
PLOT_BLUEPRINT_CLEAR_EDITOR_FLAG = "_plot_blueprint_clear_editor"
MIN_PLOT_BLUEPRINT_CHARS = 200


def uses_plot_blueprint_workflow(script_path: str) -> bool:
    return (script_path or "") in PLOT_BLUEPRINT_MODES


def build_plot_blueprint_fingerprint(
    *,
    mode: str,
    video_path: str,
    subtitle_path: str = "",
    analysis_path: str = "",
    video_episode_analysis_path: str = "",
    video_theme: str = "",
    append_prompt: str = "",
    enable_frame_analysis: bool = True,
) -> str:
    parts = [
        mode,
        (video_path or "").strip(),
        (video_episode_analysis_path or "").strip() if enable_frame_analysis else "",
        (analysis_path or "").strip() if enable_frame_analysis else "",
        (video_theme or "").strip(),
        (append_prompt or "").strip(),
        "visual" if enable_frame_analysis else "subtitle_only",
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def save_plot_blueprint(
    *,
    content: str,
    fingerprint: str,
    mode: str,
    meta: dict[str, Any] | None = None,
) -> None:
    """保存正式构思方案；下次 rerun 时由 _prepare_blueprint_editor 同步到编辑器。"""
    text = (content or "").strip()
    st.session_state["plot_blueprint_content"] = text
    st.session_state["plot_blueprint_fingerprint"] = fingerprint
    st.session_state["plot_blueprint_mode"] = mode
    st.session_state["plot_blueprint_meta"] = meta or {}
    st.session_state[PLOT_BLUEPRINT_SYNC_EDITOR_FLAG] = True


def clear_plot_blueprint() -> None:
    for key in (
        "plot_blueprint_content",
        "plot_blueprint_fingerprint",
        "plot_blueprint_mode",
        "plot_blueprint_meta",
        PLOT_BLUEPRINT_SYNC_EDITOR_FLAG,
    ):
        st.session_state.pop(key, None)
    st.session_state[PLOT_BLUEPRINT_CLEAR_EDITOR_FLAG] = True


def get_plot_blueprint() -> str:
    return str(st.session_state.get("plot_blueprint_content") or "").strip()


def get_working_blueprint_text() -> str:
    """读取编辑器当前内容；widget 未渲染时回退到已保存内容。"""
    if PLOT_BLUEPRINT_EDITOR_KEY in st.session_state:
        return str(st.session_state.get(PLOT_BLUEPRINT_EDITOR_KEY) or "").strip()
    return get_plot_blueprint()


def is_plot_blueprint_valid(fingerprint: str, mode: str) -> bool:
    stored_fp = str(st.session_state.get("plot_blueprint_fingerprint") or "")
    stored_mode = str(st.session_state.get("plot_blueprint_mode") or "")
    content = get_plot_blueprint()
    return bool(content) and stored_fp == fingerprint and stored_mode == mode


def is_plot_blueprint_ready(
    *,
    fingerprint: str,
    mode: str,
    min_chars: int = MIN_PLOT_BLUEPRINT_CHARS,
) -> bool:
    """是否可进入第二步（已保存或编辑器内已有足够内容）。"""
    if not fingerprint:
        return False
    text = get_working_blueprint_text()
    if len(text) < min_chars:
        return False
    if is_plot_blueprint_valid(fingerprint, mode):
        return True
    return True


def commit_plot_blueprint_draft(
    *,
    fingerprint: str,
    mode: str,
    min_chars: int = MIN_PLOT_BLUEPRINT_CHARS,
) -> str:
    """将编辑器内容提交为正式构思方案（第二步前自动调用）。"""
    draft = get_working_blueprint_text()
    if len(draft) < min_chars:
        raise ValueError(
            f"构思方案过短（{len(draft)} 字），请至少填写 {min_chars} 字后再生成脚本。"
        )

    meta = dict(st.session_state.get("plot_blueprint_meta") or {})
    saved = get_plot_blueprint()
    saved_valid = is_plot_blueprint_valid(fingerprint, mode)

    if not saved or not saved_valid:
        if meta.get("source_label") in (None, ""):
            meta["source_label"] = "手动填写"
    elif draft != saved:
        base = meta.get("source_label") or "字幕×抽帧联合分析"
        if "修正" not in base and "手动" not in base:
            meta["source_label"] = f"{base}（人工修正）"
        meta["edited"] = True

    save_plot_blueprint(content=draft, fingerprint=fingerprint, mode=mode, meta=meta)
    return draft


def _prepare_blueprint_editor(*, fingerprint: str, mode: str) -> None:
    """必须在 st.text_area 之前调用，避免 widget 实例化后改 session_state。"""
    if st.session_state.pop(PLOT_BLUEPRINT_CLEAR_EDITOR_FLAG, False):
        st.session_state[PLOT_BLUEPRINT_EDITOR_KEY] = ""
        return

    if st.session_state.pop(PLOT_BLUEPRINT_SYNC_EDITOR_FLAG, False):
        st.session_state[PLOT_BLUEPRINT_EDITOR_KEY] = get_plot_blueprint()
        return

    if PLOT_BLUEPRINT_EDITOR_KEY not in st.session_state:
        if is_plot_blueprint_valid(fingerprint, mode):
            st.session_state[PLOT_BLUEPRINT_EDITOR_KEY] = get_plot_blueprint()
        else:
            st.session_state[PLOT_BLUEPRINT_EDITOR_KEY] = ""


def render_plot_blueprint_panel(*, fingerprint: str, mode: str) -> bool:
    """展示/编辑构思方案；返回 True 表示当前可进入第二步。"""
    saved_valid = is_plot_blueprint_valid(fingerprint, mode)
    saved_content = get_plot_blueprint()

    if saved_content and not saved_valid:
        st.warning(
            "素材或主题已变更，已保存的构思方案可能已过期。"
            "请重新生成，或在下方修正后点击「保存构思方案」。"
        )
    elif not saved_content and PLOT_BLUEPRINT_EDITOR_KEY not in st.session_state:
        st.info(
            "请先 **① 自动生成** 构思方案，或在下方 **直接填写 / 粘贴** Markdown，"
            "修正后点击 **保存构思方案**，再 **② 生成 JSON 脚本**。"
            "① 需要抽帧 JSON，建议同时提供 SRT 字幕（对白/时间戳更准）；② 仅需构思方案 + 视频。"
        )

    _prepare_blueprint_editor(fingerprint=fingerprint, mode=mode)

    st.markdown("**完美剧情构思方案（蓝图）**")
    st.caption(
        "支持自动生成与手动修正。可直接编辑下方 Markdown；"
        f"建议不少于 {MIN_PLOT_BLUEPRINT_CHARS} 字（逐帧精剪通常需更多）。"
    )

    st.text_area(
        "构思方案内容",
        height=420,
        key=PLOT_BLUEPRINT_EDITOR_KEY,
        placeholder=(
            "在此编写或粘贴剧情构思方案（Markdown）…\n\n"
            "可包含：主要人物表、时间线、高潮前置建议、必讲情节点等。"
        ),
        label_visibility="collapsed",
    )

    working = get_working_blueprint_text()
    col_save, col_preview = st.columns([1, 1])
    with col_save:
        if st.button("保存构思方案", key="save_plot_blueprint_manual", use_container_width=True):
            try:
                commit_plot_blueprint_draft(fingerprint=fingerprint, mode=mode)
                st.success(f"✅ 已保存构思方案（约 {len(get_plot_blueprint())} 字）")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    with col_preview:
        with st.popover("预览渲染效果", use_container_width=True):
            preview_text = working or "（暂无内容）"
            st.markdown(preview_text)

    if saved_valid:
        meta = st.session_state.get("plot_blueprint_meta") or {}
        source_label = meta.get("source_label") or "字幕×抽帧×剧情联合分析"
        st.success(
            f"✅ 构思方案已保存（{source_label}，约 {len(get_plot_blueprint())} 字）。"
            "可在上方继续修正并保存，或直接 **② 生成 JSON 脚本**。"
        )
    elif working and len(working) >= MIN_PLOT_BLUEPRINT_CHARS:
        st.caption(
            f"编辑器中约 {len(working)} 字（未保存）。"
            "可直接点 **② 生成 JSON 脚本**（将自动采用当前内容），或先点「保存构思方案」。"
        )

    return is_plot_blueprint_ready(fingerprint=fingerprint, mode=mode)
