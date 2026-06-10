"""人声音频导出：将视频中互不重复的人声导出为 MP3，供用户自行重命名。"""

from __future__ import annotations

import os

import streamlit as st

from app.services.documentary.voice_sample_extraction import (
    default_export_index_path,
    export_distinct_voice_mp3s,
    load_export_index,
    voice_export_dir,
)
from app.services.subtitle_video_pairing import find_paired_subtitle_path
from webui.components.drama_character_input import get_drama_id


def render_voice_calibration_panel() -> None:
    st.caption(
        "将当前视频中**互不重复的人声**导出为 MP3 文件到本地目录。"
        "导出后请自行重命名，例如 `秦枫.mp3`、`罗博.mp3`，用于听音辨人。"
    )

    video_path = (st.session_state.get("video_origin_path") or "").strip()
    if not video_path or not os.path.isfile(video_path):
        st.info("请先在页面上方选择视频文件，再导出人声音频。")
        return

    export_dir = voice_export_dir(video_path)
    st.caption(f"当前视频：{os.path.basename(video_path)}")
    st.caption(f"导出目录：`{export_dir}`")

    drama_id = get_drama_id()
    paired_subtitle = find_paired_subtitle_path(video_path)
    use_subtitle = st.checkbox(
        "优先按字幕时间轴取样（需已转录 SRT，去重更准）",
        value=bool(paired_subtitle),
        key="voice_export_use_subtitle",
    )
    if use_subtitle:
        if paired_subtitle:
            st.caption(f"将使用字幕：{os.path.basename(paired_subtitle)}")
        else:
            st.warning("未找到配对字幕，将改用静音检测取样。")

    existing = load_export_index(default_export_index_path(video_path))
    if existing:
        st.caption(
            f"上次导出：{existing.get('distinct_voice_count', 0)} 个不重复人声 · "
            f"候选 {existing.get('candidate_count', 0)} 段"
        )

    if st.button("导出不重复人声音频", type="primary", key="voice_export_run"):
        progress = st.progress(0, text="准备导出...")
        status = st.empty()

        def _progress(pct: int, msg: str) -> None:
            progress.progress(min(100, max(0, pct)) / 100.0, text=msg)
            status.caption(msg)

        try:
            result = export_distinct_voice_mp3s(
                video_path,
                subtitle_path=paired_subtitle if use_subtitle and paired_subtitle else "",
                drama_id=drama_id,
                progress_callback=_progress,
            )
            st.session_state["voice_export_result"] = result
            st.success(
                f"已导出 {result.get('distinct_voice_count', 0)} 个 MP3 到 {result.get('export_dir', export_dir)}"
            )
        except Exception as exc:
            st.error(f"导出失败：{exc}")
        finally:
            progress.empty()
            status.empty()

    result = st.session_state.get("voice_export_result") or existing
    if not result:
        return

    exports = result.get("exports") or []
    if not exports:
        st.warning("未导出任何人声文件。")
        return

    st.markdown("**导出文件**（可直接到文件夹中重命名）：")
    for item in exports:
        file_name = str(item.get("file_name") or "")
        file_path = str(item.get("file_path") or os.path.join(export_dir, file_name))
        timestamp = str(item.get("timestamp") or "")
        hint = str(item.get("hint_text") or "").strip()
        similar = int(item.get("similar_segment_count") or 0)
        label = f"`{file_name}` · {timestamp}"
        if hint:
            label += f" · {hint}"
        if similar > 1:
            label += f" · 约 {similar} 段相似人声"
        st.markdown(label)
        if os.path.isfile(file_path):
            with open(file_path, "rb") as fp:
                st.audio(fp.read(), format="audio/mp3")
