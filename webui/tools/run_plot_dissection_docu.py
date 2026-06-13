# 剧情解剖 WebUI 入口
import os
import traceback

import streamlit as st
from loguru import logger

from app.services.documentary.plot_dissection_service import (
    default_plot_dissection_path,
    run_plot_dissection,
)


def run_plot_dissection_docu(
    params,
    *,
    analysis_path: str,
    use_character_relationship: bool = True,
    use_plot_reference: bool = False,
) -> None:
    path = (analysis_path or "").strip()
    if not path or not os.path.isfile(path):
        st.error("请先完成整片视频分析")
        return

    status = st.empty()

    def update_progress(message: str) -> None:
        status.markdown(f"📝 {message}")

    try:
        with st.spinner("正在进行剧情解剖…"):
            result = run_plot_dissection(
                video_episode_analysis_path=path,
                character_relationship=st.session_state.get("doc_character_relationship", ""),
                plot_reference=st.session_state.get("doc_plot_reference", ""),
                use_character_relationship=use_character_relationship,
                use_plot_reference=use_plot_reference,
                progress_callback=update_progress,
            )
        out_path = str(result.get("output_path") or default_plot_dissection_path(path))
        st.session_state["video_episode_dissection_json_path"] = out_path
        st.success(f"✅ 剧情解剖完成: {out_path}")
        corrected = result.get("corrected") or {}
        with st.expander("预览校正摘要", expanded=True):
            st.write(corrected.get("overall_summary") or "")
            corrections = corrected.get("name_corrections") or []
            if corrections:
                st.caption(f"人名校正 {len(corrections)} 处")
                for item in corrections[:12]:
                    if isinstance(item, dict):
                        st.text(
                            f"{item.get('wrong')} → {item.get('correct')} "
                            f"({item.get('reason') or ''})"
                        )
        with st.expander("完整 JSON", expanded=False):
            st.json(corrected)
    except Exception as err:
        st.error(f"❌ 剧情解剖失败: {err}")
        logger.exception(f"剧情解剖失败\n{traceback.format_exc()}")
