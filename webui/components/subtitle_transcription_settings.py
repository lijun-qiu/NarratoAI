"""逐帧解说素材：字幕转录与文件管理（独立功能）。"""

from __future__ import annotations

import os
import time
import traceback

import streamlit as st
from loguru import logger

from app.config import config
from app.services.subtitle_text import decode_subtitle_bytes
from app.services.subtitle_video_pairing import (
    find_paired_subtitle_path,
    get_transcription_subtitle_path,
    load_subtitle_content,
    resolve_transcription_media_path,
)
from app.utils import utils
from webui.utils.documentary_file_picker import (
    apply_subtitle_path,
    clear_subtitle_path,
    render_saved_file_picker,
)
from webui.components.documentary_output_split import render_output_split_control


def render_fun_asr_transcription(tr, *, show_output_split: bool = True):
    """音视频字幕转录：Fun-ASR / Whisper API / Gemini 兼容 API，失败自动切换。"""
    def clear_subtitle_state():
        st.session_state['subtitle_path'] = None
        st.session_state['subtitle_content'] = None
        st.session_state['subtitle_file_processed'] = False

    def _apply_subtitle_result(generated_path: str, provider_label: str):
        if not generated_path or not os.path.exists(generated_path):
            clear_subtitle_state()
            st.error(f"{provider_label} 转写失败：未生成字幕文件")
            return False
        with open(generated_path, "r", encoding="utf-8") as f:
            subtitle_content = f.read()
        st.session_state['subtitle_path'] = generated_path
        st.session_state['subtitle_content'] = subtitle_content
        st.session_state['subtitle_file_processed'] = True
        st.session_state['doc_subtitle_file_processed'] = True
        st.success(f"字幕转写成功（{provider_label}）: {os.path.basename(generated_path)}")
        return True

    st.caption(
        "上传本地音频/视频生成 SRT。大文件会自动提取并压缩音频后再转写。"
        "若使用 api.4022543.xyz 等 LLM 网关，Whisper/Gemini 转写可能不可用，请优先选 Fun-ASR。"
    )

    from app.services.media_transcription import (
        PROVIDER_FUN_ASR,
        PROVIDER_GEMINI,
        PROVIDER_WHISPER,
        PROVIDER_LABELS,
    )

    col1, col2 = st.columns(2)
    with col1:
        fun_key = st.text_input(
            "阿里百炼 API Key（Fun-ASR）",
            value=config.fun_asr.get("api_key", ""),
            type="password",
            key="fun_asr_api_key",
        )
    with col2:
        whisper_key = st.text_input(
            "Whisper / 网关 API Key",
            value=config.whisper_asr.get("api_key", ""),
            type="password",
            key="whisper_asr_api_key",
        )

    gemini_key = st.text_input(
        "Gemini 兼容 API Key",
        value=config.gemini_asr.get("api_key", "") if hasattr(config, "gemini_asr") else "",
        type="password",
        key="gemini_asr_api_key",
    )

    enable_fallback = st.checkbox(
        "失败时自动尝试其他转录方式",
        value=config.transcription.get("enable_fallback", True) if hasattr(config, "transcription") else True,
        key="transcription_enable_fallback",
    )

    video_origin_path = (st.session_state.get("video_origin_path") or "").strip()
    use_selected_video = st.checkbox(
        "未单独上传时，默认使用上方已选视频转写",
        value=st.session_state.get("transcription_use_selected_video", True),
        key="transcription_use_selected_video",
    )
    if use_selected_video and video_origin_path and os.path.isfile(video_origin_path):
        st.caption(
            f"默认转录: {os.path.basename(video_origin_path)} → "
            f"{os.path.basename(get_transcription_subtitle_path(video_origin_path))}"
        )
    elif use_selected_video:
        st.warning("请先在上方选择或上传视频文件（也可在下方单独上传转录）")

    uploaded_media = st.file_uploader(
        tr("上传需要转录的音频/视频（可选，上传后将优先于默认视频）"),
        type=[
            "aac", "amr", "avi", "flac", "flv", "m4a", "mkv", "mov",
            "mp3", "mp4", "mpeg", "ogg", "opus", "wav", "webm", "wma", "wmv",
        ],
        accept_multiple_files=False,
        key="media_transcription_uploader",
    )
    if uploaded_media is not None:
        st.info(f"将优先转录上传文件: {uploaded_media.name}")

    provider_choice = st.radio(
        "转录方式",
        options=["auto", PROVIDER_FUN_ASR, PROVIDER_WHISPER, PROVIDER_GEMINI],
        format_func=lambda x: {
            "auto": "自动（按顺序尝试已配置的 API）",
            PROVIDER_FUN_ASR: PROVIDER_LABELS[PROVIDER_FUN_ASR],
            PROVIDER_WHISPER: PROVIDER_LABELS[PROVIDER_WHISPER],
            PROVIDER_GEMINI: PROVIDER_LABELS[PROVIDER_GEMINI],
        }.get(x, x),
        horizontal=True,
        key="transcription_provider_choice",
    )

    st.markdown(
        "API Key 说明：Fun-ASR → "
        "[阿里百炼](https://bailian.console.aliyun.com/?tab=model#/api-key)；"
        "Whisper / Gemini → OpenAI 兼容网关（如 api.openai.com 或自建代理）"
    )

    if show_output_split:
        render_output_split_control(key="doc_output_split_parts")

    if st.button("转写生成字幕", key="media_transcribe_btn", use_container_width=True):
        uploaded_temp_path = ""
        if uploaded_media is not None:
            temp_dir = utils.temp_dir("transcription")
            safe_filename = os.path.basename(uploaded_media.name)
            uploaded_temp_path = os.path.join(temp_dir, safe_filename)
            file_name, file_extension = os.path.splitext(safe_filename)
            if os.path.exists(uploaded_temp_path):
                timestamp = time.strftime("%Y%m%d%H%M%S")
                uploaded_temp_path = os.path.join(
                    temp_dir, f"{file_name}_{timestamp}{file_extension}"
                )
            with open(uploaded_temp_path, "wb") as f:
                f.write(uploaded_media.getbuffer())

        media_path = resolve_transcription_media_path(
            video_origin_path,
            uploaded_temp_path,
            prefer_video=use_selected_video,
            uploaded_first=True,
        )
        if not media_path:
            clear_subtitle_state()
            st.error("请先上传转录文件，或在上方选择视频并勾选默认使用视频转写")
            return

        if provider_choice == PROVIDER_FUN_ASR and not fun_key.strip():
            st.error("请先填写 Fun-ASR API Key")
            return
        if provider_choice == PROVIDER_WHISPER and not whisper_key.strip():
            st.error("请先填写 Whisper API Key")
            return
        if provider_choice == PROVIDER_GEMINI and not gemini_key.strip():
            st.error("请先填写 Gemini API Key")
            return
        if provider_choice == "auto" and not any([
            fun_key.strip(), whisper_key.strip(), gemini_key.strip()
        ]):
            st.error("请至少填写一种转录 API Key")
            return

        try:
            clear_subtitle_state()
            from app.services import media_transcription

            if fun_key.strip():
                config.fun_asr["api_key"] = fun_key.strip()
            if whisper_key.strip():
                config.whisper_asr["api_key"] = whisper_key.strip()
            if gemini_key.strip():
                config.gemini_asr["api_key"] = gemini_key.strip()
            if hasattr(config, "transcription"):
                config.transcription["enable_fallback"] = enable_fallback
            config.save_config()

            subtitle_path = get_transcription_subtitle_path(media_path)
            os.makedirs(os.path.dirname(subtitle_path) or utils.subtitle_dir(), exist_ok=True)

            with st.spinner("正在转写字幕，失败时将自动切换其他方式..."):
                generated_path, used_provider = media_transcription.transcribe_media_to_srt(
                    media_path,
                    subtitle_path,
                    provider=provider_choice,
                    enable_fallback=enable_fallback,
                )

            split_parts = int(st.session_state.get("doc_output_split_parts") or 1)
            part_paths: list[str] = []
            if split_parts > 1:
                from app.services.documentary.material_output_split import save_split_srt_files
                from app.services.srt_utils import parse_srt_file

                entries = parse_srt_file(generated_path)
                split_result = save_split_srt_files(entries, generated_path, split_parts)
                part_paths = split_result.get("part_paths") or []

            label = PROVIDER_LABELS.get(used_provider, used_provider)
            if _apply_subtitle_result(generated_path, label):
                st.session_state["_subtitle_synced_video_path"] = video_origin_path or media_path
                if part_paths:
                    st.info(
                        "已另存 "
                        + "、".join(f"`{os.path.basename(path)}`" for path in part_paths)
                        + "。当前仍使用完整字幕文件。"
                    )
        except Exception as e:
            clear_subtitle_state()
            logger.error(f"字幕转写失败: {traceback.format_exc()}")
            st.error(f"字幕转写失败（已尝试所有可用方式）: {str(e)}")


def render_documentary_subtitle_file_picker(
    tr,
    *,
    path_input_key: str = "doc_subtitle_path_input",
    pick_key: str = "doc_subtitle_saved_pick",
    confirm_button_key: str = "doc_confirm_subtitle_path",
    clear_button_key: str = "doc_clear_subtitle",
    import_key: str = "docu_subtitle_uploader",
    on_clear=None,
) -> None:
    """从默认字幕目录选用或导入字幕文件。"""
    if "doc_subtitle_file_processed" not in st.session_state:
        st.session_state["doc_subtitle_file_processed"] = False

    video_path = (st.session_state.get("video_origin_path") or "").strip()
    material_source = (st.session_state.get("doc_material_source_video_path") or "").strip()
    from app.services.documentary.documentary_material_resolver import resolve_subtitle_path_for_documentary

    paired_path = ""
    if video_path:
        paired_path = resolve_subtitle_path_for_documentary(
            video_path,
            material_source_video_path=material_source,
            explicit_path=None,
        ) or find_paired_subtitle_path(video_path) or ""
    active_path = (st.session_state.get("subtitle_path") or "").strip()

    if active_path and os.path.isfile(active_path):
        upload_hint = "（已确认）" if st.session_state.get("doc_subtitle_file_processed") else ""
        st.info(f"当前字幕: {os.path.basename(active_path)}{upload_hint}")

    def _import_subtitle_file(subtitle_file) -> None:
        try:
            safe_filename = os.path.basename(subtitle_file.name)
            decoded = decode_subtitle_bytes(subtitle_file.getvalue())
            script_content = decoded.text
            if not script_content:
                st.error(tr("无法读取字幕文件，请检查文件编码（支持 UTF-8、UTF-16、GBK、GB2312）"))
                st.stop()

            subtitle_save_dir = utils.subtitle_dir()
            os.makedirs(subtitle_save_dir, exist_ok=True)
            script_file_path = os.path.join(subtitle_save_dir, safe_filename)
            if os.path.exists(script_file_path):
                timestamp = time.strftime("%Y%m%d%H%M%S")
                name, ext = os.path.splitext(safe_filename)
                script_file_path = os.path.join(subtitle_save_dir, f"{name}_{timestamp}{ext}")

            with open(script_file_path, "w", encoding="utf-8") as f:
                f.write(script_content)

            from webui.utils.documentary_file_picker import queue_picker_paths

            apply_subtitle_path(script_file_path)
            queue_picker_paths(path_input_key, pick_key, script_file_path)
            st.success(f"字幕已导入: {safe_filename}")
            st.rerun()
        except Exception as e:
            st.error(f"{tr('Upload failed')}: {str(e)}")

    render_saved_file_picker(
        label="字幕文件",
        directory=utils.subtitle_dir(),
        glob_pattern="*.srt",
        path_input_key=path_input_key,
        pick_key=pick_key,
        confirm_button_key=confirm_button_key,
        clear_button_key=clear_button_key,
        active_path=active_path,
        paired_path=paired_path,
        on_confirm=apply_subtitle_path,
        on_clear=on_clear
        or (lambda: clear_subtitle_path(
            path_input_key=path_input_key,
            pick_key=pick_key,
        )),
        import_label=tr("导入新字幕到字幕目录"),
        import_types=["srt"],
        import_key=import_key,
        on_import=_import_subtitle_file,
    )


def render_subtitle_transcription_panel(tr, *, show_output_split: bool = True) -> None:
    """字幕转录独立面板：ASR 转写 + 字幕文件管理。"""
    render_fun_asr_transcription(tr, show_output_split=show_output_split)
    st.divider()
    st.caption("从默认字幕目录选用已有文件，或导入新字幕。")
    render_documentary_subtitle_file_picker(tr)
