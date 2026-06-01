import os
import glob
import json
import time
import traceback
from copy import deepcopy
import streamlit as st
from loguru import logger

from app.config import config
from app.models.schema import VideoClipParams
from app.services.subtitle_text import decode_subtitle_bytes
from app.utils import utils, check_script
from webui.tools.generate_script_docu import generate_script_docu
from webui.tools.generate_script_short import generate_script_short
from webui.tools.generate_short_summary import generate_script_short_sunmmary
from webui.tools.generate_film_tv_summary import generate_script_film_tv_summary
from app.services.film_tv_settings import (
    FILM_TV_DEFAULTS,
    get_film_tv_settings,
    save_film_tv_settings_to_config,
)


def render_script_panel(tr):
    """渲染脚本配置面板"""
    with st.container(border=True):
        st.write(tr("Video Script Configuration"))
        params = VideoClipParams()

        # 渲染脚本文件选择
        render_script_file(tr, params)

        # 渲染视频文件选择
        render_video_file(tr, params)

        # 获取当前选择的脚本类型
        script_path = st.session_state.get('video_clip_json_path', '')

        # 根据脚本类型显示不同的布局
        if script_path == "auto":
            # 画面解说
            render_video_details(tr)
        elif script_path == "short":
            # 短剧混剪
            render_short_generate_options(tr)
        elif script_path == "summary":
            # 短剧解说
            short_drama_summary(tr)
        elif script_path == "film_tv":
            # 影视解说
            film_tv_narration(tr)
        else:
            # 默认为空
            pass

        # 渲染脚本操作按钮
        render_script_buttons(tr, params)


def render_script_file(tr, params):
    """渲染脚本文件选择"""
    # 定义功能模式
    MODE_FILE = "file_selection"
    MODE_AUTO = "auto"
    MODE_SHORT = "short"
    MODE_SUMMARY = "summary"
    MODE_FILM_TV = "film_tv"

    # 处理保存脚本后的模式切换（必须在 widget 实例化之前）
    if st.session_state.get('_switch_to_file_mode'):
        st.session_state['script_mode_selection'] = tr("Select/Upload Script")
        del st.session_state['_switch_to_file_mode']

    # 模式选项映射
    mode_options = {
        tr("Select/Upload Script"): MODE_FILE,
        tr("Auto Generate"): MODE_AUTO,
        tr("Short Generate"): MODE_SHORT,
        tr("Short Drama Summary"): MODE_SUMMARY,
        tr("Film TV Narration"): MODE_FILM_TV,
    }
    
    # 获取当前状态
    current_path = st.session_state.get('video_clip_json_path', '')
    
    # 确定当前选中的模式索引
    default_index = 0
    mode_keys = list(mode_options.keys())
    
    if current_path == "auto":
        default_index = mode_keys.index(tr("Auto Generate"))
    elif current_path == "short":
        default_index = mode_keys.index(tr("Short Generate"))
    elif current_path == "summary":
        default_index = mode_keys.index(tr("Short Drama Summary"))
    elif current_path == "film_tv":
        default_index = mode_keys.index(tr("Film TV Narration"))
    elif not current_path:
        default_index = mode_keys.index(tr("Film TV Narration"))
    else:
        default_index = mode_keys.index(tr("Select/Upload Script"))

    # 1. 渲染功能选择下拉框
    # 使用 segmented_control 替代 selectbox，提供更好的视觉体验
    default_mode_label = mode_keys[default_index]
    
    # 定义回调函数来处理状态更新
    def update_script_mode():
        # 获取当前选中的标签
        selected_label = st.session_state.script_mode_selection
        if selected_label:
            # 更新实际的 path 状态
            new_mode = mode_options[selected_label]
            st.session_state.video_clip_json_path = new_mode
            params.video_clip_json_path = new_mode
        else:
            # 如果用户取消选择（segmented_control 允许取消），恢复到默认或上一个状态
            # 这里我们强制保持当前状态，或者重置为默认
            st.session_state.script_mode_selection = default_mode_label

    # 渲染组件
    selected_mode_label = st.segmented_control(
        tr("Video Type"),
        options=mode_keys,
        default=default_mode_label,
        key="script_mode_selection",
        on_change=update_script_mode
    )
    
    # 处理未选择的情况（虽然有default，但在某些交互下可能为空）
    if not selected_mode_label:
        selected_mode_label = default_mode_label
        
    selected_mode = mode_options[selected_mode_label]

    # 2. 根据选择的模式处理逻辑
    if selected_mode == MODE_FILE:
        # --- 文件选择模式 ---
        script_list = [
            (tr("None"), ""),
            (tr("Upload Script"), "upload_script")
        ]

        # 获取已有脚本文件
        suffix = "*.json"
        script_dir = utils.script_dir()
        files = glob.glob(os.path.join(script_dir, suffix))
        file_list = []

        for file in files:
            file_list.append({
                "name": os.path.basename(file),
                "file": file,
                "ctime": os.path.getctime(file)
            })

        file_list.sort(key=lambda x: x["ctime"], reverse=True)
        for file in file_list:
            display_name = file['file'].replace(config.root_dir, "")
            script_list.append((display_name, file['file']))

        # 找到保存的脚本文件在列表中的索引
        # 如果当前path是特殊值(auto/short/summary)，则重置为空
        saved_script_path = current_path if current_path not in [MODE_AUTO, MODE_SHORT, MODE_SUMMARY, MODE_FILM_TV] else ""
        
        selected_index = 0
        for i, (_, path) in enumerate(script_list):
            if path == saved_script_path:
                selected_index = i
                break

        # 如果找到了保存的脚本，同步更新 selectbox 的 key 状态
        if saved_script_path and selected_index > 0:
            st.session_state['script_file_selection'] = selected_index

        selected_script_index = st.selectbox(
            tr("Script Files"),
            index=selected_index,
            options=range(len(script_list)),
            format_func=lambda x: script_list[x][0],
            key="script_file_selection"
        )

        script_path = script_list[selected_script_index][1]
        # 只有当用户实际选择了脚本时才更新路径，避免覆盖已保存的路径
        if script_path:
            st.session_state['video_clip_json_path'] = script_path
            params.video_clip_json_path = script_path
        elif saved_script_path:
            # 如果用户选择了 "None" 但之前有保存的脚本，保持原有路径
            st.session_state['video_clip_json_path'] = saved_script_path
            params.video_clip_json_path = saved_script_path

        # 处理脚本上传
        if script_path == "upload_script":
            uploaded_file = st.file_uploader(
                tr("Upload Script File"),
                type=["json"],
                accept_multiple_files=False,
            )

            if uploaded_file is not None:
                try:
                    # 读取上传的JSON内容并验证格式
                    script_content = uploaded_file.read().decode('utf-8')
                    json_data = json.loads(script_content)

                    # 保存到脚本目录
                    safe_filename = os.path.basename(uploaded_file.name)
                    script_file_path = os.path.join(script_dir, safe_filename)
                    file_name, file_extension = os.path.splitext(safe_filename)

                    # 如果文件已存在,添加时间戳
                    if os.path.exists(script_file_path):
                        timestamp = time.strftime("%Y%m%d%H%M%S")
                        file_name_with_timestamp = f"{file_name}_{timestamp}"
                        script_file_path = os.path.join(script_dir, file_name_with_timestamp + file_extension)

                    # 写入文件
                    with open(script_file_path, "w", encoding='utf-8') as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=2)

                    # 更新状态
                    st.success(tr("Script Uploaded Successfully"))
                    st.session_state['video_clip_json_path'] = script_file_path
                    params.video_clip_json_path = script_file_path
                    time.sleep(1)
                    st.rerun()

                except json.JSONDecodeError:
                    st.error(tr("Invalid JSON format"))
                except Exception as e:
                    st.error(f"{tr('Upload failed')}: {str(e)}")
    else:
        # --- 功能生成模式 ---
        st.session_state['video_clip_json_path'] = selected_mode
        params.video_clip_json_path = selected_mode


def render_video_file(tr, params):
    """渲染视频文件选择"""
    video_list = [(tr("None"), ""), (tr("Upload Local Files"), "upload_local")]

    # 获取已有视频文件
    for suffix in ["*.mp4", "*.mov", "*.avi", "*.mkv"]:
        video_files = glob.glob(os.path.join(utils.video_dir(), suffix))
        for file in video_files:
            display_name = file.replace(config.root_dir, "")
            video_list.append((display_name, file))

    selected_video_index = st.selectbox(
        tr("Video File"),
        index=0,
        options=range(len(video_list)),
        format_func=lambda x: video_list[x][0]
    )

    video_path = video_list[selected_video_index][1]
    st.session_state['video_origin_path'] = video_path
    params.video_origin_path = video_path

    if video_path == "upload_local":
        uploaded_file = st.file_uploader(
            tr("Upload Local Files"),
            type=["mp4", "mov", "avi", "flv", "mkv"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            safe_filename = os.path.basename(uploaded_file.name)
            video_file_path = os.path.join(utils.video_dir(), safe_filename)
            file_name, file_extension = os.path.splitext(safe_filename)

            if os.path.exists(video_file_path):
                timestamp = time.strftime("%Y%m%d%H%M%S")
                file_name_with_timestamp = f"{file_name}_{timestamp}"
                video_file_path = os.path.join(utils.video_dir(), file_name_with_timestamp + file_extension)

            with open(video_file_path, "wb") as f:
                f.write(uploaded_file.read())
                st.success(tr("File Uploaded Successfully"))
                st.session_state['video_origin_path'] = video_file_path
                params.video_origin_path = video_file_path
                time.sleep(1)
                st.rerun()


def render_short_generate_options(tr):
    """
    渲染Short Generate模式下的特殊选项
    在Short Generate模式下，替换原有的输入框为自定义片段选项
    """
    short_drama_summary(tr)
    # 显示自定义片段数量选择器
    custom_clips = st.number_input(
        tr("自定义片段"),
        min_value=1,
        max_value=20,
        value=st.session_state.get('custom_clips', 5),
        help=tr("设置需要生成的短视频片段数量"),
        key="custom_clips_input"
    )
    st.session_state['custom_clips'] = custom_clips


def render_video_details(tr):
    """画面解说 渲染视频主题和提示词"""
    video_theme = st.text_input(tr("Video Theme"))
    custom_prompt = st.text_area(
        tr("Generation Prompt"),
        value=st.session_state.get('video_plot', ''),
        help=tr("Custom prompt for LLM, leave empty to use default prompt"),
        height=180
    )
    # 非短视频模式下显示原有的三个输入框
    input_cols = st.columns(2)

    with input_cols[0]:
        st.number_input(
            tr("Frame Interval (seconds)"),
            min_value=0,
            value=st.session_state.get('frame_interval_input', config.frames.get('frame_interval_input', 3)),
            help=tr("Frame Interval (seconds) (More keyframes consume more tokens)"),
            key="frame_interval_input"
        )

    with input_cols[1]:
        st.number_input(
            tr("Batch Size"),
            min_value=0,
            value=st.session_state.get('vision_batch_size', config.frames.get('vision_batch_size', 10)),
            help=tr("Batch Size (More keyframes consume more tokens)"),
            key="vision_batch_size"
        )
    st.session_state['video_theme'] = video_theme
    st.session_state['custom_prompt'] = custom_prompt
    return video_theme, custom_prompt


def short_drama_summary(tr):
    """短剧解说 渲染视频主题和提示词"""
    return render_subtitle_narration_panel(tr, work_name_label="短剧名称", uploader_key="subtitle_file_uploader")


def film_tv_narration(tr):
    """影视解说 渲染视频主题和提示词"""
    video_theme = render_subtitle_narration_panel(
        tr, work_name_label="Film Title", uploader_key="film_tv_subtitle_uploader"
    )
    render_film_tv_rules_settings(tr)
    return video_theme


def render_film_tv_rules_settings(tr):
    """影视解说规则参数调节面板（默认与 config.toml [film_tv] 一致）。"""
    defaults = get_film_tv_settings()
    saved = st.session_state.get("film_tv_settings")
    base = saved if isinstance(saved, dict) else defaults

    def _clamp(value, lo, hi):
        return max(lo, min(hi, int(value)))

    with st.expander("影视解说规则参数", expanded=True):
        st.caption(
            "调节生成脚本与后处理规则；默认值来自 config.toml，可临时调整或保存为默认。"
            "「最少段数」会写入 AI 提示词并在生成后校验，未达标时自动重试一次。"
        )

        c1, c2 = st.columns(2)
        with c1:
            target_duration_percent = st.slider(
                "成片时长占原片比例 (%)",
                min_value=10, max_value=90, value=_clamp(base["target_duration_percent"], 10, 90),
                help="例如 40 表示 6 分钟原片 → 约 2.4 分钟成片",
                key="ftv_target_duration_percent",
            )
            ost1_duration_min = st.slider(
                "原声片段最短 (秒)", 3, 30, _clamp(base["ost1_duration_min"], 3, 30),
                key="ftv_ost1_duration_min",
            )
            ost1_duration_max = st.slider(
                "原声片段最长 (秒)", 5, 60, _clamp(base["ost1_duration_max"], 5, 60),
                key="ftv_ost1_duration_max",
            )
            ost1_duration_long_max = st.slider(
                "名场面原声最长 (秒)", 8, 60, _clamp(base["ost1_duration_long_max"], 8, 60),
                key="ftv_ost1_duration_long_max",
            )
            original_audio_percent = st.slider(
                "原声占比目标 (%)", 30, 95, _clamp(base["original_audio_percent"], 30, 95),
                key="ftv_original_audio_percent",
            )
        with c2:
            ost1_segment_min = st.slider(
                "原声段数最少", 3, 40, _clamp(base["ost1_segment_min"], 3, 40),
                key="ftv_ost1_segment_min",
            )
            ost1_segment_max = st.slider(
                "原声段数最多", 5, 50, _clamp(base["ost1_segment_max"], 5, 50),
                key="ftv_ost1_segment_max",
            )
            ost0_segment_min = st.slider(
                "解说段数最少", 2, 25, _clamp(base["ost0_segment_min"], 2, 25),
                key="ftv_ost0_segment_min",
            )
            ost0_segment_max = st.slider(
                "解说段数最多", 3, 30, _clamp(base["ost0_segment_max"], 3, 30),
                key="ftv_ost0_segment_max",
            )
            narration_percent = st.slider(
                "解说占比目标 (%)", 5, 70, _clamp(base["narration_percent"], 5, 70),
                key="ftv_narration_percent",
            )

        c3, c4, c5 = st.columns(3)
        with c3:
            narration_chars_min = st.slider(
                "解说字数下限", 20, 150, _clamp(base["narration_chars_min"], 20, 150),
                key="ftv_narration_chars_min",
            )
        with c4:
            narration_chars_max = st.slider(
                "解说字数上限", 40, 250, _clamp(base["narration_chars_max"], 40, 250),
                key="ftv_narration_chars_max",
            )
        with c5:
            opening_chars_max = st.slider(
                "开场解说字数上限", 60, 300, _clamp(base["opening_chars_max"], 60, 300),
                key="ftv_opening_chars_max",
            )

        allow_consecutive_ost1 = st.checkbox(
            "允许连续多段原声（不打断）",
            value=bool(base.get("allow_consecutive_ost1", True)),
            key="ftv_allow_consecutive_ost1",
        )
        enforce_narration_after_ost1 = st.checkbox(
            "原声播放期间禁止插入解说（自动修正脚本顺序）",
            value=bool(base.get("enforce_narration_after_ost1", True)),
            key="ftv_enforce_narration_after_ost1",
        )

        if ost1_duration_min > ost1_duration_max:
            st.warning("原声最短时长不能大于最长时长，生成时将自动对调。")
        if ost1_segment_min > ost1_segment_max:
            st.warning("原声段数最少不能大于最多，生成时将自动对调。")
        if ost0_segment_min > ost0_segment_max:
            st.warning("解说段数最少不能大于最多，生成时将自动对调。")
        if narration_chars_min > narration_chars_max:
            st.warning("解说字数下限不能大于上限，生成时将自动对调。")

        settings = {
            "target_duration_percent": target_duration_percent,
            "ost1_duration_min": min(ost1_duration_min, ost1_duration_max),
            "ost1_duration_max": max(ost1_duration_min, ost1_duration_max),
            "ost1_duration_long_max": ost1_duration_long_max,
            "ost1_segment_min": min(ost1_segment_min, ost1_segment_max),
            "ost1_segment_max": max(ost1_segment_min, ost1_segment_max),
            "ost0_segment_min": min(ost0_segment_min, ost0_segment_max),
            "ost0_segment_max": max(ost0_segment_min, ost0_segment_max),
            "original_audio_percent": original_audio_percent,
            "narration_percent": narration_percent,
            "narration_chars_min": min(narration_chars_min, narration_chars_max),
            "narration_chars_max": max(narration_chars_min, narration_chars_max),
            "opening_chars_max": opening_chars_max,
            "allow_consecutive_ost1": allow_consecutive_ost1,
            "enforce_narration_after_ost1": enforce_narration_after_ost1,
        }
        st.session_state["film_tv_settings"] = settings

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("恢复默认规则", key="ftv_reset_defaults", use_container_width=True):
                st.session_state["film_tv_settings"] = deepcopy(FILM_TV_DEFAULTS)
                st.rerun()
        with btn2:
            if st.button("保存为 config.toml 默认", key="ftv_save_config", use_container_width=True):
                if save_film_tv_settings_to_config(settings):
                    st.success("已保存到 config.toml [film_tv]")
                else:
                    st.error("保存失败，请查看日志")


def render_subtitle_narration_panel(tr, work_name_label: str, uploader_key: str):
    """字幕解说类模式共用面板（短剧解说 / 影视解说）"""
    # 检查是否已经处理过字幕文件
    if 'subtitle_file_processed' not in st.session_state:
        st.session_state['subtitle_file_processed'] = False

    render_fun_asr_transcription(tr)
    
    subtitle_file = st.file_uploader(
        tr("上传字幕文件"),
        type=["srt"],
        accept_multiple_files=False,
        key=uploader_key
    )
    
    # 显示当前已上传的字幕文件路径
    if 'subtitle_path' in st.session_state and st.session_state['subtitle_path']:
        st.info(f"已上传字幕: {os.path.basename(st.session_state['subtitle_path'])}")
        if st.button(tr("清除已上传字幕")):
            st.session_state['subtitle_path'] = None
            st.session_state['subtitle_content'] = None
            st.session_state['subtitle_file_processed'] = False
            st.rerun()
    
    # 只有当有文件上传且尚未处理时才执行处理逻辑
    if subtitle_file is not None and not st.session_state['subtitle_file_processed']:
        try:
            # 清理文件名，防止路径污染和路径遍历攻击
            safe_filename = os.path.basename(subtitle_file.name)

            decoded = decode_subtitle_bytes(subtitle_file.getvalue())
            script_content = decoded.text
            detected_encoding = decoded.encoding

            if not script_content:
                st.error(tr("无法读取字幕文件，请检查文件编码（支持 UTF-8、UTF-16、GBK、GB2312）"))
                st.stop()

            # 验证字幕内容（简单检查）
            if len(script_content.strip()) < 10:
                st.warning(tr("字幕文件内容似乎为空，请检查文件"))

            # 保存到字幕目录
            script_file_path = os.path.join(utils.subtitle_dir(), safe_filename)
            file_name, file_extension = os.path.splitext(safe_filename)

            # 如果文件已存在,添加时间戳
            if os.path.exists(script_file_path):
                timestamp = time.strftime("%Y%m%d%H%M%S")
                file_name_with_timestamp = f"{file_name}_{timestamp}"
                script_file_path = os.path.join(utils.subtitle_dir(), file_name_with_timestamp + file_extension)

            # 直接写入SRT内容（统一使用 UTF-8）
            with open(script_file_path, "w", encoding='utf-8') as f:
                f.write(script_content)

            # 更新状态
            st.success(
                f"{tr('字幕上传成功')} "
                f"(编码: {detected_encoding.upper()}, "
                f"大小: {len(script_content)} 字符)"
            )
            st.session_state['subtitle_path'] = script_file_path
            st.session_state['subtitle_content'] = script_content
            st.session_state['subtitle_file_processed'] = True  # 标记已处理

            # 避免使用rerun，使用更新状态的方式
            # st.rerun()

        except Exception as e:
            st.error(f"{tr('Upload failed')}: {str(e)}")

    # 名称输入框
    video_theme = st.text_input(tr(work_name_label))
    st.session_state['video_theme'] = video_theme
    # 数字输入框
    temperature = st.slider("temperature", 0.0, 2.0, 0.7)
    st.session_state['temperature'] = temperature
    return video_theme


def render_fun_asr_transcription(tr):
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
        st.success(f"字幕转写成功（{provider_label}）: {os.path.basename(generated_path)}")
        return True

    with st.expander("字幕转录（三种方式 + 自动回退）", expanded=False):
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

        uploaded_media = st.file_uploader(
            tr("上传需要转录的音频/视频"),
            type=[
                "aac", "amr", "avi", "flac", "flv", "m4a", "mkv", "mov",
                "mp3", "mp4", "mpeg", "ogg", "opus", "wav", "webm", "wma", "wmv",
            ],
            accept_multiple_files=False,
            key="media_transcription_uploader",
        )

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

        if st.button("转写生成字幕", key="media_transcribe_btn", use_container_width=True):
            if uploaded_media is None:
                clear_subtitle_state()
                st.error("请先上传需要转录的音频或视频文件")
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

                temp_dir = utils.temp_dir("transcription")
                safe_filename = os.path.basename(uploaded_media.name)
                media_path = os.path.join(temp_dir, safe_filename)
                file_name, file_extension = os.path.splitext(safe_filename)
                if os.path.exists(media_path):
                    timestamp = time.strftime("%Y%m%d%H%M%S")
                    media_path = os.path.join(temp_dir, f"{file_name}_{timestamp}{file_extension}")

                with open(media_path, "wb") as f:
                    f.write(uploaded_media.getbuffer())

                subtitle_name = f"{os.path.splitext(os.path.basename(media_path))[0]}_transcribed.srt"
                subtitle_path = os.path.join(utils.subtitle_dir(), subtitle_name)

                with st.spinner("正在转写字幕，失败时将自动切换其他方式..."):
                    generated_path, used_provider = media_transcription.transcribe_media_to_srt(
                        media_path,
                        subtitle_path,
                        provider=provider_choice,
                        enable_fallback=enable_fallback,
                    )

                label = PROVIDER_LABELS.get(used_provider, used_provider)
                _apply_subtitle_result(generated_path, label)
            except Exception as e:
                clear_subtitle_state()
                logger.error(f"字幕转写失败: {traceback.format_exc()}")
                st.error(f"字幕转写失败（已尝试所有可用方式）: {str(e)}")


def render_script_buttons(tr, params):
    """渲染脚本操作按钮"""
    # 获取当前选择的脚本类型
    script_path = st.session_state.get('video_clip_json_path', '')

    # 生成/加载按钮
    if script_path == "auto":
        button_name = tr("Generate Video Script")
    elif script_path == "short":
        button_name = tr("Generate Short Video Script")
    elif script_path == "summary":
        button_name = tr("生成短剧解说脚本")
    elif script_path == "film_tv":
        button_name = tr("Generate Film TV Script")
    elif script_path.endswith("json"):
        button_name = tr("Load Video Script")
    else:
        button_name = tr("Please Select Script File")

    if st.button(button_name, key="script_action", disabled=not script_path):
        if script_path == "auto":
            # 执行纪录片视频脚本生成（视频无字幕无配音）
            generate_script_docu(params)
        elif script_path == "short":
            # 执行 短剧混剪 脚本生成
            custom_clips = st.session_state.get('custom_clips')
            generate_script_short(tr, params, custom_clips)
        elif script_path == "summary":
            # 执行 短剧解说 脚本生成
            subtitle_path = st.session_state.get('subtitle_path')
            video_theme = st.session_state.get('video_theme')
            temperature = st.session_state.get('temperature')
            generate_script_short_sunmmary(params, subtitle_path, video_theme, temperature)
        elif script_path == "film_tv":
            # 执行 影视解说 脚本生成
            subtitle_path = st.session_state.get('subtitle_path')
            video_theme = st.session_state.get('video_theme')
            temperature = st.session_state.get('temperature')
            film_tv_settings = st.session_state.get("film_tv_settings")
            generate_script_film_tv_summary(
                params, subtitle_path, video_theme, temperature, film_tv_settings=film_tv_settings
            )
        else:
            load_script(tr, script_path)

    # 视频脚本编辑区
    video_clip_json_details = st.text_area(
        tr("Video Script"),
        value=json.dumps(st.session_state.get('video_clip_json', []), indent=2, ensure_ascii=False),
        height=500
    )

    # 操作按钮行 - 合并格式检查和保存功能
    if st.button(tr("Save Script"), key="save_script", use_container_width=True):
        save_script_with_validation(tr, video_clip_json_details)


def load_script(tr, script_path):
    """加载脚本文件"""
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            script = f.read()
            script = utils.clean_model_output(script)
            st.session_state['video_clip_json'] = json.loads(script)
            st.success(tr("Script loaded successfully"))
            st.rerun()
    except Exception as e:
        logger.error(f"加载脚本文件时发生错误\n{traceback.format_exc()}")
        st.error(f"{tr('Failed to load script')}: {str(e)}")


def save_script_with_validation(tr, video_clip_json_details):
    """保存视频脚本（包含格式验证）"""
    if not video_clip_json_details:
        st.error(tr("请输入视频脚本"))
        st.stop()

    # 第一步：格式验证
    with st.spinner("正在验证脚本格式..."):
        try:
            result = check_script.check_format(video_clip_json_details)
            if not result.get('success'):
                # 格式验证失败，显示详细错误信息
                error_message = result.get('message', '未知错误')
                error_details = result.get('details', '')

                st.error(f"**脚本格式验证失败**")
                st.error(f"**错误信息：** {error_message}")
                if error_details:
                    st.error(f"**详细说明：** {error_details}")

                # 显示正确格式示例
                st.info("**正确的脚本格式示例：**")
                example_script = [
                    {
                        "_id": 1,
                        "timestamp": "00:00:00,600-00:00:07,559",
                        "picture": "工地上，蔡晓艳奋力救人，场面混乱",
                        "narration": "灾后重建，工地上险象环生！泼辣女工蔡晓艳挺身而出，救人第一！",
                        "OST": 0
                    },
                    {
                        "_id": 2,
                        "timestamp": "00:00:08,240-00:00:12,359",
                        "picture": "领导视察，蔡晓艳不屑一顾",
                        "narration": "播放原片4",
                        "OST": 1
                    }
                ]
                st.code(json.dumps(example_script, ensure_ascii=False, indent=2), language='json')
                st.stop()

        except Exception as e:
            st.error(f"格式验证过程中发生错误: {str(e)}")
            st.stop()

    # 第二步：保存脚本
    with st.spinner(tr("Save Script")):
        script_dir = utils.script_dir()
        timestamp = time.strftime("%Y-%m%d-%H%M%S")
        save_path = os.path.join(script_dir, f"{timestamp}.json")

        try:
            data = json.loads(video_clip_json_details)
            with open(save_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
                st.session_state['video_clip_json'] = data
                st.session_state['video_clip_json_path'] = save_path
                
                # 标记需要切换到文件选择模式（在下次渲染前处理）
                st.session_state['_switch_to_file_mode'] = True

                # 更新配置
                config.app["video_clip_json_path"] = save_path

                # 显示成功消息
                st.success("✅ 脚本格式验证通过，保存成功！")

                # 强制重新加载页面更新选择框
                time.sleep(0.5)  # 给一点时间让用户看到成功消息
                st.rerun()

        except Exception as err:
            st.error(f"{tr('Failed to save script')}: {str(err)}")
            st.stop()


# crop_video函数已移除 - 现在使用统一裁剪策略，不再需要预裁剪步骤


def get_script_params():
    """获取脚本参数"""
    return {
        'video_language': st.session_state.get('video_language', ''),
        'video_clip_json_path': st.session_state.get('video_clip_json_path', ''),
        'video_origin_path': st.session_state.get('video_origin_path', ''),
        'video_name': st.session_state.get('video_name', ''),
        'video_plot': st.session_state.get('video_plot', ''),
        'source_subtitle_path': st.session_state.get('subtitle_path', ''),
    }
