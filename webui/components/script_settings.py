import os
import glob
import json
import time
import traceback
import streamlit as st
from loguru import logger

from app.config import config
from app.models.schema import VideoClipParams
from app.services.subtitle_text import decode_subtitle_bytes
from app.utils import utils, check_script
from webui.tools.generate_script_docu import generate_script_docu
from webui.tools.generate_script_short import generate_script_short
from webui.tools.generate_short_summary import generate_script_short_sunmmary
from webui.tools.generate_script_enhanced import generate_script_enhanced


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
        elif script_path == "enhanced":
            # 智能混剪解说
            render_enhanced_mix_options(tr)
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
    MODE_ENHANCED = "enhanced"
    SCRIPT_WORK_MODES = {MODE_AUTO, MODE_SHORT, MODE_SUMMARY, MODE_ENHANCED}

    if "script_work_mode" not in st.session_state:
        st.session_state["script_work_mode"] = MODE_ENHANCED
    if not st.session_state.get("video_clip_json_path"):
        st.session_state["video_clip_json_path"] = MODE_ENHANCED
        st.session_state["processing_mode"] = "enhanced"

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
        tr("Enhanced Mix Narration"): MODE_ENHANCED,
    }
    
    current_path = st.session_state.get("video_clip_json_path", MODE_ENHANCED)
    if current_path in SCRIPT_WORK_MODES:
        st.session_state["script_work_mode"] = current_path

    work_mode = st.session_state.get("script_work_mode", MODE_ENHANCED)
    mode_keys = list(mode_options.keys())

    if work_mode == MODE_AUTO:
        default_index = mode_keys.index(tr("Auto Generate"))
    elif work_mode == MODE_SHORT:
        default_index = mode_keys.index(tr("Short Generate"))
    elif work_mode == MODE_SUMMARY:
        default_index = mode_keys.index(tr("Short Drama Summary"))
    elif work_mode == MODE_ENHANCED:
        default_index = mode_keys.index(tr("Enhanced Mix Narration"))
    else:
        default_index = mode_keys.index(tr("Select/Upload Script"))

    # 1. 渲染功能选择下拉框
    # 使用 segmented_control 替代 selectbox，提供更好的视觉体验
    default_mode_label = mode_keys[default_index]
    
    # 定义回调函数来处理状态更新
    def update_script_mode():
        selected_label = st.session_state.script_mode_selection
        if selected_label:
            new_mode = mode_options[selected_label]
            st.session_state["script_work_mode"] = new_mode
            current = st.session_state.get("video_clip_json_path", "")
            if new_mode == MODE_FILE:
                restored = st.session_state.get("last_saved_script_json_path", "")
                if restored and os.path.isfile(restored):
                    st.session_state["video_clip_json_path"] = restored
                elif current and current not in SCRIPT_WORK_MODES and os.path.isfile(current):
                    st.session_state["video_clip_json_path"] = current
                else:
                    st.session_state["video_clip_json_path"] = ""
            else:
                if current and current not in SCRIPT_WORK_MODES and os.path.isfile(current):
                    st.session_state["last_saved_script_json_path"] = current
                st.session_state["video_clip_json_path"] = new_mode
                if new_mode == MODE_ENHANCED:
                    st.session_state["processing_mode"] = "enhanced"
                elif new_mode in (MODE_AUTO, MODE_SHORT, MODE_SUMMARY):
                    st.session_state["processing_mode"] = "standard"
            params.video_clip_json_path = st.session_state["video_clip_json_path"]
        else:
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
        saved_script_path = current_path if current_path not in SCRIPT_WORK_MODES else ""
        if saved_script_path:
            st.session_state["last_saved_script_json_path"] = saved_script_path
        
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
                    st.session_state["video_clip_json_path"] = script_file_path
                    st.session_state["last_saved_script_json_path"] = script_file_path
                    st.session_state["script_work_mode"] = MODE_FILE
                    params.video_clip_json_path = script_file_path
                    time.sleep(1)
                    st.rerun()

                except json.JSONDecodeError:
                    st.error(tr("Invalid JSON format"))
                except Exception as e:
                    st.error(f"{tr('Upload failed')}: {str(e)}")
    else:
        st.session_state["script_work_mode"] = selected_mode
        st.session_state["video_clip_json_path"] = selected_mode
        params.video_clip_json_path = selected_mode
        if selected_mode == MODE_ENHANCED:
            st.session_state["processing_mode"] = "enhanced"
        elif selected_mode in (MODE_AUTO, MODE_SHORT, MODE_SUMMARY):
            st.session_state["processing_mode"] = "standard"


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


def render_enhanced_mix_options(tr):
    """智能混剪解说：解说+原声混合、BGM、分段字幕"""
    short_drama_summary(tr, script_mode="enhanced")

    with st.expander(tr("Enhanced Mix Narration"), expanded=False):
        from app.utils.media_duration import get_video_duration_seconds
        from app.utils.enhanced_mix_duration import build_enhanced_mix_duration_plan

        video_path = st.session_state.get("video_origin_path")
        if video_path:
            video_sec = get_video_duration_seconds(video_path)
            if video_sec:
                plan = build_enhanced_mix_duration_plan(video_sec)
                st.caption(f"当前上传视频：{plan.plan_summary}")
        elif st.session_state.get("enhanced_mix_duration_plan"):
            st.caption(st.session_state["enhanced_mix_duration_plan"])
        else:
            st.caption("上传视频后将根据原片时长自动计算成片目标与片段数量。")
        st.info(tr("Enhanced Mix Mode Description"))

        if st.session_state.get("subtitle_path"):
            st.session_state["source_subtitle_path"] = st.session_state["subtitle_path"]
            st.session_state["processing_mode"] = "enhanced"

        mood_options = {
            tr("BGM Mood Auto"): "",
            tr("BGM Mood Suspense"): "suspense",
            tr("BGM Mood Emotional"): "emotional",
            tr("BGM Mood Action"): "action",
            tr("BGM Mood Comedy"): "comedy",
        }
        mood_labels = list(mood_options.keys())
        current_mood = st.session_state.get("bgm_mood", "")
        default_mood_index = next(
            (index for index, label in enumerate(mood_labels) if mood_options[label] == current_mood),
            0,
        )
        selected_mood_label = st.selectbox(
            tr("BGM Mood"),
            options=mood_labels,
            index=default_mood_index,
            help=tr("BGM Mood Help"),
        )
        st.session_state["bgm_mood"] = mood_options[selected_mood_label]


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


def short_drama_summary(tr, script_mode="summary"):
    """短剧解说 / 智能混剪解说：渲染字幕上传与作品名称"""
    # 检查是否已经处理过字幕文件
    if 'subtitle_file_processed' not in st.session_state:
        st.session_state['subtitle_file_processed'] = False

    with st.expander(tr("Subtitle Transcription"), expanded=False):
        render_whisper_transcription(tr)
        render_gemini_transcription(tr)
        render_fun_asr_transcription(tr)

    subtitle_file = st.file_uploader(
        tr("上传字幕文件"),
        type=["srt"],
        accept_multiple_files=False,
        key="subtitle_file_uploader"  # 添加唯一key
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

    # 作品名称输入框
    if script_mode == "enhanced":
        video_theme = st.text_input(
            tr("Movie or TV Show Name"),
            value=st.session_state.get("video_theme", ""),
            help=tr("Movie or TV Show Name Help"),
            placeholder=tr("Movie or TV Show Name Placeholder"),
        )
    else:
        video_theme = st.text_input(
            tr("Short Drama Name"),
            value=st.session_state.get("video_theme", ""),
        )
    st.session_state['video_theme'] = video_theme
    # 数字输入框
    temperature = st.slider("temperature", 0.0, 2.0, 0.7)
    st.session_state['temperature'] = temperature
    return video_theme


def render_fun_asr_transcription(tr):
    """使用阿里百炼 Fun-ASR 从本地音视频转写生成字幕。"""
    def clear_fun_asr_subtitle_state():
        st.session_state['subtitle_path'] = None
        st.session_state['subtitle_content'] = None
        st.session_state['subtitle_file_processed'] = False

    with st.expander("阿里百炼 Fun-ASR 字幕转录", expanded=False):
        st.caption("上传本地音频/视频后，将自动上传到阿里百炼临时存储并通过 fun-asr 生成 SRT 字幕。")
        st.markdown(
            "API Key 获取地址："
            "[https://bailian.console.aliyun.com/?tab=model#/api-key]"
            "(https://bailian.console.aliyun.com/?tab=model#/api-key)"
        )

        api_key = st.text_input(
            "阿里百炼 API Key",
            value=config.fun_asr.get("api_key", ""),
            type="password",
            help="请输入你自己的阿里百炼 API Key；保存配置后会写入本地 config.toml",
            key="fun_asr_api_key",
        )
        uploaded_media = st.file_uploader(
            "上传需要转录的音频/视频",
            type=[
                "aac", "amr", "avi", "flac", "flv", "m4a", "mkv", "mov",
                "mp3", "mp4", "mpeg", "ogg", "opus", "wav", "webm", "wma", "wmv",
            ],
            accept_multiple_files=False,
            key="fun_asr_media_uploader",
        )

        if st.button("转写生成字幕", key="fun_asr_transcribe"):
            if not api_key.strip():
                clear_fun_asr_subtitle_state()
                st.error("请先输入阿里百炼 API Key")
                return
            if uploaded_media is None:
                clear_fun_asr_subtitle_state()
                st.error("请先上传需要转录的音频或视频文件")
                return

            try:
                clear_fun_asr_subtitle_state()
                from app.services import fun_asr_subtitle

                config.fun_asr["api_key"] = api_key.strip()
                config.fun_asr["model"] = "fun-asr"
                config.save_config()

                temp_dir = utils.temp_dir("fun_asr")
                safe_filename = os.path.basename(uploaded_media.name)
                media_path = os.path.join(temp_dir, safe_filename)
                file_name, file_extension = os.path.splitext(safe_filename)
                if os.path.exists(media_path):
                    timestamp = time.strftime("%Y%m%d%H%M%S")
                    media_path = os.path.join(temp_dir, f"{file_name}_{timestamp}{file_extension}")

                with open(media_path, "wb") as f:
                    f.write(uploaded_media.getbuffer())

                subtitle_name = f"{os.path.splitext(os.path.basename(media_path))[0]}_fun_asr.srt"
                subtitle_path = os.path.join(utils.subtitle_dir(), subtitle_name)

                with st.spinner("正在使用阿里百炼 Fun-ASR 转写字幕，请稍候..."):
                    generated_path = fun_asr_subtitle.create_with_fun_asr(
                        local_file=media_path,
                        subtitle_file=subtitle_path,
                        api_key=api_key.strip(),
                    )

                if not generated_path or not os.path.exists(generated_path):
                    clear_fun_asr_subtitle_state()
                    st.error("Fun-ASR 转写失败：未生成字幕文件")
                    return

                with open(generated_path, "r", encoding="utf-8") as f:
                    subtitle_content = f.read()

                st.session_state['subtitle_path'] = generated_path
                st.session_state['subtitle_content'] = subtitle_content
                st.session_state['subtitle_file_processed'] = True
                st.success(f"字幕转写成功: {os.path.basename(generated_path)}")
            except Exception as e:
                clear_fun_asr_subtitle_state()
                logger.error(f"Fun-ASR 字幕转写失败: {traceback.format_exc()}")
                st.error(f"Fun-ASR 字幕转写失败: {str(e)}")


def render_gemini_transcription(tr):
    """使用 Gemini 从本地音视频转写生成字幕。"""
    def clear_gemini_subtitle_state():
        st.session_state['subtitle_path'] = None
        st.session_state['subtitle_content'] = None
        st.session_state['subtitle_file_processed'] = False

    default_api_key = (
        config.gemini_asr.get("api_key", "")
        or config.app.get("vision_openai_api_key", "")
    )
    default_model = (
        config.gemini_asr.get("model", "")
        or config.app.get("vision_openai_model_name", "gemini-2.0-flash")
    )
    default_base_url = (
        config.gemini_asr.get("base_url", "")
        or config.app.get("vision_openai_base_url", "")
    )
    default_provider = config.gemini_asr.get("provider", "auto") or "auto"

    with st.expander("Gemini 字幕转录", expanded=False):
        st.caption(
            "上传本地音频/视频，使用 Gemini 多模态模型转写为 SRT 字幕。"
            "若代理返回 502，将自动重试并回退到 Whisper API。"
        )
        st.warning("请确认视频含音轨。纯画面素材无法转写。")
        fallback_whisper = st.checkbox(
            "Gemini 失败时自动切换 Whisper",
            value=bool(config.gemini_asr.get("fallback_whisper", True)),
            key="gemini_asr_fallback_whisper",
        )

        api_key = st.text_input(
            "Gemini API Key",
            value=default_api_key,
            type="password",
            key="gemini_asr_api_key",
        )
        model_name = st.text_input(
            "Gemini 模型",
            value=default_model,
            help="如 gemini-2.0-flash、gemini-1.5-flash",
            key="gemini_asr_model_name",
        )
        base_url = st.text_input(
            "API Base URL（可选）",
            value=default_base_url,
            help="代理填写如 https://api.example.com/v1",
            key="gemini_asr_base_url",
        )
        provider = st.selectbox(
            "调用方式",
            options=["auto", "rest", "openai", "sdk"],
            index=["auto", "rest", "openai", "sdk"].index(default_provider)
            if default_provider in {"auto", "rest", "openai", "sdk"}
            else 0,
            help="国内 Gemini 代理通常选 auto 或 rest",
            key="gemini_asr_provider",
        )
        uploaded_media = st.file_uploader(
            "上传需要转录的音频/视频",
            type=[
                "aac", "amr", "avi", "flac", "flv", "m4a", "mkv", "mov",
                "mp3", "mp4", "mpeg", "ogg", "opus", "wav", "webm", "wma", "wmv",
            ],
            accept_multiple_files=False,
            key="gemini_asr_media_uploader",
        )

        if st.button("Gemini 转写生成字幕", key="gemini_asr_transcribe"):
            if not api_key.strip():
                clear_gemini_subtitle_state()
                st.error("请先输入 Gemini API Key")
                return
            if uploaded_media is None:
                clear_gemini_subtitle_state()
                st.error("请先上传需要转录的音频或视频文件")
                return

            try:
                clear_gemini_subtitle_state()
                from app.services import gemini_subtitle

                config.gemini_asr["api_key"] = api_key.strip()
                config.gemini_asr["model"] = model_name.strip() or "gemini-2.0-flash"
                config.gemini_asr["base_url"] = base_url.strip()
                config.gemini_asr["provider"] = provider
                config.gemini_asr["fallback_whisper"] = fallback_whisper
                config.save_config()

                temp_dir = utils.temp_dir("gemini_asr")
                safe_filename = os.path.basename(uploaded_media.name)
                media_path = os.path.join(temp_dir, safe_filename)
                file_name, file_extension = os.path.splitext(safe_filename)
                if os.path.exists(media_path):
                    timestamp = time.strftime("%Y%m%d%H%M%S")
                    media_path = os.path.join(temp_dir, f"{file_name}_{timestamp}{file_extension}")

                with open(media_path, "wb") as f:
                    f.write(uploaded_media.getbuffer())

                subtitle_name = f"{os.path.splitext(os.path.basename(media_path))[0]}_gemini.srt"
                subtitle_path = os.path.join(utils.subtitle_dir(), subtitle_name)

                with st.spinner("正在使用 Gemini 转写字幕，请稍候..."):
                    generated_path = gemini_subtitle.create_with_gemini(
                        local_file=media_path,
                        subtitle_file=subtitle_path,
                        api_key=api_key.strip(),
                        model_name=model_name.strip(),
                        base_url=base_url.strip(),
                        provider=provider,
                    )

                if not generated_path or not os.path.exists(generated_path):
                    clear_gemini_subtitle_state()
                    st.error("Gemini 转写失败：未生成字幕文件")
                    return

                with open(generated_path, "r", encoding="utf-8") as f:
                    subtitle_content = f.read()

                st.session_state['subtitle_path'] = generated_path
                st.session_state['subtitle_content'] = subtitle_content
                st.session_state['subtitle_file_processed'] = True
                st.success(f"字幕转写成功: {os.path.basename(generated_path)}")
            except Exception as e:
                clear_gemini_subtitle_state()
                logger.error(f"Gemini 字幕转写失败: {traceback.format_exc()}")
                st.error(f"Gemini 字幕转写失败: {str(e)}")


def render_whisper_transcription(tr):
    """使用 OpenAI Whisper API 转写字幕。"""
    def clear_whisper_subtitle_state():
        st.session_state['subtitle_path'] = None
        st.session_state['subtitle_content'] = None
        st.session_state['subtitle_file_processed'] = False

    default_api_key = (
        config.whisper_asr.get("api_key", "")
        or config.app.get("vision_openai_api_key", "")
    )
    default_base_url = (
        config.whisper_asr.get("base_url", "")
        or config.app.get("vision_openai_base_url", "")
    )

    with st.expander("Whisper 字幕转录（推荐）", expanded=False):
        st.caption("OpenAI Whisper 转写，中文稳定，适合代理网关不稳定时使用。")
        api_key = st.text_input(
            "API Key",
            value=default_api_key,
            type="password",
            key="whisper_asr_api_key",
        )
        base_url = st.text_input(
            "API Base URL",
            value=default_base_url,
            key="whisper_asr_base_url",
        )
        uploaded_media = st.file_uploader(
            "上传需要转录的音频/视频",
            type=[
                "aac", "amr", "avi", "flac", "flv", "m4a", "mkv", "mov",
                "mp3", "mp4", "mpeg", "ogg", "opus", "wav", "webm", "wma", "wmv",
            ],
            accept_multiple_files=False,
            key="whisper_asr_media_uploader",
        )
        if st.button("Whisper 转写生成字幕", key="whisper_asr_transcribe"):
            if not api_key.strip():
                clear_whisper_subtitle_state()
                st.error("请先输入 API Key")
                return
            if uploaded_media is None:
                clear_whisper_subtitle_state()
                st.error("请先上传需要转录的音频或视频文件")
                return
            try:
                clear_whisper_subtitle_state()
                from app.services import whisper_subtitle

                config.whisper_asr["api_key"] = api_key.strip()
                config.whisper_asr["base_url"] = base_url.strip()
                config.save_config()

                temp_dir = utils.temp_dir("whisper_asr")
                safe_filename = os.path.basename(uploaded_media.name)
                media_path = os.path.join(temp_dir, safe_filename)
                file_name, file_extension = os.path.splitext(safe_filename)
                if os.path.exists(media_path):
                    timestamp = time.strftime("%Y%m%d%H%M%S")
                    media_path = os.path.join(temp_dir, f"{file_name}_{timestamp}{file_extension}")

                with open(media_path, "wb") as f:
                    f.write(uploaded_media.getbuffer())

                subtitle_name = f"{os.path.splitext(os.path.basename(media_path))[0]}_whisper.srt"
                subtitle_path = os.path.join(utils.subtitle_dir(), subtitle_name)

                with st.spinner("正在使用 Whisper 转写字幕，请稍候..."):
                    generated_path = whisper_subtitle.create_with_whisper(
                        local_file=media_path,
                        subtitle_file=subtitle_path,
                        api_key=api_key.strip(),
                        base_url=base_url.strip(),
                    )

                if not generated_path or not os.path.exists(generated_path):
                    clear_whisper_subtitle_state()
                    st.error("Whisper 转写失败：未生成字幕文件")
                    return

                with open(generated_path, "r", encoding="utf-8") as f:
                    subtitle_content = f.read()

                st.session_state['subtitle_path'] = generated_path
                st.session_state['subtitle_content'] = subtitle_content
                st.session_state['subtitle_file_processed'] = True
                st.success(f"字幕转写成功: {os.path.basename(generated_path)}")
            except Exception as e:
                clear_whisper_subtitle_state()
                logger.error(f"Whisper 字幕转写失败: {traceback.format_exc()}")
                st.error(f"Whisper 字幕转写失败: {str(e)}")


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
    elif script_path == "enhanced":
        button_name = tr("Generate Enhanced Mix Script")
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
        elif script_path == "enhanced":
            subtitle_path = st.session_state.get('subtitle_path')
            video_theme = st.session_state.get('video_theme')
            temperature = st.session_state.get('temperature')
            generate_script_enhanced(params, subtitle_path, video_theme, temperature)
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
                st.session_state["last_saved_script_json_path"] = save_path

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
        'processing_mode': st.session_state.get('processing_mode', 'standard'),
        'source_subtitle_path': st.session_state.get('source_subtitle_path', ''),
        'bgm_mood': st.session_state.get('bgm_mood', ''),
    }
