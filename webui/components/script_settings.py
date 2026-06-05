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
from app.services.subtitle_video_pairing import (
    find_paired_subtitle_path,
    get_transcription_subtitle_path,
    load_subtitle_content,
    resolve_transcription_media_path,
)
from app.utils import utils, check_script
from webui.tools.generate_script_docu import generate_script_docu
from webui.tools.generate_script_short import generate_script_short
from webui.tools.generate_short_summary import generate_script_short_sunmmary
from webui.tools.generate_film_tv_summary import generate_script_film_tv_summary
from webui.components.frame_analysis_settings import render_frame_analysis_panel
from webui.utils.script_stats import render_script_ost_summary
from app.services.documentary.documentary_settings import get_documentary_compact_settings, compute_ost1_segment_bounds
from app.services.film_tv_settings import (
    FILM_TV_DEFAULTS,
    get_film_tv_settings,
    save_film_tv_settings_to_config,
)
from app.services.film_tv_rule_presets import (
    apply_preset_to_settings,
    get_film_tv_preset,
    get_preset_default_work_name,
    list_film_tv_presets,
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
            render_video_details(tr, params, compact=False)
        elif script_path == "auto_compact":
            # 逐帧精剪（纯解说快剪）
            render_video_details(tr, params, compact=True)
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
    MODE_AUTO_COMPACT = "auto_compact"
    MODE_SHORT = "short"
    MODE_SUMMARY = "summary"
    MODE_FILM_TV = "film_tv"

    # 处理保存脚本后的模式切换（必须在 widget 实例化之前）
    if st.session_state.get('_switch_to_file_mode'):
        st.session_state['script_mode_selection'] = tr("Select/Upload Script")
        del st.session_state['_switch_to_file_mode']

    # 模式选项映射
    mode_options = {
        tr("Compact Frame Narration"): MODE_AUTO_COMPACT,
        tr("Auto Generate"): MODE_AUTO,
        tr("Film TV Narration"): MODE_FILM_TV,
        tr("Short Generate"): MODE_SHORT,
        tr("Short Drama Summary"): MODE_SUMMARY,
        tr("Select/Upload Script"): MODE_FILE,
    }
    
    # 获取当前状态
    current_path = st.session_state.get('video_clip_json_path', '')
    
    # 确定当前选中的模式索引
    default_index = 0
    mode_keys = list(mode_options.keys())
    
    if current_path == "auto":
        default_index = mode_keys.index(tr("Auto Generate"))
    elif current_path == "auto_compact":
        default_index = mode_keys.index(tr("Compact Frame Narration"))
    elif current_path == "short":
        default_index = mode_keys.index(tr("Short Generate"))
    elif current_path == "summary":
        default_index = mode_keys.index(tr("Short Drama Summary"))
    elif current_path == "film_tv":
        default_index = mode_keys.index(tr("Film TV Narration"))
    elif not current_path:
        default_index = mode_keys.index(tr("Compact Frame Narration"))
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
            if new_mode == MODE_AUTO_COMPACT:
                st.session_state["documentary_script_mode"] = MODE_AUTO_COMPACT
                from app.services.documentary.documentary_settings import (
                    get_compact_custom_prompt_display,
                )

                prompt_key = "custom_prompt_input_compact"
                if prompt_key not in st.session_state:
                    st.session_state[prompt_key] = get_compact_custom_prompt_display()
            elif new_mode == MODE_AUTO:
                st.session_state["documentary_script_mode"] = MODE_AUTO
            else:
                st.session_state.pop("documentary_script_mode", None)
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
        saved_script_path = current_path if current_path not in [
            MODE_AUTO, MODE_AUTO_COMPACT, MODE_SHORT, MODE_SUMMARY, MODE_FILM_TV
        ] else ""
        
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


def _sync_subtitle_with_video(video_path: str) -> None:
    """所选视频变更时，自动关联同名/已转录字幕。"""
    video_path = (video_path or "").strip()
    if not video_path or not os.path.isfile(video_path):
        return

    last_video = st.session_state.get("_subtitle_synced_video_path")
    if last_video == video_path and st.session_state.get("subtitle_path"):
        return

    st.session_state["_subtitle_synced_video_path"] = video_path
    paired = find_paired_subtitle_path(video_path)
    if paired:
        content = load_subtitle_content(paired)
        if content.strip():
            st.session_state["subtitle_path"] = paired
            st.session_state["subtitle_content"] = content
            st.session_state["subtitle_file_processed"] = True
            return

    if last_video and last_video != video_path:
        st.session_state["subtitle_path"] = None
        st.session_state["subtitle_content"] = None
        st.session_state["subtitle_file_processed"] = False


def _resolve_active_subtitle_path() -> str:
    """当前生效的字幕路径（session 或视频配对字幕）。"""
    subtitle_path = (st.session_state.get("subtitle_path") or "").strip()
    if subtitle_path and os.path.isfile(subtitle_path):
        return subtitle_path
    video_path = (st.session_state.get("video_origin_path") or "").strip()
    if video_path:
        _sync_subtitle_with_video(video_path)
        subtitle_path = (st.session_state.get("subtitle_path") or "").strip()
        if subtitle_path:
            return subtitle_path
        paired = find_paired_subtitle_path(video_path)
        if paired:
            return paired
    return ""


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
    if video_path and video_path not in ("", "upload_local") and os.path.isfile(video_path):
        st.session_state['video_origin_path'] = video_path
        params.video_origin_path = video_path
        _sync_subtitle_with_video(video_path)
    elif video_path == "upload_local":
        params.video_origin_path = st.session_state.get('video_origin_path', '')
    else:
        st.session_state['video_origin_path'] = ''
        params.video_origin_path = ''

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
                _sync_subtitle_with_video(video_file_path)
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


def _ensure_doc_video_theme_default(doc_settings: dict, *, compact: bool) -> None:
    """逐帧解说/精剪：初始化视频主题默认值。"""
    theme_key = "doc_video_theme_compact" if compact else "doc_video_theme_full"
    default_theme = str(doc_settings.get("default_video_theme") or "罚罪2").strip()
    if theme_key not in st.session_state:
        existing = str(st.session_state.get("video_theme") or "").strip()
        st.session_state[theme_key] = existing or default_theme


def _apply_compact_hook_session_overrides(doc_settings: dict) -> dict:
    """将 WebUI 逐帧精剪开场/结尾配置合并进 settings。"""
    merged = dict(doc_settings)
    for key in (
        "enable_opening_closing_hook",
        "opening_hook_template",
        "closing_hook_template",
        "append_custom_prompt",
    ):
        if key in st.session_state:
            merged[key] = st.session_state[key]
    theme = str(st.session_state.get("doc_video_theme_compact") or "").strip()
    if theme:
        merged["default_video_theme"] = theme
    return merged


def render_video_details(tr, params, *, compact: bool = False):
    """画面解说 / 逐帧精剪：渲染视频主题和提示词"""
    from app.services.documentary.documentary_settings import (
        get_compact_custom_prompt_display,
        get_documentary_compact_settings,
        get_documentary_settings,
        save_documentary_compact_settings_to_config,
    )

    doc_settings = get_documentary_compact_settings() if compact else get_documentary_settings()
    if compact:
        doc_settings = _apply_compact_hook_session_overrides(doc_settings)
    default_interval = float(
        doc_settings.get("frame_interval_input")
        or config.frames.get("frame_interval_input", 3)
    )
    prompt_key = "custom_prompt_input_compact" if compact else "custom_prompt_input_full"

    if compact:
        if prompt_key not in st.session_state:
            st.session_state[prompt_key] = get_compact_custom_prompt_display(doc_settings)
        default_prompt = st.session_state[prompt_key]
        st.caption(
            "默认「逐帧精剪」：下方为完整规则（可改）；首尾招呼按模板自动生成。"
            "视频主题填剧名集数（如《罚罪2》第1集）。"
        )
        with st.expander("开场白 / 结尾（可配置）", expanded=False):
            enable_hook = st.checkbox(
                "启用固定开场白与结尾",
                value=bool(doc_settings.get("enable_opening_closing_hook", True)),
                key="doc_compact_enable_opening_closing_hook",
                help="关闭后不在首尾段自动插入模板；规则区也会标注为已关闭",
            )
            opening_tpl = st.text_input(
                "开场白模板（{work_name} 替换为视频主题）",
                value=str(
                    doc_settings.get("opening_hook_template")
                    or "宝子们，我们开始《{work_name}》啦！"
                ),
                key="doc_compact_opening_hook_template",
                disabled=not enable_hook,
            )
            closing_tpl = st.text_input(
                "结尾模板",
                value=str(
                    doc_settings.get("closing_hook_template") or "宝子们，我们下期再见！"
                ),
                key="doc_compact_closing_hook_template",
                disabled=not enable_hook,
            )
            st.session_state["enable_opening_closing_hook"] = enable_hook
            st.session_state["opening_hook_template"] = opening_tpl.strip()
            st.session_state["closing_hook_template"] = closing_tpl.strip()
            save_cols = st.columns([1, 3])
            with save_cols[0]:
                if st.button("保存到 config.toml", key="doc_compact_save_hooks"):
                    payload = _apply_compact_hook_session_overrides(
                        get_documentary_compact_settings()
                    )
                    if save_documentary_compact_settings_to_config(payload):
                        st.success("已保存 [documentary_compact]")
                    else:
                        st.error("保存失败，请查看日志")
        prompt_height = 260
        prompt_help = "故事讲述型完整规则，修改后参与脚本生成"
    else:
        default_prompt = str(doc_settings.get("default_custom_prompt") or "")
        if prompt_key not in st.session_state:
            st.session_state[prompt_key] = default_prompt
        prompt_height = 120
        prompt_help = tr("Custom prompt for LLM, leave empty to use default prompt")

    render_documentary_subtitle_options(tr, doc_settings, params=params, compact=compact)
    render_frame_analysis_panel(tr, params, compact=compact)

    _ensure_doc_video_theme_default(doc_settings, compact=compact)
    theme_key = "doc_video_theme_compact" if compact else "doc_video_theme_full"
    video_theme = st.text_input(
        tr("Video Theme"),
        key=theme_key,
        help="默认「罚罪2」；精剪模式建议写清集数，如《罚罪2》第1集",
    )
    if compact:
        reset_cols = st.columns([1, 4])
        with reset_cols[0]:
            if st.button("恢复默认规则", key="reset_compact_prompt_rules"):
                fresh = get_documentary_compact_settings()
                fresh = _apply_compact_hook_session_overrides(fresh)
                st.session_state[prompt_key] = get_compact_custom_prompt_display(fresh)
                st.rerun()
    custom_prompt = st.text_area(
        tr("Generation Prompt"),
        help=prompt_help,
        height=prompt_height,
        key=prompt_key,
    )
    append_key = "append_prompt_input_compact" if compact else "append_prompt_input_full"
    if append_key not in st.session_state:
        st.session_state[append_key] = str(doc_settings.get("append_custom_prompt") or "")
    append_prompt = st.text_area(
        "追加提示词",
        help=(
            "叠加在上方自定义提示词之后，仅参与脚本生成（不参与抽帧视觉分析）。"
            "适合写本集固定要求，如必讲情节、人物关系、留白段落等。"
        ),
        height=72,
        key=append_key,
    )
    st.session_state["video_theme"] = video_theme
    st.session_state["custom_prompt"] = custom_prompt
    st.session_state["append_custom_prompt"] = append_prompt
    return video_theme, custom_prompt


def render_documentary_subtitle_options(tr, doc_settings, *, params=None, compact: bool = False):
    """逐帧解说 / 精剪：可选字幕与抽帧结合。"""
    default_enabled = bool(doc_settings.get("enable_subtitle_enrichment", True))
    st.checkbox(
        "结合字幕分析（有 SRT 时与抽帧交叉验证）",
        value=st.session_state.get("doc_enable_subtitle_enrichment", default_enabled),
        key="doc_enable_subtitle_enrichment",
        help="上传/转录字幕后，抽帧分析会对照对白；并生成字幕×画面对照分析再写脚本",
    )
    if not st.session_state.get("doc_enable_subtitle_enrichment", default_enabled):
        return

    st.caption("请先完成字幕转写/上传，再执行下方「抽帧并分析」，以便一次完成硬字幕校准与脚本生成。")

    render_fun_asr_transcription(tr)

    if "doc_subtitle_file_processed" not in st.session_state:
        st.session_state["doc_subtitle_file_processed"] = False

    subtitle_file = st.file_uploader(
        tr("上传字幕文件"),
        type=["srt"],
        accept_multiple_files=False,
        key="docu_subtitle_uploader",
    )

    if st.session_state.get("subtitle_path"):
        st.info(f"已关联字幕: {os.path.basename(st.session_state['subtitle_path'])}")
        if st.button(tr("清除已上传字幕"), key="doc_clear_subtitle"):
            st.session_state["subtitle_path"] = None
            st.session_state["subtitle_content"] = None
            st.session_state["doc_subtitle_file_processed"] = False
            st.rerun()

    if subtitle_file is not None and not st.session_state.get("doc_subtitle_file_processed"):
        try:
            safe_filename = os.path.basename(subtitle_file.name)
            decoded = decode_subtitle_bytes(subtitle_file.getvalue())
            script_content = decoded.text
            if not script_content:
                st.error(tr("无法读取字幕文件，请检查文件编码（支持 UTF-8、UTF-16、GBK、GB2312）"))
                st.stop()

            script_dir = utils.script_dir()
            os.makedirs(script_dir, exist_ok=True)
            script_file_path = os.path.join(script_dir, safe_filename)
            if os.path.exists(script_file_path):
                timestamp = time.strftime("%Y%m%d%H%M%S")
                name, ext = os.path.splitext(safe_filename)
                script_file_path = os.path.join(script_dir, f"{name}_{timestamp}{ext}")

            with open(script_file_path, "w", encoding="utf-8") as f:
                f.write(script_content)

            st.session_state["subtitle_path"] = script_file_path
            st.session_state["subtitle_content"] = script_content
            st.session_state["doc_subtitle_file_processed"] = True
            st.success(f"字幕已保存: {safe_filename}")
            st.rerun()
        except Exception as e:
            st.error(f"{tr('Upload failed')}: {str(e)}")

    if doc_settings.get("enable_subtitle_refinement", True):
        _render_subtitle_refinement_panel(tr, params=params, compact=compact)


def _render_subtitle_refinement_panel(tr, *, params=None, compact: bool = False):
    """对照抽帧分析校正 ASR 字幕（产出 *_refined.srt / *_ocr_refined.srt）。"""
    from app.services.documentary.documentary_settings import get_documentary_compact_settings, get_documentary_settings
    from app.services.documentary.frame_analysis_pairing import find_paired_frame_analysis_path
    from app.services.documentary.hard_subtitle_ocr_service import get_ocr_refined_subtitle_path
    from app.services.documentary.subtitle_refinement_service import get_refined_subtitle_path
    from webui.tools.ocr_calibrate_subtitle_docu import ocr_calibrate_subtitle_docu
    from webui.tools.refine_subtitle_docu import refine_subtitle_docu

    video_path = (st.session_state.get("video_origin_path") or "").strip()
    if not video_path:
        return

    doc_settings = get_documentary_compact_settings() if compact else get_documentary_settings()

    with st.expander("对照抽帧校正字幕（手动重跑）", expanded=False):
        st.caption(
            "抽帧/上传分析 JSON 时会**自动**完成硬字幕 OCR + LLM 校正（与抽帧同一次视觉调用，不重复计费）。"
            "此处按钮仅用于手动重跑。优先使用 `*_ocr_refined.srt` / `*_refined.srt`。"
        )

        ocr_path = get_ocr_refined_subtitle_path(video_path)
        refined_path = get_refined_subtitle_path(video_path)
        if ocr_path and os.path.isfile(ocr_path):
            st.success(f"已有 OCR 校准字幕: **{os.path.basename(ocr_path)}**")
        elif refined_path and os.path.isfile(refined_path):
            st.success(f"已有 LLM 校正字幕: **{os.path.basename(refined_path)}**")
        else:
            st.caption("尚未生成校正字幕")

        analysis_path = (st.session_state.get("frame_analysis_json_path") or "").strip()
        if not analysis_path or not os.path.isfile(analysis_path):
            analysis_path = find_paired_frame_analysis_path(video_path) or ""
        if analysis_path:
            st.caption(f"将对照: `{os.path.basename(analysis_path)}`")
        else:
            st.warning("请先完成抽帧分析，再校正字幕")

        subtitle_ready = bool(
            (st.session_state.get("subtitle_path") or find_paired_subtitle_path(video_path))
        )
        can_refine = bool(analysis_path and subtitle_ready and params is not None)

        if doc_settings.get("enable_hard_subtitle_ocr", True):
            st.caption(
                "手动重跑 OCR：旧版 JSON 无 burned_in_subtitle 时会二次调用视觉模型裁剪 OCR。"
            )
            if st.button(
                "手动重跑硬字幕 OCR",
                key="doc_ocr_calibrate_subtitle_btn",
                use_container_width=True,
                disabled=not can_refine,
            ):
                ocr_calibrate_subtitle_docu(params, compact=compact)

        if doc_settings.get("enable_subtitle_refinement", True):
            if st.button(
                "手动重跑 LLM 校正",
                key="doc_refine_subtitle_btn",
                use_container_width=True,
                disabled=not can_refine,
            ):
                refine_subtitle_docu(params, compact=compact)


def short_drama_summary(tr):
    """短剧解说 渲染视频主题和提示词"""
    return render_subtitle_narration_panel(tr, work_name_label="短剧名称", uploader_key="subtitle_file_uploader")


def film_tv_narration(tr):
    """影视解说 渲染视频主题和提示词"""
    render_subtitle_narration_panel(
        tr,
        work_name_label="Film Title",
        uploader_key="film_tv_subtitle_uploader",
        show_work_name=False,
        show_temperature=False,
    )
    selected_preset_id = render_film_tv_rules_settings(tr)
    video_theme = render_film_tv_work_name(tr, selected_preset_id)
    temperature = st.slider(
        "temperature",
        0.0,
        2.0,
        float(st.session_state.get("temperature", 0.7)),
        key="film_tv_temperature",
    )
    st.session_state["temperature"] = temperature
    return video_theme


def _sync_work_name_from_preset(preset_id: str) -> None:
    """切换专题方案时，自动填入该方案绑定的默认作品名。"""
    default_name = get_preset_default_work_name(preset_id)
    last_preset_id = st.session_state.get("film_tv_last_preset_id")
    if preset_id != last_preset_id:
        st.session_state["film_tv_last_preset_id"] = preset_id
        if default_name:
            st.session_state["film_tv_video_theme"] = default_name
            st.session_state["video_theme"] = default_name
    elif default_name and not str(st.session_state.get("video_theme") or "").strip():
        st.session_state.setdefault("film_tv_video_theme", default_name)
        st.session_state["video_theme"] = default_name


def render_film_tv_work_name(tr, preset_id: str) -> str:
    """影视作品名称输入（随专题方案自动填充默认剧名）。"""
    if "film_tv_video_theme" not in st.session_state:
        st.session_state["film_tv_video_theme"] = st.session_state.get("video_theme", "")
    _sync_work_name_from_preset(preset_id)
    default_hint = get_preset_default_work_name(preset_id)
    if default_hint:
        st.caption(f"当前方案默认作品名：**{default_hint}**（可手动修改）")
    video_theme = st.text_input(
        tr("Film Title"),
        key="film_tv_video_theme",
    )
    st.session_state["video_theme"] = video_theme
    return video_theme


def render_film_tv_rules_settings(tr) -> str:
    """影视解说规则参数调节面板（模块化方案 + 细调）。"""
    defaults = get_film_tv_settings()
    saved = st.session_state.get("film_tv_settings")
    base = saved if isinstance(saved, dict) else defaults
    current_preset_id = base.get("preset_id") or defaults.get("preset_id")

    def _clamp(value, lo, hi):
        return max(lo, min(hi, int(value)))

    with st.expander("影视解说规则方案", expanded=True):
        st.caption("勾选一套方案后，数值参数与 AI 剪辑法则一并生效；下方滑块可微调。")

        presets = list_film_tv_presets()
        preset_ids = [p["id"] for p in presets]
        if current_preset_id not in preset_ids:
            current_preset_id = preset_ids[0]

        selected_preset_id = st.radio(
            "选择剪辑方案",
            options=preset_ids,
            index=preset_ids.index(current_preset_id),
            format_func=lambda pid: next(p["name"] for p in presets if p["id"] == pid),
            key="ftv_preset_radio",
            horizontal=True,
        )

        active_preset = get_film_tv_preset(selected_preset_id) or {}
        st.info(f"**{active_preset.get('name', '')}** · {active_preset.get('subtitle', '')}")
        st.markdown(active_preset.get("description", ""))

        if st.checkbox("展开查看本方案剪辑师法则（将写入 AI 提示词）", key="ftv_show_preset_law"):
            st.markdown(f"**剪辑师身份**\n\n{active_preset.get('editor_persona', '')}")
            st.markdown(f"**专项法则**\n\n{active_preset.get('style_directive', '')}")

        if selected_preset_id != base.get("preset_id"):
            base = apply_preset_to_settings(base, selected_preset_id)

        default_work = get_preset_default_work_name(selected_preset_id)
        if default_work:
            st.caption(f"作品名将默认填入：**{default_work}**")

    with st.expander("影视解说规则参数", expanded=False):
        st.caption(
            "调节生成脚本与后处理规则；「最少段数」会写入 AI 提示词并在生成后校验，未达标时自动重试。"
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
            max_total_segments = st.slider(
                "总分段上限", 20, 50, _clamp(int(base.get("max_total_segments", 36)), 20, 50),
                help="OST=0+OST=1 合计，超出会导致成片过长（罚罪2 建议 36）",
                key="ftv_max_total_segments",
            )
            min_total_segments = st.slider(
                "总分段下限", 20, 50, _clamp(int(base.get("min_total_segments", 30)), 20, 50),
                help="OST=0+OST=1 合计，低于此值会触发补段（罚罪2 建议 30）",
                key="ftv_min_total_segments",
            )
            picture_chars_max = st.slider(
                "原声旁白字数上限", 6, 24, _clamp(int(base.get("picture_chars_max", 12)), 6, 24),
                help="OST=1 原声段左侧 picture 字幕，精简承上启下",
                key="ftv_picture_chars_max",
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

        st.markdown("**开场白 / 结尾**")
        enable_opening_closing_hook = st.checkbox(
            "启用固定开场白与结尾",
            value=bool(base.get("enable_opening_closing_hook", True)),
            key="ftv_enable_opening_closing_hook",
            help="开启后，首段解说替换为开场白，末段解说替换为结尾（在视觉优化之后写入）",
        )
        opening_hook_template = st.text_area(
            "开场白模板（首段仅短招呼，悬念解说由模型生成）",
            value=str(base.get("opening_hook_template") or "宝子们，今天咱们一起追《{work_name}》。"),
            key="ftv_opening_hook_template",
            disabled=not enable_opening_closing_hook,
            help="仅作简短招呼，会与首段悬念剧情解说合并；不要写「开看之前先捋主线」类引导",
            height=68,
        )
        closing_hook_template = st.text_area(
            "结尾模板（末段 OST=0，含本集总结+道别）",
            value=str(
                base.get("closing_hook_template")
                or "本集的核心冲突、留下的悬念和下一集的火药桶，就先帮大家梳理到这儿。宝子们，觉得讲清楚了点个赞，咱们下期再见。"
            ),
            key="ftv_closing_hook_template",
            disabled=not enable_opening_closing_hook,
            help="会与模型生成的末段总结合并；若已有总结则只补道别",
            height=80,
        )

        st.markdown("**视觉模型增强**（字幕 + 关键帧，使用「基础设置」中的 vision 模型）")
        enable_vision_enrichment = st.checkbox(
            "启用视觉模型辅助（推荐罚罪2等悬疑剧）",
            value=bool(base.get("enable_vision_enrichment", True)),
            key="ftv_enable_vision_enrichment",
            help="文字模型分析字幕后，视觉模型抽帧补充场面信息，并优化 picture 旁白描述",
        )
        if enable_vision_enrichment:
            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                vision_scene_interval_sec = st.slider(
                    "剧情拉片间隔 (秒)",
                    15, 120, _clamp(base.get("vision_scene_interval_sec", 30), 15, 120),
                    key="ftv_vision_scene_interval_sec",
                    help="剧情分析阶段抽帧间隔，默认 30 秒一帧并对照字幕",
                )
            with vc2:
                vision_max_scene_samples = st.slider(
                    "剧情拉片最多帧数",
                    20, 100, _clamp(base.get("vision_max_scene_samples", 80), 20, 100),
                    key="ftv_vision_max_scene_samples",
                )
            with vc3:
                vision_segment_max_items = st.slider(
                    "旁白优化最多片段数",
                    10, 50, _clamp(base.get("vision_segment_max_items", base.get("vision_picture_max_items", 30)), 10, 50),
                    key="ftv_vision_segment_max_items",
                )
            vision_enrich_picture = st.checkbox(
                "优化原声段 picture 旁白（对照画面）",
                value=bool(base.get("vision_enrich_picture", True)),
                key="ftv_vision_enrich_picture",
            )
            vision_enrich_narration = st.checkbox(
                "优化解说段 narration 文案（对照画面，更贴视频）",
                value=bool(base.get("vision_enrich_narration", True)),
                key="ftv_vision_enrich_narration",
            )
        else:
            vision_scene_interval_sec = int(base.get("vision_scene_interval_sec", 30))
            vision_max_scene_samples = int(base.get("vision_max_scene_samples", 80))
            vision_segment_max_items = int(
                base.get("vision_segment_max_items", base.get("vision_picture_max_items", 30))
            )
            vision_enrich_picture = bool(base.get("vision_enrich_picture", True))
            vision_enrich_narration = bool(base.get("vision_enrich_narration", True))

        if ost1_duration_min > ost1_duration_max:
            st.warning("原声最短时长不能大于最长时长，生成时将自动对调。")
        if ost1_segment_min > ost1_segment_max:
            st.warning("原声段数最少不能大于最多，生成时将自动对调。")
        if ost0_segment_min > ost0_segment_max:
            st.warning("解说段数最少不能大于最多，生成时将自动对调。")
        if narration_chars_min > narration_chars_max:
            st.warning("解说字数下限不能大于上限，生成时将自动对调。")

        settings = {
            "preset_id": selected_preset_id,
            "target_duration_percent": target_duration_percent,
            "ost1_duration_min": min(ost1_duration_min, ost1_duration_max),
            "ost1_duration_max": max(ost1_duration_min, ost1_duration_max),
            "ost1_duration_long_max": ost1_duration_long_max,
            "ost1_segment_min": min(ost1_segment_min, ost1_segment_max),
            "ost1_segment_max": max(ost1_segment_min, ost1_segment_max),
            "ost0_segment_min": min(ost0_segment_min, ost0_segment_max),
            "ost0_segment_max": max(ost0_segment_min, ost0_segment_max),
            "max_total_segments": max_total_segments,
            "min_total_segments": min(min_total_segments, max_total_segments),
            "picture_chars_max": picture_chars_max,
            "original_audio_percent": original_audio_percent,
            "narration_percent": narration_percent,
            "narration_chars_min": min(narration_chars_min, narration_chars_max),
            "narration_chars_max": max(narration_chars_min, narration_chars_max),
            "opening_chars_max": opening_chars_max,
            "allow_consecutive_ost1": allow_consecutive_ost1,
            "enforce_narration_after_ost1": enforce_narration_after_ost1,
            "enable_opening_closing_hook": enable_opening_closing_hook,
            "opening_hook_template": opening_hook_template.strip(),
            "closing_hook_template": closing_hook_template.strip(),
            "enable_vision_enrichment": enable_vision_enrichment,
            "vision_scene_interval_sec": vision_scene_interval_sec,
            "vision_max_scene_samples": vision_max_scene_samples,
            "vision_enrich_picture": vision_enrich_picture,
            "vision_enrich_narration": vision_enrich_narration,
            "vision_picture_max_items": vision_segment_max_items,
            "vision_segment_max_items": vision_segment_max_items,
        }
        st.session_state["film_tv_settings"] = settings

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("恢复默认规则", key="ftv_reset_defaults", use_container_width=True):
                st.session_state["film_tv_settings"] = apply_preset_to_settings(
                    deepcopy(FILM_TV_DEFAULTS), FILM_TV_DEFAULTS.get("preset_id")
                )
                st.rerun()
        with btn2:
            if st.button("保存为 config.toml 默认", key="ftv_save_config", use_container_width=True):
                if save_film_tv_settings_to_config(settings):
                    st.success("已保存到 config.toml [film_tv]")
                else:
                    st.error("保存失败，请查看日志")

    return selected_preset_id


def render_subtitle_narration_panel(
    tr,
    work_name_label: str,
    uploader_key: str,
    *,
    show_work_name: bool = True,
    show_temperature: bool = True,
):
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
    video_theme = ""
    if show_work_name:
        video_theme = st.text_input(tr(work_name_label))
        st.session_state['video_theme'] = video_theme
    if show_temperature:
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

                label = PROVIDER_LABELS.get(used_provider, used_provider)
                if _apply_subtitle_result(generated_path, label):
                    st.session_state["_subtitle_synced_video_path"] = video_origin_path or media_path
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
    elif script_path == "auto_compact":
        button_name = tr("Generate Compact Frame Script")
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
        elif script_path == "auto_compact":
            generate_script_docu(params, compact=True)
        elif script_path == "short":
            # 执行 短剧混剪 脚本生成
            custom_clips = st.session_state.get('custom_clips')
            generate_script_short(tr, params, custom_clips)
        elif script_path == "summary":
            # 执行 短剧解说 脚本生成
            subtitle_path = _resolve_active_subtitle_path()
            video_theme = st.session_state.get('video_theme')
            temperature = st.session_state.get('temperature')
            generate_script_short_sunmmary(params, subtitle_path, video_theme, temperature)
        elif script_path == "film_tv":
            # 执行 影视解说 脚本生成
            subtitle_path = _resolve_active_subtitle_path()
            video_theme = st.session_state.get('video_theme')
            temperature = st.session_state.get('temperature')
            film_tv_settings = st.session_state.get("film_tv_settings")
            generate_script_film_tv_summary(
                params, subtitle_path, video_theme, temperature, film_tv_settings=film_tv_settings
            )
        else:
            load_script(tr, script_path)

    # 视频脚本编辑区
    script_items = st.session_state.get("video_clip_json") or []
    script_path = st.session_state.get("video_clip_json_path", "")
    min_ost1_hint = None
    max_ost1_hint = None
    if script_path == "auto_compact":
        min_ost1_hint, max_ost1_hint = compute_ost1_segment_bounds(
            len(script_items), get_documentary_compact_settings()
        )
    render_script_ost_summary(
        script_items, min_ost1=min_ost1_hint, max_ost1=max_ost1_hint
    )

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
    video_origin_path = st.session_state.get('video_origin_path', '')
    if video_origin_path and os.path.isfile(video_origin_path):
        _sync_subtitle_with_video(video_origin_path)

    subtitle_path = st.session_state.get('subtitle_path', '')
    if not subtitle_path and video_origin_path:
        subtitle_path = find_paired_subtitle_path(video_origin_path) or ""

    return {
        'video_language': st.session_state.get('video_language', ''),
        'video_clip_json_path': st.session_state.get('video_clip_json_path', ''),
        'video_origin_path': video_origin_path,
        'video_name': st.session_state.get('video_name', ''),
        'video_plot': st.session_state.get('video_plot', ''),
        'source_subtitle_path': subtitle_path,
    }
