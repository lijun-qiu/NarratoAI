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
    load_subtitle_content,
)
from app.utils import utils, check_script
from webui.tools.generate_script_docu import (
    generate_plot_blueprint_docu,
    generate_script_docu,
)
from webui.tools.generate_script_short import generate_script_short
from webui.tools.generate_short_summary import (
    generate_plot_blueprint_short,
    generate_script_short_sunmmary,
)
from webui.tools.generate_film_tv_summary import generate_script_film_tv_summary
from webui.tools.plot_blueprint_workflow import (
    build_plot_blueprint_fingerprint,
    clear_plot_blueprint,
    commit_plot_blueprint_draft,
    get_plot_blueprint,
    is_plot_blueprint_ready,
    is_plot_blueprint_valid,
    render_plot_blueprint_panel,
    uses_plot_blueprint_workflow,
)
from webui.components.documentary_material_pickers import render_documentary_material_pickers
from webui.components.documentary_preprocess_panel import render_documentary_preprocess_panel
from webui.components.subtitle_transcription_settings import (
    render_documentary_subtitle_file_picker,
    render_fun_asr_transcription,
)
from webui.utils.script_stats import render_script_ost_summary
from app.services.documentary.documentary_settings import (
    get_documentary_compact_settings,
    get_documentary_settings,
    compute_ost1_segment_bounds,
)
from app.services.short_drama_settings import get_short_drama_settings
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
        elif script_path == "preprocess":
            # 素材预处理（字幕转录 / 抽帧分析 / 校准字幕）
            render_documentary_preprocess_panel(tr, params)
        else:
            # 默认为空
            pass

        # 两步生成：先展示剧情构思方案，再生成脚本
        script_path = st.session_state.get("video_clip_json_path", "")
        if uses_plot_blueprint_workflow(script_path):
            fingerprint = _compute_plot_blueprint_fingerprint(params, script_path)
            render_plot_blueprint_panel(fingerprint=fingerprint, mode=script_path)

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
    MODE_PREPROCESS = "preprocess"

    # 处理保存脚本后的模式切换（必须在 widget 实例化之前）
    if st.session_state.get('_switch_to_file_mode'):
        st.session_state['script_mode_selection'] = tr("Select/Upload Script")
        del st.session_state['_switch_to_file_mode']

    # 模式选项映射
    mode_options = {
        tr("Compact Frame Narration"): MODE_AUTO_COMPACT,
        tr("Auto Generate"): MODE_AUTO,
        tr("Material Preprocess"): MODE_PREPROCESS,
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
    elif current_path == "preprocess":
        default_index = mode_keys.index(tr("Material Preprocess"))
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
            if new_mode == MODE_SUMMARY:
                st.session_state["narration_workflow_mode"] = MODE_SUMMARY
                config.app["narration_workflow_mode"] = MODE_SUMMARY
            elif new_mode == MODE_FILM_TV:
                st.session_state["narration_workflow_mode"] = MODE_FILM_TV
                config.app["narration_workflow_mode"] = MODE_FILM_TV
            elif new_mode in (MODE_AUTO_COMPACT, MODE_AUTO):
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
            MODE_AUTO, MODE_AUTO_COMPACT, MODE_SHORT, MODE_SUMMARY, MODE_FILM_TV, MODE_PREPROCESS
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
    """所选视频变更时，自动关联字幕（含素材来源视频回退）。"""
    from app.services.documentary.documentary_material_resolver import (
        resolve_subtitle_path_for_documentary,
    )

    video_path = (video_path or "").strip()
    if not video_path or not os.path.isfile(video_path):
        return

    last_video = st.session_state.get("_subtitle_synced_video_path")
    material_source = (st.session_state.get("doc_material_source_video_path") or "").strip()
    explicit = (st.session_state.get("subtitle_path") or "").strip()

    if last_video == video_path and explicit and os.path.isfile(explicit):
        return

    if explicit and os.path.isfile(explicit) and os.path.getsize(explicit) > 0:
        content = load_subtitle_content(explicit).strip()
        if content:
            st.session_state["_subtitle_synced_video_path"] = video_path
            st.session_state["subtitle_content"] = content
            st.session_state["subtitle_file_processed"] = True
            return

    st.session_state["_subtitle_synced_video_path"] = video_path
    paired = resolve_subtitle_path_for_documentary(
        video_path,
        material_source_video_path=material_source,
        explicit_path=explicit or None,
    )
    if paired:
        content = load_subtitle_content(paired)
        if content.strip():
            st.session_state["subtitle_path"] = paired
            st.session_state["subtitle_content"] = content
            st.session_state["subtitle_file_processed"] = True
            return

    if last_video and last_video != video_path and not st.session_state.get("doc_subtitle_file_processed"):
        st.session_state.pop("doc_subtitle_path_input", None)
        st.session_state.pop("doc_subtitle_saved_pick", None)
    if (
        last_video
        and last_video != video_path
        and not material_source
        and not st.session_state.get("doc_subtitle_file_processed")
    ):
        st.session_state["subtitle_path"] = None
        st.session_state["subtitle_content"] = None
        st.session_state["subtitle_file_processed"] = False


def _resolve_active_subtitle_path() -> str:
    """当前生效的字幕路径（session、成片或素材来源视频配对）。"""
    from app.services.documentary.documentary_material_resolver import (
        resolve_subtitle_path_for_documentary,
    )

    video_path = (st.session_state.get("video_origin_path") or "").strip()
    if video_path:
        _sync_subtitle_with_video(video_path)
    material_source = (st.session_state.get("doc_material_source_video_path") or "").strip()
    return resolve_subtitle_path_for_documentary(
        video_path,
        material_source_video_path=material_source,
        explicit_path=st.session_state.get("subtitle_path"),
    )


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
        "transition_hook_template",
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
            "默认「逐帧精剪」：在下方选用或导入字幕与抽帧分析 JSON（也可在「素材预处理」完成）"
            "→ **① 生成剧情构思方案** 或 **在下方直接填写/修正** → 保存或确认 "
            "→ **② 生成 JSON 脚本**（有构思方案后无需字幕/抽帧）。视频主题填剧名集数（如《罚罪2》第1集）。"
        )
        with st.expander("转场句 / 结尾（可配置）", expanded=False):
            enable_hook = st.checkbox(
                "启用固定结尾道别",
                value=bool(doc_settings.get("enable_opening_closing_hook", True)),
                key="doc_compact_enable_opening_closing_hook",
                help="关闭后不在末段自动插入道别模板；开头高潮由模型按规则生成",
            )
            transition_tpl = st.text_input(
                "转场句模板（第 2 段「宝子们」之后）",
                value=str(
                    doc_settings.get("transition_hook_template")
                    or "故事，得从头讲起。"
                ),
                key="doc_compact_transition_hook_template",
                help="第 2 段 OST=0 以「宝子们」开头后接此句，再进入正叙。",
            )
            closing_tpl = st.text_input(
                "结尾道别模板",
                value=str(
                    doc_settings.get("closing_hook_template") or "宝子们，我们下期再见！"
                ),
                key="doc_compact_closing_hook_template",
                disabled=not enable_hook,
            )
            st.session_state["enable_opening_closing_hook"] = enable_hook
            st.session_state["transition_hook_template"] = transition_tpl.strip()
            st.session_state["opening_hook_template"] = ""
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
        prompt_help = "罚罪2 V2 完整规则，修改后参与脚本生成"
    else:
        default_prompt = str(doc_settings.get("default_custom_prompt") or "")
        if prompt_key not in st.session_state:
            st.session_state[prompt_key] = default_prompt
        prompt_height = 120
        prompt_help = tr("Custom prompt for LLM, leave empty to use default prompt")

    render_documentary_subtitle_options(tr, doc_settings, compact=compact)
    _render_documentary_material_status(tr)
    render_documentary_material_pickers(
        tr,
        expanded=compact,
        key_prefix="doc_compact" if compact else "doc_full",
    )

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
            "置于脚本生成 prompt **首位**（最高优先级），不参与抽帧视觉分析。"
            "联合构思可注入「剧集人物关系对照」（config: enable_subtitle_analysis_drama_knowledge）；"
            "抽帧分析默认不注入（enable_frame_analysis_drama_knowledge=false），避免脑补全剧名场面。"
            "适合写本集固定要求，如开头高潮名场面、必讲情节等。"
        ),
        height=72,
        key=append_key,
    )
    st.session_state["video_theme"] = video_theme
    st.session_state["custom_prompt"] = custom_prompt
    st.session_state["append_custom_prompt"] = append_prompt
    return video_theme, custom_prompt


def _render_documentary_material_status(tr) -> None:
    """短剧解说：展示字幕与整片视频分析就绪状态。"""
    from app.services.documentary.documentary_material_resolver import (
        resolve_subtitle_path_for_documentary,
        resolve_video_episode_analysis_path_for_documentary,
    )
    from webui.components.video_episode_analysis_settings import (
        sync_video_episode_analysis_with_video,
    )

    video_path = (st.session_state.get("video_origin_path") or "").strip()
    if not video_path:
        return

    sync_video_episode_analysis_with_video(video_path)
    material_source = (st.session_state.get("doc_material_source_video_path") or "").strip()
    subtitle_path = resolve_subtitle_path_for_documentary(
        video_path,
        material_source_video_path=material_source,
        explicit_path=st.session_state.get("subtitle_path"),
    )
    video_episode_path = resolve_video_episode_analysis_path_for_documentary(
        video_path,
        material_source_video_path=material_source,
        explicit_path=st.session_state.get("video_episode_analysis_json_path"),
    )
    cols = st.columns(2)
    with cols[0]:
        if subtitle_path and os.path.isfile(subtitle_path):
            st.success(f"字幕: {os.path.basename(subtitle_path)}")
        else:
            st.warning("字幕: 未就绪")
    with cols[1]:
        if video_episode_path and os.path.isfile(video_episode_path):
            st.success(f"视频分析: {os.path.basename(video_episode_path)}")
        else:
            st.warning("视频分析: 未就绪")

    st.caption(
        f"视频分析 JSON 会按当前视频自动配对；在「{tr('Material Preprocess')}」中生成或补全。"
    )


def render_documentary_subtitle_options(tr, doc_settings, *, compact: bool = False):
    """逐帧解说 / 精剪：字幕分析开关（素材准备在预处理模式中完成）。"""
    default_enabled = bool(doc_settings.get("enable_subtitle_enrichment", True))
    st.checkbox(
        "结合字幕分析（有 SRT 时与抽帧交叉验证）",
        value=st.session_state.get("doc_enable_subtitle_enrichment", default_enabled),
        key="doc_enable_subtitle_enrichment",
        help="上传/转录字幕后，生成脚本时会对照抽帧画面；并做字幕×画面对照分析",
    )


def _render_short_drama_video_analysis_block(tr) -> None:
    """短剧解说：整片视频分析 JSON 状态（自动配对当前视频）。"""
    from app.services.documentary.documentary_material_resolver import (
        resolve_video_episode_analysis_path_for_documentary,
    )
    from webui.components.video_episode_analysis_settings import (
        sync_video_episode_analysis_with_video,
    )

    video_path = (st.session_state.get("video_origin_path") or "").strip()
    if video_path:
        sync_video_episode_analysis_with_video(video_path)

    material_source = (st.session_state.get("doc_material_source_video_path") or "").strip()
    video_episode_explicit = (
        st.session_state.get("video_episode_analysis_json_path") or ""
    ).strip() or None
    video_episode_resolved = resolve_video_episode_analysis_path_for_documentary(
        video_path,
        material_source_video_path=material_source,
        explicit_path=video_episode_explicit,
    )
    st.markdown("**整片视频分析 JSON**（构思蓝图与脚本选材依据）")
    if video_episode_resolved and os.path.isfile(video_episode_resolved):
        source = "已确认选用" if video_episode_explicit else "自动配对"
        st.success(f"{source}: `{os.path.basename(video_episode_resolved)}`")
        st.session_state["video_episode_analysis_json_path"] = video_episode_resolved
    else:
        st.info("请先在「素材预处理 → 整片视频分析」生成 JSON，生成后将自动配对当前视频。")


def short_drama_summary(tr):
    """短剧解说 渲染视频主题和提示词（整片视频分析 + 字幕）。"""
    st.session_state.setdefault("narration_workflow_mode", "summary")
    config.app.setdefault("narration_workflow_mode", "summary")
    st.caption(
        "推荐流程：完成字幕转录与「整片视频分析」→ 下方选用字幕 → "
        "**① 自动生成** 或 **手动填写/修正** 剧情构思方案 → **② 生成 JSON 脚本**（有方案后无需重复分析）。"
    )
    _render_short_drama_video_analysis_block(tr)
    st.divider()
    _render_documentary_material_status(tr)
    with st.expander("字幕（从默认目录选择）", expanded=False):
        render_documentary_subtitle_file_picker(
            tr,
            path_input_key="sd_summary_subtitle_path_input",
            pick_key="sd_summary_subtitle_saved_pick",
            confirm_button_key="sd_summary_confirm_subtitle_path",
            clear_button_key="sd_summary_clear_subtitle",
            import_key="sd_summary_subtitle_uploader",
        )
    doc_settings = get_documentary_settings()
    default_enabled = bool(doc_settings.get("enable_subtitle_enrichment", True))
    st.checkbox(
        "结合整片视频分析（与字幕交叉验证，生成构思蓝图）",
        value=st.session_state.get("sd_enable_video_episode_analysis", default_enabled),
        key="sd_enable_video_episode_analysis",
        help="勾选后须先完成「素材预处理 → 整片视频分析」；取消勾选则构思蓝图仅用字幕",
    )
    return render_subtitle_narration_panel(
        tr,
        work_name_label="短剧名称",
        uploader_key="subtitle_file_uploader",
        temperature_default=float(
            get_short_drama_settings().get("narration_script_temperature", 0.4)
        ),
    )


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
    temperature_default: float = 0.7,
):
    """字幕解说类模式共用面板（短剧解说 / 影视解说）"""
    # 检查是否已经处理过字幕文件
    if 'subtitle_file_processed' not in st.session_state:
        st.session_state['subtitle_file_processed'] = False

    with st.expander("字幕转录（三种方式 + 自动回退）", expanded=False):
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
        temperature = st.slider(
            "temperature",
            0.0,
            2.0,
            float(st.session_state.get("temperature", temperature_default)),
        )
        st.session_state['temperature'] = temperature
    return video_theme


def _compute_plot_blueprint_fingerprint(params, script_path: str) -> str:
    subtitle_path = _resolve_active_subtitle_path()
    video_episode_analysis_path = (
        st.session_state.get("video_episode_analysis_json_path") or ""
    ).strip()
    enable_video = True
    if script_path == "summary":
        enable_video = bool(st.session_state.get("sd_enable_video_episode_analysis", True))
    return build_plot_blueprint_fingerprint(
        mode=script_path,
        video_path=(params.video_origin_path or "").strip(),
        subtitle_path=subtitle_path,
        analysis_path="",
        video_episode_analysis_path=video_episode_analysis_path,
        video_theme=str(st.session_state.get("video_theme") or ""),
        append_prompt=str(st.session_state.get("append_custom_prompt") or ""),
        enable_frame_analysis=enable_video,
    )


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
    elif script_path == "preprocess":
        button_name = tr("Material Preprocess Mode")
    elif script_path.endswith("json"):
        button_name = tr("Load Video Script")
    else:
        button_name = tr("Please Select Script File")

    is_preprocess = script_path == "preprocess"
    if uses_plot_blueprint_workflow(script_path):
        fingerprint = _compute_plot_blueprint_fingerprint(params, script_path)
        blueprint_ready = is_plot_blueprint_ready(fingerprint=fingerprint, mode=script_path)
        col_blueprint, col_script = st.columns(2)
        with col_blueprint:
            if st.button(
                "① 生成剧情构思方案",
                key="generate_plot_blueprint",
                disabled=is_preprocess,
                use_container_width=True,
            ):
                if script_path == "auto_compact":
                    generate_plot_blueprint_docu(params, compact=True, fingerprint=fingerprint)
                elif script_path == "summary":
                    subtitle_path = _resolve_active_subtitle_path()
                    video_theme = st.session_state.get("video_theme")
                    generate_plot_blueprint_short(
                        params,
                        subtitle_path,
                        video_theme,
                        fingerprint=fingerprint,
                    )
        with col_script:
            if st.button(
                "② 生成 JSON 脚本",
                key="generate_script_from_blueprint",
                disabled=is_preprocess or not blueprint_ready,
                use_container_width=True,
            ):
                try:
                    commit_plot_blueprint_draft(fingerprint=fingerprint, mode=script_path)
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
                blueprint = get_plot_blueprint()
                if script_path == "auto_compact":
                    generate_script_docu(params, compact=True, plot_blueprint=blueprint)
                elif script_path == "summary":
                    subtitle_path = _resolve_active_subtitle_path()
                    video_theme = st.session_state.get("video_theme")
                    temperature = st.session_state.get("temperature")
                    generate_script_short_sunmmary(
                        params,
                        subtitle_path,
                        video_theme,
                        temperature,
                        plot_blueprint=blueprint,
                    )
        if blueprint_ready and st.button(
            "清除构思方案（重新构思）",
            key="clear_plot_blueprint",
            use_container_width=True,
        ):
            clear_plot_blueprint()
            st.rerun()
    elif st.button(
        button_name,
        key="script_action",
        disabled=not script_path or is_preprocess,
    ):
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

    if is_preprocess:
        return

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


def _normalize_script_timestamp_fields(script_content: str) -> tuple[str, bool]:
    """保存前规范 timestamp 为 HH:MM:SS,mmm-HH:MM:SS,mmm。"""
    from app.services.srt_utils import normalize_script_timestamp_range

    payload = json.loads(script_content)
    if not isinstance(payload, list):
        return script_content, False
    changed = False
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_ts = str(item.get("timestamp") or "").strip()
        fixed_ts = normalize_script_timestamp_range(raw_ts)
        if fixed_ts != raw_ts:
            item["timestamp"] = fixed_ts
            changed = True
    if not changed:
        return script_content, False
    return json.dumps(payload, ensure_ascii=False, indent=2), True


def save_script_with_validation(tr, video_clip_json_details):
    """保存视频脚本（包含格式验证）"""
    if not video_clip_json_details:
        st.error(tr("请输入视频脚本"))
        st.stop()

    normalized_content, timestamps_fixed = _normalize_script_timestamp_fields(
        video_clip_json_details
    )
    if timestamps_fixed:
        st.session_state["video_clip_json"] = json.loads(normalized_content)
        video_clip_json_details = normalized_content
        st.info("已自动规范部分 timestamp 格式，正在重新验证…")

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
                workflow = st.session_state.get("narration_workflow_mode") or MODE_SUMMARY
                st.session_state["narration_workflow_mode"] = workflow
                config.app["narration_workflow_mode"] = workflow
                
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
        'narration_workflow_mode': (
            st.session_state.get('narration_workflow_mode')
            or config.app.get('narration_workflow_mode')
            or ''
        ),
        'video_origin_path': video_origin_path,
        'video_name': st.session_state.get('video_name', ''),
        'video_plot': st.session_state.get('video_plot', ''),
        'source_subtitle_path': subtitle_path,
    }
