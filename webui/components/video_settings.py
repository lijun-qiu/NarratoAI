import streamlit as st
from app.config import config
from app.models.schema import VideoClipParams, VideoAspect, AudioVolumeDefaults
from app.services.video_output_settings import (
    VIDEO_OUTPUT_DEFAULTS,
    get_video_output_settings,
    save_video_output_settings_to_config,
)


def render_video_panel(tr):
    """渲染视频配置面板"""
    with st.container(border=True):
        st.write(tr("Video Settings"))
        params = VideoClipParams()
        render_video_config(tr, params)
        render_video_output_settings(tr)


def render_video_config(tr, params):
    """渲染视频配置"""
    # 视频比例
    video_aspect_ratios = [
        (tr("Portrait"), VideoAspect.portrait.value),
        (tr("Landscape"), VideoAspect.landscape.value),
    ]
    selected_index = st.selectbox(
        tr("Video Ratio"),
        options=range(len(video_aspect_ratios)),
        format_func=lambda x: video_aspect_ratios[x][0],
    )
    params.video_aspect = VideoAspect(video_aspect_ratios[selected_index][1])
    st.session_state['video_aspect'] = params.video_aspect.value

    # 视频画质
    video_qualities = [
        ("4K (2160p)", "2160p"),
        ("2K (1440p)", "1440p"),
        ("Full HD (1080p)", "1080p"),
        ("HD (720p)", "720p"),
        ("SD (480p)", "480p"),
    ]
    quality_index = st.selectbox(
        tr("Video Quality"),
        options=range(len(video_qualities)),
        format_func=lambda x: video_qualities[x][0],
        index=2  # 默认选择 1080p
    )
    st.session_state['video_quality'] = video_qualities[quality_index][1]

    # 原声音量 - 使用统一的默认值
    params.original_volume = st.slider(
        tr("Original Volume"),
        min_value=AudioVolumeDefaults.MIN_VOLUME,
        max_value=AudioVolumeDefaults.MAX_VOLUME,
        value=AudioVolumeDefaults.ORIGINAL_VOLUME,
        step=0.01,
        help=tr("Adjust the volume of the original audio")
    )
    st.session_state['original_volume'] = params.original_volume


def render_video_output_settings(tr):
    """成片输出：水印与原声旁白字幕。"""
    defaults = get_video_output_settings()
    saved = st.session_state.get("video_output_settings")
    base = saved if isinstance(saved, dict) else defaults

    with st.expander("成片输出（水印 / 原声旁白）", expanded=False):
        st.caption("16:9 与 9:16 下，旁白字幕固定居中靠左；水印居中靠右，上下浮动约 10% 画面高度，缓慢漂移。")

        watermark_text = st.text_input(
            "水印文字",
            value=str(base.get("watermark_text", VIDEO_OUTPUT_DEFAULTS["watermark_text"])),
            help="留空则不添加水印",
            key="vo_watermark_text",
        )
        enable_picture_narration = st.checkbox(
            "原声段显示旁白描述字幕",
            value=bool(base.get("enable_picture_narration", True)),
            help="开启后，OST=1 原声段在画面居中靠左显示 picture 描述字幕",
            key="vo_enable_picture_narration",
        )

        c1, c2 = st.columns(2)
        with c1:
            picture_narration_font_size = st.slider(
                "旁白字幕字号",
                min_value=24,
                max_value=72,
                value=int(base.get("picture_narration_font_size", 44)),
                key="vo_picture_narration_font_size",
            )
        with c2:
            picture_narration_color = st.color_picker(
                "旁白字幕颜色",
                value=str(base.get("picture_narration_color", "#FFE066")),
                key="vo_picture_narration_color",
            )

        picture_narration_duration = st.slider(
            "旁白字幕停留时长（秒）",
            min_value=1.0,
            max_value=15.0,
            value=float(base.get("picture_narration_duration", 2.0)),
            step=0.5,
            help="每段 OST=1 原声开始时显示 picture 描述字幕的时长；原声段更长时字幕不会全程显示",
            key="vo_picture_narration_duration",
        )

        settings = {
            "watermark_text": watermark_text.strip(),
            "enable_picture_narration": enable_picture_narration,
            "picture_narration_font_size": picture_narration_font_size,
            "picture_narration_color": picture_narration_color,
            "picture_narration_max_chars": int(base.get("picture_narration_max_chars", 16)),
            "picture_narration_duration": picture_narration_duration,
        }
        st.session_state["video_output_settings"] = settings
        config.video_output = settings

        if st.button("保存为默认配置", key="vo_save_defaults"):
            if save_video_output_settings_to_config(settings):
                st.success("已保存到 config.toml [video_output]")


def get_video_params():
    """获取视频参数"""
    vo = st.session_state.get("video_output_settings") or get_video_output_settings()
    return {
        'video_aspect': st.session_state.get('video_aspect', VideoAspect.portrait.value),
        'video_quality': st.session_state.get('video_quality', '1080p'),
        'original_volume': st.session_state.get('original_volume', AudioVolumeDefaults.ORIGINAL_VOLUME),
        'watermark_text': vo.get('watermark_text', VIDEO_OUTPUT_DEFAULTS['watermark_text']),
        'enable_picture_narration': vo.get('enable_picture_narration', True),
    }
