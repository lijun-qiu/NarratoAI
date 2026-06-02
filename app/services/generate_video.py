#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project: NarratoAI
@File   : generate_video
@Author : Viccy同学
@Date   : 2025/5/7 上午11:55 
'''

import os
import math
import traceback
import tempfile
from typing import Optional, Dict, Any
from loguru import logger
from moviepy import (
    VideoFileClip,
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    TextClip,
    afx
)
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import ImageFont

from app.utils import utils
from app.models.schema import AudioVolumeDefaults, VideoAspect
from app.services.audio_normalizer import AudioNormalizer, normalize_audio_for_mixing


def _is_landscape_video(video_width: float, video_height: float, video_aspect: Any = None) -> bool:
    """根据用户选择的画幅比例判断横屏/竖屏布局。"""
    aspect_value = video_aspect.value if hasattr(video_aspect, "value") else video_aspect
    aspect_text = str(aspect_value or "").strip().lower()
    if aspect_text in {VideoAspect.landscape.value, VideoAspect.landscape_2.value, "16:9", "4:3"}:
        return True
    if aspect_text in {VideoAspect.portrait.value, VideoAspect.portrait_2.value, "9:16", "3:4"}:
        return False
    if video_width > 0 and video_height > 0:
        return video_width / video_height >= 1.55
    return False


def _fixed_center_right_position(
    clip_w: float,
    clip_h: float,
    canvas_w: float,
    canvas_h: float,
) -> tuple[float, float]:
    """垂直居中、水平靠右半区中部（16:9 / 9:16 通用）。"""
    margin = max(12, int(min(canvas_w, canvas_h) * 0.02))
    pos_y = (canvas_h - clip_h) / 2
    right_zone_center_x = canvas_w * 0.75
    pos_x = right_zone_center_x - clip_w / 2
    pos_x = max(canvas_w * 0.5 + margin, min(pos_x, canvas_w - clip_w - margin))
    return pos_x, pos_y


def _fixed_center_left_position(
    clip_w: float,
    clip_h: float,
    canvas_w: float,
    canvas_h: float,
) -> tuple[float, float]:
    """垂直居中、水平靠左半区中部（16:9 / 9:16 通用）。"""
    margin = max(12, int(min(canvas_w, canvas_h) * 0.02))
    pos_y = (canvas_h - clip_h) / 2
    left_zone_center_x = canvas_w * 0.25
    pos_x = left_zone_center_x - clip_w / 2
    pos_x = max(margin, min(pos_x, canvas_w * 0.5 - clip_w - margin))
    return pos_x, pos_y


def is_valid_subtitle_file(subtitle_path: str) -> bool:
    """
    检查字幕文件是否有效

    参数:
        subtitle_path: 字幕文件路径

    返回:
        bool: 如果字幕文件存在且包含有效内容则返回True，否则返回False
    """
    if not subtitle_path or not os.path.exists(subtitle_path):
        return False

    try:
        with open(subtitle_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # 检查文件是否为空
        if not content:
            return False

        # 检查是否包含时间戳格式（SRT格式的基本特征）
        # SRT格式应该包含类似 "00:00:00,000 --> 00:00:00,000" 的时间戳
        import re
        time_pattern = r'\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}'
        if not re.search(time_pattern, content):
            return False

        return True
    except Exception as e:
        logger.warning(f"检查字幕文件时出错: {str(e)}")
        return False


def _resolve_subtitle_font_path(subtitle_font: str) -> Optional[str]:
    if not subtitle_font:
        return None
    font_path = os.path.join(utils.font_dir(), subtitle_font)
    if os.name == "nt":
        font_path = font_path.replace("\\", "/")
    return font_path


def _parse_subtitle_style_options(options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """解析字幕样式选项，供合成与延迟烧录共用。"""
    options = options or {}
    subtitle_bg_color = options.get("subtitle_bg_color", "transparent")
    if subtitle_bg_color == "transparent":
        subtitle_bg_color = None
    try:
        custom_position = float(options.get("custom_position", 60))
    except (TypeError, ValueError):
        custom_position = 60.0
    try:
        stroke_width = float(options.get("stroke_width", 1))
    except (TypeError, ValueError):
        stroke_width = 1.0
    return {
        "subtitle_font": options.get("subtitle_font", ""),
        "subtitle_font_size": options.get("subtitle_font_size", 40),
        "subtitle_color": options.get("subtitle_color", "#FFFFFF"),
        "subtitle_bg_color": subtitle_bg_color,
        "subtitle_position": options.get("subtitle_position", "custom"),
        "custom_position": custom_position,
        "stroke_color": options.get("stroke_color", "#000000"),
        "stroke_width": stroke_width,
        "picture_narration_font_size": options.get("picture_narration_font_size", 44),
        "picture_narration_color": options.get("picture_narration_color", "#FFE066"),
        "video_aspect": options.get("video_aspect"),
    }


def _position_subtitle_clip(
    clip,
    *,
    video_width: float,
    video_height: float,
    subtitle_position: str,
    custom_position: float,
    position_mode: str = "default",
):
    """主字幕/旁白字幕的统一位置计算（与 merge_materials 保持一致）。"""
    if position_mode == "picture_narration":
        pic_x, pic_y = _fixed_center_left_position(
            clip.w, clip.h, video_width, video_height
        )
        return clip.with_position((pic_x, pic_y))
    if subtitle_position == "bottom":
        return clip.with_position(("center", video_height * 0.95 - clip.h))
    if subtitle_position == "top":
        return clip.with_position(("center", video_height * 0.05))
    if subtitle_position == "custom":
        margin = 10
        max_y = video_height - clip.h - margin
        min_y = margin
        custom_y = (video_height - clip.h) * (custom_position / 100)
        custom_y = max(min_y, min(custom_y, max_y))
        return clip.with_position(("center", custom_y))
    return clip.with_position(("center", "center"))


def _create_timed_subtitle_clip(
    subtitle_item,
    *,
    video_width: float,
    video_height: float,
    font_path: Optional[str],
    style: Dict[str, Any],
    position_mode: str = "default",
    is_landscape: bool = False,
):
    phrase = subtitle_item[1]
    font_size = style["subtitle_font_size"]
    color = style["subtitle_color"]
    max_width_ratio = 0.9
    clip_stroke_color = style["stroke_color"]
    clip_stroke_width = style["stroke_width"]
    subtitle_bg_color = style["subtitle_bg_color"]

    if position_mode == "picture_narration":
        font_size = style["picture_narration_font_size"]
        color = style["picture_narration_color"]
        max_width_ratio = 0.42 if is_landscape else 0.45
        clip_stroke_color = "#000000"
        clip_stroke_width = max(2, style["stroke_width"])

    max_width = video_width * max_width_ratio
    wrapped_txt = phrase
    if font_path:
        wrapped_txt, _ = wrap_text(
            phrase,
            max_width=max_width,
            font=font_path,
            fontsize=font_size,
        )

    try:
        clip = TextClip(
            text=wrapped_txt,
            font=font_path,
            font_size=font_size,
            color=color,
            bg_color=subtitle_bg_color,
            stroke_color=clip_stroke_color,
            stroke_width=clip_stroke_width,
        )
    except Exception as e:
        logger.error(f"创建字幕片段失败: {str(e)}, 使用简化参数重试")
        clip = TextClip(
            text=wrapped_txt,
            font=font_path,
            font_size=font_size,
            color=color,
        )

    duration = subtitle_item[0][1] - subtitle_item[0][0]
    clip = clip.with_start(subtitle_item[0][0])
    clip = clip.with_end(subtitle_item[0][1])
    clip = clip.with_duration(duration)
    return _position_subtitle_clip(
        clip,
        video_width=video_width,
        video_height=video_height,
        subtitle_position=style["subtitle_position"],
        custom_position=style["custom_position"],
        position_mode=position_mode,
    )


def load_subtitle_overlay_clips(
    subtitle_path: str,
    *,
    video_width: float,
    video_height: float,
    options: Optional[Dict[str, Any]] = None,
    position_mode: str = "default",
) -> list:
    """从 SRT 加载字幕叠加层，位置/样式与 merge_materials 主字幕一致。"""
    if not subtitle_path or not is_valid_subtitle_file(subtitle_path):
        return []

    style = _parse_subtitle_style_options(options)
    font_path = _resolve_subtitle_font_path(style["subtitle_font"])
    is_landscape = _is_landscape_video(video_width, video_height, style.get("video_aspect"))

    def make_textclip(text):
        return TextClip(
            text=text,
            font=font_path,
            font_size=style["subtitle_font_size"],
            color=style["subtitle_color"],
        )

    try:
        sub = SubtitlesClip(
            subtitles=subtitle_path,
            encoding="utf-8",
            make_textclip=make_textclip,
        )
        clips = []
        for item in sub.subtitles:
            clips.append(
                _create_timed_subtitle_clip(
                    item,
                    video_width=video_width,
                    video_height=video_height,
                    font_path=font_path,
                    style=style,
                    position_mode=position_mode,
                    is_landscape=is_landscape,
                )
            )
        return clips
    except Exception as e:
        logger.error(f"处理字幕失败 ({subtitle_path}): \n{traceback.format_exc()}")
        return []


def merge_materials(
    video_path: str,
    audio_path: str,
    output_path: str,
    subtitle_path: Optional[str] = None,
    bgm_path: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None
) -> str:
    """
    合并视频、音频、BGM和字幕素材生成最终视频
    
    参数:
        video_path: 视频文件路径
        audio_path: 音频文件路径
        output_path: 输出文件路径
        subtitle_path: 字幕文件路径，可选
        bgm_path: 背景音乐文件路径，可选
        options: 其他选项配置，可包含以下字段:
            - voice_volume: 人声音量，默认1.0
            - bgm_volume: 背景音乐音量，默认0.3
            - original_audio_volume: 原始音频音量，默认0.0
            - keep_original_audio: 是否保留原始音频，默认False
            - subtitle_font: 字幕字体，默认None，系统会使用默认字体
            - subtitle_font_size: 字幕字体大小，默认40
            - subtitle_color: 字幕颜色，默认白色
            - subtitle_bg_color: 字幕背景颜色，默认透明
            - subtitle_position: 字幕位置，可选值'bottom', 'top', 'center', 'custom'，默认'custom'
            - custom_position: 自定义位置（距顶部百分比），默认60
            - stroke_color: 描边颜色，默认黑色
            - stroke_width: 描边宽度，默认1
            - threads: 处理线程数，默认2
            - fps: 输出帧率，默认30
            - subtitle_enabled: 是否启用字幕，默认True
            
    返回:
        输出视频的路径
    """
    # 合并选项默认值
    if options is None:
        options = {}
    
    # 设置默认参数值 - 使用统一的音量配置
    voice_volume = options.get('voice_volume', AudioVolumeDefaults.VOICE_VOLUME)
    bgm_volume = options.get('bgm_volume', AudioVolumeDefaults.BGM_VOLUME)
    # 修复bug: 将原声音量默认值从0.0改为0.7，确保短剧解说模式下原片音量正常
    original_audio_volume = options.get('original_audio_volume', AudioVolumeDefaults.ORIGINAL_VOLUME)
    keep_original_audio = options.get('keep_original_audio', True)  # 默认保留原声
    subtitle_font = options.get('subtitle_font', '')
    subtitle_font_size = options.get('subtitle_font_size', 40)
    subtitle_color = options.get('subtitle_color', '#FFFFFF')
    subtitle_bg_color = options.get('subtitle_bg_color', 'transparent')
    subtitle_position = options.get('subtitle_position', 'custom')
    custom_position = options.get('custom_position', 60)
    stroke_color = options.get('stroke_color', '#000000')
    stroke_width = options.get('stroke_width', 1)
    threads = options.get('threads', 2)
    fps = options.get('fps', 30)
    subtitle_enabled = options.get('subtitle_enabled', True)
    watermark_text = str(options.get('watermark_text') or '').strip()
    picture_narration_path = options.get('picture_narration_path')
    picture_narration_enabled = options.get('enable_picture_narration', False)
    picture_narration_font_size = options.get('picture_narration_font_size', 44)
    picture_narration_color = options.get('picture_narration_color', '#FFE066')
    video_aspect = options.get('video_aspect')

    # 配置日志 - 便于调试问题
    logger.info(f"音量配置详情:")
    logger.info(f"  - 配音音量: {voice_volume}")
    logger.info(f"  - 背景音乐音量: {bgm_volume}")
    logger.info(f"  - 原声音量: {original_audio_volume}")
    logger.info(f"  - 是否保留原声: {keep_original_audio}")
    logger.info(f"字幕配置详情:")
    logger.info(f"  - 是否启用字幕: {subtitle_enabled}")
    logger.info(f"  - 字幕文件路径: {subtitle_path}")
    logger.info(f"成片输出配置:")
    logger.info(f"  - 水印: {watermark_text or '未启用'}")
    logger.info(f"  - 原声旁白字幕: {picture_narration_enabled}")
    if picture_narration_enabled:
        logger.info(f"  - 旁白字幕路径: {picture_narration_path or '未生成'}")

    # 音量参数验证
    def validate_volume(volume, name):
        if not (AudioVolumeDefaults.MIN_VOLUME <= volume <= AudioVolumeDefaults.MAX_VOLUME):
            logger.warning(f"{name}音量 {volume} 超出有效范围 [{AudioVolumeDefaults.MIN_VOLUME}, {AudioVolumeDefaults.MAX_VOLUME}]，将被限制")
            return max(AudioVolumeDefaults.MIN_VOLUME, min(volume, AudioVolumeDefaults.MAX_VOLUME))
        return volume

    voice_volume = validate_volume(voice_volume, "配音")
    bgm_volume = validate_volume(bgm_volume, "背景音乐")
    original_audio_volume = validate_volume(original_audio_volume, "原声")

    # 处理透明背景色问题 - MoviePy 2.1.1不支持'transparent'值
    if subtitle_bg_color == 'transparent':
        subtitle_bg_color = None  # None在新版MoviePy中表示透明背景

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"开始合并素材...")
    logger.info(f"  ① 视频: {video_path}")
    logger.info(f"  ② 音频: {audio_path}")
    if subtitle_path:
        logger.info(f"  ③ 字幕: {subtitle_path}")
    if bgm_path:
        logger.info(f"  ④ 背景音乐: {bgm_path}")
    logger.info(f"  ⑤ 输出: {output_path}")
    
    # 加载视频
    try:
        video_clip = VideoFileClip(video_path)
        logger.info(f"视频尺寸: {video_clip.size[0]}x{video_clip.size[1]}, 时长: {video_clip.duration}秒")
        
        # 提取视频原声(如果需要)
        original_audio = None
        if keep_original_audio and original_audio_volume > 0:
            try:
                original_audio = video_clip.audio
                if original_audio:
                    # 关键修复：只有当音量不为1.0时才进行音量调整，保持原声音量不变
                    if abs(original_audio_volume - 1.0) > 0.001:  # 使用小的容差值比较浮点数
                        original_audio = original_audio.with_effects([afx.MultiplyVolume(original_audio_volume)])
                        logger.info(f"已提取视频原声，音量调整为: {original_audio_volume}")
                    else:
                        logger.info("已提取视频原声，保持原始音量不变")
                else:
                    logger.warning("视频没有音轨，无法提取原声")
            except Exception as e:
                logger.error(f"提取视频原声失败: {str(e)}")
                original_audio = None
        
        # 移除原始音轨，稍后会合并新的音频
        video_clip = video_clip.without_audio()
        
    except Exception as e:
        logger.error(f"加载视频失败: {str(e)}")
        raise
    
    # 处理背景音乐和所有音频轨道合成
    audio_tracks = []

    # 智能音量调整（可选功能）
    if AudioVolumeDefaults.ENABLE_SMART_VOLUME and audio_path and os.path.exists(audio_path) and original_audio is not None:
        try:
            normalizer = AudioNormalizer()
            temp_dir = tempfile.mkdtemp()
            temp_original_path = os.path.join(temp_dir, "temp_original.wav")

            # 保存原声到临时文件进行分析
            original_audio.write_audiofile(temp_original_path, verbose=False, logger=None)

            # 计算智能音量调整
            tts_adjustment, original_adjustment = normalizer.calculate_volume_adjustment(
                audio_path, temp_original_path
            )

            # 应用智能调整，但保留用户设置的相对比例
            smart_voice_volume = voice_volume * tts_adjustment
            smart_original_volume = original_audio_volume * original_adjustment

            # 限制音量范围，避免过度调整
            smart_voice_volume = max(0.1, min(1.5, smart_voice_volume))
            smart_original_volume = max(0.1, min(2.0, smart_original_volume))

            voice_volume = smart_voice_volume
            original_audio_volume = smart_original_volume

            logger.info(f"智能音量调整 - TTS: {voice_volume:.2f}, 原声: {original_audio_volume:.2f}")

            # 清理临时文件
            import shutil
            shutil.rmtree(temp_dir)

        except Exception as e:
            logger.warning(f"智能音量分析失败，使用原始设置: {e}")

    # 先添加主音频（配音）
    if audio_path and os.path.exists(audio_path):
        try:
            voice_audio = AudioFileClip(audio_path).with_effects([afx.MultiplyVolume(voice_volume)])
            audio_tracks.append(voice_audio)
            logger.info(f"已添加配音音频，音量: {voice_volume}")
        except Exception as e:
            logger.error(f"加载配音音频失败: {str(e)}")

    # 添加原声（如果需要）
    if original_audio is not None:
        # 重新应用调整后的音量（因为original_audio已经应用了一次音量）
        # 计算需要的额外调整
        current_volume_in_original = 1.0  # original_audio中已应用的音量
        additional_adjustment = original_audio_volume / current_volume_in_original

        adjusted_original_audio = original_audio.with_effects([afx.MultiplyVolume(additional_adjustment)])
        audio_tracks.append(adjusted_original_audio)
        logger.info(f"已添加视频原声，最终音量: {original_audio_volume}")

    # 添加背景音乐（如果有）
    if bgm_path and os.path.exists(bgm_path):
        try:
            bgm_clip = AudioFileClip(bgm_path).with_effects([
                afx.MultiplyVolume(bgm_volume),
                afx.AudioFadeOut(3),
                afx.AudioLoop(duration=video_clip.duration),
            ])
            audio_tracks.append(bgm_clip)
            logger.info(f"已添加背景音乐，音量: {bgm_volume}")
        except Exception as e:
            logger.error(f"添加背景音乐失败: \n{traceback.format_exc()}")

    # 合成最终的音频轨道
    if audio_tracks:
        final_audio = CompositeAudioClip(audio_tracks)
        video_clip = video_clip.with_audio(final_audio)
        logger.info(f"已合成所有音频轨道，共{len(audio_tracks)}个")
    else:
        logger.warning("没有可用的音频轨道，输出视频将没有声音")
    
    font_path = _resolve_subtitle_font_path(subtitle_font)
    if font_path:
        logger.info(f"使用字体: {font_path}")

    video_width, video_height = video_clip.size
    is_landscape = _is_landscape_video(video_width, video_height, video_aspect)
    if is_landscape:
        logger.info("画幅 16:9：旁白字幕居中靠左，水印居中靠右（上下 10% 缓慢浮动）")
    else:
        logger.info("画幅 9:16：旁白字幕居中靠左，水印居中靠右（上下 10% 缓慢浮动）")

    subtitle_overlay_options = {
        "subtitle_font": subtitle_font,
        "subtitle_font_size": subtitle_font_size,
        "subtitle_color": subtitle_color,
        "subtitle_bg_color": subtitle_bg_color,
        "subtitle_position": subtitle_position,
        "custom_position": custom_position,
        "stroke_color": stroke_color,
        "stroke_width": stroke_width,
        "picture_narration_font_size": picture_narration_font_size,
        "picture_narration_color": picture_narration_color,
        "video_aspect": video_aspect,
    }

    def create_watermark_clip():
        if not watermark_text:
            return None
        wm_font_size = max(18, int(subtitle_font_size * 0.55))
        try:
            wm_clip = TextClip(
                text=watermark_text,
                font=font_path,
                font_size=wm_font_size,
                color="#FFFFFF",
                stroke_color="#000000",
                stroke_width=1,
            )
        except Exception:
            wm_clip = TextClip(
                text=watermark_text,
                font=font_path,
                font_size=wm_font_size,
                color="#FFFFFF",
            )
        wm_clip = wm_clip.with_opacity(0.72)
        wm_x, base_y = _fixed_center_right_position(
            wm_clip.w, wm_clip.h, video_width, video_height
        )
        margin = max(12, int(min(video_width, video_height) * 0.02))
        float_amplitude = video_height * 0.10
        min_y = max(margin, base_y - float_amplitude)
        max_y = min(video_height - wm_clip.h - margin, base_y + float_amplitude)
        float_period = 30.0

        def floating_position(t):
            offset_y = float_amplitude * math.sin(2 * math.pi * t / float_period)
            y = base_y + offset_y
            y = max(min_y, min(y, max_y))
            return (wm_x, y)

        wm_clip = wm_clip.with_position(floating_position)
        wm_clip = wm_clip.with_start(0).with_end(video_clip.duration).with_duration(video_clip.duration)
        return wm_clip

    # 处理字幕、旁白字幕与水印
    overlay_clips = []

    if subtitle_enabled and subtitle_path:
        if is_valid_subtitle_file(subtitle_path):
            logger.info("字幕已启用，开始处理字幕文件")
            main_subtitle_clips = load_subtitle_overlay_clips(
                subtitle_path,
                video_width=video_width,
                video_height=video_height,
                options=subtitle_overlay_options,
                position_mode="default",
            )
            overlay_clips.extend(main_subtitle_clips)
            if main_subtitle_clips:
                logger.info(f"已添加 {len(main_subtitle_clips)} 个主字幕片段")
        else:
            logger.warning(f"字幕文件无效或为空: {subtitle_path}，跳过字幕处理")
    elif not subtitle_enabled:
        logger.info("字幕已禁用，跳过字幕处理")
    elif not subtitle_path:
        logger.info("未提供字幕文件路径，跳过字幕处理")

    watermark_clip = create_watermark_clip()
    if watermark_clip:
        overlay_clips.append(watermark_clip)
        logger.info(f"已添加水印: {watermark_text}")

    if picture_narration_enabled and picture_narration_path:
        pic_clips = load_subtitle_overlay_clips(
            picture_narration_path,
            video_width=video_width,
            video_height=video_height,
            options=subtitle_overlay_options,
            position_mode="picture_narration",
        )
        if pic_clips:
            overlay_clips.extend(pic_clips)
            logger.info(f"已添加 {len(pic_clips)} 个原声旁白字幕片段")

    if overlay_clips:
        video_clip = CompositeVideoClip([video_clip, *overlay_clips])
    else:
        logger.info("警告：没有叠加层被添加到视频中")
    
    # 导出最终视频
    try:
        video_clip.write_videofile(
            output_path,
            audio_codec="aac",
            temp_audiofile_path=output_dir,
            threads=threads,
            fps=fps,
        )
        logger.success(f"素材合并完成: {output_path}")
    except Exception as e:
        logger.error(f"导出视频失败: {str(e)}")
        raise
    finally:
        # 释放资源
        video_clip.close()
        del video_clip
    
    return output_path


def burn_subtitles_on_video(
    video_path: str,
    subtitle_path: str,
    output_path: str,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """
    在已合成视频上仅叠加主字幕，保留原有音轨及已烧录的水印/旁白字幕。

    用于「先合成 → API 转写 → 再烧主字幕」流程的最后一步。
    """
    if options is None:
        options = {}

    if not is_valid_subtitle_file(subtitle_path):
        raise ValueError(f"无效字幕文件: {subtitle_path}")

    style = _parse_subtitle_style_options(options)
    threads = options.get("threads", 2)
    fps = options.get("fps", 30)

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    write_path = output_path
    if os.path.abspath(video_path) == os.path.abspath(output_path):
        fd, write_path = tempfile.mkstemp(suffix=".mp4", dir=output_dir)
        os.close(fd)

    logger.info(
        f"在成片上烧录主字幕: {video_path} -> {output_path} "
        f"(位置={style['subtitle_position']}, custom={style['custom_position']})"
    )

    video_clip = VideoFileClip(video_path)
    video_width, video_height = video_clip.size

    subtitle_clips = load_subtitle_overlay_clips(
        subtitle_path,
        video_width=video_width,
        video_height=video_height,
        options=options,
        position_mode="default",
    )
    if not subtitle_clips:
        video_clip.close()
        raise RuntimeError(f"加载字幕失败: {subtitle_path}")
    logger.info(f"已加载 {len(subtitle_clips)} 个主字幕片段")

    if subtitle_clips:
        final_video = CompositeVideoClip([video_clip, *subtitle_clips])
    else:
        final_video = video_clip

    try:
        final_video.write_videofile(
            write_path,
            audio_codec="aac",
            temp_audiofile_path=output_dir,
            threads=threads,
            fps=fps,
        )
        logger.success(f"主字幕烧录完成: {output_path}")
    finally:
        if final_video is not video_clip:
            final_video.close()
        video_clip.close()

    if write_path != output_path:
        os.replace(write_path, output_path)

    return output_path


def wrap_text(text, max_width, font="Arial", fontsize=60):
    """
    文本换行函数，使长文本适应指定宽度
    
    参数:
        text: 需要换行的文本
        max_width: 最大宽度（像素）
        font: 字体路径
        fontsize: 字体大小
        
    返回:
        换行后的文本和文本高度
    """
    # 创建ImageFont对象
    try:
        font_obj = ImageFont.truetype(font, fontsize)
    except:
        # 如果无法加载指定字体，使用默认字体
        font_obj = ImageFont.load_default()
    
    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        left, top, right, bottom = font_obj.getbbox(inner_text)
        return right - left, bottom - top

    width, height = get_text_size(text)
    if width <= max_width:
        return text, height

    processed = True

    _wrapped_lines_ = []
    words = text.split(" ")
    _txt_ = ""
    for word in words:
        _before = _txt_
        _txt_ += f"{word} "
        _width, _height = get_text_size(_txt_)
        if _width <= max_width:
            continue
        else:
            if _txt_.strip() == word.strip():
                processed = False
                break
            _wrapped_lines_.append(_before)
            _txt_ = f"{word} "
    _wrapped_lines_.append(_txt_)
    if processed:
        _wrapped_lines_ = [line.strip() for line in _wrapped_lines_]
        result = "\n".join(_wrapped_lines_).strip()
        height = len(_wrapped_lines_) * height
        return result, height

    _wrapped_lines_ = []
    chars = list(text)
    _txt_ = ""
    for word in chars:
        _txt_ += word
        _width, _height = get_text_size(_txt_)
        if _width <= max_width:
            continue
        else:
            _wrapped_lines_.append(_txt_)
            _txt_ = ""
    _wrapped_lines_.append(_txt_)
    result = "\n".join(_wrapped_lines_).strip()
    height = len(_wrapped_lines_) * height
    return result, height


if __name__ == '__main__':
    merger_mp4 = '/Users/apple/Desktop/home/NarratoAI/storage/tasks/qyn2-2-demo/merger.mp4'
    merger_sub = '/Users/apple/Desktop/home/NarratoAI/storage/tasks/qyn2-2-demo/merged_subtitle_00_00_00-00_01_30.srt'
    merger_audio = '/Users/apple/Desktop/home/NarratoAI/storage/tasks/qyn2-2-demo/merger_audio.mp3'
    bgm_path = '/Users/apple/Desktop/home/NarratoAI/resource/songs/bgm.mp3'
    output_video = '/Users/apple/Desktop/home/NarratoAI/storage/tasks/qyn2-2-demo/combined_test.mp4'
    
    # 调用示例
    options = {
        'voice_volume': 1.0,            # 配音音量
        'bgm_volume': 0.1,              # 背景音乐音量
        'original_audio_volume': 1.0,   # 视频原声音量，0表示不保留
        'keep_original_audio': True,    # 是否保留原声
        'subtitle_enabled': True,       # 是否启用字幕 - 修复字幕开关bug
        'subtitle_font': 'MicrosoftYaHeiNormal.ttc',  # 这里使用相对字体路径，会自动在 font_dir() 目录下查找
        'subtitle_font_size': 40,
        'subtitle_color': '#FFFFFF',
        'subtitle_bg_color': None,      # 直接使用None表示透明背景
        'subtitle_position': 'custom',
        'custom_position': 60,
        'threads': 2
    }
    
    try:
        merge_materials(
            video_path=merger_mp4,
            audio_path=merger_audio,
            subtitle_path=merger_sub,
            bgm_path=bgm_path,
            output_path=output_video,
            options=options
        )
    except Exception as e:
        logger.error(f"合并素材失败: \n{traceback.format_exc()}")
