import math
import json
import os
import os.path
import re
import traceback
from os import path
from typing import Optional
from loguru import logger

from app.config import config
from app.config.audio_config import AudioConfig, get_recommended_volumes_for_content
from app.models import const
from app.models.schema import VideoClipParams
from app.services import (voice, audio_merger, subtitle_merger, clip_video, merger_video, update_script, generate_video)
from app.services.update_script import probe_media_duration
from app.services.perfect_subtitle_service import (
    build_merged_subtitle_path,
    build_picture_narration_subtitle_path,
    is_deferred_subtitle_enabled,
    is_perfect_subtitle_enabled,
)
from app.services.video_output_settings import get_video_output_settings
from app.services.update_script import is_valid_video_file
from app.services import state as sm
from app.utils import utils
from app.services.film_tv_script_optimizer import (
    finalize_film_tv_playback_order,
    is_ost_grouped_by_type,
)

_SCRIPT_PATH_MODES = frozenset({
    "auto",
    "short",
    "summary",
    "film_tv",
    "file_selection",
})


def _is_writable_script_json_path(script_path: str) -> bool:
    if not script_path or script_path in _SCRIPT_PATH_MODES:
        return False
    if not str(script_path).lower().endswith(".json"):
        return False
    return bool(path.dirname(path.abspath(script_path)))


def _persist_synced_script(
    task_id: str,
    script_list: list,
    source_script_path: str = "",
) -> str:
    """裁剪并 update_script 后回写 JSON，使 duration / editedTimeRange / video 等与成片一致。"""
    task_script_path = path.join(utils.task_dir(task_id), "script_after_clip.json")
    os.makedirs(path.dirname(task_script_path), exist_ok=True)

    def _write_json(target: str) -> None:
        with open(target, "w", encoding="utf-8") as fp:
            json.dump(script_list, fp, ensure_ascii=False, indent=2)

    _write_json(task_script_path)
    logger.info(f"已回写任务脚本 -> {task_script_path}")

    if _is_writable_script_json_path(source_script_path):
        abs_source = path.abspath(source_script_path)
        try:
            os.makedirs(path.dirname(abs_source), exist_ok=True)
            _write_json(abs_source)
            logger.info(f"已回写源脚本 -> {abs_source}")
            return abs_source
        except OSError as exc:
            logger.warning(f"回写源脚本失败 ({abs_source}): {exc}")

    return task_script_path


def _prepare_video_script_list(list_script: list) -> list:
    """若 OST 按类型分组排列，则按原片时间轴重排为穿插播放顺序。"""
    if is_ost_grouped_by_type(list_script):
        logger.info("检测到 OST 分组排列，按原片时间轴重排播放顺序")
        return finalize_film_tv_playback_order(list_script)
    return list_script


def _build_merge_video_options(
    params: VideoClipParams,
    *,
    list_script: list,
    picture_narration_path: str = "",
) -> dict:
    """构建 merge_materials 的 options 字典。"""
    optimized_volumes = get_recommended_volumes_for_content('mixed')

    final_tts_volume = float(
        getattr(params, 'tts_volume', optimized_volumes['tts_volume'])
        or optimized_volumes['tts_volume']
    )
    final_original_volume = float(
        getattr(params, 'original_volume', optimized_volumes['original_volume'])
        or optimized_volumes['original_volume']
    )
    final_bgm_volume = float(
        getattr(params, 'bgm_volume', optimized_volumes['bgm_volume'])
        or optimized_volumes['bgm_volume']
    )

    video_output = get_video_output_settings()
    from app.services.short_drama_settings import resolve_video_output_for_script_mode

    script_mode = str(getattr(params, "video_clip_json_path", "") or "").strip().lower()
    workflow_mode = str(getattr(params, "narration_workflow_mode", "") or "").strip().lower()
    video_output = resolve_video_output_for_script_mode(
        video_output, script_path=script_mode, workflow_mode=workflow_mode
    )
    if hasattr(params, 'watermark_text') and params.watermark_text is not None:
        video_output["watermark_text"] = params.watermark_text
    if hasattr(params, 'enable_picture_narration') and params.enable_picture_narration is not None:
        video_output["enable_picture_narration"] = params.enable_picture_narration

    logger.info(f"音量配置 - TTS: {final_tts_volume}, 原声: {final_original_volume}, BGM: {final_bgm_volume}")

    return {
        'voice_volume': final_tts_volume,
        'bgm_volume': final_bgm_volume,
        'original_audio_volume': final_original_volume,
        'keep_original_audio': True,
        'subtitle_enabled': params.subtitle_enabled,
        'subtitle_font': params.font_name,
        'subtitle_font_size': params.font_size,
        'subtitle_color': params.text_fore_color,
        'subtitle_bg_color': None,
        'subtitle_position': params.subtitle_position,
        'custom_position': params.custom_position,
        'stroke_color': getattr(params, 'stroke_color', '#000000'),
        'stroke_width': getattr(params, 'stroke_width', 1.5),
        'threads': params.n_threads,
        'watermark_text': video_output.get('watermark_text', ''),
        'enable_picture_narration': bool(video_output.get('enable_picture_narration', True)),
        'picture_narration_path': picture_narration_path,
        'picture_narration_font_size': int(video_output.get('picture_narration_font_size', 44)),
        'picture_narration_color': video_output.get('picture_narration_color', '#FFE066'),
        'original_subtitle_color': video_output.get('original_subtitle_color', '#FFE066'),
        'video_aspect': getattr(params, 'video_aspect', None),
    }


def _should_defer_subtitle_asr(params: VideoClipParams) -> bool:
    """完美字幕开启且配置延迟转写时，合成阶段跳过 ASR。"""
    return is_perfect_subtitle_enabled() and is_deferred_subtitle_enabled()


def _merge_task_subtitles(
    task_id: str,
    new_script_list: list,
    params: VideoClipParams,
    *,
    force: bool = False,
) -> str:
    """Generate merged subtitles (perfect dual-track preferred, legacy fallback)."""
    if not force and _should_defer_subtitle_asr(params):
        logger.info("延迟字幕模式：跳过同步字幕合并，将在成片合成后通过 API 转写并烧录")
        return ""

    source_subtitle_path = (getattr(params, "source_subtitle_path", None) or "").strip()
    if not source_subtitle_path:
        source_subtitle_path = (config.app.get("source_subtitle_path") or "").strip()

    merged_subtitle_path = build_merged_subtitle_path(
        new_script_list,
        task_id=task_id,
        source_subtitle_path=source_subtitle_path or None,
    )
    if merged_subtitle_path:
        logger.info(f"完美字幕合并成功 -> {merged_subtitle_path}")
        return merged_subtitle_path

    merged_subtitle_path = subtitle_merger.merge_subtitle_files(new_script_list)
    if merged_subtitle_path:
        logger.info(f"字幕文件合并成功 -> {merged_subtitle_path}")
        return merged_subtitle_path

    logger.warning("没有有效的字幕内容，将生成无字幕视频")
    return ""


def _collect_video_clips_from_script(
    new_script_list: list,
    subclip_path_videos: dict = None,
) -> tuple[list, list]:
    """收集可合并的视频片段路径及对应 OST（跳过无有效视频的片段，二者索引对齐）。"""
    video_clips = []
    video_ost = []

    def _append_clip(path: str, ost: int) -> None:
        video_clips.append(path)
        video_ost.append(int(ost))

    for new_script in new_script_list:
        ost = int(new_script.get("OST", 0) or 0)
        video_path = new_script.get("video")
        if video_path and is_valid_video_file(video_path):
            _append_clip(video_path, ost)
            continue

        logger.warning(
            f"片段 {new_script.get('_id')} 的视频文件不存在、损坏或未生成: {video_path}"
        )
        if subclip_path_videos and new_script.get("_id") in subclip_path_videos:
            backup_video = subclip_path_videos[new_script.get("_id")]
            if is_valid_video_file(backup_video):
                _append_clip(backup_video, ost)
                logger.info(f"使用备用视频: {backup_video}")
            else:
                logger.error(f"备用视频也不存在: {backup_video}")
        else:
            logger.error(f"无法找到片段 {new_script.get('_id')} 的视频文件")
    return video_clips, video_ost


def _merge_video_clips_and_sync_timeline(
    task_id: str,
    new_script_list: list,
    video_clips: list,
    video_ost: list,
    params: VideoClipParams,
    source_script_path: str = "",
) -> tuple[str, list]:
    """合并视频片段，并按重编码后的 processed 时长校正成片时间轴。"""
    combined_video_path = path.join(utils.task_dir(task_id), "merger.mp4")
    logger.info(f"\n\n## 合并视频: => {combined_video_path}")
    logger.info(f"准备合并 {len(video_clips)} 个视频片段")

    merger_video.combine_clip_videos(
        output_video_path=combined_video_path,
        video_paths=video_clips,
        video_ost_list=video_ost,
        video_aspect=params.video_aspect,
        threads=params.n_threads,
    )

    new_script_list = update_script.sync_script_timeline_after_video_merge(
        task_id,
        new_script_list,
        len(video_clips),
    )
    _persist_synced_script(task_id, new_script_list, source_script_path)
    return combined_video_path, new_script_list


def _merge_audio_and_subtitles(
    task_id: str,
    new_script_list: list,
    params: VideoClipParams,
    *,
    tts_segments: list,
) -> tuple[str, str, str]:
    """在成片时间轴校正后合并 TTS 音轨与字幕。"""
    logger.info("\n\n## 合并音频和字幕")
    merger_video_path = path.join(utils.task_dir(task_id), "merger.mp4")
    probed_merger = probe_media_duration(merger_video_path) if os.path.isfile(merger_video_path) else 0.0
    script_total = sum(float(script.get("duration") or 0) for script in new_script_list)
    total_duration = probed_merger or script_total
    if probed_merger > 0 and abs(probed_merger - script_total) >= 0.05:
        logger.info(
            f"配音轨对齐 merger.mp4: 脚本 {script_total:.3f}s, 实测 {probed_merger:.3f}s"
        )
    merged_audio_path = ""
    merged_subtitle_path = ""
    picture_narration_path = ""
    try:
        if tts_segments:
            merged_audio_path = audio_merger.merge_audio_files(
                task_id=task_id,
                total_duration=total_duration,
                list_script=new_script_list,
            )
            logger.info(f"音频文件合并成功->{merged_audio_path}")

        merged_subtitle_path = _merge_task_subtitles(task_id, new_script_list, params)
        from app.services.short_drama_settings import resolve_video_output_for_script_mode

        video_output = get_video_output_settings()
        script_path = str(getattr(params, "video_clip_json_path", "") or "").strip().lower()
        workflow_mode = str(getattr(params, "narration_workflow_mode", "") or "").strip().lower()
        video_output = resolve_video_output_for_script_mode(
            video_output, script_path=script_path, workflow_mode=workflow_mode
        )
        if hasattr(params, "enable_picture_narration") and params.enable_picture_narration is not None:
            video_output["enable_picture_narration"] = params.enable_picture_narration
        picture_narration_path = _build_picture_narration_path(
            task_id, new_script_list, video_output=video_output
        )
    except Exception as e:
        logger.error(f"合并音频/字幕文件失败: {str(e)}")
    return merged_audio_path, merged_subtitle_path, picture_narration_path


def _build_picture_narration_path(
    task_id: str,
    new_script_list: list,
    video_output: Optional[dict] = None,
) -> str:
    try:
        path = build_picture_narration_subtitle_path(
            new_script_list,
            task_id=task_id,
            video_output=video_output,
        )
        if path:
            logger.info(f"原声旁白字幕生成成功 -> {path}")
        return path or ""
    except Exception as exc:
        logger.warning(f"原声旁白字幕生成失败: {exc}")
        return ""


def _produce_final_video(
    task_id: str,
    params: VideoClipParams,
    *,
    combined_video_path: str,
    merged_audio_path: str,
    merged_subtitle_path: str,
    picture_narration_path: str,
    list_script: list,
    new_script_list: list,
) -> str:
    """合并 BGM/配音/旁白/水印，并在延迟模式下于最后 API 转写后烧录主字幕。"""
    task_dir = utils.task_dir(task_id)
    output_video_path = path.join(task_dir, "combined.mp4")
    bgm_path = utils.get_bgm_file()
    options = _build_merge_video_options(
        params,
        list_script=list_script,
        picture_narration_path=picture_narration_path,
    )

    if _should_defer_subtitle_asr(params):
        base_output_path = path.join(task_dir, "combined_base.mp4")
        logger.info(
            f"\n\n## 6. 合成成片（无主字幕）: 水印/旁白/BGM/配音 -> {base_output_path}"
        )
        merge_options = dict(options)
        merge_options["subtitle_enabled"] = False

        generate_video.merge_materials(
            video_path=combined_video_path,
            audio_path=merged_audio_path,
            subtitle_path=None,
            bgm_path=bgm_path,
            output_path=base_output_path,
            options=merge_options,
        )

        if not getattr(params, "subtitle_enabled", True):
            logger.info("主字幕已禁用，跳过延迟 API 转写")
            os.replace(base_output_path, output_video_path)
            return output_video_path

        logger.info("\n\n## 7. 延迟字幕：API 转写")
        sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=90)
        deferred_subtitle_path = _merge_task_subtitles(
            task_id, new_script_list, params, force=True
        )

        if not deferred_subtitle_path:
            logger.warning("延迟字幕转写未生成有效字幕，保留无主字幕成片")
            os.replace(base_output_path, output_video_path)
            return output_video_path

        logger.info(f"\n\n## 8. 烧录主字幕 -> {output_video_path}")
        sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=95)
        generate_video.burn_subtitles_on_video(
            video_path=base_output_path,
            subtitle_path=deferred_subtitle_path,
            output_path=output_video_path,
            options=options,
        )
        return output_video_path

    logger.info(f"\n\n## 6. 最后一步: 合并字幕/BGM/配音/视频 -> {output_video_path}")
    generate_video.merge_materials(
        video_path=combined_video_path,
        audio_path=merged_audio_path,
        subtitle_path=merged_subtitle_path,
        bgm_path=bgm_path,
        output_path=output_video_path,
        options=options,
    )
    return output_video_path


def start_subclip(task_id: str, params: VideoClipParams, subclip_path_videos: dict = None):
    """
    后台任务（统一视频裁剪处理）- 优化版本

    实施基于OST类型的统一视频裁剪策略，消除双重裁剪问题：
    - OST=0: 根据TTS音频时长动态裁剪，移除原声
    - OST=1: 严格按照脚本timestamp精确裁剪，保持原声
    - OST=2: 根据TTS音频时长动态裁剪，保持原声

    Args:
        task_id: 任务ID
        params: 视频参数
        subclip_path_videos: 视频片段路径（可选，仅作为备用方案）
    """
    global merged_audio_path, merged_subtitle_path

    logger.info(f"\n\n## 开始任务: {task_id}")
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=0)

    """
    1. 加载剪辑脚本
    """
    logger.info("\n\n## 1. 加载视频脚本")
    video_script_path = path.join(params.video_clip_json_path)
    
    if path.exists(video_script_path):
        try:
            with open(video_script_path, "r", encoding="utf-8") as f:
                list_script = json.load(f)
                list_script = _prepare_video_script_list(list_script)
                video_list = [i['narration'] for i in list_script]
                video_ost = [i['OST'] for i in list_script]
                time_list = [i['timestamp'] for i in list_script]

                video_script = " ".join(video_list)
                logger.debug(f"解说完整脚本: \n{video_script}")
                logger.debug(f"解说 OST 列表: \n{video_ost}")
                logger.debug(f"解说时间戳列表: \n{time_list}")
        except Exception as e:
            logger.error(f"无法读取视频json脚本，请检查脚本格式是否正确")
            raise ValueError("无法读取视频json脚本，请检查脚本格式是否正确")
    else:
        logger.error(f"解说脚本文件不存在: {video_script_path}，请先点击【保存脚本】按钮保存脚本后再生成视频")
        raise ValueError("解说脚本文件不存在！请先点击【保存脚本】按钮保存脚本后再生成视频。")

    """
    2. 使用 TTS 生成音频素材
    """
    logger.info("\n\n## 2. 根据OST设置生成音频列表")
    # 只为OST=0 or 2的判断生成音频， OST=0 仅保留解说 OST=2 保留解说和原声
    tts_segments = [
        segment for segment in list_script 
        if segment['OST'] in [0, 2]
    ]
    logger.debug(f"需要生成TTS的片段数: {len(tts_segments)}")

    tts_results = voice.tts_multiple(
        task_id=task_id,
        list_script=tts_segments,  # 只传入需要TTS的片段
        tts_engine=params.tts_engine,
        voice_name=params.voice_name,
        voice_rate=params.voice_rate,
        voice_pitch=params.voice_pitch,
    )

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)

    # """
    # 3. (可选) 使用 whisper 生成字幕
    # """
    # if merged_subtitle_path is None:
    #     if audio_files:
    #         merged_subtitle_path = path.join(utils.task_dir(task_id), f"subtitle.srt")
    #         subtitle_provider = config.app.get("subtitle_provider", "").strip().lower()
    #         logger.info(f"\n\n使用 {subtitle_provider} 生成字幕")
    #
    #         subtitle.create(
    #             audio_file=merged_audio_path,
    #             subtitle_file=merged_subtitle_path,
    #         )
    #         subtitle_lines = subtitle.file_to_subtitles(merged_subtitle_path)
    #         if not subtitle_lines:
    #             logger.warning(f"字幕文件无效: {merged_subtitle_path}")
    #
    # sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)

    """
    3. 统一视频裁剪 - 基于OST类型的差异化裁剪策略
    """
    logger.info("\n\n## 3. 统一视频裁剪（基于OST类型）")

    # 使用新的统一裁剪策略
    video_clip_result = clip_video.clip_video_unified(
        video_origin_path=params.video_origin_path,
        script_list=list_script,
        tts_results=tts_results
    )

    # 更新 list_script 中的时间戳和路径信息
    tts_clip_result = {tts_result['_id']: tts_result['audio_file'] for tts_result in tts_results}
    subclip_clip_result = {
        tts_result['_id']: tts_result['subtitle_file'] for tts_result in tts_results
    }
    tts_duration_by_id = {
        tts_result["_id"]: float(tts_result.get("duration") or 0)
        for tts_result in tts_results
    }
    new_script_list = update_script.update_script_timestamps(
        list_script,
        video_clip_result,
        tts_clip_result,
        subclip_clip_result,
        tts_duration_by_id=tts_duration_by_id,
    )
    _persist_synced_script(task_id, new_script_list, video_script_path)

    logger.info(f"统一裁剪完成，处理了 {len(video_clip_result)} 个视频片段")

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=60)

    """
    4. 合并视频（先合并，再按重编码后时长校正时间轴）
    """
    final_video_paths = []
    combined_video_paths = []

    video_clips, clip_ost_list = _collect_video_clips_from_script(
        new_script_list, subclip_path_videos
    )
    combined_video_path, new_script_list = _merge_video_clips_and_sync_timeline(
        task_id, new_script_list, video_clips, clip_ost_list, params, video_script_path
    )
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=75)

    """
    5. 合并音频和字幕（使用与 merger.mp4 一致的时间轴）
    """
    merged_audio_path, merged_subtitle_path, picture_narration_path = _merge_audio_and_subtitles(
        task_id, new_script_list, params, tts_segments=tts_segments
    )
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=85)

    """
    6. 合并字幕/BGM/配音/视频（延迟模式下先合成再 API 转写烧字幕）
    """
    output_video_path = _produce_final_video(
        task_id,
        params,
        combined_video_path=combined_video_path,
        merged_audio_path=merged_audio_path,
        merged_subtitle_path=merged_subtitle_path,
        picture_narration_path=picture_narration_path,
        list_script=list_script,
        new_script_list=new_script_list,
    )

    final_video_paths.append(output_video_path)
    combined_video_paths.append(combined_video_path)

    logger.success(f"任务 {task_id} 已完成, 生成 {len(final_video_paths)} 个视频.")

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths
    }
    sm.state.update_task(task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs)
    return kwargs


def start_subclip_unified(task_id: str, params: VideoClipParams):
    """
    统一视频裁剪处理函数 - 完全基于OST类型的新实现

    这是优化后的版本，完全移除了对预裁剪视频的依赖，
    实现真正的统一裁剪策略。

    Args:
        task_id: 任务ID
        params: 视频参数
    """
    global merged_audio_path, merged_subtitle_path

    logger.info(f"\n\n## 开始统一视频处理任务: {task_id}")
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=0)

    """
    1. 加载剪辑脚本
    """
    logger.info("\n\n## 1. 加载视频脚本")
    video_script_path = path.join(params.video_clip_json_path)

    if path.exists(video_script_path):
        try:
            with open(video_script_path, "r", encoding="utf-8") as f:
                list_script = json.load(f)
                list_script = _prepare_video_script_list(list_script)
                video_list = [i['narration'] for i in list_script]
                video_ost = [i['OST'] for i in list_script]
                time_list = [i['timestamp'] for i in list_script]

                video_script = " ".join(video_list)
                logger.debug(f"解说完整脚本: \n{video_script}")
                logger.debug(f"解说 OST 列表: \n{video_ost}")
                logger.debug(f"解说时间戳列表: \n{time_list}")
        except Exception as e:
            logger.error(f"无法读取视频json脚本，请检查脚本格式是否正确")
            raise ValueError("无法读取视频json脚本，请检查脚本格式是否正确")
    else:
        logger.error(f"解说脚本文件不存在: {video_script_path}，请先点击【保存脚本】按钮保存脚本后再生成视频")
        raise ValueError("解说脚本文件不存在！请先点击【保存脚本】按钮保存脚本后再生成视频。")

    """
    2. 使用 TTS 生成音频素材
    """
    logger.info("\n\n## 2. 根据OST设置生成音频列表")
    # 只为OST=0 or 2的判断生成音频， OST=0 仅保留解说 OST=2 保留解说和原声
    tts_segments = [
        segment for segment in list_script
        if segment['OST'] in [0, 2]
    ]
    logger.debug(f"需要生成TTS的片段数: {len(tts_segments)}")

    tts_results = voice.tts_multiple(
        task_id=task_id,
        list_script=tts_segments,  # 只传入需要TTS的片段
        tts_engine=params.tts_engine,
        voice_name=params.voice_name,
        voice_rate=params.voice_rate,
        voice_pitch=params.voice_pitch,
    )

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)

    """
    3. 统一视频裁剪 - 基于OST类型的差异化裁剪策略
    """
    logger.info("\n\n## 3. 统一视频裁剪（基于OST类型）")

    # 使用新的统一裁剪策略
    video_clip_result = clip_video.clip_video_unified(
        video_origin_path=params.video_origin_path,
        script_list=list_script,
        tts_results=tts_results
    )

    # 更新 list_script 中的时间戳和路径信息
    tts_clip_result = {tts_result['_id']: tts_result['audio_file'] for tts_result in tts_results}
    subclip_clip_result = {
        tts_result['_id']: tts_result['subtitle_file'] for tts_result in tts_results
    }
    tts_duration_by_id = {
        tts_result["_id"]: float(tts_result.get("duration") or 0)
        for tts_result in tts_results
    }
    new_script_list = update_script.update_script_timestamps(
        list_script,
        video_clip_result,
        tts_clip_result,
        subclip_clip_result,
        tts_duration_by_id=tts_duration_by_id,
    )
    _persist_synced_script(task_id, new_script_list, video_script_path)

    logger.info(f"统一裁剪完成，处理了 {len(video_clip_result)} 个视频片段")

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=60)

    """
    4. 合并视频（先合并，再按重编码后时长校正时间轴）
    """
    final_video_paths = []
    combined_video_paths = []

    video_clips, clip_ost_list = _collect_video_clips_from_script(new_script_list)
    combined_video_path, new_script_list = _merge_video_clips_and_sync_timeline(
        task_id, new_script_list, video_clips, clip_ost_list, params, video_script_path
    )
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=75)

    """
    5. 合并音频和字幕（使用与 merger.mp4 一致的时间轴）
    """
    merged_audio_path, merged_subtitle_path, picture_narration_path = _merge_audio_and_subtitles(
        task_id, new_script_list, params, tts_segments=tts_segments
    )
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=85)

    """
    6. 合并字幕/BGM/配音/视频（延迟模式下先合成再 API 转写烧字幕）
    """
    output_video_path = _produce_final_video(
        task_id,
        params,
        combined_video_path=combined_video_path,
        merged_audio_path=merged_audio_path,
        merged_subtitle_path=merged_subtitle_path,
        picture_narration_path=picture_narration_path,
        list_script=list_script,
        new_script_list=new_script_list,
    )

    final_video_paths.append(output_video_path)
    combined_video_paths.append(combined_video_path)

    logger.success(f"统一处理任务 {task_id} 已完成, 生成 {len(final_video_paths)} 个视频.")

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths
    }
    sm.state.update_task(task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs)
    return kwargs


def validate_params(video_path, audio_path, output_file, params):
    """
    验证输入参数
    Args:
        video_path: 视频文件路径
        audio_path: 音频文件路径（可以为空字符串）
        output_file: 输出文件路径
        params: 视频参数

    Raises:
        FileNotFoundError: 文件不存在时抛出
        ValueError: 参数无效时抛出
    """
    if not video_path:
        raise ValueError("视频路径不能为空")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
    # 如果提供了音频路径，则验证文件是否存在
    if audio_path and not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
    if not output_file:
        raise ValueError("输出文件路径不能为空")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not params:
        raise ValueError("视频参数不能为空")


if __name__ == "__main__":
    task_id = "demo"

    # 提前裁剪是为了方便检查视频
    subclip_path_videos = {
        1: '/Users/apple/Desktop/home/NarratoAI/storage/temp/clip_video/113343d127b5a09d0bf84b68bd1b3b97/vid_00-00-05-390@00-00-57-980.mp4',
        2: '/Users/apple/Desktop/home/NarratoAI/storage/temp/clip_video/113343d127b5a09d0bf84b68bd1b3b97/vid_00-00-28-900@00-00-43-700.mp4',
        3: '/Users/apple/Desktop/home/NarratoAI/storage/temp/clip_video/113343d127b5a09d0bf84b68bd1b3b97/vid_00-01-17-840@00-01-27-600.mp4',
        4: '/Users/apple/Desktop/home/NarratoAI/storage/temp/clip_video/113343d127b5a09d0bf84b68bd1b3b97/vid_00-02-35-460@00-02-52-380.mp4',
        5: '/Users/apple/Desktop/home/NarratoAI/storage/temp/clip_video/113343d127b5a09d0bf84b68bd1b3b97/vid_00-06-59-520@00-07-29-500.mp4',
    }

    params = VideoClipParams(
        video_clip_json_path="/Users/apple/Desktop/home/NarratoAI/resource/scripts/2025-0507-223311.json",
        video_origin_path="/Users/apple/Desktop/home/NarratoAI/resource/videos/merged_video_4938.mp4",
    )
    start_subclip(task_id, params, subclip_path_videos)
