#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""剧集人物关系知识库：仅当配置显式指定文件路径时注入，不自动匹配剧名。"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

from loguru import logger

from app.services.documentary.documentary_settings import (
    FRAME_UNKNOWN_CHARACTER_FEMALE,
    FRAME_UNKNOWN_CHARACTER_MALE,
)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_NAME_SECTION_RE = re.compile(
    r"^###\s+(?:\d+\.\s*)?([\u4e00-\u9fffA-Za-z·]{2,6})（",
    re.MULTILINE,
)
_NAME_LIST_ITEM_RE = re.compile(
    r"^-\s+\*\*([\u4e00-\u9fffA-Za-z·]{2,6})\*\*",
    re.MULTILINE,
)
_NAME_TABLE_RE = re.compile(
    r"^\|\s*([\u4e00-\u9fffA-Za-z·]{2,6})\s*\|",
    re.MULTILINE,
)


def resolve_obvious_character_relations(drama_id: str) -> tuple:
    """通用版不维护剧专属人物关系表。"""
    return ()


def build_plot_blueprint_character_relationship_table_section(
    theme: str,
    settings: dict[str, Any] | None = None,
    *,
    use_video_episode_analysis: bool = False,
) -> tuple[str, set[str]]:
    """构思蓝图：注入文字版人物关系表（须配置 knowledge 文件路径）。"""
    if not _is_subtitle_analysis_enabled(settings):
        return "", set()

    max_chars = int(
        (settings or {}).get("subtitle_analysis_drama_knowledge_max_chars", 10000) or 10000
    )
    work = (theme or "本剧").strip()
    visual_ref = (
        "整片视频分析人物索引"
        if use_video_episode_analysis
        else "抽帧字幕人物索引"
    )
    material_ref = "整片视频分析" if use_video_episode_analysis else "字幕/抽帧"
    header = f"""## 人物关系表（文字 · **分析前必读** · 与{material_ref}联合分析）

以下为 **{work}** 人物关系表，请**先通读本表**再写「主要人物表」与时间线：
- 写蓝图时须**同时对照**本表、{material_ref} 与字幕，交叉验证人名/关系/阵营
- {material_ref}出现的人名须能在本表中找到对应身份与关系
- **禁止张冠李戴**：勿把 A 的台词、遭遇、关系写成 B
- **关系须与本表一致**；硬字幕简称须与对照表核对后再写入人物表"""
    return _build_drama_knowledge_block(
        theme=theme,
        settings=settings,
        max_chars=max_chars,
        header=header,
        log_label="人物关系表",
    )


def build_frame_obvious_relationship_hint(drama_id: str = "") -> str:
    """抽帧 prompt：已写入姓名的两人之间可补明显关系（通用版无预设关系）。"""
    return ""


def _resolve_knowledge_path(theme: str, settings: dict[str, Any] | None) -> str:
    cfg = settings or {}
    explicit = str(
        cfg.get("subtitle_analysis_drama_knowledge_file")
        or cfg.get("drama_knowledge_file")
        or ""
    ).strip()
    if explicit:
        if os.path.isabs(explicit):
            return explicit
        return os.path.join(_PROJECT_ROOT, explicit.replace("/", os.sep))
    return ""


def _is_subtitle_analysis_enabled(settings: dict[str, Any] | None) -> bool:
    cfg = settings or {}
    if cfg.get("enable_subtitle_analysis_drama_knowledge") is False:
        return False
    if cfg.get("enable_drama_knowledge") is False:
        return False
    return bool(_resolve_knowledge_path("", cfg))


def _is_frame_analysis_enabled(settings: dict[str, Any] | None) -> bool:
    cfg = settings or {}
    if cfg.get("enable_drama_knowledge") is False:
        return False
    frame_flag = cfg.get("enable_frame_analysis_drama_knowledge")
    if frame_flag is False:
        return False
    if frame_flag is True:
        return bool(_resolve_knowledge_path("", cfg))
    return False


@lru_cache(maxsize=4)
def _load_knowledge_file(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8") as fp:
            return fp.read().strip()
    except OSError as exc:
        logger.warning(f"读取剧集知识库失败 {path}: {exc}")
        return ""


def extract_drama_knowledge_names(content: str) -> set[str]:
    """从知识库 Markdown 提取人物姓名。"""
    if not content:
        return set()
    names: set[str] = set()
    for pattern in (_NAME_SECTION_RE, _NAME_LIST_ITEM_RE, _NAME_TABLE_RE):
        for match in pattern.finditer(content):
            name = match.group(1).strip()
            if len(name) >= 2:
                names.add(name)
    return names


def _build_drama_knowledge_block(
    *,
    theme: str,
    settings: dict[str, Any] | None,
    max_chars: int,
    header: str,
    log_label: str,
) -> tuple[str, set[str]]:
    path = _resolve_knowledge_path(theme, settings)
    if not path:
        return "", set()

    raw = _load_knowledge_file(path)
    if not raw:
        return "", set()

    body = raw
    if len(body) > max_chars:
        body = body[: max_chars - 24].rstrip() + "\n\n…（人物关系对照已截断）"

    known_names = extract_drama_knowledge_names(raw)
    block = f"{header.strip()}\n\n{body}\n"
    logger.info(
        f"已注入{log_label}：{os.path.basename(path)}，"
        f"约 {len(body)} 字，{len(known_names)} 个姓名"
    )
    return block, known_names


def build_short_drama_drama_knowledge_section(
    theme: str,
    settings: dict[str, Any] | None = None,
    *,
    use_video_episode_analysis: bool = False,
) -> tuple[str, set[str]]:
    """
    返回 (注入 prompt 的 Markdown 块, 知识库人物名集合)。
    无配置知识库路径时返回 ("", set())。
    """
    if not _is_subtitle_analysis_enabled(settings):
        return "", set()

    max_chars = int(
        (settings or {}).get("subtitle_analysis_drama_knowledge_max_chars", 10000) or 10000
    )
    work = (theme or "本剧").strip()
    visual_ref = (
        "整片视频分析人物索引"
        if use_video_episode_analysis
        else "抽帧字幕人物索引"
    )
    material_ref = "整片视频分析" if use_video_episode_analysis else "字幕/抽帧"
    header = f"""## 剧集人物关系对照（**分析前必读** · 熟悉后再对照{material_ref}）

以下为 **{work}** 官方人物关系与身份，请**先通读本节**再写「主要人物表」与时间线：
- {material_ref}出现的人名须在本对照或下方「{visual_ref}」中可对应
- **禁止张冠李戴**：勿把 A 的台词、遭遇、关系写成 B
- 硬字幕简称须与对照表核对后再写入人物表"""
    return _build_drama_knowledge_block(
        theme=theme,
        settings=settings,
        max_chars=max_chars,
        header=header,
        log_label="剧集人物关系知识库",
    )


def build_frame_analysis_drama_knowledge_section(
    theme: str,
    settings: dict[str, Any] | None = None,
) -> tuple[str, set[str]]:
    """抽帧视觉分析：注入精简版人物关系对照（须配置 knowledge 文件路径）。"""
    if not _is_frame_analysis_enabled(settings):
        return "", set()

    max_chars = int(
        (settings or {}).get("frame_analysis_drama_knowledge_max_chars", 5000) or 5000
    )
    work = (theme or "本剧").strip()
    header = f"""## 剧集人物关系对照（抽帧 · {work} · **仅作校正，不可猜人**）

本节用于**校正**本批已完成**头像/定妆照面孔匹配**的人名写法，**不是**出场名单：
- **唯一写名途径**：本批画面中脸与头像/定妆照一致（≥1 帧匹配）→ 写规范姓名；或后帧匹配后，前序**同一身形+同一服装**可确认同一人时回溯写名
- **硬字幕/SRT** 仅摘录对白；**禁止**凭称呼猜人，也**禁止**整批便衣统一替换
- 面孔无法匹配任一头像 → 「{FRAME_UNKNOWN_CHARACTER_MALE}」「{FRAME_UNKNOWN_CHARACTER_FEMALE}」或便衣/警员等可见描述
- **两人姓名均已由面孔匹配写入**时，可补明显关系（师徒/父子/上下级等）"""
    return _build_drama_knowledge_block(
        theme=theme,
        settings=settings,
        max_chars=max_chars,
        header=header,
        log_label="抽帧人物关系知识库",
    )


def find_name_mistakes_in_text(text: str) -> list[str]:
    """通用版不维护剧专属人名纠错表。"""
    return []


def build_plot_blueprint_name_unification_section(
    theme: str = "",
    settings: dict[str, Any] | None = None,
    *,
    use_video_episode_analysis: bool = False,
) -> str:
    """构思蓝图：整片视频分析/抽帧与字幕谐音、ASR 错字、简称归并为同一人物。"""
    source_hint = (
        "整片视频分析 important_dialogues / involved_characters"
        if use_video_episode_analysis
        else "抽帧 subtitle_entries、硬字幕、observation"
    )
    index_label = (
        "整片视频分析人物索引"
        if use_video_episode_analysis
        else "抽帧字幕人物索引"
    )
    visual_source = "整片视频分析" if use_video_episode_analysis else "抽帧"
    return "\n".join(
        [
            "## 人名谐音/ASR 归并（硬性 · 同一人）",
            f"分析{visual_source}与字幕索引时，**谐音/ASR 错字/简称**若明显指同一人，"
            "须**归并为同一角色**，人物表只列一条规范全名；",
            f"- {source_hint} 与 SRT 人名冲突时：以**{index_label} + 关系对照表**归并到同一人，勿新增虚构角色",
            "- 时间线、OST 清单、叙事顺序中的**说话人/当事人**须用**规范名**；"
            "「建议保留原声」条目内**台词原文**可保留 SRT 原字",
            "- **禁止**因写法不同拆成两个角色",
        ]
    )


def correct_name_mistakes_in_text(text: str) -> str:
    """通用版不做剧专属人名替换。"""
    return text or ""


def apply_name_corrections_to_segment(segment: dict[str, Any]) -> None:
    """通用版不修改 segment 人名。"""
    return


def apply_name_corrections_to_observation(observation: dict[str, Any]) -> None:
    return


def apply_name_corrections_to_frame_analysis_artifact(artifact: dict[str, Any]) -> None:
    """通用版抽帧 artifact 不做剧专属人名后处理。"""
    return
