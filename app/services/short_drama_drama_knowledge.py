#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""短剧解说：剧集人物关系知识库（分析前须熟悉）。"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

from loguru import logger

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_DEFAULT_KNOWLEDGE_BY_THEME: tuple[tuple[str, str], ...] = (
    ("罚罪2", "app/data/drama_knowledge/fazu2_relationships.md"),
    ("罚罪", "app/data/drama_knowledge/fazu2_relationships.md"),
    ("fazu", "app/data/drama_knowledge/fazu2_relationships.md"),
)

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

# 常见张冠李戴 / 错别字（左为错误写法）
COMMON_NAME_MISTAKES: tuple[tuple[str, str, str], ...] = (
    ("胡小月", "胡小跃", "胡小跃是男刑警，叶天佑弟子，非「胡小月」"),
    ("胡晓月", "胡小跃", "同上"),
    ("小月", "小跃", "胡小跃简称应为小跃"),
    ("秦峰", "秦枫", "男一号秦枫，禁止写秦峰"),
    ("罗伯", "罗博", "马金手下罗博，禁止写罗伯"),
)

# 构思蓝图：规范名 + 常见谐音/ASR/简称（均视为同一人，勿拆成两个角色）
PLOT_BLUEPRINT_NAME_ALIAS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("胡小跃", ("胡小月", "胡晓月", "小月", "小跃", "胡队")),
    ("秦枫", ("秦峰", "峰啊")),
    ("罗博", ("罗伯",)),
    ("叶天佑", ("老叶", "叶局")),
    ("麦洪超", ("麦队",)),
    ("彭含章", ("彭姐",)),
    ("文琴", ("文妈",)),
)


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

    theme_text = (theme or "").strip().lower()
    for keyword, rel_path in _DEFAULT_KNOWLEDGE_BY_THEME:
        if keyword.lower() in theme_text:
            return os.path.join(_PROJECT_ROOT, rel_path.replace("/", os.sep))
    return ""


def _is_subtitle_analysis_enabled(settings: dict[str, Any] | None) -> bool:
    cfg = settings or {}
    if cfg.get("enable_subtitle_analysis_drama_knowledge") is False:
        return False
    if cfg.get("enable_drama_knowledge") is False:
        return False
    return True


def _is_frame_analysis_enabled(settings: dict[str, Any] | None) -> bool:
    cfg = settings or {}
    if cfg.get("enable_drama_knowledge") is False:
        return False
    frame_flag = cfg.get("enable_frame_analysis_drama_knowledge")
    if frame_flag is False:
        return False
    if frame_flag is True:
        return True
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
) -> tuple[str, set[str]]:
    """
    返回 (注入 prompt 的 Markdown 块, 知识库人物名集合)。
    无匹配知识库时返回 ("", set())。
    """
    if not _is_subtitle_analysis_enabled(settings):
        return "", set()

    max_chars = int(
        (settings or {}).get("subtitle_analysis_drama_knowledge_max_chars", 10000) or 10000
    )
    work = (theme or "本剧").strip()
    header = f"""## 剧集人物关系对照（**分析前必读** · 熟悉后再对照字幕/抽帧）

以下为 **{work}** 官方人物关系与身份，请**先通读本节**再写「主要人物表」与时间线：
- 字幕/抽帧出现的人名须在本对照或下方「抽帧字幕人物索引」中可对应
- **谐音/ASR 错字须归并为同一人**（小月/胡小月→胡小跃，秦峰→秦枫，罗伯→罗博，老叶→叶天佑）；人物表每人只列规范名一条
- **禁止张冠李戴**：勿把 A 的台词、遭遇、关系写成 B（如秦枫≠刘天也，文江燕是刘天也妹妹）
- **关系须准确**：三兄妹为秦枫、刘天也、文江燕（文琴养子女）；叶天佑是局长、秦枫/胡小跃/麦洪超的师傅
- 硬字幕简称（如老叶）须与对照表核对后再写入人物表"""
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
    """抽帧视觉分析：注入精简版人物关系对照（与联合构思共用同一 md 源文件）。"""
    if not _is_frame_analysis_enabled(settings):
        return "", set()

    max_chars = int(
        (settings or {}).get("frame_analysis_drama_knowledge_max_chars", 5000) or 5000
    )
    work = (theme or "本剧").strip()
    header = f"""## 剧集人物关系对照（抽帧 · {work} · **仅作校正，不可猜人**）

本节用于**校正**已出现在本批字幕/硬字幕/画面中的人物身份，**不是**出场名单：
- **先**看本批硬字幕/SRT/subtitle_entries，**再**用本节校正谐音与关系（秦峰→秦枫，老叶→叶天佑）
- **禁止**因对照表里有某角色，就把该名字写到本批未在字幕/画面中出现的人身上
- 仅字幕或硬字幕出现姓名时才写真名；面孔无法确认时用「未名人员(男/女)」
- **禁止**：胡小月/小月→须写胡小跃；秦峰→秦枫；罗伯→罗博
- **勿混**：叶天佑/老叶（局长）≠ 伟业；秦枫≠刘天也；文江燕是刘天也亲妹妹"""
    return _build_drama_knowledge_block(
        theme=theme,
        settings=settings,
        max_chars=max_chars,
        header=header,
        log_label="抽帧人物关系知识库",
    )


def find_name_mistakes_in_text(text: str) -> list[str]:
    """检测构思输出中的常见人名/关系笔误。"""
    content = text or ""
    issues: list[str] = []
    for wrong, correct, hint in COMMON_NAME_MISTAKES:
        if wrong in content and wrong != correct:
            issues.append(f"出现错误写法「{wrong}」，应为「{correct}」——{hint}")
    return issues


def build_plot_blueprint_name_unification_section(
    theme: str = "",
    settings: dict[str, Any] | None = None,
) -> str:
    """构思蓝图：抽帧/字幕谐音、ASR 错字、简称归并为同一人物。"""
    path = _resolve_knowledge_path(theme, settings)
    if not path and not any(
        keyword in (theme or "").lower() for keyword, _ in _DEFAULT_KNOWLEDGE_BY_THEME
    ):
        lines = [
            "## 人名谐音/简称归并（硬性）",
            "- 抽帧 subtitle_entries、硬字幕、observation 中的**谐音/ASR 错字/简称**，"
            "若明显指同一人，须**归并为同一角色**，人物表只列一条",
            "- 人物表、时间线、OST 清单用**规范全名**；引用原声台词时可保留抽帧原文",
            "- **禁止**因写法不同拆成两个角色（如「小月」与「胡小跃」不得各占一行）",
        ]
        return "\n".join(lines)

    alias_lines = [
        f"- **{canonical}** ← {(' / '.join(aliases))}"
        for canonical, aliases in PLOT_BLUEPRINT_NAME_ALIAS_GROUPS
    ]
    return "\n".join(
        [
            "## 人名谐音/ASR 归并（硬性 · 同一人）",
            "分析抽帧与字幕索引时，下列写法**一律视为同一人**；"
            "**主要人物表只写规范名一条**，括号内可注「又名/字幕常写：…」：",
            *alias_lines,
            "- **叶天佑（老叶）≠ 伟业**：不同人物，禁止合并",
            "- **秦枫 ≠ 刘天也 ≠ 文江燕**：禁止因关系相近而合并",
            "- 时间线、OST 清单、叙事顺序中的**说话人/当事人**用规范名；"
            "「建议保留原声」条目内**台词原文**可保留抽帧 subtitle_entries 原字（如「小月」）",
            "- 抽帧 observation 与 subtitle 人名冲突时：以**关系对照表 + 上下文**归并到同一人，勿新增虚构角色",
        ]
    )


def correct_name_mistakes_in_text(text: str) -> str:
    """将文本中的常见人名笔误替换为规范写法。"""
    corrected = text or ""
    if not corrected:
        return corrected
    for wrong, correct, _ in COMMON_NAME_MISTAKES:
        if wrong and wrong != correct:
            corrected = corrected.replace(wrong, correct)
    return corrected


def apply_name_corrections_to_segment(segment: dict[str, Any]) -> None:
    """就地修正 scene_segment 各字段中的常见人名笔误（字幕字段保持 SRT 原文）。"""
    if not isinstance(segment, dict):
        return
    for key in ("scene", "observation", "action", "emotion", "key_visual"):
        value = segment.get(key)
        if isinstance(value, str) and value.strip():
            segment[key] = correct_name_mistakes_in_text(value)


def apply_name_corrections_to_observation(observation: dict[str, Any]) -> None:
    if not isinstance(observation, dict):
        return
    value = observation.get("observation")
    if isinstance(value, str) and value.strip():
        observation["observation"] = correct_name_mistakes_in_text(value)


def apply_name_corrections_to_frame_analysis_artifact(artifact: dict[str, Any]) -> None:
    """抽帧 artifact 完成后：统一修正 scene_segments / frame_observations 中的人名笔误。"""
    if not isinstance(artifact, dict):
        return
    for segment in artifact.get("scene_segments") or []:
        apply_name_corrections_to_segment(segment)
    for batch in artifact.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        for segment in batch.get("scene_segments") or []:
            apply_name_corrections_to_segment(segment)
        for observation in batch.get("frame_observations") or batch.get("observations") or []:
            apply_name_corrections_to_observation(observation)
    for observation in artifact.get("frame_observations") or []:
        apply_name_corrections_to_observation(observation)
