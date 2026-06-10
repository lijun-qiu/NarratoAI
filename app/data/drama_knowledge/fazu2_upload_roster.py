#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""《罚罪2》人物头像上传名单（按出镜频率分级，供抽帧定妆照槽位）。"""

from __future__ import annotations

from typing import Any

# tier: core=高频 | medium=中等 | low=低频率/仅提及
FAZU2_CHARACTER_UPLOAD_ROSTER: tuple[dict[str, str], ...] = (
    # —— 高频（建议优先上传）——
    {"name": "秦枫", "tier": "core", "role_hint": "男一号"},
    {"name": "刘天也", "tier": "core", "role_hint": "大反派"},
    {"name": "文江燕", "tier": "core", "role_hint": "女主"},
    {"name": "叶天佑", "tier": "core", "role_hint": "局长"},
    {"name": "胡小跃", "tier": "core", "role_hint": "关键牺牲者"},
    {"name": "罗博", "tier": "core", "role_hint": "嚣张打手"},
    {"name": "文琴", "tier": "core", "role_hint": "族长"},
    {"name": "赵鹏", "tier": "core", "role_hint": "村主任"},
    {"name": "马金", "tier": "core", "role_hint": "金鼎董事长"},
    {"name": "麦洪超", "tier": "core", "role_hint": "大师兄"},
    {"name": "彭含章", "tier": "core", "role_hint": "副局长"},
    {"name": "钟雁宁", "tier": "core", "role_hint": "支队长"},
    {"name": "严明", "tier": "core", "role_hint": "常务副局长"},
    {"name": "吉竹江", "tier": "core", "role_hint": "内鬼"},
    {"name": "陈水发", "tier": "core", "role_hint": "腐败主任"},
    {"name": "张欣", "tier": "core", "role_hint": "白手套"},
    {"name": "贺彪", "tier": "core", "role_hint": "儒颂集团"},
    {"name": "王旭", "tier": "core", "role_hint": "商会副会长"},
    {"name": "宋浩", "tier": "core", "role_hint": "洗钱相关"},
    {"name": "钱雨虹", "tier": "core", "role_hint": "洗钱相关"},
    # —— 中等频率 ——
    {"name": "秦立志", "tier": "medium", "role_hint": "父亲，已故，仅回忆"},
    {"name": "汪涛", "tier": "medium", "role_hint": "秦枫下属"},
    {"name": "杨振刚", "tier": "medium", "role_hint": "秦枫下属"},
    {"name": "刘如意", "tier": "medium", "role_hint": "秦枫下属"},
    {"name": "边静", "tier": "medium", "role_hint": "秦枫下属"},
    {"name": "秦陶义", "tier": "medium", "role_hint": "刘天也跟班"},
    {"name": "刘天飞", "tier": "medium", "role_hint": "刘天也跟班"},
    {"name": "赵子怡", "tier": "medium", "role_hint": "联姻妻子"},
    {"name": "周思思", "tier": "medium", "role_hint": "洗钱同学"},
    {"name": "郑镐", "tier": "medium", "role_hint": "顶包"},
    {"name": "楚青桐", "tier": "medium", "role_hint": "省厅副厅长"},
    {"name": "叶斯远", "tier": "medium", "role_hint": "侄子"},
    {"name": "徐丽", "tier": "medium", "role_hint": "受害者相关"},
    {"name": "冷珊", "tier": "medium", "role_hint": "受害者相关"},
    {"name": "文波", "tier": "medium", "role_hint": "下一代"},
    {"name": "丁小帅", "tier": "medium", "role_hint": "下一代"},
    {"name": "吴代南", "tier": "medium", "role_hint": "镇长"},
    {"name": "弘沐寿", "tier": "medium", "role_hint": "最大保护伞"},
    {"name": "文江勇", "tier": "medium", "role_hint": "文琴家人"},
    {"name": "孙娜", "tier": "medium", "role_hint": "文琴家人"},
    {"name": "秦立民", "tier": "medium", "role_hint": "村民"},
    {"name": "丁铁军", "tier": "medium", "role_hint": "丁小帅父亲"},
    {"name": "赵老大", "tier": "medium", "role_hint": "渔霸"},
    {"name": "老林", "tier": "medium", "role_hint": "马金手下"},
    {"name": "瓜子佬", "tier": "medium", "role_hint": "马金手下"},
    {"name": "刀疤", "tier": "medium", "role_hint": "马金手下"},
    {"name": "老九", "tier": "medium", "role_hint": "马金手下"},
    {"name": "方磊", "tier": "medium", "role_hint": "马金手下"},
    {"name": "王依依", "tier": "medium", "role_hint": "王旭女儿"},
    {"name": "贺刚", "tier": "medium", "role_hint": "贺彪弟弟"},
    {"name": "徐家俊", "tier": "medium", "role_hint": "记者"},
    {"name": "孟雨", "tier": "medium", "role_hint": "受害者"},
    {"name": "乔德福", "tier": "medium", "role_hint": "受害者"},
    {"name": "冯江", "tier": "medium", "role_hint": "伸冤"},
    {"name": "黎政", "tier": "medium", "role_hint": "好官"},
    {"name": "赵文轩", "tier": "medium", "role_hint": "边缘角色"},
    {"name": "文葆宝", "tier": "medium", "role_hint": "边缘角色"},
    # —— 低频率 / 仅提及 ——
    {"name": "乔德福老婆", "tier": "low", "role_hint": "仅提及"},
    {"name": "罗小美", "tier": "low", "role_hint": "仅提及"},
)

TIER_LABELS: dict[str, str] = {
    "core": "高频（建议优先上传）",
    "medium": "中等频率",
    "low": "低频率 / 仅提及",
}

TIER_ORDER: tuple[str, ...] = ("core", "medium", "low")

DRAMA_UPLOAD_ROSTERS: dict[str, tuple[dict[str, str], ...]] = {
    "罚罪": FAZU2_CHARACTER_UPLOAD_ROSTER,
    "罚罪2": FAZU2_CHARACTER_UPLOAD_ROSTER,
}


def roster_for_drama(drama_id: str) -> tuple[dict[str, str], ...] | None:
    drama_id = (drama_id or "").strip()
    return DRAMA_UPLOAD_ROSTERS.get(drama_id)


def roster_groups_for_drama(drama_id: str) -> list[dict[str, Any]]:
    roster = roster_for_drama(drama_id)
    if not roster:
        return []
    grouped: dict[str, list[dict[str, str]]] = {tier: [] for tier in TIER_ORDER}
    for item in roster:
        tier = str(item.get("tier") or "medium")
        grouped.setdefault(tier, []).append(dict(item))
    return [
        {"tier": tier, "label": TIER_LABELS.get(tier, tier), "characters": grouped.get(tier, [])}
        for tier in TIER_ORDER
        if grouped.get(tier)
    ]
