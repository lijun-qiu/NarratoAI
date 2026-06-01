#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project: NarratoAI
@Description: 电影/电视剧解说提示词模块
"""

from .plot_analysis import PlotAnalysisPrompt
from .script_generation import ScriptGenerationPrompt
from .work_briefing import WorkBriefingPrompt
from ..manager import PromptManager


def register_prompts():
    """注册影视解说相关的提示词"""
    work_briefing_prompt = WorkBriefingPrompt()
    PromptManager.register_prompt(work_briefing_prompt, is_default=True)

    plot_analysis_prompt = PlotAnalysisPrompt()
    PromptManager.register_prompt(plot_analysis_prompt, is_default=True)

    script_generation_prompt = ScriptGenerationPrompt()
    PromptManager.register_prompt(script_generation_prompt, is_default=True)


__all__ = [
    "WorkBriefingPrompt",
    "PlotAnalysisPrompt",
    "ScriptGenerationPrompt",
    "register_prompts",
]
