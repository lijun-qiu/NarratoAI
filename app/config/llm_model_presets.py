"""LLM 模型与网关预设，供 WebUI 选择与默认配置。"""

from __future__ import annotations

CUSTOM_MODEL_OPTION = "__custom__"

DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# (显示名称, 模型 ID)
VISION_MODEL_PRESETS: list[tuple[str, str]] = [
<<<<<<< HEAD
    ("Gemini 3.1 Flash Lite（推荐）", "gemini-3.1-flash-lite"),
    ("Gemini 3 Flash Preview", "gemini-3-flash-preview"),
=======
    ("Gemini 3 Flash Preview（推荐）", "gemini-3-flash-preview"),
    ("Gemini 3.1 Flash Lite", "gemini-3.1-flash-lite"),
>>>>>>> 5ff28823f54ba6da58cf214b2e2d57a09a1015df
    ("Qwen-VL-Max", "qwen-vl-max"),
    ("Qwen-VL-Plus", "qwen-vl-plus"),
    ("Qwen2.5-VL-72B-Instruct", "qwen2.5-vl-72b-instruct"),
    ("Qwen2.5-VL-32B-Instruct", "qwen2.5-vl-32b-instruct"),
    ("GPT-4o", "gpt-4o"),
    ("Gemini 2.0 Flash", "gemini-2.0-flash"),
    ("自定义模型", CUSTOM_MODEL_OPTION),
]

TEXT_MODEL_PRESETS: list[tuple[str, str]] = [
    ("DeepSeek V4 Flash（推荐）", "deepseek-v4-flash"),
    ("GPT-4o", "gpt-4o"),
    ("Qwen-Max", "qwen-max"),
    ("Qwen-Plus", "qwen-plus"),
    ("Qwen-Turbo", "qwen-turbo"),
    ("Qwen-Long", "qwen-long"),
    ("DeepSeek-V3", "deepseek-v3"),
    ("GLM-4-Plus", "glm-4-plus"),
    ("自定义模型", CUSTOM_MODEL_OPTION),
]

DEFAULT_ALT_BASE_URL = "https://api.4022543.xyz/v1"

LLM_GATEWAY_PRESETS: list[tuple[str, str]] = [
    ("阿里百炼 DashScope", DEFAULT_DASHSCOPE_BASE_URL),
    ("4022 网关", DEFAULT_ALT_BASE_URL),
    ("SiliconFlow", "https://api.siliconflow.cn/v1"),
    ("OpenRouter", "https://openrouter.ai/api/v1"),
    ("自定义网关", ""),
]

VISION_PRESET_MODEL_IDS = {model_id for _, model_id in VISION_MODEL_PRESETS if model_id != CUSTOM_MODEL_OPTION}
TEXT_PRESET_MODEL_IDS = {model_id for _, model_id in TEXT_MODEL_PRESETS if model_id != CUSTOM_MODEL_OPTION}


def match_preset_index(
    presets: list[tuple[str, str]],
    model_id: str,
) -> int:
    normalized = (model_id or "").strip()
    for index, (_, value) in enumerate(presets):
        if value == normalized:
            return index
    return len(presets) - 1


def resolve_gateway_label(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    for label, value in LLM_GATEWAY_PRESETS:
        if value and normalized == value.rstrip("/"):
            return label
    return LLM_GATEWAY_PRESETS[-1][0]
