import tempfile
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from app.config import config as cfg
from app.config.defaults import (
    DEFAULT_ALT_BASE_URL,
    get_openai_compatible_ui_values,
    normalize_openai_compatible_model_name,
)


class ConfigBootstrapDefaultsTests(unittest.TestCase):
    def test_load_config_bootstraps_webui_llm_defaults(self):
        original_root_dir = cfg.root_dir
        original_config_file = cfg.config_file

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            example_file = tmp_path / "config.example.toml"
            example_file.write_text(
                """
[app]
vision_llm_provider = "openai"
vision_openai_model_name = "gemini/gemini-2.0-flash-lite"
vision_openai_api_key = ""
vision_openai_base_url = ""
text_llm_provider = "openai"
text_openai_model_name = "deepseek/deepseek-chat"
text_openai_api_key = ""
text_openai_base_url = ""
hide_config = true
""".strip()
                + "\n",
                encoding="utf-8",
            )

            config_path = tmp_path / "config.toml"
            try:
                cfg.root_dir = str(tmp_path)
                cfg.config_file = str(config_path)

                config_data = cfg.load_config()
                saved_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
            finally:
                cfg.root_dir = original_root_dir
                cfg.config_file = original_config_file

        self.assertEqual("openai", config_data["app"]["vision_llm_provider"])
        self.assertEqual("gemini-3.1-flash-lite", config_data["app"]["vision_openai_model_name"])
        self.assertEqual(DEFAULT_ALT_BASE_URL, config_data["app"]["vision_openai_base_url"])
        self.assertEqual("openai", config_data["app"]["text_llm_provider"])
        self.assertEqual("deepseek-v4-flash", config_data["app"]["text_openai_model_name"])
        self.assertEqual(DEFAULT_ALT_BASE_URL, config_data["app"]["text_openai_base_url"])
        self.assertEqual("gemini-3.1-flash-lite", saved_config["app"]["vision_openai_model_name"])
        self.assertEqual("deepseek-v4-flash", saved_config["app"]["text_openai_model_name"])
        self.assertTrue(saved_config["app"]["hide_config"])


class OpenAICompatibleModelDefaultsTests(unittest.TestCase):
    def test_ui_keeps_full_model_name_and_openai_provider(self):
        provider, model_name = get_openai_compatible_ui_values(
            "qwen-vl-max",
            "fallback-model",
        )

        self.assertEqual("openai", provider)
        self.assertEqual("qwen-vl-max", model_name)

    def test_normalize_only_strips_openai_prefix(self):
        self.assertEqual(
            "qwen-max",
            normalize_openai_compatible_model_name("openai/qwen-max"),
        )
        self.assertEqual(
            "qwen-max",
            normalize_openai_compatible_model_name("qwen-max"),
        )


if __name__ == "__main__":
    unittest.main()
