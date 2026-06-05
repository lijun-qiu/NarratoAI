import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config.instance import init_instance_paths


class InstancePathsTests(unittest.TestCase):
    def test_single_instance_uses_project_storage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_config = os.path.join(tmp_dir, "config.toml")
            Path(base_config).write_text("[app]\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("NARRATO_INSTANCE_ID", None)
                os.environ.pop("NARRATO_PORT", None)
                paths = init_instance_paths(tmp_dir, base_config)

            self.assertEqual("", paths.instance_id)
            self.assertEqual(base_config, paths.config_file)
            self.assertEqual(os.path.join(tmp_dir, "storage"), paths.storage_root)
            self.assertEqual(8501, paths.port)

    def test_numeric_instance_gets_isolated_paths_and_port(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_config = os.path.join(tmp_dir, "config.toml")
            Path(base_config).write_text("[app]\nhide_config = true\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {"NARRATO_INSTANCE_ID": "2"},
                clear=False,
            ):
                os.environ.pop("NARRATO_PORT", None)
                paths = init_instance_paths(tmp_dir, base_config)

            instance_config = os.path.join(tmp_dir, "instances", "2", "config.toml")
            self.assertEqual("2", paths.instance_id)
            self.assertEqual(instance_config, paths.config_file)
            self.assertTrue(os.path.isfile(instance_config))
            self.assertEqual(
                os.path.join(tmp_dir, "instances", "2", "storage"),
                paths.storage_root,
            )
            self.assertEqual(8502, paths.port)
            self.assertTrue(os.path.isdir(os.path.join(paths.storage_root, "tasks")))

    def test_custom_port_override(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_config = os.path.join(tmp_dir, "config.toml")
            Path(base_config).write_text("[app]\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {"NARRATO_INSTANCE_ID": "dev", "NARRATO_PORT": "8600"},
                clear=False,
            ):
                paths = init_instance_paths(tmp_dir, base_config)

            self.assertEqual("dev", paths.instance_id)
            self.assertEqual(8600, paths.port)


if __name__ == "__main__":
    unittest.main()
