import unittest

from app.services.documentary.subtitle_typo_calibration import (
    calibrate_typo_from_screen_subtitle,
    normalize_subtitle_text,
    should_apply_typo_correction,
    subtitle_text_similarity,
)


class SubtitleTypoCalibrationTest(unittest.TestCase):
    def test_typo_correction_applies_for_similar_line(self):
        corrected = calibrate_typo_from_screen_subtitle("你号好吗", "你好吗")
        self.assertEqual("你好吗", corrected)

    def test_typo_correction_skips_unrelated_line(self):
        corrected = calibrate_typo_from_screen_subtitle("今天天气不错", "完全不同的对白")
        self.assertIsNone(corrected)

    def test_should_apply_typo_correction_rejects_rewrite(self):
        self.assertTrue(should_apply_typo_correction("你号", "你好"))
        self.assertFalse(should_apply_typo_correction("短句", "这是一句完全不同的长对白内容"))

    def test_similarity(self):
        self.assertGreaterEqual(subtitle_text_similarity("你号", "你好"), 0.5)
        self.assertEqual("你好", normalize_subtitle_text("你好！"))


if __name__ == "__main__":
    unittest.main()
