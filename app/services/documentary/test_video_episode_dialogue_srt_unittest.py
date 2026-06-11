#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest

from app.services.documentary.video_episode_dialogue_srt import (
    dedupe_important_dialogues,
    enrich_important_dialogues_with_srt,
)


class VideoEpisodeDialogueSrtTests(unittest.TestCase):
    def test_dedupe_near_duplicate_with_conflicting_speakers(self):
        dialogues = [
            {
                "speaker": "秦枫",
                "timestamp": "00:00:17",
                "quote": "有什么不能告诉我和大师兄的",
                "significance": "a",
            },
            {
                "speaker": "胡小跃",
                "timestamp": "00:00:19",
                "quote": "有什么不能告诉我和我大师兄的",
                "significance": "b",
            },
        ]
        merged, warnings = dedupe_important_dialogues(dialogues)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["timestamp"], "00:00:17")
        self.assertEqual(merged[0]["speaker"], "剧中未明确交代")
        self.assertTrue(warnings)

    def test_enrich_keeps_video_quote_when_srt_differs(self):
        srt = """1
00:00:16,500 --> 00:00:18,200
有什么不能告诉我和大师兄的
"""
        dialogues = [
            {
                "speaker": "秦枫",
                "timestamp": "00:00:17",
                "quote": "有什么不能告诉我和大师兄的",
                "significance": "试探",
            },
            {
                "speaker": "胡小跃",
                "timestamp": "00:00:19",
                "quote": "有什么不能告诉我和我大师兄的",
                "significance": "误标",
            },
        ]
        enriched, _warnings = enrich_important_dialogues_with_srt(dialogues, srt)
        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0]["timestamp"], "00:00:16")
        self.assertEqual(enriched[0]["speaker"], "剧中未明确交代")

    def test_enrich_prefers_on_screen_quote_over_wrong_srt(self):
        srt = """1
00:00:16,500 --> 00:00:18,200
ASR错字：有什么不能告诉我和大师兄
"""
        dialogues = [
            {
                "speaker": "秦枫",
                "timestamp": "00:00:17",
                "quote": "有什么不能告诉我和大师兄的",
                "significance": "画面硬字幕",
            },
        ]
        enriched, warnings = enrich_important_dialogues_with_srt(dialogues, srt)
        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0]["quote"], "有什么不能告诉我和大师兄的")
        self.assertEqual(enriched[0]["timestamp"], "00:00:16")
        self.assertTrue(any("差异较大" in item for item in warnings))


if __name__ == "__main__":
    unittest.main()
