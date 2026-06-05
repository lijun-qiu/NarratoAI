import json
import tempfile
import unittest
from unittest.mock import patch

from app.services.documentary.subtitle_refinement_service import (
    _build_batch_frame_context,
    _entries_in_range,
    _parse_refinement_response,
    refine_subtitle_with_frame_analysis,
)
from app.services.srt_utils import SrtEntry, write_srt_file


class SubtitleRefinementServiceTest(unittest.TestCase):
    def test_parse_refinement_response(self):
        payload = '[{"index": 1, "text": "校正后"}, {"index": 2, "text": "第二句"}]'
        parsed = _parse_refinement_response(payload, 2)
        self.assertEqual("校正后", parsed.get(1))
        self.assertEqual("第二句", parsed.get(2))

    def test_entries_in_range(self):
        entries = [
            SrtEntry(start_ms=0, end_ms=2000, text="a"),
            SrtEntry(start_ms=5000, end_ms=7000, text="b"),
        ]
        matched = _entries_in_range(entries, 0, 3000)
        self.assertEqual(1, len(matched))
        self.assertEqual(0, matched[0][0])

    def test_build_batch_frame_context(self):
        context = _build_batch_frame_context(
            {
                "time_range": "00:00:00,000-00:00:05,000",
                "overall_activity_summary": "测试摘要",
                "frame_observations": [
                    {
                        "timestamp": "00:00:01,000",
                        "observation": "人物对话",
                        "burned_in_subtitle": "硬字幕原文",
                    },
                ],
            }
        )
        self.assertIn("测试摘要", context)
        self.assertIn("硬字幕原文", context)
        self.assertIn("画面硬字幕", context)

    @patch(
        "app.services.documentary.subtitle_refinement_service._refine_subtitle_chunk",
        return_value={0: "校正对白"},
    )
    def test_refine_subtitle_with_frame_analysis_writes_file(self, _mock_chunk):
        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_path = f"{tmp_dir}/demo_transcribed.srt"
            write_srt_file(
                [SrtEntry(start_ms=1000, end_ms=3000, text="错字对白")],
                subtitle_path,
            )
            analysis_path = f"{tmp_dir}/demo_frame_analysis.json"
            with open(analysis_path, "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "video_path": f"{tmp_dir}/demo.mp4",
                        "batches": [
                            {
                                "batch_index": 0,
                                "time_range": "00:00:00,000-00:00:05,000",
                                "overall_activity_summary": "摘要",
                                "frame_observations": [],
                            }
                        ],
                    },
                    fp,
                    ensure_ascii=False,
                )
            output_path = f"{tmp_dir}/demo_refined.srt"
            result = refine_subtitle_with_frame_analysis(
                subtitle_path=subtitle_path,
                analysis_json_path=analysis_path,
                output_path=output_path,
            )
            self.assertEqual(output_path, result)
            with open(result, "r", encoding="utf-8") as fp:
                content = fp.read()
            self.assertIn("校正对白", content)


if __name__ == "__main__":
    unittest.main()
