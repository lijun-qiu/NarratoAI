import json
import tempfile
import unittest
from unittest.mock import patch

import PIL.Image

from app.services.documentary.hard_subtitle_ocr_service import (
    OcrFrameHit,
    _apply_ocr_to_entries,
    _collect_ocr_frames,
    _parse_ocr_batch_response,
    _pick_ocr_text,
    _timestamp_from_keyframe_name,
    calibrate_subtitle_with_hard_subtitle_ocr,
    extract_ocr_hits_from_artifact,
)
from app.services.documentary.subtitle_typo_calibration import normalize_subtitle_text
from app.services.srt_utils import SrtEntry, write_srt_file


class HardSubtitleOcrServiceTest(unittest.TestCase):
    def test_timestamp_from_keyframe_name(self):
        ts = _timestamp_from_keyframe_name("/tmp/keyframe_000075_000003000.jpg")
        self.assertEqual("00:00:03,000", ts)

    def test_parse_ocr_batch_response(self):
        payload = (
            '{"frame_results": ['
            '{"index": 1, "has_subtitle": true, "text": "你好"}, '
            '{"index": 2, "has_subtitle": false, "text": ""}'
            "]}"
        )
        rows = _parse_ocr_batch_response(payload, 2)
        self.assertTrue(rows[0]["has_subtitle"])
        self.assertEqual("你好", rows[0]["text"])
        self.assertFalse(rows[1]["has_subtitle"])

    def test_normalize_subtitle_text(self):
        self.assertEqual("你好世界", normalize_subtitle_text("你好，世界！"))

    def test_pick_ocr_text_majority(self):
        hits = [
            OcrFrameHit("a.jpg", 1000, "00:00:01,000", "你好", True),
            OcrFrameHit("b.jpg", 1200, "00:00:01,200", "你好", True),
            OcrFrameHit("c.jpg", 1400, "00:00:01,400", "再见", True),
        ]
        self.assertEqual("你好", _pick_ocr_text(hits, 2))

    def test_collect_ocr_frames_from_batches(self):
        artifact = {
            "batches": [
                {
                    "frame_paths": ["/tmp/keyframe_000000_000001000.jpg"],
                    "frame_observations": [
                        {"timestamp": "00:00:01,000", "observation": "test"},
                    ],
                }
            ]
        }
        with patch("app.services.documentary.hard_subtitle_ocr_service.os.path.isfile", return_value=True):
            frames = _collect_ocr_frames(artifact)
        self.assertEqual(1, len(frames))
        self.assertEqual(1000, frames[0]["timestamp_ms"])

    def test_apply_ocr_skips_unrelated_screen_subtitle(self):
        entries = [SrtEntry(start_ms=900, end_ms=1500, text="今天天气很好")]
        hits = [
            OcrFrameHit("a.jpg", 1000, "00:00:01,000", "完全不同的对白", True),
        ]
        calibrated, changed = _apply_ocr_to_entries(
            entries,
            hits,
            pad_ms=500,
            min_frames=1,
            min_similarity=0.5,
            max_length_ratio_delta=0.35,
        )
        self.assertEqual(0, changed)
        self.assertEqual("今天天气很好", calibrated[0].text)

    def test_apply_ocr_to_entries(self):
        entries = [SrtEntry(start_ms=900, end_ms=1500, text="你号")]
        hits = [
            OcrFrameHit("a.jpg", 1000, "00:00:01,000", "你好", True),
        ]
        calibrated, changed = _apply_ocr_to_entries(
            entries,
            hits,
            pad_ms=500,
            min_frames=1,
            min_similarity=0.5,
            max_length_ratio_delta=0.35,
        )
        self.assertEqual(1, changed)
        self.assertEqual("你好", calibrated[0].text)

    def test_extract_ocr_hits_from_artifact(self):
        artifact = {
            "batches": [
                {
                    "frame_paths": ["/tmp/keyframe_000000_000001000.jpg"],
                    "frame_observations": [
                        {
                            "timestamp": "00:00:01,000",
                            "burned_in_subtitle": "硬字幕",
                            "has_burned_in_subtitle": True,
                        },
                    ],
                }
            ]
        }
        hits = extract_ocr_hits_from_artifact(artifact)
        self.assertEqual(1, len(hits))
        self.assertEqual("硬字幕", hits[0].text)

    def test_extract_ocr_hits_from_compact_artifact_without_frame_path(self):
        artifact = {
            "video_path": "/tmp/demo.mp4",
            "frame_interval_seconds": 2.0,
            "frame_observations": [
                {
                    "timestamp": "00:00:03,020",
                    "burned_in_subtitle": "我说句没觉悟的话啊",
                    "has_burned_in_subtitle": True,
                    "batch_index": 0,
                },
                {
                    "timestamp": "00:00:06,540",
                    "burned_in_subtitle": "干嘛非要要求回去当局长",
                    "has_burned_in_subtitle": True,
                    "batch_index": 0,
                },
            ],
        }
        hits = extract_ocr_hits_from_artifact(artifact)
        self.assertEqual(2, len(hits))
        self.assertEqual("我说句没觉悟的话啊", hits[0].text)
        self.assertEqual(3020, hits[0].timestamp_ms)
        self.assertEqual("", hits[0].frame_path)

    @patch("app.services.documentary.hard_subtitle_ocr_service._run_async_safely")
    def test_calibrate_writes_ocr_refined_file(self, mock_run_async):
        mock_run_async.return_value = [
            OcrFrameHit("a.jpg", 1000, "00:00:01,000", "校正对白", True),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_path = f"{tmp_dir}/keyframe_000000_000001000.jpg"
            image = PIL.Image.new("RGB", (640, 360), color=(0, 0, 0))
            image.save(frame_path, format="JPEG")

            subtitle_path = f"{tmp_dir}/demo_transcribed.srt"
            write_srt_file(
                [SrtEntry(start_ms=900, end_ms=1500, text="错字对白")],
                subtitle_path,
            )
            analysis_path = f"{tmp_dir}/demo_frame_analysis.json"
            with open(analysis_path, "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "video_path": f"{tmp_dir}/demo.mp4",
                        "batches": [
                            {
                                "frame_paths": [frame_path],
                                "frame_observations": [
                                    {
                                        "timestamp": "00:00:01,000",
                                        "observation": "",
                                        "burned_in_subtitle": "你好对白",
                                        "has_burned_in_subtitle": True,
                                    },
                                ],
                            }
                        ],
                    },
                    fp,
                    ensure_ascii=False,
                )
            output_path = f"{tmp_dir}/demo_ocr_refined.srt"
            result = calibrate_subtitle_with_hard_subtitle_ocr(
                subtitle_path=subtitle_path,
                analysis_json_path=analysis_path,
                output_path=output_path,
            )
            self.assertEqual(output_path, result)
            mock_run_async.assert_not_called()
            with open(result, "r", encoding="utf-8") as fp:
                content = fp.read()
            self.assertIn("你好对白", content)


if __name__ == "__main__":
    unittest.main()
