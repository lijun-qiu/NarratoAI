import unittest
import os
import json
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.services.documentary.frame_analysis_models import DocumentaryAnalysisConfig
from app.services.documentary.frame_analysis_service import DocumentaryFrameAnalysisService
from app.utils import utils


class DocumentaryFrameAnalysisServiceTests(unittest.TestCase):
    def test_build_analysis_prompt_formats_real_frame_count(self):
        service = DocumentaryFrameAnalysisService()

        prompt = service._build_analysis_prompt(frame_count=3)

        self.assertIn("我提供了 3 张视频帧", prompt)
        self.assertNotIn("%s", prompt)
        self.assertIn("scene_segments", prompt)
        self.assertIn("frame_observations", prompt)
        self.assertIn("overall_activity_summary", prompt)

    def test_parse_failed_batch_keeps_raw_response_and_time_range(self):
        service = DocumentaryFrameAnalysisService()

        batch = service._build_failed_batch_result(
            batch_index=2,
            raw_response="not-json",
            error_message="JSON decode failed",
            frame_paths=["/tmp/keyframe_000000_000000000.jpg"],
            time_range="00:00:00,000-00:00:03,000",
        )

        self.assertEqual("failed", batch.status)
        self.assertEqual("not-json", batch.raw_response)
        self.assertEqual("00:00:00,000-00:00:03,000", batch.time_range)
        self.assertTrue(batch.fallback_summary)

    def test_parse_failed_batch_uses_non_empty_fallback_when_raw_response_is_empty(self):
        service = DocumentaryFrameAnalysisService()

        batch = service._build_failed_batch_result(
            batch_index=3,
            raw_response="",
            error_message="Empty model response",
            frame_paths=["/tmp/keyframe_000001_000001000.jpg"],
            time_range="00:00:03,000-00:00:06,000",
        )

        self.assertEqual("failed", batch.status)
        self.assertEqual("", batch.raw_response)
        self.assertTrue(batch.fallback_summary)

    def test_failed_batch_result_uses_prompt_contract_field_names(self):
        service = DocumentaryFrameAnalysisService()

        batch = service._build_failed_batch_result(
            batch_index=4,
            raw_response="not-json",
            error_message="JSON decode failed",
            frame_paths=["/tmp/keyframe_000002_000002000.jpg"],
            time_range="00:00:06,000-00:00:09,000",
        )

        self.assertEqual([], batch.frame_observations)
        self.assertEqual("", batch.overall_activity_summary)
        self.assertFalse(hasattr(batch, "observations"))
        self.assertFalse(hasattr(batch, "summary"))

    def test_parse_batch_returns_failed_result_when_json_is_invalid(self):
        service = DocumentaryFrameAnalysisService()

        batch = service._parse_batch_response(
            batch_index=0,
            raw_response="plain text",
            frame_paths=["/tmp/keyframe_000000_000000000.jpg"],
            time_range="00:00:00,000-00:00:03,000",
        )

        self.assertEqual("failed", batch.status)
        self.assertEqual("plain text", batch.raw_response)
        self.assertEqual(["/tmp/keyframe_000000_000000000.jpg"], batch.frame_paths)
        self.assertEqual([], batch.frame_observations)
        self.assertEqual("", batch.overall_activity_summary)

    def test_parse_batch_returns_failed_result_for_empty_json_object(self):
        service = DocumentaryFrameAnalysisService()

        batch = service._parse_batch_response(
            batch_index=0,
            raw_response="{}",
            frame_paths=["/tmp/keyframe_000000_000000000.jpg"],
            time_range="00:00:00,000-00:00:03,000",
        )

        self.assertEqual("failed", batch.status)
        self.assertEqual("{}", batch.raw_response)
        self.assertIn("scene_segments", batch.error_message)

    def test_parse_batch_parses_scene_segments_contract(self):
        service = DocumentaryFrameAnalysisService()
        raw_response = """
{
  "scene_segments": [
    {
      "timestamp": "00:00:01,940-00:00:09,940",
      "scene": "办公室",
      "characters": ["领导", "伟业"],
      "action": "领导直视伟业，伟业站着",
      "emotion": "严肃、压抑",
      "key_visual": "桌上堆着文件，光从侧面打来",
      "audio_cue": "只有对话，无背景音乐",
      "importance": "高（主线剧情）"
    }
  ],
  "frame_observations": [
    {"observation": "办公室内对话"},
    {"observation": "领导特写"}
  ],
  "overall_activity_summary": "办公室对峙"
}
""".strip()

        batch = service._parse_batch_response(
            batch_index=1,
            raw_response=raw_response,
            frame_paths=[
                "/tmp/keyframe_000000_000001940.jpg",
                "/tmp/keyframe_000075_000009940.jpg",
            ],
            time_range="00:00:01,940-00:00:09,940",
        )

        self.assertEqual("success", batch.status)
        self.assertEqual(1, len(batch.scene_segments))
        self.assertEqual("办公室", batch.scene_segments[0]["scene"])
        self.assertEqual(["领导", "伟业"], batch.scene_segments[0]["characters"])
        self.assertEqual("高（主线剧情）", batch.scene_segments[0]["importance"])
        self.assertEqual(2, len(batch.frame_observations))
        self.assertEqual("办公室对峙", batch.overall_activity_summary)

    def test_parse_batch_synthesizes_frames_when_only_scene_segments(self):
        service = DocumentaryFrameAnalysisService()
        raw_response = """
{
  "scene_segments": [
    {
      "timestamp": "00:00:00,000-00:00:06,000",
      "scene": "街道",
      "characters": ["张三"],
      "action": "奔跑",
      "emotion": "紧张",
      "key_visual": "雨夜霓虹",
      "audio_cue": "脚步声",
      "importance": "中"
    }
  ],
  "overall_activity_summary": "雨中奔跑"
}
""".strip()

        batch = service._parse_batch_response(
            batch_index=0,
            raw_response=raw_response,
            frame_paths=[
                "/tmp/keyframe_000000_000000000.jpg",
                "/tmp/keyframe_000075_000003000.jpg",
            ],
            time_range="00:00:00,000-00:00:06,000",
        )

        self.assertEqual("success", batch.status)
        self.assertEqual(1, len(batch.scene_segments))
        self.assertEqual(2, len(batch.frame_observations))
        self.assertTrue(all(frame.get("observation") for frame in batch.frame_observations))

    def test_parse_batch_returns_failed_result_when_observations_are_too_short(self):
        service = DocumentaryFrameAnalysisService()
        raw_response = """
{
  "frame_observations": [
    {"observation": "第一帧画面"}
  ],
  "overall_activity_summary": "只有一条帧观察"
}
""".strip()

        batch = service._parse_batch_response(
            batch_index=1,
            raw_response=raw_response,
            frame_paths=[
                "/tmp/keyframe_000000_000000000.jpg",
                "/tmp/keyframe_000075_000003000.jpg",
            ],
            time_range="00:00:00,000-00:00:06,000",
        )

        self.assertEqual("failed", batch.status)
        self.assertEqual(raw_response, batch.raw_response)
        self.assertIn("frame_observations", batch.error_message)

    def test_parse_batch_parses_code_fenced_json_into_structured_result(self):
        service = DocumentaryFrameAnalysisService()
        raw_response = """```json
{
  "frame_observations": [
    {"observation": "第一帧画面"},
    {"observation": "第二帧画面"}
  ],
  "overall_activity_summary": "人物从房间走到街道"
}
```"""

        batch = service._parse_batch_response(
            batch_index=1,
            raw_response=raw_response,
            frame_paths=[
                "/tmp/keyframe_000000_000000000.jpg",
                "/tmp/keyframe_000075_000003000.jpg",
            ],
            time_range="00:00:00,000-00:00:06,000",
        )

        self.assertEqual("success", batch.status)
        self.assertEqual(
            [
                {
                    "frame_path": "/tmp/keyframe_000000_000000000.jpg",
                    "timestamp": "00:00:00,000",
                    "observation": "第一帧画面",
                    "burned_in_subtitle": "",
                    "has_burned_in_subtitle": False,
                },
                {
                    "frame_path": "/tmp/keyframe_000075_000003000.jpg",
                    "timestamp": "00:00:03,000",
                    "observation": "第二帧画面",
                    "burned_in_subtitle": "",
                    "has_burned_in_subtitle": False,
                },
            ],
            batch.frame_observations,
        )
        self.assertEqual("人物从房间走到街道", batch.overall_activity_summary)
        self.assertEqual("", batch.fallback_summary)

    def test_parse_batch_preserves_frames_when_summary_is_missing(self):
        service = DocumentaryFrameAnalysisService()
        raw_response = """
{
  "frame_observations": [
    {"observation": "第一帧画面"},
    {"observation": "第二帧画面"}
  ]
}
""".strip()

        batch = service._parse_batch_response(
            batch_index=2,
            raw_response=raw_response,
            frame_paths=[
                "/tmp/keyframe_000000_000000000.jpg",
                "/tmp/keyframe_000075_000003000.jpg",
            ],
            time_range="00:00:00,000-00:00:06,000",
        )

        self.assertEqual("success", batch.status)
        self.assertEqual(1, len(batch.scene_segments))
        self.assertEqual(2, len(batch.frame_observations))
        self.assertEqual("", batch.overall_activity_summary)

    def test_cache_key_changes_when_interval_changes(self):
        service = DocumentaryFrameAnalysisService()

        with patch("app.services.documentary.frame_extraction_service.os.path.getmtime", return_value=100.0):
            key_a = service._build_cache_key("video.mp4", 3.0, "prompt-v1", "model-a", 10, 2)
            key_b = service._build_cache_key("video.mp4", 5.0, "prompt-v1", "model-a", 10, 2)

        self.assertNotEqual(key_a, key_b)

    def test_cache_key_changes_when_model_changes(self):
        service = DocumentaryFrameAnalysisService()

        with patch("app.services.documentary.frame_extraction_service.os.path.getmtime", return_value=100.0):
            key_a = service._build_cache_key("video.mp4", 3.0, "prompt-v1", "model-a", 10, 2)
            key_b = service._build_cache_key("video.mp4", 3.0, "prompt-v1", "model-b", 10, 2)

        self.assertNotEqual(key_a, key_b)

    def test_cache_key_starts_with_legacy_video_hash_prefix(self):
        service = DocumentaryFrameAnalysisService()

        with patch("app.services.documentary.frame_extraction_service.os.path.getmtime", return_value=123.0):
            key = service._build_cache_key("video.mp4", 3.0, "prompt-v1", "model-a", 10, 2)

        expected_prefix = utils.md5("video.mp4" + "123.0")
        self.assertTrue(key.startswith(expected_prefix))

    def test_clear_keyframes_cache_respects_scope_and_prefix_match(self):
        with TemporaryDirectory() as temp_root:
            service = DocumentaryFrameAnalysisService()
            analysis_dir = os.path.join(temp_root, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            with patch("app.services.documentary.frame_extraction_service.os.path.getmtime", return_value=123.0):
                target_key_a = service._build_cache_key("video.mp4", 3.0, "prompt-v1", "model-a", 10, 2)
                target_key_b = service._build_cache_key("video.mp4", 5.0, "prompt-v1", "model-a", 10, 2)
                keep_key = service._build_cache_key("other.mp4", 3.0, "prompt-v1", "model-a", 10, 2)

            target_dir_a = os.path.join(analysis_dir, target_key_a)
            target_dir_b = os.path.join(analysis_dir, target_key_b)
            keep_dir = os.path.join(analysis_dir, keep_key)
            os.makedirs(target_dir_a, exist_ok=True)
            os.makedirs(target_dir_b, exist_ok=True)
            os.makedirs(keep_dir, exist_ok=True)

            with patch("app.utils.utils.temp_dir", return_value=temp_root), patch(
                "app.utils.utils.os.path.getmtime", return_value=123.0
            ):
                utils.clear_keyframes_cache(video_path="video.mp4", cache_scope="analysis")

            self.assertFalse(os.path.exists(target_dir_a))
            self.assertFalse(os.path.exists(target_dir_b))
            self.assertTrue(os.path.exists(keep_dir))


    def test_default_analysis_path_uses_video_stem(self):
        with TemporaryDirectory() as temp_dir:
            with patch.object(utils, "storage_dir", return_value=temp_dir):
                path = DocumentaryFrameAnalysisService.default_analysis_path_for_video(
                    r"C:\videos\6月4日(1).mp4"
                )
                self.assertTrue(path.endswith("6月4日(1)_frame_analysis.json"))
                self.assertIn(os.path.join("temp", "analysis"), path)

    def test_resolve_reusable_analysis_path_prefers_default_video_file(self):
        service = DocumentaryFrameAnalysisService()
        with TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "demo.mp4")
            with open(video_path, "wb") as fp:
                fp.write(b"demo")

            with patch.object(utils, "storage_dir", return_value=temp_dir):
                default_path = service.default_analysis_path_for_video(video_path)
                os.makedirs(os.path.dirname(default_path), exist_ok=True)
                artifact = {
                    "artifact_version": "documentary-frame-analysis-v2",
                    "video_path": video_path,
                    "batches": [{"batch_index": 1, "frame_observations": []}],
                }
                with open(default_path, "w", encoding="utf-8") as fp:
                    json.dump(artifact, fp)

                resolved = service.resolve_reusable_analysis_path(
                    video_path,
                    reuse=True,
                )
                self.assertEqual(default_path, resolved)

    def test_save_analysis_artifact_writes_video_named_file(self):
        service = DocumentaryFrameAnalysisService()
        with TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "episode1.mp4")
            with patch.object(utils, "storage_dir", return_value=temp_dir):
                saved_path = service._save_analysis_artifact(
                    {
                        "artifact_version": "documentary-frame-analysis-v2",
                        "video_path": video_path,
                        "batches": [],
                    },
                    video_path=video_path,
                )
                self.assertEqual(
                    service.default_analysis_path_for_video(video_path),
                    saved_path,
                )
                self.assertTrue(os.path.isfile(saved_path))

    def test_count_failed_batches(self):
        service = DocumentaryFrameAnalysisService()
        artifact = {
            "batches": [
                {"batch_index": 0, "status": "success"},
                {"batch_index": 1, "status": "failed"},
                {"batch_index": 2, "status": "failed"},
            ]
        }
        self.assertEqual(2, service.count_failed_batches(artifact))

    def test_coerce_batch_payload_accepts_script_clip_array(self):
        service = DocumentaryFrameAnalysisService()
        raw = """[
  {
    "_id": 1,
    "timestamp": "00:11:03,060-00:11:09,380",
    "picture": "伟业在警局背景墙前打电话",
    "narration": "领导来电",
    "OST": 1
  }
]"""
        payload_raw = service._load_batch_payload_json(raw)
        payload = service._coerce_batch_payload(
            payload_raw,
            time_range="00:11:00,000-00:11:09,000",
        )
        self.assertIn("scene_segments", payload)
        self.assertEqual(1, len(payload["scene_segments"]))
        self.assertEqual("00:11:03,060-00:11:09,380", payload["scene_segments"][0]["timestamp"])

        batch = service._parse_batch_response(
            batch_index=66,
            raw_response=raw,
            frame_paths=[
                "/tmp/keyframe_006600_000660000.jpg",
                "/tmp/keyframe_006660_000666000.jpg",
            ],
            time_range="00:11:00,000-00:11:09,000",
        )
        self.assertEqual("success", batch.status)
        self.assertEqual(1, len(batch.scene_segments))
        self.assertEqual(2, len(batch.frame_observations))
        service = DocumentaryFrameAnalysisService()
        batch = service._batch_dict_to_result(
            {
                "batch_index": 3,
                "status": "failed",
                "time_range": "00:00:30,000-00:00:39,000",
                "frame_paths": ["/tmp/a.jpg"],
                "error_message": "parse error",
            }
        )
        self.assertEqual(3, batch.batch_index)
        self.assertEqual("failed", batch.status)
        self.assertEqual(["/tmp/a.jpg"], batch.frame_paths)
        self.assertEqual("parse error", batch.error_message)


class DocumentaryFrameAnalysisCompactTests(unittest.TestCase):
    def _sample_artifact(self) -> dict:
        return {
            "artifact_version": "documentary-frame-analysis-v3",
            "generated_at": "2026-06-07T12:00:00",
            "video_path": "/tmp/demo.mp4",
            "frame_interval_seconds": 1.0,
            "scene_segments": [
                {
                    "timestamp": "00:00:01,000-00:00:03,000",
                    "scene": "天台",
                    "action": "对话",
                    "batch_index": 0,
                    "time_range": "00:00:00,000-00:00:09,000",
                    "subtitle_entries": [
                        {"start": "00:00:01,940", "end": "00:00:02,420", "text": "白爷"},
                        {"start": "00:00:02,500", "end": "00:00:02,900", "text": "你好"},
                    ],
                    "subtitle": "白爷；你好",
                }
            ],
            "frame_observations": [
                {
                    "frame_path": "/tmp/keyframe_000000_000000000.jpg",
                    "timestamp": "00:00:00,000",
                    "observation": "阴天",
                    "batch_index": 0,
                    "time_range": "00:00:00,000-00:00:09,000",
                }
            ],
            "overall_activity_summaries": [
                {
                    "batch_index": 0,
                    "time_range": "00:00:00,000-00:00:09,000",
                    "summary": "开场",
                }
            ],
            "batches": [
                {
                    "batch_index": 0,
                    "status": "success",
                    "time_range": "00:00:00,000-00:00:09,000",
                    "raw_response": "x" * 1000,
                    "frame_paths": ["/tmp/keyframe_000000_000000000.jpg"],
                    "scene_segments": [
                        {
                            "timestamp": "00:00:01,000-00:00:03,000",
                            "scene": "天台",
                            "action": "对话",
                        }
                    ],
                    "frame_observations": [
                        {
                            "frame_path": "/tmp/keyframe_000000_000000000.jpg",
                            "timestamp": "00:00:00,000",
                            "observation": "阴天",
                        }
                    ],
                    "overall_activity_summary": "开场",
                }
            ],
        }

    def test_compact_analysis_artifact_removes_debug_fields(self):
        from app.services.documentary.frame_analysis_compact import compact_analysis_artifact

        compact = compact_analysis_artifact(self._sample_artifact(), include_frame_observations=True)
        self.assertIn("field_comments", compact)
        self.assertEqual("documentary-frame-analysis-v3-compact", compact["artifact_version"])
        self.assertEqual(1, len(compact["scene_segments"]))
        payload_without_comments = {k: v for k, v in compact.items() if k != "field_comments"}
        self.assertNotIn("raw_response", json.dumps(payload_without_comments))
        self.assertNotIn("frame_paths", json.dumps(payload_without_comments))
        self.assertEqual(1, len(compact["batches"]))
        self.assertEqual(["batch_index", "time_range", "status"], list(compact["batches"][0].keys()))

    def test_rebuild_batches_from_compact_artifact(self):
        from app.services.documentary.frame_analysis_compact import (
            compact_analysis_artifact,
            rebuild_batches_from_artifact,
        )

        compact = compact_analysis_artifact(self._sample_artifact(), include_frame_observations=True)
        batches = rebuild_batches_from_artifact(compact)
        self.assertEqual(1, len(batches))
        self.assertEqual("00:00:00,000-00:00:09,000", batches[0]["time_range"])
        self.assertEqual(1, len(batches[0]["scene_segments"]))
        self.assertEqual(1, len(batches[0]["frame_observations"]))
        self.assertNotIn("frame_path", batches[0]["frame_observations"][0])

    def test_save_compact_analysis_artifact(self):
        artifact = self._sample_artifact()
        with TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "demo_frame_analysis.json")
            with open(source_path, "w", encoding="utf-8") as fp:
                json.dump(artifact, fp, ensure_ascii=False)

            result = DocumentaryFrameAnalysisService.save_compact_analysis_artifact(
                source_path,
                include_frame_observations=True,
            )
            self.assertTrue(os.path.isfile(result["output_path"]))
            if result["original_bytes"] > 8000:
                self.assertLess(result["compact_bytes"], result["original_bytes"])
            self.assertIn("_frame_analysis_compact.json", result["output_path"])

    def test_save_minimal_scene_analysis_artifact(self):
        from app.services.documentary.frame_analysis_compact import save_minimal_scene_analysis_artifact

        artifact = self._sample_artifact()
        with TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "demo_frame_analysis.json")
            with open(source_path, "w", encoding="utf-8") as fp:
                json.dump(artifact, fp, ensure_ascii=False)

            result = save_minimal_scene_analysis_artifact(source_path)
            with open(result["output_path"], encoding="utf-8") as fp:
                payload = json.load(fp)
            self.assertTrue(os.path.isfile(result["output_path"]))
            self.assertLess(result["compact_bytes"], result["original_bytes"])
            self.assertIn("_frame_analysis_minimal.json", result["output_path"])
            self.assertIn("field_comments", payload)
            self.assertEqual(1, len(payload["scene_segments"]))
            segment = payload["scene_segments"][0]
            self.assertEqual(
                {
                    "timestamp",
                    "time_range",
                    "scene",
                    "observation",
                    "action",
                    "subtitle",
                },
                set(segment.keys()),
            )
            self.assertEqual("00:00:01,940-00:00:02,900", payload["scene_segments"][0]["time_range"])
            self.assertEqual("对话", payload["scene_segments"][0]["action"])
            self.assertEqual("对话", payload["scene_segments"][0]["observation"])
            self.assertNotIn("batches", payload)


class DocumentaryMaterialOutputSplitTests(unittest.TestCase):
    def test_compute_equal_split_ranges(self):
        from app.services.documentary.material_output_split import compute_equal_split_ranges

        ranges = compute_equal_split_ranges(100_000, 4)
        self.assertEqual(4, len(ranges))
        self.assertEqual(0, ranges[0]["start_ms"])
        self.assertEqual(100_000, ranges[-1]["end_ms"])

    def test_split_frame_analysis_artifact_by_batches(self):
        from app.services.documentary.material_output_split import split_frame_analysis_artifact

        artifact = {
            "video_path": "",
            "batches": [
                {"batch_index": 0, "time_range": "00:00:00,000-00:00:09,000", "status": "success"},
                {"batch_index": 1, "time_range": "00:00:10,000-00:00:19,000", "status": "success"},
            ],
            "scene_segments": [
                {"timestamp": "00:00:01,000-00:00:03,000", "scene": "A", "batch_index": 0},
                {"timestamp": "00:00:11,000-00:00:13,000", "scene": "B", "batch_index": 1},
            ],
            "frame_observations": [],
            "overall_activity_summaries": [],
        }
        parts = split_frame_analysis_artifact(artifact, 2)
        self.assertEqual(2, len(parts))
        self.assertEqual(1, len(parts[0]["batches"]))
        self.assertEqual(1, len(parts[1]["batches"]))

    def test_split_srt_entries_keep_global_timestamps(self):
        from app.services.documentary.material_output_split import split_srt_entries
        from app.services.srt_utils import SrtEntry

        entries = [
            SrtEntry(start_ms=0, end_ms=1000, text="a"),
            SrtEntry(start_ms=5000, end_ms=6000, text="b"),
        ]
        windows, grouped = split_srt_entries(entries, 2, total_ms=10_000)
        self.assertEqual(2, len(grouped))
        self.assertEqual(1, len(grouped[0]))
        self.assertEqual(1, len(grouped[1]))
        self.assertEqual(5000, grouped[1][0].start_ms)


    def test_split_assigns_each_batch_to_single_part(self):
        from app.services.documentary.material_output_split import split_frame_analysis_artifact

        artifact = {
            "video_path": "",
            "batches": [
                {"batch_index": i, "time_range": f"00:00:{i*9:02d},000-00:00:{i*9+8:02d},000", "status": "success"}
                for i in range(20)
            ],
            "scene_segments": [],
            "frame_observations": [],
            "overall_activity_summaries": [],
        }
        parts = split_frame_analysis_artifact(artifact, 4)
        batch_ids = [batch["batch_index"] for part in parts for batch in part.get("batches") or []]
        self.assertEqual(len(batch_ids), len(set(batch_ids)))
        self.assertEqual(20, len(batch_ids))


class DocumentaryAnalysisConfigTests(unittest.TestCase):
    def test_config_rejects_non_positive_frame_interval(self):
        with self.assertRaises(ValueError):
            DocumentaryAnalysisConfig(
                video_path="/tmp/demo.mp4",
                frame_interval_seconds=0,
                vision_batch_size=5,
                vision_llm_provider="openai",
                vision_model_name="gpt-4o-mini",
            )

    def test_config_rejects_non_positive_batch_size(self):
        with self.assertRaises(ValueError):
            DocumentaryAnalysisConfig(
                video_path="/tmp/demo.mp4",
                frame_interval_seconds=5,
                vision_batch_size=0,
                vision_llm_provider="openai",
                vision_model_name="gpt-4o-mini",
            )

    def test_config_rejects_non_positive_max_concurrency(self):
        with self.assertRaises(ValueError):
            DocumentaryAnalysisConfig(
                video_path="/tmp/demo.mp4",
                frame_interval_seconds=5,
                vision_batch_size=5,
                vision_llm_provider="openai",
                vision_model_name="gpt-4o-mini",
                max_concurrency=0,
            )


class DocumentaryFrameGenderHintTests(unittest.TestCase):
    def test_analysis_prompt_requires_gender_in_characters(self):
        service = DocumentaryFrameAnalysisService()
        prompt = service._build_analysis_prompt(frame_count=2)
        self.assertIn("姓名(男)", prompt)
        self.assertIn("姓名(女)", prompt)
        self.assertIn("未名人员(男)", prompt)
        self.assertIn("未名人员(女)", prompt)
        self.assertIn("禁止把「领导」", prompt)

    def test_batch_prompt_includes_gender_hint(self):
        service = DocumentaryFrameAnalysisService()
        prompt = service._build_batch_prompt(
            frame_count=2,
            video_theme="罚罪2",
            custom_prompt="",
            documentary_settings={"documentary_compact_mode": True, "documentary_compact_style": "fazu2"},
            time_range="00:00:00,000-00:00:06,000",
        )
        self.assertIn("人物性别**仅据画面**判断", prompt)
        self.assertIn("胡小跃=男", prompt)

    def test_batch_prompt_includes_naming_hint(self):
        service = DocumentaryFrameAnalysisService()
        prompt = service._build_batch_prompt(
            frame_count=2,
            video_theme="罚罪2",
            custom_prompt="",
            documentary_settings={"documentary_compact_mode": True, "documentary_compact_style": "fazu2"},
            time_range="00:00:00,000-00:00:06,000",
        )
        self.assertIn("无法从字幕确认身份时", prompt)
        self.assertIn("老叶与伟业并肩", prompt)
        self.assertIn("勿把与之对话的上级/长辈称为伟业", prompt)

    @patch("app.services.documentary.documentary_settings.logger.warning")
    def test_warn_frame_analysis_gender_mismatch_detects_female_tag(self, mock_warning):
        from app.services.documentary.documentary_settings import warn_frame_analysis_gender_mismatch

        warn_frame_analysis_gender_mismatch(
            scene_segments=[{"characters": ["胡小跃(女)"], "action": "女警胡小跃站立"}],
            frame_observations=[{"observation": "胡小跃(女)在楼道"}],
            batch_index=1,
            time_range="00:01:00,000-00:01:06,000",
            settings={"documentary_compact_mode": True, "documentary_compact_style": "fazu2"},
        )
        self.assertTrue(mock_warning.called)
        self.assertGreaterEqual(mock_warning.call_count, 1)


class DocumentaryFrameSubtitleAttachmentTests(unittest.TestCase):
    SAMPLE_SRT = """1
00:00:01,940 --> 00:00:02,420
老叶，

2
00:00:03,020 --> 00:00:04,580
我说句没觉悟的话啊，

3
00:00:05,100 --> 00:00:06,340
你都到厅级了，
"""

    def test_attach_subtitles_adds_frame_and_segment_fields(self):
        from app.services.documentary.documentary_subtitle_enrichment import (
            attach_subtitles_to_frame_analysis_artifact,
        )

        artifact = {
            "batches": [
                {
                    "batch_index": 0,
                    "time_range": "00:00:00,000-00:00:09,000",
                    "frame_observations": [
                        {"timestamp": "00:00:02,000", "observation": "出现老叶字样"},
                        {"timestamp": "00:00:05,000", "observation": "对话中"},
                    ],
                    "scene_segments": [
                        {"timestamp": "00:00:01,940-00:00:09,940", "scene": "楼顶天台"},
                    ],
                }
            ],
            "frame_observations": [
                {"timestamp": "00:00:02,000", "observation": "出现老叶字样", "batch_index": 0},
                {"timestamp": "00:00:05,000", "observation": "对话中", "batch_index": 0},
            ],
            "scene_segments": [
                {"timestamp": "00:00:01,940-00:00:09,940", "scene": "楼顶天台", "batch_index": 0},
            ],
        }

        enriched = attach_subtitles_to_frame_analysis_artifact(artifact, self.SAMPLE_SRT)
        self.assertTrue(enriched.get("subtitle_attached"))
        frame0 = enriched["frame_observations"][0]
        self.assertIn("老叶", frame0["subtitle"])
        self.assertEqual("00:00:01,940", frame0["subtitle_start"])
        self.assertEqual("00:00:02,420", frame0["subtitle_end"])
        self.assertIn("你都到厅级了", enriched["frame_observations"][1]["subtitle"])
        self.assertIn("subtitle_entries", enriched["batches"][0])
        self.assertIn("subtitle", enriched["scene_segments"][0])
        entries = enriched["scene_segments"][0]["subtitle_entries"]
        self.assertTrue(any(item.get("start") == "00:00:01,940" for item in entries))

    def test_attach_subtitles_prefers_burned_in_text_with_srt_time(self):
        from app.services.documentary.documentary_subtitle_enrichment import (
            attach_subtitles_to_frame_analysis_artifact,
        )

        srt = """1
00:00:16,940 --> 00:00:18,700
胡小月是我的徒弟
"""
        artifact = {
            "batches": [],
            "frame_observations": [
                {
                    "timestamp": "00:00:17,000",
                    "observation": "伟业说话",
                    "burned_in_subtitle": "胡小跃是我的徒弟",
                    "has_burned_in_subtitle": True,
                }
            ],
            "scene_segments": [],
        }
        enriched = attach_subtitles_to_frame_analysis_artifact(artifact, srt)
        frame = enriched["frame_observations"][0]
        self.assertEqual("胡小跃是我的徒弟", frame["subtitle"])
        self.assertEqual("00:00:16,940", frame["subtitle_start"])
        self.assertEqual("00:00:18,700", frame["subtitle_end"])
        self.assertEqual("burned_in_corrected", frame["subtitle_text_source"])
