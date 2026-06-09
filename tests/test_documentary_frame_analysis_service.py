import unittest
import os
import json
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.services.documentary.frame_analysis_models import DocumentaryAnalysisConfig
from app.services.documentary.frame_analysis_service import DocumentaryFrameAnalysisService
from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService
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
                    "timestamp": "00:00:00,000",
                    "observation": "第一帧画面",
                    "burned_in_subtitle": "",
                    "has_burned_in_subtitle": False,
                },
                {
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
        self.assertTrue(batch.overall_activity_summary.startswith("本批次："))

    def test_parse_batch_overrides_model_timestamp_with_keyframe_filename(self):
        service = DocumentaryFrameAnalysisService()
        raw_response = """
{
  "frame_observations": [
    {"timestamp": "00:01:01,000", "observation": "[特写] 车内，秦枫(男)侧脸"},
    {"timestamp": "00:01:02,000", "observation": "[中景] 车内，秦枫(男)侧脸"}
  ],
  "overall_activity_summary": "车顶追逐"
}
""".strip()

        batch = service._parse_batch_response(
            batch_index=0,
            raw_response=raw_response,
            frame_paths=[
                "/tmp/keyframe_009030_000501000.jpg",
                "/tmp/keyframe_009060_000502000.jpg",
            ],
            time_range="00:05:01,000-00:05:02,000",
        )

        self.assertEqual("success", batch.status)
        self.assertEqual("00:05:01,000", batch.frame_observations[0]["timestamp"])
        self.assertEqual("00:05:02,000", batch.frame_observations[1]["timestamp"])

    def test_keyframe_cache_key_changes_when_interval_changes(self):
        service = DocumentaryFrameExtractionService()

        with patch("app.services.documentary.frame_extraction_service.os.path.getmtime", return_value=100.0):
            key_a = service._build_keyframe_cache_key("video.mp4", 3.0)
            key_b = service._build_keyframe_cache_key("video.mp4", 5.0)

        self.assertNotEqual(key_a, key_b)

    def test_keyframe_cache_key_starts_with_video_hash_prefix(self):
        service = DocumentaryFrameExtractionService()

        with patch("app.services.documentary.frame_extraction_service.os.path.getmtime", return_value=123.0):
            key = service._build_keyframe_cache_key("video.mp4", 3.0)

        expected_prefix = utils.md5("video.mp4" + "123.0")
        self.assertTrue(key.startswith(expected_prefix))

    def test_clear_keyframes_cache_respects_scope_and_prefix_match(self):
        with TemporaryDirectory() as temp_root:
            service = DocumentaryFrameExtractionService()
            keyframes_dir = os.path.join(temp_root, "keyframes")
            os.makedirs(keyframes_dir, exist_ok=True)

            with patch("app.services.documentary.frame_extraction_service.os.path.getmtime", return_value=123.0):
                target_key_a = service._build_keyframe_cache_key("video.mp4", 3.0)
                target_key_b = service._build_keyframe_cache_key("video.mp4", 5.0)
                keep_key = service._build_keyframe_cache_key("other.mp4", 3.0)

            target_dir_a = os.path.join(keyframes_dir, target_key_a)
            target_dir_b = os.path.join(keyframes_dir, target_key_b)
            keep_dir = os.path.join(keyframes_dir, keep_key)
            os.makedirs(target_dir_a, exist_ok=True)
            os.makedirs(target_dir_b, exist_ok=True)
            os.makedirs(keep_dir, exist_ok=True)

            with patch("app.utils.utils.temp_dir", return_value=temp_root), patch(
                "app.utils.utils.os.path.getmtime", return_value=123.0
            ):
                utils.clear_keyframes_cache(video_path="video.mp4", cache_scope="keyframes")

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

    def test_rejects_narration_script_clip_array(self):
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
        self.assertTrue(service._is_narration_script_clip_payload(payload_raw))
        payload = service._coerce_batch_payload(
            payload_raw,
            time_range="00:11:00,000-00:11:09,000",
        )
        self.assertEqual({}, payload)

        batch = service._parse_batch_response(
            batch_index=66,
            raw_response=raw,
            frame_paths=[
                "/tmp/keyframe_006600_000660000.jpg",
                "/tmp/keyframe_006660_000666000.jpg",
            ],
            time_range="00:11:00,000-00:11:09,000",
        )
        self.assertEqual("failed", batch.status)
        self.assertIn("narration script", batch.error_message.lower())

    def test_batch_dict_to_result(self):
        service = DocumentaryFrameAnalysisService()
        artifact = {"keyframe_cache_key": "test_cache_key"}
        with patch("os.path.isfile", return_value=True):
            batch = service._batch_dict_to_result(
                {
                    "batch_index": 3,
                    "status": "failed",
                    "time_range": "00:00:30,000-00:00:39,000",
                    "frame_files": ["keyframe_000030_000030000.jpg"],
                    "error_message": "parse error",
                },
                artifact=artifact,
            )
        self.assertEqual(3, batch.batch_index)
        self.assertEqual("failed", batch.status)
        self.assertEqual(1, len(batch.frame_paths))
        self.assertEqual("parse error", batch.error_message)


class DocumentaryFrameAnalysisCompactTests(unittest.TestCase):
    def _sample_artifact(self) -> dict:
        return {
            "artifact_version": "documentary-frame-analysis-v4",
            "generated_at": "2026-06-07T12:00:00",
            "video_path": "/tmp/demo.mp4",
            "keyframe_cache_key": "abc123_def456",
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
                    "timestamp": "00:00:00,000",
                    "observation": "阴天",
                    "batch_index": 0,
                    "time_range": "00:00:00,000-00:00:09,000",
                }
            ],
            "video_segment_overview": {
                "segment_count": 1,
                "segments": [{"index": 1, "scene": "天台", "summary": "对话"}],
            },
            "batches": [
                {
                    "batch_index": 0,
                    "status": "success",
                    "time_range": "00:00:00,000-00:00:09,000",
                    "raw_response": "x" * 1000,
                    "frame_files": ["keyframe_000000_000000000.jpg"],
                    "scene_segments": [
                        {
                            "timestamp": "00:00:01,000-00:00:03,000",
                            "scene": "天台",
                            "action": "对话",
                        }
                    ],
                    "frame_observations": [
                        {
                            "timestamp": "00:00:00,000",
                            "observation": "阴天",
                        }
                    ],
                    "overall_activity_summary": "开场",
                }
            ],
        }

    def test_compact_script_export_includes_segment_time_range(self):
        from app.services.documentary.frame_analysis_compact import compact_analysis_artifact

        compact = compact_analysis_artifact(
            self._sample_artifact(),
            include_frame_observations=False,
            include_summaries=True,
            include_batch_index=True,
            keep_batch_meta=True,
        )
        segment = compact["scene_segments"][0]
        self.assertIn("time_range", segment)
        self.assertEqual("00:00:01,940-00:00:02,900", segment["time_range"])

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
        self.assertEqual(
            ["batch_index", "time_range", "status", "frame_files"],
            list(compact["batches"][0].keys()),
        )

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

    def test_rebuild_batches_merges_top_level_observations_when_batch_has_scene_segments(self):
        from app.services.documentary.frame_analysis_compact import rebuild_batches_from_artifact

        artifact = {
            "scene_segments": [
                {
                    "batch_index": 0,
                    "timestamp": "00:00:01,000-00:00:03,000",
                    "scene": "楼顶",
                    "subtitle": "硬字幕A",
                }
            ],
            "frame_observations": [
                {
                    "batch_index": 0,
                    "timestamp": "00:00:01,940",
                    "burned_in_subtitle": "硬字幕B",
                    "has_burned_in_subtitle": True,
                },
                {
                    "batch_index": 1,
                    "timestamp": "00:00:20,220",
                    "burned_in_subtitle": "硬字幕C",
                    "has_burned_in_subtitle": True,
                },
            ],
            "batches": [
                {
                    "batch_index": 0,
                    "time_range": "00:00:00,000-00:00:18,000",
                    "status": "success",
                    "scene_segments": [
                        {
                            "timestamp": "00:00:01,000-00:00:03,000",
                            "scene": "楼顶",
                            "subtitle": "硬字幕A",
                        }
                    ],
                    "overall_activity_summary": "批次0摘要",
                },
                {
                    "batch_index": 1,
                    "time_range": "00:00:20,000-00:00:38,000",
                    "status": "success",
                    "scene_segments": [],
                    "overall_activity_summary": "批次1摘要",
                },
            ],
        }
        batches = rebuild_batches_from_artifact(artifact)
        self.assertEqual(2, len(batches))
        self.assertEqual(1, len(batches[0]["frame_observations"]))
        self.assertEqual("硬字幕B", batches[0]["frame_observations"][0]["burned_in_subtitle"])
        self.assertEqual(1, len(batches[1]["frame_observations"]))
        self.assertEqual("硬字幕C", batches[1]["frame_observations"][0]["burned_in_subtitle"])
        self.assertEqual("批次1摘要", batches[1]["overall_activity_summary"])

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
                    "scene",
                    "observation",
                    "action",
                    "subtitle",
                },
                set(segment.keys()),
            )
            self.assertEqual("白爷；你好", segment["subtitle"])
            self.assertNotIn("subtitle_entries", segment)
            self.assertNotIn("time_range", segment)
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
        }
        parts = split_frame_analysis_artifact(artifact, 4)
        batch_ids = [batch["batch_index"] for part in parts for batch in part.get("batches") or []]
        self.assertEqual(len(batch_ids), len(set(batch_ids)))
        self.assertEqual(20, len(batch_ids))


class FrameAnalysisPairingTests(unittest.TestCase):
    def test_load_analysis_artifact_rejects_legacy_frame_paths(self):
        from app.services.documentary.frame_analysis_pairing import (
            _LEGACY_ARTIFACT_ERROR,
            load_analysis_artifact,
        )

        with TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "legacy.json")
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "batches": [
                            {
                                "batch_index": 0,
                                "frame_paths": ["/tmp/keyframe.jpg"],
                            }
                        ]
                    },
                    fp,
                )
            with self.assertRaises(ValueError) as ctx:
                load_analysis_artifact(path)
            self.assertIn(_LEGACY_ARTIFACT_ERROR, str(ctx.exception))

    def test_load_analysis_artifact_accepts_v4_payload(self):
        from app.services.documentary.frame_analysis_pairing import load_analysis_artifact

        with TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "v4.json")
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "artifact_version": "documentary-frame-analysis-v4",
                        "keyframe_cache_key": "abc",
                        "scene_segments": [{"timestamp": "00:00:00,000-00:00:01,000", "scene": "A"}],
                        "batches": [{"batch_index": 0, "frame_files": ["keyframe_000000_000000000.jpg"]}],
                    },
                    fp,
                )
            artifact = load_analysis_artifact(path)
            self.assertEqual("documentary-frame-analysis-v4", artifact["artifact_version"])


class FrameTimelineRefinementTests(unittest.TestCase):
    def test_offset_scene_segments_to_absolute(self):
        from app.services.documentary.frame_timeline_refinement import (
            offset_scene_segments_to_absolute,
        )

        segments = offset_scene_segments_to_absolute(
            [
                {
                    "timestamp": "00:00:00,000-00:00:15,000",
                    "scene": "问询室",
                    "observation": "对话",
                },
                {
                    "timestamp": "00:00:14,000-00:00:20,000",
                    "scene": "走廊",
                    "observation": "交接",
                },
            ],
            time_range="00:06:40,000-00:06:59,000",
        )
        self.assertEqual("00:06:40,000-00:06:55,000", segments[0]["timestamp"])
        self.assertEqual("00:06:54,000-00:07:00,000", segments[1]["timestamp"])

    def test_build_batch_timeline_summary_uses_milestones_not_every_frame(self):
        from app.services.documentary.frame_timeline_refinement import build_batch_timeline_summary

        frames = [
            {
                "timestamp": "00:06:50,000",
                "observation": "[特写] 问询室，秦枫(男)低头陈述",
                "burned_in_subtitle": "这件事情最后什么结果",
            },
            {
                "timestamp": "00:06:51,000",
                "observation": "[特写] 问询室，秦枫(男)面部阴影加深",
                "burned_in_subtitle": "这件事情最后什么结果",
            },
            {
                "timestamp": "00:06:53,000",
                "observation": "[特写] 问询室，秦枫(男)嘴部微张",
                "burned_in_subtitle": "我都认",
            },
        ]
        summary = build_batch_timeline_summary(frames)
        self.assertIn("本批次：", summary)
        self.assertLessEqual(summary.count("→"), 2)

    def test_failed_batch_does_not_receive_untagged_segments(self):
        from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService

        artifact = {
            "scene_segments": [
                {
                    "batch_index": 1,
                    "timestamp": "00:06:50,000-00:06:59,000",
                    "scene": "问询室",
                    "observation": "秦枫(男)陈述",
                }
            ],
            "batches": [
                {"batch_index": 0, "status": "failed", "time_range": "00:06:40,000-00:06:49,000"},
                {"batch_index": 1, "status": "success", "time_range": "00:06:50,000-00:06:59,000"},
            ],
        }
        DocumentaryFrameExtractionService._finalize_scene_segments_in_artifact(artifact)
        self.assertEqual([], artifact["batches"][0]["scene_segments"])
        self.assertEqual(1, len(artifact["batches"][1]["scene_segments"]))

    def test_character_field_separation_strips_names_from_descriptions(self):
        from app.services.documentary.frame_character_naming import (
            apply_character_field_separation_to_artifact,
        )

        artifact = {
            "character_references": [{"name": "秦枫"}, {"name": "胡小跃"}],
            "scene_segments": [
                {
                    "batch_index": 0,
                    "scene": "审讯室",
                    "observation": "秦枫(男)低头，胡小跃(男)搭肩安抚",
                    "action": "秦枫(男)沉默，胡小跃(男)俯身耳语",
                }
            ],
            "frame_observations": [
                {
                    "batch_index": 0,
                    "timestamp": "00:06:40,000",
                    "observation": "[近景] 审讯室，秦枫(男)低头，逆光暖调",
                }
            ],
            "batches": [],
        }
        apply_character_field_separation_to_artifact(artifact)
        segment = artifact["scene_segments"][0]
        frame = artifact["frame_observations"][0]
        self.assertEqual(["秦枫", "胡小跃"], segment["characters"])
        self.assertNotIn("秦枫", segment["observation"])
        self.assertNotIn("胡小跃", segment["action"])
        self.assertEqual(["秦枫"], frame["characters"])
        self.assertEqual("[近景] 审讯室，低头，逆光暖调", frame["observation"])


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
        self.assertIn("未名人员(男/女)", prompt)
        self.assertIn("须含可见性别", prompt)
        self.assertIn("仅可见画面", prompt)

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
        self.assertIn("禁止写姓名(男/女)", prompt)
        self.assertIn("只允许在 characters 写**已上传头像名单内**", prompt)
        self.assertIn("禁止写名单外旧称（如伟业、老叶", prompt)

    def test_batch_prompt_default_skips_drama_knowledge_for_fazu_theme(self):
        service = DocumentaryFrameAnalysisService()
        prompt = service._build_batch_prompt(
            frame_count=2,
            video_theme="罚罪2",
            custom_prompt="",
            documentary_settings={
                "documentary_compact_mode": True,
                "documentary_compact_style": "fazu2",
            },
            time_range="00:00:00,000-00:00:06,000",
        )
        self.assertNotIn("剧集人物关系对照（抽帧分析必读", prompt)
        self.assertIn("仅可见画面", prompt)

    def test_batch_prompt_includes_drama_knowledge_for_fazu_theme(self):
        service = DocumentaryFrameAnalysisService()
        prompt = service._build_batch_prompt(
            frame_count=2,
            video_theme="罚罪2",
            custom_prompt="",
            documentary_settings={
                "documentary_compact_mode": True,
                "documentary_compact_style": "fazu2",
                "enable_frame_analysis_drama_knowledge": True,
            },
            time_range="00:00:00,000-00:00:06,000",
        )
        self.assertIn("剧集人物关系对照（抽帧分析必读", prompt)
        self.assertIn("秦枫", prompt)
        self.assertIn("胡小跃", prompt)

    def test_batch_prompt_skips_drama_knowledge_when_disabled(self):
        service = DocumentaryFrameAnalysisService()
        prompt = service._build_batch_prompt(
            frame_count=2,
            video_theme="罚罪2",
            custom_prompt="",
            documentary_settings={
                "documentary_compact_mode": True,
                "documentary_compact_style": "fazu2",
                "enable_frame_analysis_drama_knowledge": False,
                "enable_subtitle_analysis_drama_knowledge": False,
            },
            time_range="00:00:00,000-00:00:06,000",
        )
        self.assertNotIn("剧集人物关系对照（抽帧分析必读", prompt)

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
                        {
                            "timestamp": "00:00:02,000",
                            "observation": "出现老叶字样",
                            "burned_in_subtitle": "老叶，",
                            "has_burned_in_subtitle": True,
                        },
                        {
                            "timestamp": "00:00:05,000",
                            "observation": "对话中",
                            "burned_in_subtitle": "你都到厅级了，",
                            "has_burned_in_subtitle": True,
                        },
                    ],
                    "scene_segments": [
                        {"timestamp": "00:00:01,940-00:00:09,940", "scene": "楼顶天台"},
                    ],
                }
            ],
            "frame_observations": [
                {
                    "timestamp": "00:00:02,000",
                    "observation": "出现老叶字样",
                    "batch_index": 0,
                    "burned_in_subtitle": "老叶，",
                    "has_burned_in_subtitle": True,
                },
                {
                    "timestamp": "00:00:05,000",
                    "observation": "对话中",
                    "batch_index": 0,
                    "burned_in_subtitle": "你都到厅级了，",
                    "has_burned_in_subtitle": True,
                },
            ],
            "scene_segments": [
                {"timestamp": "00:00:01,940-00:00:09,940", "scene": "楼顶天台", "batch_index": 0},
            ],
        }

        enriched = attach_subtitles_to_frame_analysis_artifact(artifact, self.SAMPLE_SRT)
        self.assertTrue(enriched.get("subtitle_attached"))
        self.assertEqual("burned_in_only", enriched.get("subtitle_source"))
        segment = enriched["scene_segments"][0]
        self.assertIn("subtitle", segment)
        self.assertIn("老叶", segment["subtitle"])
        self.assertIn("厅级", segment["subtitle"])
        self.assertNotIn("subtitle_entries", segment)
        self.assertNotIn("time_range", segment)
        self.assertNotIn("subtitle_entries", enriched["batches"][0])

    def test_dedupe_scene_environment_across_segments(self):
        from app.services.documentary.frame_timeline_sampling import (
            dedupe_scene_environment_across_segments,
        )

        segments = [
            {
                "timestamp": "00:00:00,000-00:00:10,000",
                "scene": "楼顶天台",
                "key_visual": "阴天冷色调，云层低垂",
                "emotion": "压抑",
                "observation": "阴天冷色调；胡小跃站在天台边缘",
                "action": "胡小跃(男)站立",
            },
            {
                "timestamp": "00:00:10,000-00:00:20,000",
                "scene": "楼顶天台",
                "key_visual": "阴天冷色调，云层低垂",
                "emotion": "压抑",
                "observation": "阴天冷色调；秦枫与胡小跃对话",
                "action": "秦枫(男)与胡小跃交谈",
            },
            {
                "timestamp": "00:00:20,000-00:00:30,000",
                "scene": "地下仓库",
                "key_visual": "昏暗仓库",
                "action": "扭打",
            },
        ]
        dedupe_scene_environment_across_segments(segments)
        self.assertIn("scene", segments[0])
        self.assertNotIn("scene", segments[1])
        self.assertNotIn("key_visual", segments[1])
        self.assertNotIn("emotion", segments[1])
        self.assertIn("秦枫", segments[1]["observation"])
        self.assertNotIn("阴天冷色调", segments[1]["observation"])
        self.assertIn("scene", segments[2])

    def test_compress_analysis_artifact_strips_debug_and_subtitle_dup(self):
        from app.services.documentary.frame_analysis_compact import compress_analysis_artifact

        artifact = {
            "scene_segments": [
                {
                    "timestamp": "00:00:00,000-00:00:05,000",
                    "scene": "办公室",
                    "subtitle": "老叶，",
                    "subtitle_entries": [
                        {"start": "00:00:01,940", "end": "00:00:02,420", "text": "老叶，"},
                    ],
                    "batch_index": 0,
                }
            ],
            "batches": [
                {
                    "batch_index": 0,
                    "status": "success",
                    "time_range": "00:00:00,000-00:00:05,000",
                    "raw_response": "x" * 5000,
                    "frame_files": ["keyframe_000002_000002000.jpg"],
                    "subtitle": "老叶，",
                    "subtitle_entries": [{"start": "00:00:01,940", "end": "00:00:02,420", "text": "老叶，"}],
                    "frame_observations": [{"timestamp": "00:00:02,000", "observation": "画面"}],
                }
            ],
            "frame_observations": [
                {
                    "timestamp": "00:00:02,000",
                    "observation": "画面",
                    "subtitle": "老叶，",
                    "batch_index": 0,
                }
            ],
        }
        compress_analysis_artifact(artifact)
        segment = artifact["scene_segments"][0]
        self.assertIn("subtitle", segment)
        self.assertNotIn("subtitle_entries", segment)
        batch = artifact["batches"][0]
        self.assertNotIn("raw_response", batch)
        self.assertNotIn("frame_paths", batch)
        self.assertNotIn("frame_observations", batch)
        frame = artifact["frame_observations"][0]
        self.assertNotIn("observation", frame)
        self.assertIn("subtitle", frame)

    def test_compress_analysis_artifact_preserves_full_when_strip_debug_false(self):
        from app.services.documentary.frame_analysis_compact import compress_analysis_artifact

        artifact = {
            "scene_segments": [
                {
                    "timestamp": "00:00:00,000-00:00:05,000",
                    "scene": "办公室",
                    "observation": "对话",
                    "batch_index": 0,
                }
            ],
            "batches": [
                {
                    "batch_index": 0,
                    "status": "success",
                    "time_range": "00:00:00,000-00:00:05,000",
                    "raw_response": "debug",
                    "frame_files": ["keyframe_000002_000002000.jpg"],
                    "frame_observations": [
                        {
                            "timestamp": "00:00:02,000",
                            "observation": "画面",
                            "burned_in_subtitle": "对白",
                            "has_burned_in_subtitle": True,
                        }
                    ],
                }
            ],
            "frame_observations": [
                {
                    "timestamp": "00:00:02,000",
                    "observation": "画面",
                    "burned_in_subtitle": "对白",
                    "has_burned_in_subtitle": True,
                    "batch_index": 0,
                }
            ],
        }
        compress_analysis_artifact(artifact, strip_debug=False)
        self.assertIn("raw_response", artifact["batches"][0])
        self.assertIn("frame_files", artifact["batches"][0])
        self.assertNotIn("frame_paths", artifact["batches"][0])
        self.assertIn("frame_observations", artifact["batches"][0])
        self.assertIn("observation", artifact["frame_observations"][0])
        self.assertEqual("对白", artifact["frame_observations"][0]["burned_in_subtitle"])

    def test_compact_frame_storage_uses_short_filenames(self):
        from app.services.documentary.frame_analysis_compact import (
            compact_frame_storage_in_artifact,
            resolve_batch_frame_files,
        )

        cache_key = "abc123_def456"
        cache_dir = os.path.join(utils.temp_dir(), "keyframes", cache_key)
        artifact = {
            "keyframe_cache_key": cache_key,
            "batches": [
                {
                    "batch_index": 0,
                    "frame_files": [
                        "keyframe_000000_000000000.jpg",
                        "keyframe_000075_000003000.jpg",
                    ],
                    "frame_observations": [
                        {
                            "timestamp": "00:00:00,000",
                            "observation": "画面",
                        }
                    ],
                }
            ],
            "frame_observations": [
                {
                    "timestamp": "00:00:00,000",
                    "observation": "画面",
                }
            ],
        }
        compact_frame_storage_in_artifact(artifact)
        batch = artifact["batches"][0]
        self.assertEqual(
            ["keyframe_000000_000000000.jpg", "keyframe_000075_000003000.jpg"],
            batch.get("frame_files"),
        )
        self.assertNotIn("frame_paths", batch)
        self.assertNotIn("frame_path", artifact["frame_observations"][0])

        with patch("os.path.isfile", return_value=True):
            resolved = resolve_batch_frame_files(artifact, batch)
        self.assertEqual(
            [
                os.path.join(cache_dir, "keyframe_000000_000000000.jpg"),
                os.path.join(cache_dir, "keyframe_000075_000003000.jpg"),
            ],
            resolved,
        )

    def test_build_video_segment_overview(self):
        from app.services.documentary.frame_analysis_compact import build_video_segment_overview

        artifact = {
            "scene_segments": [
                {
                    "timestamp": "00:00:00,000-00:00:10,000",
                    "scene": "楼顶天台",
                    "observation": "叶天佑与楚青桐对话",
                    "action": "两人对峙",
                },
                {
                    "timestamp": "00:00:10,000-00:00:20,000",
                    "scene": "停车场",
                    "observation": "秦枫奔跑追捕",
                    "action": "秦枫(男)持枪奔跑",
                },
            ]
        }
        overview = build_video_segment_overview(artifact)
        self.assertEqual(2, overview["segment_count"])
        self.assertEqual("00:00:00,000-00:00:20,000", overview["time_span"])
        self.assertEqual(2, len(overview["segments"]))
        self.assertIn("全片共 2 个片段", overview["narrative_outline"])
        self.assertIn("楼顶天台", overview["narrative_outline"])

    def test_assign_subtitle_entries_to_segments_assigns_once(self):
        from app.services.documentary.documentary_subtitle_enrichment import (
            assign_subtitle_entries_to_segments,
            extract_subtitle_entries_from_frame_analysis,
        )
        from app.services.srt_utils import parse_srt

        srt = """1
00:00:29,150 --> 00:00:31,550
我了解小月。

2
00:00:31,990 --> 00:00:35,230
她不是对组织、对自己失去

3
00:00:36,750 --> 00:00:38,790
更不是害怕和逃避。
"""
        rooftop = {
            "timestamp": "00:00:25,000-00:00:35,000",
            "scene": "楼顶天台",
            "batch_index": 0,
        }
        warehouse = {
            "timestamp": "00:00:30,000-00:00:40,000",
            "scene": "地下仓库",
            "batch_index": 0,
        }
        segments = [rooftop, warehouse]
        assign_subtitle_entries_to_segments(segments, parse_srt(srt))

        rooftop_starts = {item["start"] for item in rooftop.get("subtitle_entries") or []}
        warehouse_starts = {item["start"] for item in warehouse.get("subtitle_entries") or []}
        self.assertEqual(set(), rooftop_starts & warehouse_starts)
        self.assertIn("00:00:29,150", rooftop_starts)
        self.assertIn("00:00:36,750", warehouse_starts)

        extracted = extract_subtitle_entries_from_frame_analysis(
            {"scene_segments": segments, "batches": []}
        )
        starts = [entry.start_ms for entry in extracted]
        self.assertEqual(len(starts), len(set(starts)))

    def test_partition_subtitle_entries_removes_overlap_duplicates(self):
        from app.services.documentary.documentary_subtitle_enrichment import (
            partition_subtitle_entries_across_segments,
            resolve_segment_time_range,
        )

        rooftop = {
            "timestamp": "00:00:25,000-00:00:35,000",
            "scene": "楼顶天台",
            "subtitle_entries": [
                {"start": "00:00:23,340", "end": "00:00:26,740", "text": "还有一些关于举报他个人的材料也正在核实。"},
                {"start": "00:00:28,390", "end": "00:00:29,070", "text": "我的徒弟，"},
                {"start": "00:00:29,150", "end": "00:00:31,550", "text": "我了解小月。"},
                {"start": "00:00:31,990", "end": "00:00:35,230", "text": "她不是对组织、对自己失去"},
                {"start": "00:00:35,230", "end": "00:00:35,830", "text": "信心，"},
            ],
        }
        warehouse = {
            "timestamp": "00:00:30,000-00:00:40,000",
            "scene": "地下仓库",
            "subtitle_entries": [
                {"start": "00:00:29,150", "end": "00:00:31,550", "text": "我了解小月。"},
                {"start": "00:00:31,990", "end": "00:00:35,230", "text": "她不是对组织、对自己失去"},
                {"start": "00:00:35,230", "end": "00:00:35,830", "text": "信心，"},
                {"start": "00:00:36,750", "end": "00:00:38,790", "text": "更不是害怕和逃避。"},
            ],
        }
        segments = [rooftop, warehouse]
        partition_subtitle_entries_across_segments(segments)

        rooftop_starts = {item["start"] for item in rooftop["subtitle_entries"]}
        warehouse_starts = {item["start"] for item in warehouse["subtitle_entries"]}
        self.assertEqual(
            set(),
            rooftop_starts & warehouse_starts,
            "重叠 scene 不应共享同一条 subtitle_entries",
        )
        self.assertIn("00:00:29,150", rooftop_starts)
        self.assertIn("00:00:36,750", warehouse_starts)
        self.assertNotIn("00:00:29,150", warehouse_starts)
        self.assertEqual(
            "00:00:23,340-00:00:31,550",
            resolve_segment_time_range(rooftop),
        )
        self.assertEqual(
            "00:00:31,990-00:00:38,790",
            resolve_segment_time_range(warehouse),
        )

    def test_resolve_segment_time_range_from_subtitle_entries(self):
        from app.services.documentary.documentary_subtitle_enrichment import (
            resolve_segment_time_range,
        )

        segment = {
            "timestamp": "00:00:01,000-00:00:09,940",
            "subtitle_entries": [
                {"start": "00:00:01,940", "end": "00:00:02,420", "text": "老叶，"},
                {"start": "00:00:05,100", "end": "00:00:06,340", "text": "你都到厅级了，"},
            ],
        }
        self.assertEqual("00:00:01,940-00:00:06,340", resolve_segment_time_range(segment))

    def test_attach_subtitles_uses_burned_in_only(self):
        from app.services.documentary.documentary_subtitle_enrichment import (
            attach_burned_in_subtitles_to_artifact,
            attach_subtitles_to_frame_analysis_artifact,
            is_phantom_subtitle_fragment,
        )

        self.assertTrue(is_phantom_subtitle_fragment("了。"))
        self.assertFalse(is_phantom_subtitle_fragment("胡小跃是我的徒弟。"))

        artifact = {
            "batches": [],
            "frame_observations": [
                {
                    "timestamp": "00:00:17,000",
                    "observation": "伟业说话",
                    "burned_in_subtitle": "胡小跃是我的徒弟",
                    "has_burned_in_subtitle": True,
                    "batch_index": 0,
                }
            ],
            "scene_segments": [
                {"timestamp": "00:00:16,000-00:00:18,000", "batch_index": 0},
            ],
        }
        enriched = attach_subtitles_to_frame_analysis_artifact(
            artifact,
            """1
00:00:16,940 --> 00:00:18,700
胡小月是我的徒弟
""",
        )
        self.assertEqual("胡小跃是我的徒弟", enriched["scene_segments"][0]["subtitle"])
        self.assertNotIn("subtitle_entries", enriched["scene_segments"][0])

        artifact_with_phantom = {
            "batches": [],
            "frame_observations": [
                {
                    "timestamp": "00:00:09,940",
                    "burned_in_subtitle": "了。",
                    "has_burned_in_subtitle": True,
                    "batch_index": 0,
                },
                {
                    "timestamp": "00:00:12,000",
                    "burned_in_subtitle": "胡小跃是我的徒弟。",
                    "has_burned_in_subtitle": True,
                    "batch_index": 0,
                },
            ],
            "scene_segments": [
                {"timestamp": "00:00:09,000-00:00:16,000", "batch_index": 0},
            ],
        }
        attach_burned_in_subtitles_to_artifact(artifact_with_phantom)
        subtitle = artifact_with_phantom["scene_segments"][0].get("subtitle", "")
        self.assertIn("胡小跃", subtitle)
        self.assertNotIn("了。", subtitle)


class FrameAnalysisDramaKnowledgeTests(unittest.TestCase):
    def test_correct_name_mistakes_in_text(self):
        from app.services.short_drama_drama_knowledge import correct_name_mistakes_in_text

        self.assertEqual(
            "秦枫与胡小跃对话",
            correct_name_mistakes_in_text("秦峰与胡小月对话"),
        )

    def test_apply_name_corrections_to_frame_analysis_artifact(self):
        from app.services.short_drama_drama_knowledge import (
            apply_name_corrections_to_frame_analysis_artifact,
        )

        artifact = {
            "scene_segments": [
                {
                    "action": "秦峰(男)与罗伯对峙",
                    "observation": "胡小月在旁",
                    "subtitle": "秦峰说",
                    "subtitle_entries": [{"start": "00:00:29,150", "end": "00:00:31,550", "text": "我了解小月。"}],
                },
            ],
            "batches": [
                {
                    "scene_segments": [{"subtitle": "秦峰说"}],
                    "frame_observations": [{"observation": "罗伯出现", "burned_in_subtitle": "罗伯出现"}],
                }
            ],
        }
        apply_name_corrections_to_frame_analysis_artifact(artifact)
        self.assertIn("秦枫", artifact["scene_segments"][0]["action"])
        self.assertIn("胡小跃", artifact["scene_segments"][0]["observation"])
        self.assertIn("秦峰", artifact["scene_segments"][0]["subtitle"])
        self.assertEqual("我了解小月。", artifact["scene_segments"][0]["subtitle_entries"][0]["text"])
        self.assertIn("秦峰", artifact["batches"][0]["scene_segments"][0]["subtitle"])
        self.assertIn("罗博", artifact["batches"][0]["frame_observations"][0]["observation"])
        self.assertEqual("罗伯出现", artifact["batches"][0]["frame_observations"][0]["burned_in_subtitle"])

    def test_extract_subtitle_srt_from_subtitle_entries(self):
        from app.services.documentary.documentary_subtitle_enrichment import (
            extract_subtitle_entries_from_frame_analysis,
        )

        data = {
            "scene_segments": [
                {
                    "timestamp": "00:00:29,150-00:00:38,790",
                    "subtitle_entries": [
                        {
                            "start": "00:00:29,150",
                            "end": "00:00:31,550",
                            "text": "我了解小月。",
                        },
                        {
                            "start": "00:00:31,990",
                            "end": "00:00:35,230",
                            "text": "她不是对组织、对自己失去",
                        },
                    ],
                }
            ],
            "batches": [],
        }
        entries = extract_subtitle_entries_from_frame_analysis(data)
        self.assertEqual(2, len(entries))
        self.assertEqual("我了解小月。", entries[0].text)
        self.assertEqual("她不是对组织、对自己失去", entries[1].text)

    def test_build_plot_blueprint_material_principles(self):
        from app.services.documentary.documentary_subtitle_enrichment import (
            build_plot_blueprint_material_principles,
        )

        with_srt_and_frame = build_plot_blueprint_material_principles(
            has_srt_subtitle=True,
            has_frame_subtitle=True,
            theme="罚罪",
        )
        self.assertIn("抽帧（主·画面/场景）", with_srt_and_frame)
        self.assertIn("SRT 字幕（对白/时间戳主）", with_srt_and_frame)
        self.assertIn("抽帧内字幕（辅）", with_srt_and_frame)
        self.assertIn("人名谐音/ASR 归并", with_srt_and_frame)

        with_frame_only = build_plot_blueprint_material_principles(
            has_frame_subtitle=True,
            theme="罚罪",
        )
        self.assertIn("对白字幕（取自抽帧）", with_frame_only)
        self.assertNotIn("SRT 字幕", with_frame_only)

        without_sub = build_plot_blueprint_material_principles(
            has_srt_subtitle=False,
            has_frame_subtitle=False,
        )
        self.assertIn("对白字幕（暂无）", without_sub)

    def test_resolve_subtitles_for_plot_blueprint(self):
        import json
        import tempfile
        from app.services.documentary.documentary_subtitle_enrichment import (
            resolve_subtitles_for_plot_blueprint,
        )

        srt = "1\n00:00:01,000 --> 00:00:02,000\n测试对白"
        payload = {
            "scene_segments": [
                {
                    "subtitle_entries": [
                        {
                            "start": "00:00:29,150",
                            "end": "00:00:31,550",
                            "text": "我了解小月。",
                        }
                    ]
                }
            ],
            "batches": [],
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as fp:
            json.dump(payload, fp, ensure_ascii=False)
            path = fp.name
        try:
            srt_text, frame_text, source = resolve_subtitles_for_plot_blueprint(
                subtitle_content=srt,
                frame_json_path=path,
            )
            self.assertEqual("srt_file", source)
            self.assertIn("测试对白", srt_text)
            self.assertIn("我了解小月。", frame_text)
        finally:
            os.remove(path)

    def test_build_plot_blueprint_name_unification_section(self):
        from app.services.short_drama_drama_knowledge import (
            build_plot_blueprint_name_unification_section,
        )

        fazu = build_plot_blueprint_name_unification_section(theme="罚罪2")
        self.assertIn("胡小跃", fazu)
        self.assertIn("秦峰", fazu)
        self.assertIn("叶天佑（老叶）≠ 伟业", fazu)
        self.assertIn("禁止", fazu)

        generic = build_plot_blueprint_name_unification_section(theme="某新剧")
        self.assertIn("人名谐音/简称归并", generic)
        self.assertNotIn("胡小跃 ←", generic)

    def test_resolve_frame_subtitle_for_plot_blueprint(self):
        import json
        import tempfile
        from app.services.documentary.documentary_subtitle_enrichment import (
            resolve_frame_subtitle_for_plot_blueprint,
        )

        payload = {
            "scene_segments": [
                {
                    "subtitle_entries": [
                        {
                            "start": "00:00:29,150",
                            "end": "00:00:31,550",
                            "text": "我了解小月。",
                        }
                    ]
                }
            ],
            "batches": [],
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as fp:
            json.dump(payload, fp, ensure_ascii=False)
            path = fp.name
        try:
            text = resolve_frame_subtitle_for_plot_blueprint(path)
            self.assertIn("我了解小月。", text)
        finally:
            os.remove(path)

    def test_finalize_scene_segments_applies_name_corrections(self):
        from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService

        artifact = {
            "scene_segments": [
                {
                    "timestamp": "00:00:00,000-00:00:05,000",
                    "scene": "办公室",
                    "action": "秦峰与胡小月交谈",
                }
            ],
            "batches": [],
        }
        DocumentaryFrameExtractionService._finalize_scene_segments_in_artifact(artifact)
        self.assertIn("秦枫", artifact["scene_segments"][0]["action"])
        self.assertIn("胡小跃", artifact["scene_segments"][0]["action"])


class SceneSegmentDedupTests(unittest.TestCase):
    def test_dedupe_scene_segments_picks_richer_for_identical_timestamp(self):
        from app.services.documentary.frame_timeline_sampling import dedupe_scene_segments

        segments = [
            {
                "timestamp": "00:00:00,000-00:00:10,000",
                "scene": "地下仓库",
                "observation": "打斗",
                "action": "扭打",
            },
            {
                "timestamp": "00:00:00,000-00:00:10,000",
                "scene": "楼顶天台",
                "observation": "阴天城市远景下，两名男性角色在天台边缘对峙，气氛严肃",
                "action": "老叶(男)与楚青桐(男)并肩站在天台边缘交谈",
                "emotion": "严肃、压抑",
            },
        ]
        result = dedupe_scene_segments(segments)
        self.assertEqual(1, len(result))
        kept = result[0]
        self.assertEqual("楼顶天台", kept["scene"])
        self.assertNotIn("切换至", kept["scene"])
        self.assertIn("阴天城市远景", kept["observation"])
        self.assertNotIn("画面由", kept["observation"])
        self.assertIn("老叶", kept["action"])
        self.assertEqual("严肃、压抑", kept["emotion"])

    def test_split_scene_segments_splits_scene_chain(self):
        from app.services.documentary.frame_timeline_sampling import split_scene_segments

        segments = [
            {
                "timestamp": "00:00:00,000-00:00:20,000",
                "scene": "从天空切换至地下车库切换至地下仓库",
                "observation": "画面由天空切换至地下车库切换至地下仓库；阴云密布；昏暗车库；仓库内搏斗",
                "action": "说话；站立；扭打",
                "emotion": "压抑；紧张；混乱",
                "key_visual": "冷色调；低光；杂乱",
            }
        ]
        result = split_scene_segments(segments)
        self.assertEqual(3, len(result))
        self.assertEqual("天空", result[0]["scene"])
        self.assertEqual("地下车库", result[1]["scene"])
        self.assertEqual("地下仓库", result[2]["scene"])
        self.assertNotIn("切换至", result[0]["scene"])
        self.assertNotIn("画面由", result[0]["observation"])

    def test_dedupe_scene_segments_keeps_different_scenes_when_overlapping(self):
        from app.services.documentary.frame_timeline_sampling import dedupe_scene_segments

        segments = [
            {
                "timestamp": "00:01:57,550-00:02:19,000",
                "scene": "废弃建筑内部",
                "observation": "秦枫与汪涛在破窗前对话，表情严肃互探态度，气氛紧张疑虑，环境破旧",
                "action": "秦枫(男)与汪涛(男)在破窗前对话",
            },
            {
                "timestamp": "00:01:55,000-00:01:59,000",
                "scene": "窗后特写",
                "observation": "短",
            },
            {
                "timestamp": "00:02:00,000-00:02:10,000",
                "scene": "废弃建筑楼梯间",
                "observation": "楼梯间",
            },
        ]
        result = dedupe_scene_segments(segments)
        self.assertEqual(3, len(result))
        scenes = {item["scene"] for item in result}
        self.assertIn("窗后特写", scenes)
        self.assertIn("废弃建筑内部", scenes)
        self.assertIn("废弃建筑楼梯间", scenes)
        for item in result:
            self.assertNotIn("切换至", item["scene"])

    def test_merge_same_scene_within_batch(self):
        from app.services.documentary.frame_timeline_sampling import merge_same_scene_within_batch

        segments = [
            {
                "batch_index": 0,
                "timestamp": "00:00:01,000-00:00:05,000",
                "scene": "楼顶天台",
                "observation": "两人对话",
                "action": "站立交谈",
            },
            {
                "batch_index": 0,
                "timestamp": "00:00:05,000-00:00:10,000",
                "scene": "楼顶天台",
                "observation": "继续对话",
                "action": "并肩站立",
            },
            {
                "batch_index": 0,
                "timestamp": "00:00:06,000-00:00:09,000",
                "scene": "室外停车场",
                "observation": "停车场",
            },
        ]
        merged = merge_same_scene_within_batch(segments)
        self.assertEqual(2, len(merged))
        rooftop = next(item for item in merged if item["scene"] == "楼顶天台")
        self.assertEqual("00:00:01,000-00:00:10,000", rooftop["timestamp"])
        self.assertIn("继续对话", rooftop["observation"])

    def test_merge_same_scene_skips_empty_scene_segments(self):
        from app.services.documentary.frame_timeline_sampling import merge_same_scene_within_batch

        segments = [
            {
                "batch_index": 0,
                "timestamp": "00:00:01,000-00:00:05,000",
                "scene": "",
                "observation": "审讯室对话",
            },
            {
                "batch_index": 0,
                "timestamp": "00:00:05,000-00:00:10,000",
                "scene": "",
                "observation": "走廊追逐",
            },
        ]
        merged = merge_same_scene_within_batch(segments)
        self.assertEqual(2, len(merged))

    def test_infer_scene_label_from_observation(self):
        from app.services.documentary.frame_timeline_sampling import infer_scene_label_from_segment

        label = infer_scene_label_from_segment(
            {
                "scene": "",
                "observation": "阴天楼顶，叶天佑与另一男子面对面站立，气氛严肃压抑",
            }
        )
        self.assertEqual("阴天楼顶", label)

    def test_prune_cross_scene_overlaps_keeps_richest(self):
        from app.services.documentary.frame_timeline_sampling import prune_cross_scene_overlaps

        segments = [
            {
                "batch_index": 0,
                "timestamp": "00:00:01,940-00:00:16,940",
                "scene": "楼顶天台",
                "observation": "阴天楼顶，叶天佑与另一男子面对面站立，气氛严肃压抑",
                "action": "叶天佑(男)与未名人员(男)在天台交谈",
                "emotion": "严肃、压抑",
                "key_visual": "阴天冷色调，城市建筑远景",
            },
            {
                "batch_index": 0,
                "timestamp": "00:00:02,100-00:00:06,000",
                "scene": "案发现场",
                "observation": "男警倒在血泊中",
                "action": "胡小跃(男)倒地",
            },
            {
                "batch_index": 0,
                "timestamp": "00:00:03,500-00:00:08,000",
                "scene": "龙湾村航拍",
                "observation": "航拍渔村",
            },
        ]
        pruned = prune_cross_scene_overlaps(segments, overlap_ratio=0.5)
        self.assertEqual(1, len(pruned))
        self.assertEqual("楼顶天台", pruned[0]["scene"])

    def test_normalize_scene_segments_strict_prunes_opening_hallucinations(self):
        from app.services.documentary.frame_timeline_sampling import normalize_scene_segments

        segments = [
            {
                "batch_index": 0,
                "timestamp": "00:00:00,000-00:00:15,000",
                "scene": "室外夜景与住宅区",
                "observation": "警服与住宅区",
            },
            {
                "batch_index": 0,
                "timestamp": "00:00:01,940-00:00:16,940",
                "scene": "楼顶天台",
                "observation": "阴天楼顶，两人交谈，气氛严肃",
                "action": "两人并肩对话",
                "emotion": "压抑",
            },
            {
                "batch_index": 0,
                "timestamp": "00:00:04,500-00:00:09,000",
                "scene": "室内仓库",
                "observation": "仓库突袭",
                "action": "持枪搜索",
            },
        ]
        result = normalize_scene_segments(segments, strict_scene_rules=True)
        self.assertLessEqual(len(result), 2)
        scenes = {item["scene"] for item in result}
        self.assertNotIn("室内仓库", scenes & {"室内仓库", "室外夜景与住宅区"})

    def test_split_bloated_segment_without_scene_does_not_crash(self):
        from app.services.documentary.frame_timeline_sampling import split_scene_segments

        segments = [
            {
                "batch_index": 0,
                "timestamp": "00:00:00,000-00:00:45,000",
                "observation": "第一段；第二段",
                "action": "动作一；动作二",
            },
        ]
        result = split_scene_segments(segments, max_duration_ms=15000)
        self.assertGreater(len(result), 1)
        for item in result:
            self.assertNotIn("scene", item)

    def test_normalize_scene_segments_does_not_collapse_to_two_blobs(self):
        from app.services.documentary.frame_timeline_sampling import normalize_scene_segments
        from app.services.documentary.frame_analysis_pairing import load_analysis_artifact
        from pathlib import Path

        sample_path = (
            Path(__file__).resolve().parents[1]
            / "instances/1/storage/temp/analysis/6月4日抽帧_frame_analysis.json"
        )
        if not sample_path.exists():
            self.skipTest("sample frame analysis artifact missing")
        artifact = load_analysis_artifact(str(sample_path))
        raw = [s for s in artifact.get("scene_segments", []) if isinstance(s, dict)]
        result = normalize_scene_segments(raw, strict_scene_rules=True)
        self.assertGreater(len(result), 10)
        max_duration_ms = 0
        for segment in result:
            from app.services.documentary.frame_timeline_sampling import _segment_timestamp_bounds

            start_ms, end_ms = _segment_timestamp_bounds(segment)
            max_duration_ms = max(max_duration_ms, end_ms - start_ms)
        self.assertLess(max_duration_ms, 5 * 60 * 1000)

    def test_parse_scene_chain_repairs_corrupted_cong_prefix(self):
        from app.services.documentary.frame_timeline_sampling import (
            _format_scene_chain,
            _parse_scene_chain,
        )

        scenes = _parse_scene_chain("从从从天空切换至地下车库切换至从地下仓库")
        self.assertEqual(["天空", "地下车库", "地下仓库"], scenes)
        self.assertEqual(
            "从天空切换至地下车库切换至地下仓库",
            _format_scene_chain(scenes),
        )

    def test_compact_analysis_artifact_deduplicates_scene_segments(self):
        from app.services.documentary.frame_analysis_compact import compact_analysis_artifact

        artifact = {
            "artifact_version": "documentary-frame-analysis-v4",
            "video_path": "/tmp/demo.mp4",
            "scene_segments": [
                {
                    "timestamp": "00:00:00,000-00:00:10,000",
                    "scene": "地下车库",
                    "observation": "x",
                },
                {
                    "timestamp": "00:00:00,000-00:00:10,000",
                    "scene": "天空",
                    "observation": "阴云密布的天空中，阳光从云层缝隙透出",
                },
                {
                    "timestamp": "00:00:10,900-00:00:19,000",
                    "scene": "楼顶天台",
                    "observation": "对话",
                },
            ],
            "batches": [],
        }
        compact = compact_analysis_artifact(artifact, include_frame_observations=False)
        self.assertEqual(2, len(compact["scene_segments"]))
        self.assertEqual("天空", compact["scene_segments"][0]["scene"])
        self.assertIn("阴云密布", compact["scene_segments"][0]["observation"])
        self.assertEqual("楼顶天台", compact["scene_segments"][1]["scene"])


class PlotBlueprintValidationTests(unittest.TestCase):
    def test_validate_plot_blueprint_rejects_out_of_bounds_timestamp(self):
        from app.services.documentary.documentary_plot_blueprint_validator import (
            validate_plot_blueprint,
        )

        text = (
            "## 主要人物表\n- 胡小跃(男)\n"
            "## 开头高潮方案\n00:40:00,000-00:40:10,000\n"
            "## 原片时间线\n1. 事件\n"
            "## 成片叙事顺序方案\n_id 1\n"
            "## 建议保留原声 OST=1\n"
            "1. 说话人：老叶 台词：「测试」 时间戳：00:40:05,000-00:40:15,000\n"
            "## 解说 OST=0 脉络规划\n- 解说\n"
            "## 声画对位注意\n- 无\n"
            + "补" * 2100
        )
        result = validate_plot_blueprint(
            text,
            source_duration_ms=38 * 60 * 1000 + 13 * 1000,
            frame_max_ms=38 * 60 * 1000 + 13 * 1000,
            min_chars=2000,
            relaxed=False,
        )
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("超出" in issue for issue in result.get("issues") or [])
        )

    def test_validate_plot_blueprint_rejects_short_ost1_clip(self):
        from app.services.documentary.documentary_plot_blueprint_validator import (
            validate_plot_blueprint,
        )

        text = (
            "## 主要人物表\n- 胡小跃(男)\n"
            "## 开头高潮方案\n00:00:58,370-00:01:03,410\n"
            "## 原片时间线\n1. 事件\n"
            "## 成片叙事顺序方案\n_id 1\n"
            "## 建议保留原声 OST=1\n"
            "1. 说话人：胡小跃 台词：「那你们跟着我。」 "
            "时间戳：00:02:51,080-00:02:51,920\n"
            "## 解说 OST=0 脉络规划\n- 解说\n"
            "## 声画对位注意\n- 无\n"
            + "补" * 2100
        )
        result = validate_plot_blueprint(
            text,
            frame_max_ms=40 * 60 * 1000,
            settings={"ost1_duration_min": 8, "ost1_duration_max": 18},
            min_chars=2000,
            relaxed=False,
        )
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("较短" in issue or "过短" in issue for issue in result.get("issues") or [])
        )

    def test_validate_plot_blueprint_rejects_huxiaoyue_as_female(self):
        from app.services.documentary.documentary_plot_blueprint_validator import (
            validate_plot_blueprint,
        )

        text = (
            "## 主要人物表\n- 胡小跃：女警\n"
            "## 开头高潮方案\n00:00:58,370-00:01:03,410\n"
            "## 原片时间线\n1. 事件\n"
            "## 成片叙事顺序方案\n_id 1\n"
            "## 建议保留原声 OST=1\n"
            "1. 说话人：老叶 台词：「测试」 时间戳：00:00:58,370-00:01:08,370\n"
            "## 解说 OST=0 脉络规划\n- 解说\n"
            "## 声画对位注意\n- 无\n"
            + "补" * 2100
        )
        result = validate_plot_blueprint(
            text,
            frame_max_ms=40 * 60 * 1000,
            min_chars=2000,
            relaxed=False,
        )
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("胡小跃" in issue for issue in result.get("issues") or [])
        )

    def test_build_frame_analysis_time_bounds_section(self):
        from app.services.documentary.documentary_subtitle_enrichment import (
            build_frame_analysis_time_bounds_section,
        )

        section = build_frame_analysis_time_bounds_section(
            "",
            source_duration_sec=2300.0,
        )
        self.assertIn("抽帧时间边界", section)


class Fazu2Ost0TimingTests(unittest.TestCase):
    def test_align_ost0_lead_in_before_next_ost1(self):
        from app.services.documentary.documentary_script_optimizer import (
            _align_fazu2_ost0_to_adjacent_ost1,
        )
        from app.services.documentary.documentary_settings import get_documentary_compact_settings
        from app.services.srt_utils import parse_timestamp_range

        items = [
            {
                "_id": 1,
                "timestamp": "00:37:59,940-00:38:13,520",
                "picture": "庙会冲突",
                "narration": "播放原片",
                "OST": 1,
            },
            {
                "_id": 2,
                "timestamp": "00:00:01,940-00:00:16,260",
                "picture": "解说引入正叙",
                "narration": "宝子们，我们开始看罚罪2第一集。故事，得从另一场更绝望的对峙说起。",
                "OST": 0,
            },
            {
                "_id": 3,
                "timestamp": "00:03:08,639-00:03:17,639",
                "picture": "铁笼内，胡小跃被囚禁刑讯",
                "narration": "播放原片3",
                "OST": 1,
            },
        ]
        cfg = get_documentary_compact_settings()
        result = _align_fazu2_ost0_to_adjacent_ost1(items, cfg)
        ost0 = result[1]
        start_ms, end_ms = parse_timestamp_range(ost0["timestamp"])
        next_start_ms, _ = parse_timestamp_range(result[2]["timestamp"])
        lead_ms = int(float(cfg.get("ost0_lead_before_ost1_sec", 10)) * 1000)
        self.assertGreaterEqual(next_start_ms - start_ms, lead_ms - 500)
        self.assertLess(end_ms, next_start_ms)
        self.assertNotEqual("00:00:01,940", ost0["timestamp"].split("-", 1)[0])

    def test_align_ost0_commentary_uses_previous_ost1(self):
        from app.services.documentary.documentary_script_optimizer import (
            _align_fazu2_ost0_to_adjacent_ost1,
        )
        from app.services.documentary.documentary_settings import get_documentary_compact_settings

        items = [
            {
                "_id": 1,
                "timestamp": "00:00:16,940-00:00:26,700",
                "narration": "播放原片",
                "OST": 1,
            },
            {
                "_id": 2,
                "timestamp": "00:00:01,940-00:00:16,260",
                "narration": "一句话把领导噎住了。",
                "OST": 0,
            },
            {
                "_id": 3,
                "timestamp": "00:00:30,000-00:00:40,000",
                "narration": "播放原片",
                "OST": 0,
            },
        ]
        cfg = get_documentary_compact_settings()
        result = _align_fazu2_ost0_to_adjacent_ost1(items, cfg)
        self.assertTrue(result[1]["timestamp"].startswith("00:00:16,940"))


class OpeningClimaxReplayTests(unittest.TestCase):
    def test_apply_opening_climax_chronological_replay_inserts_before_chronological_slot(self):
        from app.services.documentary.opening_climax_resolver import (
            apply_opening_climax_chronological_replay,
        )

        items = [
            {
                "_id": 1,
                "timestamp": "00:10:23,010-00:10:39,520",
                "picture": "审讯室暖黄光，胡小跃背对镜头",
                "narration": "播放原片6",
                "OST": 1,
            },
            {
                "_id": 2,
                "timestamp": "00:00:00,000-00:00:10,000",
                "picture": "黑屏",
                "narration": "宝子们，我们开始看罚罪2第1集。",
                "OST": 0,
            },
            {
                "_id": 10,
                "timestamp": "00:09:13,520-00:10:12,560",
                "picture": "审讯室对峙",
                "narration": "播放原片6",
                "OST": 1,
            },
            {
                "_id": 11,
                "timestamp": "00:11:41,920-00:12:00,900",
                "picture": "局里宣布停职",
                "narration": "局里宣布胡小跃停职检讨",
                "OST": 0,
            },
        ]
        updated = apply_opening_climax_chronological_replay(items, enabled=True)
        self.assertEqual(5, len(updated))
        replay_items = [
            item
            for item in updated
            if item.get("timestamp") == "00:10:23,010-00:10:39,520" and int(item.get("_id") or 0) != 1
        ]
        self.assertEqual(1, len(replay_items))
        replay = replay_items[0]
        self.assertEqual(1, int(replay.get("OST")))
        self.assertIn("【复现】", str(replay.get("picture") or ""))
        self.assertEqual(5, int(updated[-1]["_id"]))
        self.assertEqual("00:11:41,920-00:12:00,900", updated[-1]["timestamp"])

    def test_apply_opening_climax_chronological_replay_skips_when_already_present(self):
        from app.services.documentary.opening_climax_resolver import (
            apply_opening_climax_chronological_replay,
        )

        items = [
            {
                "_id": 1,
                "timestamp": "00:10:23,010-00:10:39,520",
                "picture": "开篇",
                "narration": "播放原片",
                "OST": 1,
            },
            {
                "_id": 2,
                "timestamp": "00:10:23,010-00:10:39,520",
                "picture": "【复现】开篇",
                "narration": "播放原片",
                "OST": 1,
            },
        ]
        updated = apply_opening_climax_chronological_replay(items, enabled=True)
        self.assertEqual(2, len(updated))


class ShortDramaScriptOptimizerTests(unittest.TestCase):
    def test_enforce_opening_head_ost1_limit(self):
        from app.services.short_drama_script_optimizer import enforce_opening_head_ost1_limit

        items = [
            {"_id": 1, "OST": 1, "picture": "开篇", "narration": "播放原片1", "timestamp": "00:19:51,659-00:19:55,659"},
            {"_id": 2, "OST": 1, "picture": "交枪", "narration": "播放原片2", "timestamp": "00:01:04,719-00:01:08,879"},
            {"_id": 3, "OST": 0, "picture": "中景", "narration": "宝子们，我们开始看。", "timestamp": "00:00:01,000-00:00:18,000"},
        ]
        updated = enforce_opening_head_ost1_limit(items, head_count=3, max_ost1=1)
        ost1_ids = [int(item["_id"]) for item in updated if int(item.get("OST", 0)) == 1]
        self.assertEqual([1], ost1_ids)

    def test_enforce_scene_ost1_after_narration(self):
        from app.services.short_drama_script_optimizer import enforce_scene_ost1_after_narration

        items = [
            {"_id": 4, "OST": 1, "narration": "播放原片4", "timestamp": "00:09:55,878-00:10:00,878"},
            {"_id": 5, "OST": 1, "narration": "播放原片5", "timestamp": "00:10:23,000-00:10:28,000"},
        ]
        updated = enforce_scene_ost1_after_narration(items)
        self.assertEqual(0, int(updated[1].get("OST", 0)))

    def test_format_ost1_max_segments_rule(self):
        from app.services.short_drama_settings import (
            format_ost1_max_segments_rule,
            resolve_ost1_max_segments,
        )

        self.assertEqual(0, resolve_ost1_max_segments({"ost1_max_segments": 0}))
        self.assertIn("不设固定", format_ost1_max_segments_rule({"ost1_max_segments": 0}))
        self.assertIn("≤8 段", format_ost1_max_segments_rule({"ost1_max_segments": 8}))

    def test_convert_excess_ost1_skips_when_unlimited(self):
        from app.services.short_drama_script_optimizer import convert_excess_ost1_to_narration

        items = [
            {"_id": i, "OST": 1, "narration": f"播放原片{i}"}
            for i in range(1, 6)
        ]
        updated = convert_excess_ost1_to_narration(items, ost1_max=0)
        self.assertEqual(5, sum(1 for item in updated if int(item.get("OST", 0)) == 1))

    def test_convert_excess_ost1_keeps_lowest_id(self):
        from app.services.short_drama_script_optimizer import convert_excess_ost1_to_narration

        items = [
            {"_id": 1, "OST": 1, "picture": "开篇", "narration": "播放原片1", "timestamp": "00:19:51,659-00:19:55,659"},
            {"_id": 6, "OST": 1, "picture": "仓库", "narration": "播放原片2", "timestamp": "00:02:38,242-00:02:42,242"},
            {"_id": 12, "OST": 1, "picture": "审讯", "narration": "播放原片4", "timestamp": "00:09:55,878-00:10:00,878"},
        ]
        updated = convert_excess_ost1_to_narration(items, ost1_max=1)
        ost1_ids = [int(item["_id"]) for item in updated if int(item.get("OST", 0)) == 1]
        self.assertEqual([1], ost1_ids)
        self.assertEqual(3, len(updated))

    def test_remove_picture_echo_narrations(self):
        from app.services.short_drama_script_optimizer import (
            is_picture_echo_narration,
            remove_picture_echo_narrations,
        )

        self.assertTrue(
            is_picture_echo_narration(
                "随后，特写：楚青桐神情严肃，语气沉重。",
                "特写：楚青桐神情严肃，语气沉重",
            )
        )
        items = [
            {
                "_id": 2,
                "OST": 0,
                "picture": "特写：楚青桐神情严肃，语气沉重",
                "narration": "随后，特写：楚青桐神情严肃，语气沉重。",
                "timestamp": "00:01:08,929-00:01:13,929",
            }
        ]
        updated = remove_picture_echo_narrations(items)
        self.assertFalse(
            is_picture_echo_narration(
                updated[0]["narration"],
                updated[0]["picture"],
            )
        )


class FrameCharacterNamingTests(unittest.TestCase):
    def test_is_character_name_evidence_backed_with_alias(self):
        from app.services.documentary.frame_character_naming import is_character_name_evidence_backed

        self.assertTrue(is_character_name_evidence_backed("胡小跃", "我了解小月。"))
        self.assertFalse(is_character_name_evidence_backed("刘天也", "秦枫在说话"))

    def test_sanitize_segment_character_names(self):
        from app.services.documentary.frame_character_naming import sanitize_segment_character_names

        segment = {"characters": ["秦枫", "刘天也"]}
        removed = sanitize_segment_character_names(
            segment,
            reliable_faces={"秦枫"},
            reference_names={"秦枫", "刘天也"},
        )
        self.assertEqual(["秦枫"], segment["characters"])
        self.assertEqual(["刘天也"], removed)

    def test_strip_legacy_hallucinated_name_weiye(self):
        from app.services.documentary.frame_character_naming import (
            apply_face_gated_names_to_artifact,
            strip_unreliable_names_in_text,
        )

        text = strip_unreliable_names_in_text(
            "叶天佑(男)与伟业(男)相对而立，伟业伫立聆听",
            reliable_faces={"叶天佑"},
            ref_names={"叶天佑", "楚青桐", "秦枫"},
        )
        self.assertNotIn("伟业", text)
        self.assertIn("叶天佑(男)", text)

        artifact = {
            "character_references": [{"name": "叶天佑"}, {"name": "楚青桐"}],
            "scene_segments": [
                {
                    "batch_index": 0,
                    "action": "伟业(男)伫立，叶天佑(男)说话",
                    "observation": "楼顶天台，叶天佑(男)与伟业(男)对话",
                }
            ],
            "frame_observations": [
                {
                    "batch_index": 0,
                    "observation": "叶天佑(男)与伟业(男)相对而立",
                }
            ],
            "batches": [
                {
                    "batch_index": 0,
                    "overall_activity_summary": "叶天佑(男)与伟业(男)对话",
                    "frame_observations": [
                        {"observation": "叶天佑(男)与伟业(男)相对而立"},
                    ],
                    "scene_segments": [{"action": "伟业(男)伫立"}],
                }
            ],
        }
        apply_face_gated_names_to_artifact(artifact)
        self.assertNotIn("伟业", artifact["scene_segments"][0]["action"])
        self.assertNotIn("伟业", artifact["batches"][0]["overall_activity_summary"])

    def test_validate_face_naming_rejects_generic_labels_when_refs_attached(self):
        from app.services.documentary.frame_character_naming import (
            validate_face_naming_when_references_attached,
        )

        err = validate_face_naming_when_references_attached(
            frame_observations=[
                {"timestamp": "00:00:04,000", "observation": "[中景] 询问室，年轻警官(男)站立俯视"},
            ],
            scene_segments=[],
            character_references=[{"name": "秦枫"}, {"name": "叶天佑"}],
            reference_images_attached=True,
        )
        self.assertIn("per-frame", err.lower())

        ok = validate_face_naming_when_references_attached(
            frame_observations=[
                {"timestamp": "00:00:04,000", "observation": "[中景] 询问室，叶天佑(男)站立俯视秦枫(男)"},
            ],
            scene_segments=[],
            character_references=[{"name": "秦枫"}, {"name": "叶天佑"}],
            reference_images_attached=True,
        )
        self.assertEqual("", ok)

        mixed_err = validate_face_naming_when_references_attached(
            frame_observations=[
                {"timestamp": "00:00:00,000", "observation": "[特写] 秦枫(男)低头"},
                {"timestamp": "00:00:04,000", "observation": "[中景] 年轻警官(男)站立"},
            ],
            scene_segments=[],
            character_references=[{"name": "秦枫"}, {"name": "叶天佑"}],
            reference_images_attached=True,
        )
        self.assertIn("00:00:04,000", mixed_err)

    def test_strip_preserves_temporary_role_labels(self):
        from app.services.documentary.frame_character_naming import strip_unreliable_names_in_text

        text = strip_unreliable_names_in_text(
            "便衣男警察(男)训斥年轻男子(男)，警服男警官(男)站立",
            reliable_faces=set(),
            ref_names={"秦枫", "叶天佑"},
        )
        self.assertIn("便衣男警察(男)", text)
        self.assertIn("年轻男子(男)", text)
        self.assertNotIn("未名人员", text)


class FrameExtractionTestModeTests(unittest.TestCase):
    def test_default_test_analysis_path_for_video(self):
        from app.services.documentary.frame_analysis_pairing import default_test_analysis_path_for_video

        path = default_test_analysis_path_for_video(r"D:\素材\6月4日.mp4", max_duration_seconds=5)
        self.assertTrue(path.endswith("6月4日_frame_analysis_test_5s.json"))

        path_from = default_test_analysis_path_for_video(
            r"D:\素材\6月4日.mp4",
            max_duration_seconds=5,
            start_time_seconds=30,
        )
        self.assertTrue(path_from.endswith("6月4日_frame_analysis_test_from30s_5s.json"))

    def test_filter_keyframes_by_window(self):
        from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService

        frames = [
            "/tmp/keyframe_000000_000000000.jpg",
            "/tmp/keyframe_000075_000003000.jpg",
            "/tmp/keyframe_000150_000006000.jpg",
            "/tmp/keyframe_000225_000009000.jpg",
        ]
        filtered = DocumentaryFrameExtractionService._filter_keyframes_by_window(
            frames,
            start_time_seconds=0.0,
            max_duration_seconds=5.0,
        )
        self.assertEqual(frames[:2], filtered)

        window_filtered = DocumentaryFrameExtractionService._filter_keyframes_by_window(
            frames,
            start_time_seconds=3.0,
            max_duration_seconds=5.0,
        )
        self.assertEqual(frames[1:3], window_filtered)

    def test_filter_keyframes_by_max_duration(self):
        from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService

        frames = [
            "/tmp/keyframe_000000_000000000.jpg",
            "/tmp/keyframe_000075_000003000.jpg",
            "/tmp/keyframe_000150_000006000.jpg",
        ]
        filtered = DocumentaryFrameExtractionService._filter_keyframes_by_max_duration(frames, 5.0)
        self.assertEqual(frames[:2], filtered)

    def test_resolve_max_duration_seconds_only_in_test_mode(self):
        from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService

        self.assertIsNone(
            DocumentaryFrameExtractionService._resolve_max_duration_seconds(5, test_mode=False)
        )
        self.assertEqual(
            5.0,
            DocumentaryFrameExtractionService._resolve_max_duration_seconds(5, test_mode=True),
        )


    def test_obvious_relationship_supplement_when_both_named(self):
        from app.services.documentary.frame_character_naming import (
            apply_obvious_character_relationships_to_artifact,
        )

        artifact = {
            "drama_id": "罚罪2",
            "character_references": [{"name": "叶天佑"}, {"name": "秦枫"}],
            "scene_segments": [
                {
                    "batch_index": 0,
                    "action": "叶天佑(男)与秦枫(男)在天台交谈",
                    "observation": "阴天楼顶，两名男子对话",
                }
            ],
            "batches": [],
            "frame_observations": [
                {"batch_index": 0, "observation": "[远景] 楼顶，叶天佑(男)与秦枫(男)对话"},
                {"batch_index": 0, "observation": "[中景] 楼顶，叶天佑(男)侧脸"},
                {"batch_index": 0, "observation": "[中景] 楼顶，秦枫(男)侧脸"},
            ],
        }
        apply_obvious_character_relationships_to_artifact(artifact)
        segment = artifact["scene_segments"][0]
        relations = segment.get("character_relationships") or []
        self.assertTrue(any(item.get("type") == "师徒" for item in relations))
        self.assertIn("师徒", segment.get("observation", ""))

    def test_no_relationship_inference_from_rank_dialogue(self):
        from app.services.documentary.frame_character_naming import (
            apply_obvious_character_relationships_to_artifact,
        )

        artifact = {
            "drama_id": "罚罪2",
            "scene_segments": [
                {
                    "batch_index": 0,
                    "action": "叶天佑(男)与未名人员(男)对话",
                    "observation": "楼顶对话",
                    "subtitle": "老叶；你都到厅级了",
                }
            ],
            "batches": [],
            "frame_observations": [],
        }
        apply_obvious_character_relationships_to_artifact(artifact)
        segment = artifact["scene_segments"][0]
        self.assertNotIn("楚青桐", segment.get("action", ""))
        self.assertNotIn("楚青桐", segment.get("observation", ""))
        self.assertFalse(segment.get("character_relationships"))


class DramaCharacterRegistryTests(unittest.TestCase):
    def _temp_image_path(self) -> str:
        import tempfile

        import PIL.Image

        handle, path = tempfile.mkstemp(suffix=".jpg")
        os.close(handle)
        PIL.Image.new("RGB", (64, 64), color=(120, 80, 40)).save(path)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def test_list_dramas_includes_fazu2(self):
        from app.services.drama_character_registry import DEFAULT_DRAMA_ID, list_dramas

        dramas = list_dramas()
        self.assertTrue(any(item["id"] == DEFAULT_DRAMA_ID for item in dramas))

    def test_list_characters_for_fazu2(self):
        from app.services.drama_character_registry import (
            list_character_head_slot_groups,
            list_characters_for_drama,
        )

        names = list_characters_for_drama("罚罪2")
        self.assertIn("秦枫", names)
        self.assertIn("胡小跃", names)
        self.assertIn("罗博", names)
        self.assertNotIn("金鼎集团", names)
        self.assertEqual(59, len(names))

        groups = list_character_head_slot_groups("罚罪2")
        self.assertEqual(3, len(groups))
        self.assertEqual(20, len(groups[0]["slots"]))
        self.assertEqual("core", groups[0]["tier"])

    def test_head_widget_session_keys_are_ascii(self):
        from app.services.drama_character_registry import (
            character_widget_slot_id,
            head_selection_session_key,
            head_uploader_session_key,
        )

        slot_id = character_widget_slot_id("罚罪2", "秦枫")
        self.assertEqual(12, len(slot_id))
        self.assertTrue(slot_id.isalnum())
        self.assertNotIn("秦枫", head_uploader_session_key("罚罪2", "秦枫"))
        self.assertNotIn("秦枫", head_selection_session_key("罚罪2", "秦枫"))

    def test_list_unrecognized_head_images(self):
        import shutil
        import tempfile

        from app.services.drama_character_registry import (
            ensure_head_img_dir,
            list_unrecognized_head_images,
            save_head_image,
        )
        from PIL import Image
        import io

        tmp_root = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmp_root, ignore_errors=True))
        drama_dir = os.path.join(tmp_root, "headImg", "test_drama")
        os.makedirs(drama_dir, exist_ok=True)

        orphan_path = os.path.join(drama_dir, "Snipaste_test.png")
        Image.new("RGB", (8, 8), color=(1, 2, 3)).save(orphan_path)

        import app.services.drama_character_registry as registry

        original_root = registry._PROJECT_ROOT
        registry._PROJECT_ROOT = tmp_root
        self.addCleanup(lambda: setattr(registry, "_PROJECT_ROOT", original_root))

        from app.data.drama_knowledge import fazu2_upload_roster as roster_mod

        original_rosters = dict(roster_mod.DRAMA_UPLOAD_ROSTERS)
        roster_mod.DRAMA_UPLOAD_ROSTERS["test_drama"] = ({"name": "秦枫", "tier": "core", "role_hint": ""},)
        self.addCleanup(lambda: roster_mod.DRAMA_UPLOAD_ROSTERS.update(original_rosters))

        buf = io.BytesIO()
        Image.new("RGB", (8, 8), color=(4, 5, 6)).save(buf, format="JPEG")
        save_head_image("test_drama", "秦枫", buf.getvalue(), original_filename="a.jpg")

        orphans = list_unrecognized_head_images("test_drama")
        self.assertIn("Snipaste_test.png", orphans)
        self.assertNotIn("秦枫.jpg", orphans)
        from app.services.drama_character_registry import build_character_reference_prompt_section

        section = build_character_reference_prompt_section(
            [{"name": "秦枫", "path": "/tmp/qin.jpg"}],
            video_frame_count=3,
        )
        self.assertIn("秦枫", section)
        self.assertIn("3", section)
        self.assertIn("参照图 #1", section)

    def test_compose_batch_vision_inputs_prepends_refs(self):
        from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService

        frames = ["/tmp/frame1.jpg", "/tmp/frame2.jpg"]
        img = self._temp_image_path()
        refs = [{"name": "秦枫", "path": img}]
        settings = {"frame_reference_token_saver": False, "frame_reference_use_collage": False}
        images, active, carryover, ref_count = DocumentaryFrameExtractionService._compose_batch_vision_inputs(
            frames,
            batch_index=0,
            character_references=refs,
            documentary_settings=settings,
        )
        self.assertEqual(3, len(images))
        self.assertTrue(images[0].endswith(".jpg"))
        self.assertNotEqual(img, images[0])
        self.assertEqual(frames, images[1:])
        self.assertEqual(1, len(active))
        self.assertEqual(1, ref_count)
        self.assertEqual("", carryover)

    def test_compose_batch_vision_inputs_first_batch_only(self):
        from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService

        frames = ["/tmp/frame1.jpg"]
        img = self._temp_image_path()
        refs = [{"name": "秦枫", "path": img}]
        settings = {"frame_reference_token_saver": True, "frame_reference_use_collage": False}
        images0, _, carryover0, count0 = DocumentaryFrameExtractionService._compose_batch_vision_inputs(
            frames,
            batch_index=0,
            character_references=refs,
            documentary_settings=settings,
        )
        images1, _, carryover1, count1 = DocumentaryFrameExtractionService._compose_batch_vision_inputs(
            frames,
            batch_index=1,
            character_references=refs,
            documentary_settings=settings,
        )
        self.assertEqual(2, len(images0))
        self.assertEqual(1, count0)
        self.assertEqual(frames, images1)
        self.assertEqual(0, count1)
        self.assertIn("沿用", carryover1)

    def test_compose_batch_vision_inputs_relationship_diagram_first(self):
        from app.services.documentary.frame_extraction_service import DocumentaryFrameExtractionService

        frames = ["/tmp/frame1.jpg"]
        rel = self._temp_image_path()
        img = self._temp_image_path()
        refs = [{"name": "秦枫", "path": img}]
        settings = {"frame_reference_token_saver": False, "frame_reference_use_collage": False}
        images, active, _, ref_count = DocumentaryFrameExtractionService._compose_batch_vision_inputs(
            frames,
            batch_index=0,
            relationship_diagram_path=rel,
            character_references=refs,
            documentary_settings=settings,
        )
        self.assertEqual(3, len(images))
        self.assertTrue(images[0].endswith(".jpg"))
        self.assertTrue(os.path.isfile(images[0]))
        self.assertEqual(frames, images[2:])
        self.assertEqual(1, len(active))
        self.assertEqual(2, ref_count)

    def test_merge_frame_analysis_settings_disables_text_by_default(self):
        from app.services.drama_character_registry import merge_frame_analysis_settings_for_drama

        merged = merge_frame_analysis_settings_for_drama({}, "罚罪2")
        self.assertFalse(merged.get("enable_frame_analysis_drama_knowledge"))

    def test_merge_frame_analysis_settings_enables_text_when_requested(self):
        from app.services.drama_character_registry import merge_frame_analysis_settings_for_drama

        merged = merge_frame_analysis_settings_for_drama({}, "罚罪2", enable_knowledge_text=True)
        self.assertTrue(merged.get("enable_frame_analysis_drama_knowledge"))
        self.assertEqual("罚罪2", merged.get("selected_drama_id"))

    def test_should_attach_reference_images_first_batch_only(self):
        from app.services.documentary.frame_reference_images import (
            ATTACH_MODE_FIRST_BATCH,
            resolve_reference_collage_mode,
            should_attach_reference_images,
        )

        settings = {"frame_reference_token_saver": True}
        self.assertTrue(should_attach_reference_images(0, settings))
        self.assertFalse(should_attach_reference_images(1, settings))
        # 拼图/少量头像：每批附上参照图以便逐脸对照
        self.assertTrue(
            should_attach_reference_images(1, settings, head_count=11, use_collage=True)
        )
        self.assertTrue(
            should_attach_reference_images(2, settings, head_count=3, use_collage=False)
        )
        settings_off = {"frame_reference_token_saver": False, "frame_reference_attach_mode": ATTACH_MODE_FIRST_BATCH}
        self.assertFalse(should_attach_reference_images(1, settings_off, head_count=11, use_collage=False))

        self.assertTrue(resolve_reference_collage_mode({"frame_reference_token_saver": True}, head_count=11))
        self.assertFalse(
            resolve_reference_collage_mode(
                {"frame_reference_token_saver": True, "frame_reference_force_individual_heads": True},
                head_count=11,
            )
        )
        self.assertFalse(resolve_reference_collage_mode({}, head_count=1))
        self.assertTrue(resolve_reference_collage_mode({}, head_count=4))
        self.assertFalse(
            resolve_reference_collage_mode({"frame_reference_use_collage": False}, head_count=4)
        )

        from app.services.drama_character_registry import resolve_active_relationship_diagram_path

        self.assertEqual("", resolve_active_relationship_diagram_path("罚罪2", enabled=False))

    def test_resolve_character_references_filters_by_selection(self):
        from app.services.drama_character_registry import resolve_character_references

        with unittest.mock.patch(
            "app.services.drama_character_registry.list_character_head_slots",
            return_value=[
                {"name": "秦枫", "image_path": __file__, "uploaded": True},
                {"name": "刘天也", "image_path": __file__, "uploaded": True},
            ],
        ):
            all_refs = resolve_character_references("罚罪2")
            self.assertEqual(2, len(all_refs))
            selected = resolve_character_references("罚罪2", selected_names={"秦枫"})
            self.assertEqual(1, len(selected))
            self.assertEqual("秦枫", selected[0]["name"])
            empty = resolve_character_references("罚罪2", selected_names=set())
            self.assertEqual([], empty)

    def test_build_batch_vision_reference_prompt_includes_relationship(self):
        from app.services.drama_character_registry import build_batch_vision_reference_prompt_section

        section = build_batch_vision_reference_prompt_section(
            relationship_diagram_path=__file__,
            character_references=[{"name": "秦枫", "path": __file__}],
            video_frame_count=3,
            drama_label="罚罪2",
        )
        self.assertIn("关系图", section)
        self.assertIn("图 #1", section)
        self.assertIn("秦枫", section)

