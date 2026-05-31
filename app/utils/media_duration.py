#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Read media duration via ffprobe."""

from __future__ import annotations

import subprocess
from typing import Optional

from loguru import logger


def get_video_duration_seconds(video_path: Optional[str]) -> Optional[float]:
    if not video_path:
        return None
    try:
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stderr=subprocess.STDOUT,
            text=True,
        )
        duration = float(output.strip())
        return duration if duration > 0 else None
    except Exception as exc:
        logger.warning(f"无法读取视频时长: {video_path} ({exc})")
        return None
