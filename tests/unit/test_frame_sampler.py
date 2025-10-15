from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pose_ai.data import extract_frames_every_n_seconds


def _create_test_video(video_path: Path, frame_count: int = 20, fps: float = 10.0) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (64, 64))
    for idx in range(frame_count):
        frame = np.full((64, 64, 3), idx * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_extract_frames_every_n_seconds(tmp_path: Path) -> None:
    video_path = tmp_path / "videos" / "sample.mp4"
    _create_test_video(video_path)

    output_root = tmp_path / "output_frames"
    result = extract_frames_every_n_seconds(
        video_path,
        interval_seconds=0.5,
        output_root=output_root,
    )

    assert result.saved_frames > 0
    assert len(result.frame_paths) == result.saved_frames
    assert all(path.exists() for path in result.frame_paths)
    assert result.manifest_path is not None and result.manifest_path.exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["video"] == str(video_path.resolve())
    assert manifest["saved_frames"] == result.saved_frames
    assert len(manifest["frames"]) == result.saved_frames
