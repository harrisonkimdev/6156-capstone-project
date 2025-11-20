from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pose_ai.data.metrics import compute_motion_scores, load_manifest, manifest_to_frame_metrics


def _write_dummy_manifest(tmp_path: Path) -> tuple[Path, Path]:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(3):
        (frame_dir / f"video_frame_{idx:04d}.jpg").write_bytes(b"\xff" * 64)
    manifest_path = frame_dir / "manifest.json"
    manifest_data = {
        "video": str((tmp_path / "videos" / "sample.mp4").resolve()),
        "fps": 10.0,
        "interval_seconds": 0.1,
        "total_frames": 30,
        "saved_frames": 3,
        "frames": [
            {"frame_index": idx * 10, "timestamp_seconds": idx * 0.5, "relative_path": f"video_frame_{idx:04d}.jpg"}
            for idx in range(3)
        ],
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    return manifest_path, frame_dir


def test_load_manifest(tmp_path: Path) -> None:
    manifest_path, _ = _write_dummy_manifest(tmp_path)
    manifest = load_manifest(manifest_path)
    assert manifest.saved_frames == 3
    assert manifest.frame_entries[1].relative_path.endswith("video_frame_0001.jpg")


def test_manifest_to_frame_metrics(tmp_path: Path) -> None:
    manifest_path, frame_dir = _write_dummy_manifest(tmp_path)
    manifest = load_manifest(manifest_path)
    motion_scores = [0.0, 0.2, 0.05]
    metrics = manifest_to_frame_metrics(
        manifest,
        frame_dir,
        motion_scores=motion_scores,
        hold_change_flags=[False, True, False],
    )
    assert len(metrics) == 3
    assert metrics[1].motion_score == 0.2
    assert metrics[1].hold_changed is True


def test_compute_motion_scores(tmp_path: Path) -> None:
    import cv2

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for idx, value in enumerate([10, 50, 90]):
        img = np.full((8, 8, 3), value, dtype=np.uint8)
        path = frame_dir / f"{idx}.jpg"
        cv2.imwrite(str(path), img)
        image_paths.append(path)

    scores = compute_motion_scores(image_paths)
    assert scores[0] == 0.0
    assert scores[1] > scores[0]
