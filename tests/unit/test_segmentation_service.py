from __future__ import annotations

import json
from pathlib import Path

from pose_ai.service import segment_video_from_manifest


def _create_manifest_with_scores(tmp_path: Path) -> Path:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    # create fake images
    for idx, value in enumerate([10, 80, 20, 5]):
        (frame_dir / f"video_frame_{idx:04d}.jpg").write_text(str(value))
    manifest = {
        "video": str((tmp_path / "videos" / "sample.mp4").resolve()),
        "fps": 10.0,
        "interval_seconds": 0.2,
        "total_frames": 40,
        "saved_frames": 4,
        "frames": [
            {"frame_index": idx * 5, "timestamp_seconds": idx * 0.5, "relative_path": f"video_frame_{idx:04d}.jpg"}
            for idx in range(4)
        ],
    }
    manifest_path = frame_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_segment_video_from_manifest(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _create_manifest_with_scores(tmp_path)

    # Monkeypatch motion scores to avoid OpenCV dependency in test
    from pose_ai.data import metrics as metrics_module

    def fake_motion_scores(paths):
        return [0.05, 0.25, 0.2, 0.05]

    monkeypatch.setattr(metrics_module, "compute_motion_scores", fake_motion_scores)

    segments = segment_video_from_manifest(manifest_path)
    assert segments
    assert any(seg.label == "movement" for seg in segments)
