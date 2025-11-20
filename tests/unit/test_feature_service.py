from __future__ import annotations

import json
from pathlib import Path

from pose_ai.service.feature_service import export_features_for_manifest


def _write_manifest_and_pose(tmp_path: Path) -> Path:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(2):
        (frame_dir / f"frame_{idx:04d}.jpg").write_text("placeholder")

    manifest = {
        "video": str((tmp_path / "videos" / "sample.mp4").resolve()),
        "fps": 10.0,
        "interval_seconds": 0.5,
        "total_frames": 20,
        "saved_frames": 2,
        "frames": [
            {"frame_index": idx * 5, "timestamp_seconds": idx * 0.5, "relative_path": f"frame_{idx:04d}.jpg"}
            for idx in range(2)
        ],
    }
    manifest_path = frame_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    pose_results = {
        "video": manifest["video"],
        "frame_count": 2,
        "frames": [
            {
                "image_path": str(frame_dir / "frame_0000.jpg"),
                "timestamp_seconds": 0.0,
                "detection_score": 0.9,
                "landmarks": [
                    {"name": "left_shoulder", "x": 0.4, "y": 0.3, "z": 0.0, "visibility": 0.9},
                    {"name": "right_shoulder", "x": 0.6, "y": 0.3, "z": 0.0, "visibility": 0.9},
                    {"name": "left_elbow", "x": 0.35, "y": 0.45, "z": 0.0, "visibility": 0.9},
                    {"name": "right_elbow", "x": 0.65, "y": 0.45, "z": 0.0, "visibility": 0.9},
                    {"name": "left_wrist", "x": 0.3, "y": 0.6, "z": 0.0, "visibility": 0.9},
                    {"name": "right_wrist", "x": 0.7, "y": 0.6, "z": 0.0, "visibility": 0.9},
                    {"name": "left_hip", "x": 0.45, "y": 0.6, "z": 0.0, "visibility": 0.9},
                    {"name": "right_hip", "x": 0.55, "y": 0.6, "z": 0.0, "visibility": 0.9},
                    {"name": "left_knee", "x": 0.47, "y": 0.8, "z": 0.0, "visibility": 0.9},
                    {"name": "right_knee", "x": 0.53, "y": 0.8, "z": 0.0, "visibility": 0.9},
                    {"name": "left_ankle", "x": 0.46, "y": 0.95, "z": 0.0, "visibility": 0.9},
                    {"name": "right_ankle", "x": 0.54, "y": 0.95, "z": 0.0, "visibility": 0.9},
                    {"name": "left_foot_index", "x": 0.46, "y": 0.98, "z": 0.0, "visibility": 0.9},
                    {"name": "right_foot_index", "x": 0.54, "y": 0.98, "z": 0.0, "visibility": 0.9},
                ],
            }
        ],
    }
    (frame_dir / "pose_results.json").write_text(json.dumps(pose_results), encoding="utf-8")
    return manifest_path


def test_export_features_for_manifest(tmp_path: Path) -> None:
    manifest_path = _write_manifest_and_pose(tmp_path)
    output_path = export_features_for_manifest(manifest_path)
    data = json.loads(Path(output_path).read_text(encoding="utf-8"))
    assert len(data) == 1
    assert "left_elbow_angle" in data[0]
