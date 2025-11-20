from __future__ import annotations

import json
from pathlib import Path

from pose_ai.segmentation import FrameMetrics, Segment
from pose_ai.service.segment_report import generate_segment_report


def _setup(tmp_path: Path) -> Path:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(3):
        (frame_dir / f"frame_{idx:04d}.jpg").write_text("placeholder")
    manifest = {
        "video": str((tmp_path / "videos" / "sample.mp4").resolve()),
        "fps": 10.0,
        "interval_seconds": 0.5,
        "total_frames": 30,
        "saved_frames": 3,
        "frames": [
            {"frame_index": idx * 5, "timestamp_seconds": idx * 0.5, "relative_path": f"frame_{idx:04d}.jpg"}
            for idx in range(3)
        ],
    }
    manifest_path = frame_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (frame_dir / "pose_results.json").write_text(json.dumps({"frames": []}), encoding="utf-8")
    return manifest_path


def test_generate_segment_report(monkeypatch, tmp_path: Path) -> None:
    manifest_path = _setup(tmp_path)

    frame_features = [
        {"timestamp": 0.0, "com_x": 0.4, "com_y": 0.3, "detection_score": 0.9, "left_elbow_angle": 150.0, "left_hand_target": "hold1"},
        {"timestamp": 0.5, "com_x": 0.45, "com_y": 0.31, "detection_score": 0.92, "left_elbow_angle": 155.0, "left_hand_target": "hold2"},
    ]

    def fake_load_pose_results(path):
        return []

    def fake_summarize(frames, holds=None):
        return frame_features

    monkeypatch.setattr("pose_ai.service.segment_report._load_pose_results", fake_load_pose_results)
    monkeypatch.setattr("pose_ai.service.segment_report.summarize_features", fake_summarize)
    monkeypatch.setattr(
        "pose_ai.service.segment_report.features_to_frame_metrics",
        lambda rows: [
            FrameMetrics(timestamp=row["timestamp"], motion_score=0.1 * idx)
            for idx, row in enumerate(rows)
        ],
    )
    monkeypatch.setattr(
        "pose_ai.service.segment_report.segment_by_activity",
        lambda metrics, **kwargs: [
            Segment(start_time=0.0, end_time=0.5, label="movement", frame_indices=(0, 1))
        ],
    )

    metrics = generate_segment_report(manifest_path)
    assert metrics
    output_path = manifest_path.parent / "segment_metrics.json"
    assert output_path.exists()
