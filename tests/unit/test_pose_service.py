from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pose_ai.pose.estimator import PoseFrame, PoseLandmark
from pose_ai.service.pose_service import estimate_poses_from_manifest


class DummyEstimator:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def process_paths(self, image_paths, *, timestamps=None):
        self.calls += 1
        frames = []
        for path, ts in zip(image_paths, timestamps or []):
            frames.append(
                PoseFrame(
                    image_path=path,
                    timestamp_seconds=ts,
                    landmarks=[
                        PoseLandmark(name="hip", x=0.1, y=0.2, z=0.0, visibility=0.9),
                    ],
                    detection_score=0.9,
                )
            )
        return frames

    def close(self):
        self.closed = True


def _write_manifest(tmp_path: Path) -> Path:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for idx in range(2):
        image_path = frame_dir / f"frame_{idx:04d}.jpg"
        image_path.write_bytes(b"fake")
        images.append(image_path)

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
    return manifest_path


def test_estimate_poses_from_manifest_writes_json(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    estimator = DummyEstimator()
    frames = estimate_poses_from_manifest(manifest_path, estimator=estimator, save_json=True)
    assert estimator.calls == 1
    assert len(frames) == 2

    output_file = Path(manifest_path).parent / "pose_results.json"
    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert data["frame_count"] == 2
    assert data["frames"][0]["landmarks"][0]["name"] == "hip"
