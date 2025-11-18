"""Unit tests for YOLO manifest filtering helpers."""

from __future__ import annotations

import json
from pathlib import Path

from pose_ai.service.yolo_service import Detection, annotate_manifest_with_yolo


class DummySelector:
    """Simple selector that returns canned detections per filename."""

    def __init__(self, detections: dict[str, list[Detection]]) -> None:
        self._detections = detections
        self.model_name = "dummy-yolo"

    def detect_many(self, image_paths: list[Path]) -> list[list[Detection]]:
        return [self._detections.get(path.name, []) for path in image_paths]


def _write_manifest(tmp_path: Path, frame_count: int = 3) -> Path:
    frames: list[dict[str, object]] = []
    for idx in range(frame_count):
        frame_name = f"frame_{idx:02d}.jpg"
        (tmp_path / frame_name).write_text("stub", encoding="utf-8")
        frames.append(
            {
                "frame_index": idx,
                "saved_index": idx,
                "timestamp_seconds": float(idx),
                "relative_path": frame_name,
            }
        )
    payload = {
        "video": "dummy.mp4",
        "fps": 30.0,
        "interval_seconds": 1.0,
        "total_frames": frame_count,
        "saved_frames": frame_count,
        "frames": frames,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def test_annotate_manifest_keeps_required_frames(tmp_path):
    manifest_path = _write_manifest(tmp_path, frame_count=3)
    selector = DummySelector(
        {
            "frame_00.jpg": [Detection(label="climber", confidence=0.9, bbox=(0, 0, 1, 1))],
            "frame_01.jpg": [],
            "frame_02.jpg": [Detection(label="hold", confidence=0.4, bbox=(0, 0, 1, 1))],
        }
    )
    result = annotate_manifest_with_yolo(
        manifest_path,
        selector=selector,
        required_labels=["climber"],
        target_labels=["climber", "hold"],
        min_confidence=0.5,
    )
    assert result.selected_frames == 1
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["saved_frames"] == 1
    assert payload["frames"][0]["relative_path"] == "frame_00.jpg"
    yolo_meta = payload["frames"][0]["metadata"]["yolo"]
    assert yolo_meta["selected"] is True
    assert yolo_meta["score"] == 0.9


def test_annotate_manifest_can_be_disabled(tmp_path):
    manifest_path = _write_manifest(tmp_path, frame_count=2)
    result = annotate_manifest_with_yolo(manifest_path, enabled=False)
    assert result.skipped_reason == "disabled"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["saved_frames"] == 2
    assert len(payload["frames"]) == 2
