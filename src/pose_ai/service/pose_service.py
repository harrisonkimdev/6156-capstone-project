"""High-level helpers for running pose estimation on extracted frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from pose_ai.data.metrics import ManifestData, load_manifest
from pose_ai.pose.estimator import PoseEstimator, PoseFrame


def _serialize_pose_frames(frames: Sequence[PoseFrame]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for frame in frames:
        payload.append(
            {
                "image_path": str(frame.image_path),
                "timestamp_seconds": frame.timestamp_seconds,
                "detection_score": frame.detection_score,
                "landmarks": [
                    {
                        "name": landmark.name,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility,
                    }
                    for landmark in frame.landmarks
                ],
            }
        )
    return payload


def estimate_poses_from_manifest(
    manifest_path: Path | str,
    *,
    estimator: Optional[PoseEstimator] = None,
    save_json: bool = True,
    output_path: Path | None = None,
    estimator_kwargs: Optional[dict] = None,
) -> list[PoseFrame]:
    manifest = load_manifest(manifest_path)
    frame_dir = Path(manifest_path).parent
    image_paths = [frame_dir / entry.relative_path for entry in manifest.frame_entries]
    timestamps = [entry.timestamp_seconds for entry in manifest.frame_entries]

    close_after = estimator is None
    estimator = estimator or PoseEstimator(**(estimator_kwargs or {}))
    try:
        frames = estimator.process_paths(image_paths, timestamps=timestamps)
    finally:
        if close_after:
            estimator.close()

    if save_json:
        output_file = output_path or frame_dir / "pose_results.json"
        payload = {
            "video": str(manifest.video),
            "frame_count": len(frames),
            "frames": _serialize_pose_frames(frames),
        }
        Path(output_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return list(frames)


def estimate_poses_for_directory(
    frames_root: Path | str,
    *,
    estimator: Optional[PoseEstimator] = None,
    save_json: bool = True,
    estimator_kwargs: Optional[dict] = None,
) -> dict[str, list[PoseFrame]]:
    root = Path(frames_root)
    results: dict[str, list[PoseFrame]] = {}
    close_after = estimator is None
    estimator = estimator or PoseEstimator(**(estimator_kwargs or {}))
    try:
        for manifest_path in sorted(root.rglob("manifest.json")):
            frames = estimate_poses_from_manifest(
                manifest_path,
                estimator=estimator,
                save_json=save_json,
                estimator_kwargs=None,
            )
            results[str(manifest_path)] = frames
    finally:
        if close_after:
            estimator.close()
    return results


__all__ = ["estimate_poses_for_directory", "estimate_poses_from_manifest"]
