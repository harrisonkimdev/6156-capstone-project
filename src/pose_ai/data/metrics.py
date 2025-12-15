"""Helpers for deriving frame-level metrics from stored manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Any

import numpy as np

from pose_ai.segmentation import FrameMetrics


@dataclass(slots=True)
class ManifestFrame:
    """Represents a single entry inside a frame manifest."""

    frame_index: int
    timestamp_seconds: float
    relative_path: str
    saved_index: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class ManifestData:
    """Loaded manifest metadata for a video."""

    video: Path
    fps: float
    interval_seconds: float
    total_frames: int
    saved_frames: int
    frame_entries: List[ManifestFrame]


def load_manifest(path: Path | str) -> ManifestData:
    """Read a manifest.json produced by ``frame_sampler``."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [
        ManifestFrame(
            frame_index=int(item["frame_index"]),
            timestamp_seconds=float(item["timestamp_seconds"]),
            relative_path=str(item["relative_path"]),
            saved_index=item.get("saved_index"),
            metadata=item.get("metadata"),
        )
        for item in payload.get("frames", [])
    ]
    # Calculate interval_seconds if not present (for backward compatibility)
    if "interval_seconds" in payload:
        interval_seconds = float(payload["interval_seconds"])
    else:
        # Estimate from frames if available
        if entries and len(entries) > 1:
            fps = float(payload.get("fps", 30.0))
            if fps > 0:
                interval_seconds = (entries[-1].timestamp_seconds - entries[0].timestamp_seconds) / max(1, len(entries) - 1)
            else:
                interval_seconds = 1.0
        else:
            interval_seconds = 1.0
    
    return ManifestData(
        video=Path(payload["video"]),
        fps=float(payload["fps"]),
        interval_seconds=interval_seconds,
        total_frames=int(payload["total_frames"]),
        saved_frames=int(payload["saved_frames"]),
        frame_entries=entries,
    )


def compute_motion_scores(image_paths: Sequence[Path]) -> List[float]:
    """Simple placeholder motion metric using grayscale frame differences."""
    import cv2

    scores: list[float] = []
    prev_gray = None

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            scores.append(0.0)
            continue
        if prev_gray is None:
            scores.append(0.0)
            prev_gray = image
            continue

        diff = cv2.absdiff(prev_gray, image)
        normalized_score = float(np.mean(diff) / 255.0)
        scores.append(normalized_score)
        prev_gray = image

    return scores


def manifest_to_frame_metrics(
    manifest: ManifestData,
    frame_directory: Path,
    *,
    hold_change_flags: Iterable[bool] | None = None,
    motion_scores: Sequence[float] | None = None,
) -> List[FrameMetrics]:
    """Convert manifest entries into ``FrameMetrics`` for segmentation."""
    image_paths = [frame_directory / entry.relative_path for entry in manifest.frame_entries]

    derived_motion = (
        list(motion_scores)
        if motion_scores is not None
        else compute_motion_scores(image_paths)
    )

    hold_flags = list(hold_change_flags) if hold_change_flags is not None else [False] * len(image_paths)

    frame_metrics: list[FrameMetrics] = []
    for entry, motion, hold_changed in zip(manifest.frame_entries, derived_motion, hold_flags):
        frame_metrics.append(
            FrameMetrics(
                timestamp=entry.timestamp_seconds,
                motion_score=float(motion),
                hold_changed=bool(hold_changed),
            )
        )
    return frame_metrics


__all__ = ["ManifestData", "ManifestFrame", "compute_motion_scores", "load_manifest", "manifest_to_frame_metrics"]
