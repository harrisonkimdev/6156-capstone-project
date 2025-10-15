"""Orchestration helpers to produce segments from stored frame manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from pose_ai.data.metrics import load_manifest, manifest_to_frame_metrics
from pose_ai.segmentation import FrameMetrics, Segment, segment_by_activity


def segment_video_from_manifest(
    manifest_path: Path | str,
    *,
    hold_change_flags: Iterable[bool] | None = None,
) -> List[Segment]:
    manifest = load_manifest(manifest_path)
    frame_dir = Path(manifest_path).parent
    metrics: list[FrameMetrics] = manifest_to_frame_metrics(
        manifest,
        frame_dir,
        hold_change_flags=hold_change_flags,
    )
    return segment_by_activity(metrics)


def segment_videos_under_directory(
    frames_root: Path | str,
) -> dict[str, List[Segment]]:
    frames_root_path = Path(frames_root)
    results: dict[str, List[Segment]] = {}
    for manifest_path in frames_root_path.rglob("manifest.json"):
        segments = segment_video_from_manifest(manifest_path)
        results[str(manifest_path)] = segments
    return results


__all__ = ["segment_video_from_manifest", "segment_videos_under_directory"]
