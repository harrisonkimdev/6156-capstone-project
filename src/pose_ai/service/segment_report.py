"""Utilities to combine pose features, segmentation, and export reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from pose_ai.features import summarize_features
from pose_ai.features.segment_metrics import SegmentMetrics, aggregate_segment_metrics
from pose_ai.service.feature_service import _load_pose_results, _load_holds
from pose_ai.service.pose_service import estimate_poses_from_manifest
from pose_ai.segmentation import features_to_frame_metrics, segment_by_activity


def generate_segment_report(
    manifest_path: Path | str,
    *,
    holds_path: Optional[Path] = None,
) -> List[SegmentMetrics]:
    manifest_path = Path(manifest_path)
    frame_dir = manifest_path.parent
    pose_results_path = frame_dir / "pose_results.json"
    if not pose_results_path.exists():
        estimate_poses_from_manifest(manifest_path)

    frames = _load_pose_results(pose_results_path)
    holds = _load_holds(holds_path) if holds_path else None
    frame_features = summarize_features(frames, holds=holds)
    metrics_input = features_to_frame_metrics(frame_features)
    segments = segment_by_activity(
        metrics_input,
        motion_threshold=None,
        min_motion_threshold=0.01,
        dynamic_percentile=70.0,
    )
    metrics = aggregate_segment_metrics(frame_features, segments)

    output_path = frame_dir / "segment_metrics.json"
    output_path.write_text(
        json.dumps([metric.as_dict() for metric in metrics], indent=2),
        encoding="utf-8",
    )
    return metrics
