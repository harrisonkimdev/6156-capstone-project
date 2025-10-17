"""Segmentation routines for climbing video analysis."""

from .rule_based import (
    FrameMetrics,
    Segment,
    features_to_frame_metrics,
    segment_by_activity,
)

__all__ = ["FrameMetrics", "Segment", "segment_by_activity", "features_to_frame_metrics"]
