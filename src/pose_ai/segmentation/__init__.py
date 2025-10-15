"""Segmentation routines for climbing video analysis."""

from .rule_based import (
    FrameMetrics,
    Segment,
    segment_by_activity,
)

__all__ = ["FrameMetrics", "Segment", "segment_by_activity"]
