"""Segmentation routines for climbing video analysis."""

from .rule_based import (
    FrameMetrics,
    Segment,
    features_to_frame_metrics,
    segment_by_activity,
)
from .hsv_segmentation import HsvSegmentationModel
from .yolo_segmentation import (
    HoldColorInfo,
    RouteGroup,
    SegmentationMask,
    SegmentationResult,
    YoloSegmentationModel,
    cluster_holds_by_color,
    extract_hold_colors,
    export_routes_json,
    export_segmentation_masks,
)

__all__ = [
    "FrameMetrics",
    "Segment",
    "segment_by_activity",
    "features_to_frame_metrics",
    "SegmentationMask",
    "SegmentationResult",
    "HoldColorInfo",
    "RouteGroup",
    "YoloSegmentationModel",
    "HsvSegmentationModel",
    "extract_hold_colors",
    "cluster_holds_by_color",
    "export_segmentation_masks",
    "export_routes_json",
]
