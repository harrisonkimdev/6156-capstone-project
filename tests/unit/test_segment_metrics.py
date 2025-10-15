from __future__ import annotations

from pose_ai.features.segment_metrics import SegmentMetrics, aggregate_segment_metrics
from pose_ai.segmentation import Segment


def test_aggregate_segment_metrics_basic():
    frame_features = [
        {"timestamp": 0.0, "com_x": 0.4, "com_y": 0.3, "detection_score": 0.9,
         "left_elbow_angle": 150.0, "left_hand_target": "hold1"},
        {"timestamp": 0.5, "com_x": 0.45, "com_y": 0.32, "detection_score": 0.92,
         "left_elbow_angle": 155.0, "left_hand_target": "hold2"},
        {"timestamp": 1.0, "com_x": 0.5, "com_y": 0.35, "detection_score": 0.95,
         "left_elbow_angle": 160.0, "left_hand_target": "hold2"},
    ]
    segments = [Segment(start_time=0.0, end_time=1.0, label="movement", frame_indices=(0, 2))]
    metrics = aggregate_segment_metrics(frame_features, segments)
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.frame_count == 3
    assert metric.com_displacement > 0
    assert metric.joint_ranges.get("left_elbow_angle") == 10.0
    assert metric.contact_changes.get("left_hand_target") == 1
