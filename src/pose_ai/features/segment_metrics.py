"""Segment-level aggregation of pose features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import math

from pose_ai.segmentation import Segment


@dataclass(slots=True)
class SegmentMetrics:
    segment: Segment
    frame_count: int
    com_displacement: float
    com_path_length: float
    avg_detection_score: float
    joint_ranges: Dict[str, float]
    joint_velocity: Dict[str, float]
    contact_changes: Dict[str, int]
    duration: float

    def as_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "label": self.segment.label,
            "start_time": self.segment.start_time,
            "end_time": self.segment.end_time,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "com_displacement": self.com_displacement,
            "com_path_length": self.com_path_length,
            "avg_detection_score": self.avg_detection_score,
        }
        payload.update({f"joint_range.{k}": v for k, v in self.joint_ranges.items()})
        payload.update({f"joint_velocity.{k}": v for k, v in self.joint_velocity.items()})
        payload.update({f"contacts.{k}": v for k, v in self.contact_changes.items()})
        return payload


def _slice_features(frame_features: Sequence[Dict[str, object]], segment: Segment) -> List[Dict[str, object]]:
    start, end = segment.frame_indices
    return list(frame_features[start : end + 1])


def _clean_numeric(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _range(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(max(values) - min(values))


def _velocity(samples: List[tuple[float, float]]) -> float:
    if len(samples) < 2:
        return float("nan")
    total = 0.0
    count = 0
    prev_t, prev_v = samples[0]
    for t, v in samples[1:]:
        dt = t - prev_t
        if dt > 0:
            total += abs(v - prev_v) / dt
            count += 1
        prev_t, prev_v = t, v
    if count == 0:
        return float("nan")
    return float(total / count)


def _contact_changes(sequence: List[str]) -> int:
    if not sequence:
        return 0
    changes = 0
    prev = sequence[0]
    for value in sequence[1:]:
        if value != prev:
            changes += 1
            prev = value
    return changes


def _com_metrics(slice_features: List[Dict[str, object]]) -> tuple[float, float]:
    points: List[tuple[float, float]] = []
    for row in slice_features:
        x = _clean_numeric(row.get("com_x"))
        y = _clean_numeric(row.get("com_y"))
        if x is not None and y is not None:
            points.append((x, y))
    if len(points) < 2:
        return (float("nan"), float("nan"))
    start_x, start_y = points[0]
    end_x, end_y = points[-1]
    displacement = math.hypot(end_x - start_x, end_y - start_y)
    path = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        path += math.hypot(x2 - x1, y2 - y1)
    return float(displacement), float(path)


def aggregate_segment_metrics(
    frame_features: Sequence[Dict[str, object]],
    segments: Sequence[Segment],
) -> List[SegmentMetrics]:
    metrics: List[SegmentMetrics] = []
    joint_keys = {key for row in frame_features for key in row if key.endswith("_angle")}
    contact_keys = {key for row in frame_features for key in row if key.endswith("_target")}

    for segment in segments:
        slice_features = _slice_features(frame_features, segment)
        if not slice_features:
            continue

        timestamps: List[float] = []
        for index, row in enumerate(slice_features):
            t_value = _clean_numeric(row.get("timestamp"))
            timestamps.append(t_value if t_value is not None else float(index))

        com_displacement, com_path_length = _com_metrics(slice_features)

        joint_ranges: Dict[str, float] = {}
        joint_velocity: Dict[str, float] = {}
        for key in joint_keys:
            joint_values: List[float] = []
            velocity_samples: List[tuple[float, float]] = []
            for idx, row in enumerate(slice_features):
                value = _clean_numeric(row.get(key))
                if value is not None:
                    joint_values.append(value)
                    velocity_samples.append((timestamps[idx], value))
            joint_ranges[key] = _range(joint_values)
            joint_velocity[key] = _velocity(velocity_samples)

        contact_changes: Dict[str, int] = {}
        for key in contact_keys:
            sequence = [str(row.get(key) or "") for row in slice_features]
            contact_changes[key] = _contact_changes(sequence)

        detection_sum = 0.0
        for row in slice_features:
            detection_sum += _clean_numeric(row.get("detection_score")) or 0.0
        avg_detection = detection_sum / len(slice_features)

        metrics.append(
            SegmentMetrics(
                segment=segment,
                frame_count=len(slice_features),
                com_displacement=com_displacement,
                com_path_length=com_path_length,
                avg_detection_score=float(avg_detection),
                joint_ranges=joint_ranges,
                joint_velocity=joint_velocity,
                contact_changes=contact_changes,
                duration=segment.duration,
            )
        )

    return metrics
