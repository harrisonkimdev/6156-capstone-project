"""Rule-based heuristics for splitting climbing sessions into segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(slots=True)
class FrameMetrics:
    """Lightweight features describing a single frame in time order."""

    timestamp: float
    motion_score: float
    hold_changed: bool = False
    detection_score: float | None = None
    visibility: float | None = None


@dataclass(slots=True)
class Segment:
    """Represents a contiguous interval of similar activity."""

    start_time: float
    end_time: float
    label: str
    frame_indices: tuple[int, int]

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


def segment_by_activity(
    frames: Sequence[FrameMetrics],
    *,
    motion_threshold: float | None = None,
    min_motion_threshold: float = 0.05,
    dynamic_percentile: float = 75.0,
    min_segment_duration: float = 0.5,
    hold_change_bonus: float = 0.12,
    sustain_frames: int = 1,
) -> List[Segment]:
    """Split a sequence of ``FrameMetrics`` into rest/movement segments."""
    if not frames:
        return []

    score_series = np.array(
        [
            max(0.0, metrics.motion_score) + (hold_change_bonus if metrics.hold_changed else 0.0)
            for metrics in frames
        ],
        dtype=float,
    )

    if motion_threshold is None:
        if score_series.size:
            dynamic_threshold = float(np.percentile(score_series, dynamic_percentile))
            if dynamic_threshold <= min_motion_threshold:
                dynamic_threshold = float(np.max(score_series, initial=0.0))
            motion_threshold_value = max(min_motion_threshold, dynamic_threshold)
        else:
            motion_threshold_value = min_motion_threshold
    else:
        motion_threshold_value = motion_threshold

    segments: list[Segment] = []
    current_label = None
    current_start_idx = 0
    active_streak = 0
    rest_streak = 0

    def classify_frame(idx: int) -> str:
        return "movement" if score_series[idx] >= motion_threshold_value else "rest"

    for idx, metrics in enumerate(frames):
        label = classify_frame(idx)
        timestamp = metrics.timestamp

        if current_label is None:
            current_label = label
            current_start_idx = idx
            continue

        if label == "movement":
            active_streak += 1
            rest_streak = 0
        else:
            rest_streak += 1
            active_streak = 0

        should_flip = label != current_label and (
            (label == "movement" and active_streak >= sustain_frames)
            or (label == "rest" and rest_streak >= sustain_frames)
        )

        if should_flip:
            start_time = frames[current_start_idx].timestamp
            end_time = frames[idx - sustain_frames + 1].timestamp
            segment = Segment(
                start_time=start_time,
                end_time=end_time,
                label=current_label or label,
                frame_indices=(current_start_idx, idx - sustain_frames + 1),
            )
            if segment.duration >= min_segment_duration:
                segments.append(segment)
            current_label = label
            current_start_idx = idx - sustain_frames + 1

    last_segment = Segment(
        start_time=frames[current_start_idx].timestamp,
        end_time=frames[-1].timestamp,
        label=current_label or classify_frame(len(frames) - 1),
        frame_indices=(current_start_idx, len(frames) - 1),
    )
    if last_segment.duration >= min_segment_duration or not segments:
        segments.append(last_segment)

    merged: list[Segment] = []
    for segment in segments:
        if segment.duration < min_segment_duration and merged:
            prev = merged[-1]
            merged[-1] = Segment(
                start_time=prev.start_time,
                end_time=segment.end_time,
                label=prev.label,
                frame_indices=(prev.frame_indices[0], segment.frame_indices[1]),
            )
        else:
            merged.append(segment)

    return merged


def features_to_frame_metrics(feature_rows: Sequence[dict[str, object]]) -> List[FrameMetrics]:
    """Convert pose feature rows into FrameMetrics for segmentation."""
    metrics: list[FrameMetrics] = []
    if not feature_rows:
        return metrics

    joint_keys = [key for key in feature_rows[0].keys() if key.endswith("_angle")]
    contact_keys = [key for key in feature_rows[0].keys() if key.endswith("_target")]

    prev_row: dict[str, object] | None = None
    prev_time: float | None = None

    for row in feature_rows:
        timestamp = float(row.get("timestamp", len(metrics)))
        detection_score = float(row.get("detection_score", 0.0) or 0.0)

        motion_components: list[float] = []
        hold_changed = False

        if prev_row is not None:
            cur_com_x = row.get("com_x")
            cur_com_y = row.get("com_y")
            prev_com_x = prev_row.get("com_x")
            prev_com_y = prev_row.get("com_y")
            if cur_com_x is not None and prev_com_x is not None:
                motion_components.append(abs(float(cur_com_x) - float(prev_com_x)))
            if cur_com_y is not None and prev_com_y is not None:
                motion_components.append(abs(float(cur_com_y) - float(prev_com_y)))

            for key in joint_keys:
                cur_val = row.get(key)
                prev_val = prev_row.get(key)
                if cur_val is not None and prev_val is not None:
                    motion_components.append(abs(float(cur_val) - float(prev_val)) / 180.0)

            if contact_keys:
                hold_changed = any(row.get(key) != prev_row.get(key) for key in contact_keys)

        motion_score = float(np.mean(motion_components)) if motion_components else 0.0

        metrics.append(
            FrameMetrics(
                timestamp=timestamp,
                motion_score=motion_score,
                hold_changed=hold_changed,
                detection_score=detection_score,
            )
        )
        prev_row = row
        prev_time = timestamp

    return metrics


__all__ = ["FrameMetrics", "Segment", "segment_by_activity", "features_to_frame_metrics"]
