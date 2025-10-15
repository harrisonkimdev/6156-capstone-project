"""Rule-based heuristics for splitting climbing sessions into segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(slots=True)
class FrameMetrics:
    """Lightweight features describing a single frame in time order."""

    timestamp: float
    motion_score: float
    hold_changed: bool = False
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
    motion_threshold: float = 0.18,
    min_segment_duration: float = 1.0,
    hold_change_bonus: float = 0.12,
    sustain_frames: int = 2,
) -> List[Segment]:
    """Split a sequence of ``FrameMetrics`` into rest/movement segments."""
    if not frames:
        return []

    segments: list[Segment] = []
    current_label = None
    current_start_idx = 0
    active_streak = 0
    rest_streak = 0

    def classify_frame(metrics: FrameMetrics) -> str:
        score = metrics.motion_score + (hold_change_bonus if metrics.hold_changed else 0.0)
        return "movement" if score >= motion_threshold else "rest"

    for idx, metrics in enumerate(frames):
        label = classify_frame(metrics)
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
        label=current_label or classify_frame(frames[-1]),
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


__all__ = ["FrameMetrics", "Segment", "segment_by_activity"]
