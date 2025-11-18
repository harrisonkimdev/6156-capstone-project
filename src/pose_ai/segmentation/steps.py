"""Step segmentation utilities driven by contact changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, MutableMapping, Sequence


@dataclass(slots=True)
class StepSegment:
    step_id: int
    start_index: int
    end_index: int
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


def _support_count(row: MutableMapping[str, object]) -> int:
    return sum(int(bool(row.get(f"{limb}_contact_on"))) for limb in ("left_hand", "right_hand", "left_foot", "right_foot"))


def _contact_signature(row: MutableMapping[str, object]) -> tuple:
    values = []
    for limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
        values.append(row.get(f"{limb}_contact_type"))
        values.append(row.get(f"{limb}_contact_hold"))
    return tuple(values)


def segment_steps_by_contacts(
    feature_rows: Sequence[MutableMapping[str, object]],
    *,
    fps: float = 25.0,
    min_duration: float = 0.2,
    max_duration: float = 4.0,
) -> List[StepSegment]:
    """Create step segments whenever exactly one limb contact changes."""
    if not feature_rows:
        return []

    min_frames = max(1, int(min_duration * fps))
    max_frames = max(min_frames, int(max_duration * fps))

    segments: List[StepSegment] = []
    current_start = 0
    current_signature = _contact_signature(feature_rows[0])
    step_id = 0

    for idx in range(1, len(feature_rows)):
        row = feature_rows[idx]
        prev_row = feature_rows[idx - 1]
        sig = _contact_signature(row)
        prev_sig = _contact_signature(prev_row)
        if sig == prev_sig:
            continue

        changed_limbs = 0
        for limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
            key_type = f"{limb}_contact_type"
            key_hold = f"{limb}_contact_hold"
            if row.get(key_type) != prev_row.get(key_type) or row.get(key_hold) != prev_row.get(key_hold):
                changed_limbs += 1
        support = _support_count(row)
        if changed_limbs == 1 and support >= 2:
            frame_count = idx - current_start
            if frame_count >= min_frames or not segments:
                start_time = float(feature_rows[current_start].get("timestamp", current_start / fps))
                end_time = float(row.get("timestamp", idx / fps))
                segment = StepSegment(
                    step_id=step_id,
                    start_index=current_start,
                    end_index=idx,
                    start_time=start_time,
                    end_time=end_time,
                )
                if segment.duration <= max_duration:
                    segments.append(segment)
                    step_id += 1
                    current_start = idx
                    current_signature = sig
            else:
                current_signature = sig

    # Tail segment
    if not segments or segments[-1].end_index != len(feature_rows) - 1:
        start_time = float(feature_rows[current_start].get("timestamp", current_start / fps))
        end_time = float(feature_rows[-1].get("timestamp", (len(feature_rows) - 1) / fps))
        segments.append(
            StepSegment(
                step_id=step_id,
                start_index=current_start,
                end_index=len(feature_rows) - 1,
                start_time=start_time,
                end_time=end_time,
            )
        )

    for segment in segments:
        for idx in range(segment.start_index, segment.end_index + 1):
            feature_rows[idx]["step_id"] = segment.step_id

    return segments


__all__ = ["segment_steps_by_contacts", "StepSegment"]
