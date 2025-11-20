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
    label: str = "unknown"

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


def _get_avg_speed(feature_rows: Sequence[MutableMapping[str, object]], start: int, end: int) -> float:
    """Compute average COM speed across a segment."""
    speeds = []
    for idx in range(start, end + 1):
        row = feature_rows[idx]
        speed = row.get("com_speed")
        if speed is not None:
            try:
                speeds.append(float(speed))
            except (ValueError, TypeError):
                pass
    return sum(speeds) / len(speeds) if speeds else 0.0


def _classify_step_label(
    feature_rows: Sequence[MutableMapping[str, object]],
    start_idx: int,
    end_idx: int,
    prev_segment: StepSegment | None,
) -> str:
    """Classify step type based on contact changes and velocity patterns."""
    if start_idx >= len(feature_rows) or end_idx >= len(feature_rows):
        return "unknown"
    
    start_row = feature_rows[start_idx]
    end_row = feature_rows[end_idx]
    
    # Count support at start and end
    start_support = _support_count(start_row)
    end_support = _support_count(end_row)
    
    # Check which limbs changed
    hand_changed = False
    foot_changed = False
    
    if prev_segment is not None and prev_segment.end_index < len(feature_rows):
        prev_row = feature_rows[prev_segment.end_index]
        for limb in ("left_hand", "right_hand"):
            if (start_row.get(f"{limb}_contact_hold") != prev_row.get(f"{limb}_contact_hold") or
                start_row.get(f"{limb}_contact_type") != prev_row.get(f"{limb}_contact_type")):
                hand_changed = True
        for limb in ("left_foot", "right_foot"):
            if (start_row.get(f"{limb}_contact_hold") != prev_row.get(f"{limb}_contact_hold") or
                start_row.get(f"{limb}_contact_type") != prev_row.get(f"{limb}_contact_type")):
                foot_changed = True
    
    # Get average speed
    avg_speed = _get_avg_speed(feature_rows, start_idx, end_idx)
    
    # Get max y position (higher y = higher on wall)
    start_com_y = start_row.get("com_y")
    end_com_y = end_row.get("com_y")
    
    # Classification rules
    # Rest: very low velocity, stable contacts
    if avg_speed < 0.01 and start_support >= 3:
        return "rest"
    
    # Finish: at top (low com_y in normalized coords), sustained stability
    if end_com_y is not None and isinstance(end_com_y, (int, float)):
        if float(end_com_y) < 0.3 and avg_speed < 0.02 and end_support >= 3:
            return "finish"
    
    # FootAdjust: only foot changed, hands stable
    if foot_changed and not hand_changed:
        return "foot_adjust"
    
    # Reach: support increased or high velocity with hand movement
    if end_support > start_support or (avg_speed > 0.05 and hand_changed):
        return "reach"
    
    # Stabilize: contacts maintained, velocity decreasing
    if start_support == end_support and avg_speed < 0.03:
        return "stabilize"
    
    # DynamicMove: multiple limbs changed or high velocity
    if (hand_changed and foot_changed) or avg_speed > 0.08:
        return "dynamic_move"
    
    # Default
    return "movement"


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
                prev_segment = segments[-1] if segments else None
                label = _classify_step_label(feature_rows, current_start, idx, prev_segment)
                segment = StepSegment(
                    step_id=step_id,
                    start_index=current_start,
                    end_index=idx,
                    start_time=start_time,
                    end_time=end_time,
                    label=label,
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
        prev_segment = segments[-1] if segments else None
        label = _classify_step_label(feature_rows, current_start, len(feature_rows) - 1, prev_segment)
        segments.append(
            StepSegment(
                step_id=step_id,
                start_index=current_start,
                end_index=len(feature_rows) - 1,
                start_time=start_time,
                end_time=end_time,
                label=label,
            )
        )

    for segment in segments:
        for idx in range(segment.start_index, segment.end_index + 1):
            feature_rows[idx]["step_id"] = segment.step_id

    return segments


__all__ = ["segment_steps_by_contacts", "StepSegment"]
