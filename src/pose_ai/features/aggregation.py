"""Feature extraction from PoseFrame sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from pose_ai.pose.estimator import PoseFrame, PoseLandmark

from .config import CENTER_OF_MASS_WEIGHTS, CONTACT_LANDMARKS, DEFAULT_CONTACT_THRESHOLD, JOINT_DEFINITIONS


def _landmarks_by_name(frame: PoseFrame) -> Dict[str, PoseLandmark]:
    return {landmark.name: landmark for landmark in frame.landmarks}


def _vector(a: PoseLandmark, b: PoseLandmark) -> np.ndarray:
    return np.array([b.x - a.x, b.y - a.y, b.z - a.z], dtype=float)


def _angle(vec_a: np.ndarray, vec_b: np.ndarray) -> Optional[float]:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return None
    cos_value = float(np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_value)))


def compute_joint_angles(frame: PoseFrame) -> Dict[str, Optional[float]]:
    lookup = _landmarks_by_name(frame)
    features: Dict[str, Optional[float]] = {}
    for name, (proximal, joint, distal) in JOINT_DEFINITIONS.items():
        p = lookup.get(proximal)
        j = lookup.get(joint)
        d = lookup.get(distal)
        if None in (p, j, d):
            features[name] = None
            continue
        features[name] = _angle(_vector(j, p), _vector(j, d))
    return features


def compute_segment_inclination(frame: PoseFrame, start: str, end: str) -> Optional[float]:
    lookup = _landmarks_by_name(frame)
    s = lookup.get(start)
    e = lookup.get(end)
    if None in (s, e):
        return None
    dx = e.x - s.x
    dy = e.y - s.y
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(dy, dx)))


def compute_center_of_mass(frame: PoseFrame) -> Dict[str, Optional[float]]:
    lookup = _landmarks_by_name(frame)
    xs, ys, weights = [], [], []
    for name, weight in CENTER_OF_MASS_WEIGHTS.items():
        landmark = lookup.get(name)
        if landmark is None:
            continue
        xs.append(landmark.x)
        ys.append(landmark.y)
        weights.append(weight)
    if not weights:
        return {"com_x": None, "com_y": None, "com_n": 0}
    w = np.array(weights, dtype=float)
    com_x = float(np.average(np.array(xs, dtype=float), weights=w))
    com_y = float(np.average(np.array(ys, dtype=float), weights=w))
    return {"com_x": com_x, "com_y": com_y, "com_n": len(weights)}


@dataclass(slots=True)
class HoldDefinition:
    name: str
    coords: Sequence[float]
    normalized: bool = False
    hold_type: str = "auto"
    notes: Optional[str] = None


def _normalize_hold_coords(hold: HoldDefinition, frame: PoseFrame) -> np.ndarray:
    if hold.normalized:
        return np.array(hold.coords, dtype=float)
    # Assume coords are pixels if not normalized; map using image dimensions if available.
    # Without explicit image shape, treat as normalized to avoid crashing.
    return np.array(hold.coords, dtype=float)


def analyze_hold_relationship(
    frame: PoseFrame,
    holds: Dict[str, HoldDefinition],
    *,
    contact_threshold: float = DEFAULT_CONTACT_THRESHOLD,
) -> Dict[str, Optional[float]]:
    lookup = _landmarks_by_name(frame)
    if not holds:
        return {}

    hold_positions = {
        name: _normalize_hold_coords(hold, frame)
        for name, hold in holds.items()
    }

    features: Dict[str, Optional[float]] = {}
    for limb, landmark_name in CONTACT_LANDMARKS.items():
        landmark = lookup.get(landmark_name)
        if landmark is None:
            features[f"{limb}_target"] = None
            features[f"{limb}_distance"] = None
            features[f"{limb}_on_hold"] = 0
            continue
        limb_point = np.array([landmark.x, landmark.y], dtype=float)
        best_name = None
        best_distance = float("inf")
        for hold_name, hold_point in hold_positions.items():
            distance = float(np.linalg.norm(limb_point - hold_point))
            if distance < best_distance:
                best_distance = distance
                best_name = hold_name
        features[f"{limb}_target"] = best_name
        features[f"{limb}_distance"] = best_distance
        features[f"{limb}_on_hold"] = int(best_distance <= contact_threshold)
    return features


def pose_to_feature_row(
    frame: PoseFrame,
    *,
    holds: Optional[Dict[str, HoldDefinition]] = None,
) -> Dict[str, object]:
    feature_row: Dict[str, object] = {
        "image_path": str(frame.image_path),
        "timestamp": frame.timestamp_seconds,
        "detection_score": frame.detection_score,
        "landmark_count": len(frame.landmarks),
    }

    feature_row.update(compute_joint_angles(frame))
    feature_row.update(
        {
            "torso_inclination": compute_segment_inclination(frame, "left_hip", "left_shoulder"),
            "spine_inclination": compute_segment_inclination(frame, "right_hip", "right_shoulder"),
            "hip_line_inclination": compute_segment_inclination(frame, "left_hip", "right_hip"),
            "shoulder_line_inclination": compute_segment_inclination(frame, "left_shoulder", "right_shoulder"),
        }
    )
    feature_row.update(compute_center_of_mass(frame))

    if holds:
        feature_row.update(analyze_hold_relationship(frame, holds))
    return feature_row


def summarize_features(
    frames: Sequence[PoseFrame],
    *,
    holds: Optional[Dict[str, HoldDefinition]] = None,
) -> List[Dict[str, object]]:
    return [
        pose_to_feature_row(frame, holds=holds)
        for frame in frames
    ]
