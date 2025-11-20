"""Feature extraction from PoseFrame sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from pose_ai.pose.estimator import PoseFrame, PoseLandmark

from .config import (
    CENTER_OF_MASS_WEIGHTS,
    CONTACT_LANDMARKS,
    DEFAULT_CONTACT_THRESHOLD,
    JOINT_DEFINITIONS,
)
try:  # optional: wall angle utilities
    from pose_ai.wall.angle import estimate_wall_angle, compute_wall_angle_from_imu  # type: ignore
except Exception:  # pragma: no cover - soft dependency
    estimate_wall_angle = None  # type: ignore
    compute_wall_angle_from_imu = None  # type: ignore

TRACKED_LANDMARKS = {
    "left_hand": "left_wrist",
    "right_hand": "right_wrist",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
}


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
    xs, ys, zs, weights = [], [], [], []
    for name, weight in CENTER_OF_MASS_WEIGHTS.items():
        landmark = lookup.get(name)
        if landmark is None:
            continue
        xs.append(landmark.x)
        ys.append(landmark.y)
        zs.append(landmark.z)
        weights.append(weight)
    if not weights:
        return {"com_x": None, "com_y": None, "com_z": None, "com_n": 0}
    w = np.array(weights, dtype=float)
    com_x = float(np.average(np.array(xs, dtype=float), weights=w))
    com_y = float(np.average(np.array(ys, dtype=float), weights=w))
    com_z = float(np.average(np.array(zs, dtype=float), weights=w))
    return {"com_x": com_x, "com_y": com_y, "com_z": com_z, "com_n": len(weights)}


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
            features[f"{limb}_target_x"] = None
            features[f"{limb}_target_y"] = None
            continue
        limb_point = np.array([landmark.x, landmark.y], dtype=float)
        best_name = None
        best_distance = float("inf")
        best_coords: Optional[np.ndarray] = None
        for hold_name, hold_point in hold_positions.items():
            distance = float(np.linalg.norm(limb_point - hold_point))
            if distance < best_distance:
                best_distance = distance
                best_name = hold_name
                best_coords = hold_point
        features[f"{limb}_target"] = best_name
        features[f"{limb}_distance"] = best_distance
        features[f"{limb}_on_hold"] = int(best_distance <= contact_threshold)
        if best_coords is not None:
            features[f"{limb}_target_x"] = float(best_coords[0])
            features[f"{limb}_target_y"] = float(best_coords[1])
        else:
            features[f"{limb}_target_x"] = None
            features[f"{limb}_target_y"] = None
    return features


def pose_to_feature_row(
    frame: PoseFrame,
    *,
    holds: Optional[Dict[str, HoldDefinition]] = None,
    wall_angle_degrees: Optional[float] = None,
    climber_height: Optional[float] = None,
    climber_wingspan: Optional[float] = None,
    climber_flexibility: Optional[float] = None,
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

    # Track limb coordinates for downstream contact/velocity logic.
    lookup = _landmarks_by_name(frame)
    for label, landmark_name in TRACKED_LANDMARKS.items():
        landmark = lookup.get(landmark_name)
        feature_row[f"{label}_x"] = float(landmark.x) if landmark else None
        feature_row[f"{label}_y"] = float(landmark.y) if landmark else None
        feature_row[f"{label}_z"] = float(landmark.z) if landmark else None
        feature_row[f"{label}_visibility"] = float(landmark.visibility) if landmark else None

    left_shoulder = lookup.get("left_shoulder")
    right_shoulder = lookup.get("right_shoulder")
    if left_shoulder and right_shoulder:
        detected_shoulder_width = float(
            np.linalg.norm(
                np.array([left_shoulder.x, left_shoulder.y]) - np.array([right_shoulder.x, right_shoulder.y])
            )
        )
        feature_row["body_scale"] = detected_shoulder_width
        
        # If climber height is provided, compute normalized body scale
        if climber_height is not None:
            # Average shoulder width is approximately 16% of height
            expected_shoulder_width_cm = climber_height * 0.16
            # Convert to normalized units (assuming shoulder width in image is in normalized coords)
            # This ratio helps adjust for different body proportions
            feature_row["body_scale_normalized"] = detected_shoulder_width * (170.0 * 0.16 / expected_shoulder_width_cm)
        else:
            feature_row["body_scale_normalized"] = None
    else:
        feature_row["body_scale"] = None
        feature_row["body_scale_normalized"] = None
    
    # Store climber physical parameters in each feature row
    feature_row["climber_height"] = climber_height
    feature_row["climber_wingspan"] = climber_wingspan
    feature_row["climber_flexibility"] = climber_flexibility

    left_hip = lookup.get("left_hip")
    right_hip = lookup.get("right_hip")
    if left_hip and right_hip:
        feature_row["hip_center_x"] = float((left_hip.x + right_hip.x) / 2.0)
        feature_row["hip_center_y"] = float((left_hip.y + right_hip.y) / 2.0)
        feature_row["hip_center_z"] = float((left_hip.z + right_hip.z) / 2.0)
    else:
        feature_row["hip_center_x"] = None
        feature_row["hip_center_y"] = None
        feature_row["hip_center_z"] = None

    if holds:
        feature_row.update(analyze_hold_relationship(frame, holds))

    if wall_angle_degrees is not None:
        feature_row["wall_angle"] = float(wall_angle_degrees)
        # Hip / COM relative to wall orientation: project inclination features.
        hip_inclination = feature_row.get("hip_line_inclination")
        if isinstance(hip_inclination, (int, float)):
            # Difference between hip line and wall angle as basic alignment metric.
            diff = abs(float(hip_inclination) - float(wall_angle_degrees)) % 180.0
            feature_row["hip_alignment_error"] = diff if diff <= 90.0 else 180.0 - diff
        com_x = feature_row.get("com_x")
        com_y = feature_row.get("com_y")
        if isinstance(com_x, (int, float)) and isinstance(com_y, (int, float)):
            # Rotate COM into wall-aligned coordinate system (wall angle w.r.t horizontal).
            theta = np.radians(float(wall_angle_degrees))
            # Horizontal axis along wall, vertical axis perpendicular (approx for overhang/vertical).
            com_along = float(com_x * np.cos(theta) + com_y * np.sin(theta))
            com_perp = float(-com_x * np.sin(theta) + com_y * np.cos(theta))
            feature_row["com_along_wall"] = com_along
            feature_row["com_perp_wall"] = com_perp
    else:
        feature_row["wall_angle"] = None
        feature_row["hip_alignment_error"] = None
        feature_row["com_along_wall"] = None
        feature_row["com_perp_wall"] = None
    return feature_row


def summarize_features(
    frames: Sequence[PoseFrame],
    *,
    holds: Optional[Dict[str, HoldDefinition]] = None,
    wall_angle_degrees: Optional[float] = None,
    auto_estimate_wall: bool = False,
    imu_quaternion: Optional[list[float]] = None,
    imu_euler_angles: Optional[list[float]] = None,
    climber_height: Optional[float] = None,
    climber_wingspan: Optional[float] = None,
    climber_flexibility: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Extract features from pose frames with optional IMU and climber personalization.
    
    Args:
        frames: Sequence of pose estimation results
        holds: Optional hold definitions for contact analysis
        wall_angle_degrees: Pre-computed wall angle (overrides all estimation)
        auto_estimate_wall: Enable vision-based wall angle estimation (fallback)
        imu_quaternion: Device orientation as quaternion [w, x, y, z] (priority over euler)
        imu_euler_angles: Device orientation as Euler angles [pitch, roll, yaw] in degrees
        climber_height: Climber height in cm (for body scale normalization)
        climber_wingspan: Climber wingspan in cm (for reach constraints)
        climber_flexibility: Flexibility score 0-1 (for personalized thresholds)
    
    Returns:
        List of feature dictionaries, one per frame
    """
    angle = wall_angle_degrees
    
    # Priority 1: Use pre-computed angle if provided
    if angle is None:
        # Priority 2: Compute from IMU sensor data
        if (imu_quaternion is not None or imu_euler_angles is not None) and compute_wall_angle_from_imu is not None:
            try:
                angle_result = compute_wall_angle_from_imu(
                    quaternion=imu_quaternion,
                    euler_angles=imu_euler_angles,
                )
                angle = angle_result.angle_degrees
            except Exception:
                angle = None
        
        # Priority 3: Vision-based estimation (fallback)
        if angle is None and auto_estimate_wall and frames and estimate_wall_angle is not None:
            try:
                angle_result = estimate_wall_angle(frames[0].image_path)
                angle = angle_result.angle_degrees
            except Exception:
                angle = None
    
    return [
        pose_to_feature_row(
            frame,
            holds=holds,
            wall_angle_degrees=angle,
            climber_height=climber_height,
            climber_wingspan=climber_wingspan,
            climber_flexibility=climber_flexibility,
        )
        for frame in frames
    ]
