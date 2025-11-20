"""Static configuration for pose feature extraction."""

from __future__ import annotations

from typing import Dict, Tuple

JOINT_DEFINITIONS: Dict[str, Tuple[str, str, str]] = {
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
    "left_shoulder_angle": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder_angle": ("right_hip", "right_shoulder", "right_elbow"),
}

CENTER_OF_MASS_WEIGHTS: Dict[str, float] = {
    "left_shoulder": 0.08,
    "right_shoulder": 0.08,
    "left_elbow": 0.05,
    "right_elbow": 0.05,
    "left_wrist": 0.02,
    "right_wrist": 0.02,
    "left_hip": 0.14,
    "right_hip": 0.14,
    "left_knee": 0.13,
    "right_knee": 0.13,
    "left_ankle": 0.08,
    "right_ankle": 0.08,
    "left_foot_index": 0.06,
    "right_foot_index": 0.06,
}

CONTACT_LANDMARKS: Dict[str, str] = {
    "left_hand": "left_wrist",
    "right_hand": "right_wrist",
    "left_foot": "left_foot_index",
    "right_foot": "right_foot_index",
}

DEFAULT_CONTACT_THRESHOLD = 0.05
