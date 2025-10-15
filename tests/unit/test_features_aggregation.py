from __future__ import annotations

from pose_ai.features import (
    HoldDefinition,
    analyze_hold_relationship,
    compute_center_of_mass,
    compute_joint_angles,
    compute_segment_inclination,
    pose_to_feature_row,
)
from pose_ai.pose.estimator import PoseFrame, PoseLandmark


def _make_pose_frame() -> PoseFrame:
    landmarks = [
        PoseLandmark(name="left_shoulder", x=0.4, y=0.3, z=0.0, visibility=0.9),
        PoseLandmark(name="right_shoulder", x=0.6, y=0.3, z=0.0, visibility=0.9),
        PoseLandmark(name="left_elbow", x=0.35, y=0.45, z=0.0, visibility=0.9),
        PoseLandmark(name="right_elbow", x=0.65, y=0.45, z=0.0, visibility=0.9),
        PoseLandmark(name="left_wrist", x=0.3, y=0.6, z=0.0, visibility=0.9),
        PoseLandmark(name="right_wrist", x=0.7, y=0.6, z=0.0, visibility=0.9),
        PoseLandmark(name="left_hip", x=0.45, y=0.6, z=0.0, visibility=0.9),
        PoseLandmark(name="right_hip", x=0.55, y=0.6, z=0.0, visibility=0.9),
        PoseLandmark(name="left_knee", x=0.47, y=0.8, z=0.0, visibility=0.9),
        PoseLandmark(name="right_knee", x=0.53, y=0.8, z=0.0, visibility=0.9),
        PoseLandmark(name="left_ankle", x=0.46, y=0.95, z=0.0, visibility=0.9),
        PoseLandmark(name="right_ankle", x=0.54, y=0.95, z=0.0, visibility=0.9),
        PoseLandmark(name="left_foot_index", x=0.46, y=0.98, z=0.0, visibility=0.9),
        PoseLandmark(name="right_foot_index", x=0.54, y=0.98, z=0.0, visibility=0.9),
    ]
    return PoseFrame(
        image_path="frame.jpg",
        timestamp_seconds=0.0,
        landmarks=landmarks,
        detection_score=0.9,
    )


def test_compute_joint_angles_returns_values():
    frame = _make_pose_frame()
    angles = compute_joint_angles(frame)
    assert "left_elbow_angle" in angles
    assert angles["left_elbow_angle"] is not None


def test_compute_center_of_mass():
    frame = _make_pose_frame()
    com = compute_center_of_mass(frame)
    assert 0.4 < com["com_x"] < 0.6
    assert 0.5 < com["com_y"] < 0.8


def test_segment_inclination():
    frame = _make_pose_frame()
    incline = compute_segment_inclination(frame, "left_hip", "left_shoulder")
    assert incline is not None


def test_pose_to_feature_row_with_holds():
    frame = _make_pose_frame()
    holds = {
        "left_hold": HoldDefinition(name="left_hold", coords=(0.28, 0.6), normalized=True),
        "right_hold": HoldDefinition(name="right_hold", coords=(0.72, 0.6), normalized=True),
    }
    row = pose_to_feature_row(frame, holds=holds)
    assert row["left_hand_target"] == "left_hold"
    assert "left_elbow_angle" in row
