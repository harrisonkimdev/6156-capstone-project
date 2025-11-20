"""Unit tests for advanced frame sampler module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pose_ai.data.advanced_sampler import (
    compute_optical_flow,
    compute_pose_similarity,
    extract_frames_with_motion,
)
from pose_ai.pose.estimator import PoseFrame, PoseLandmark


def _create_test_video(video_path: Path, frame_count: int = 30, fps: float = 30.0) -> None:
    """Create a test video with varying motion."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (320, 240))

    for idx in range(frame_count):
        # Create frames with varying content
        if idx < 10:
            # Static background
            frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        elif idx < 20:
            # Moving object
            frame = np.full((240, 320, 3), 128, dtype=np.uint8)
            x = int((idx - 10) * 10)
            cv2.rectangle(frame, (x, 100), (x + 50, 150), (255, 0, 0), -1)
        else:
            # Different static background
            frame = np.full((240, 320, 3), 200, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_compute_optical_flow() -> None:
    """Test optical flow computation."""
    # Create two frames with motion
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)

    # Add a moving rectangle
    cv2.rectangle(frame1, (10, 10), (30, 30), (255, 255, 255), -1)
    cv2.rectangle(frame2, (20, 10), (40, 30), (255, 255, 255), -1)

    motion_score, flow_magnitude = compute_optical_flow(frame1, frame2)

    assert motion_score >= 0.0
    assert flow_magnitude.shape == (100, 100)
    # Should detect some motion
    assert motion_score > 0.0


def test_compute_optical_flow_no_motion() -> None:
    """Test optical flow with identical frames."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (30, 30), (255, 255, 255), -1)

    motion_score, _ = compute_optical_flow(frame, frame)

    # Should have minimal motion
    assert motion_score >= 0.0
    assert motion_score < 1.0  # Very low motion for identical frames


def test_compute_pose_similarity() -> None:
    """Test pose similarity computation."""
    # Create two similar poses
    landmarks1 = [
        PoseLandmark("left_shoulder", 0.5, 0.3, 0.0, 0.9),
        PoseLandmark("right_shoulder", 0.5, 0.3, 0.0, 0.9),
        PoseLandmark("left_hip", 0.5, 0.6, 0.0, 0.9),
        PoseLandmark("right_hip", 0.5, 0.6, 0.0, 0.9),
    ]

    landmarks2 = [
        PoseLandmark("left_shoulder", 0.51, 0.31, 0.0, 0.9),
        PoseLandmark("right_shoulder", 0.49, 0.29, 0.0, 0.9),
        PoseLandmark("left_hip", 0.5, 0.6, 0.0, 0.9),
        PoseLandmark("right_hip", 0.5, 0.6, 0.0, 0.9),
    ]

    pose1 = PoseFrame(Path("frame1.jpg"), 0.0, landmarks1, 0.9)
    pose2 = PoseFrame(Path("frame2.jpg"), 1.0, landmarks2, 0.9)

    similarity = compute_pose_similarity(pose1, pose2)

    assert 0.0 <= similarity <= 1.0
    # Similar poses should have high similarity
    assert similarity > 0.5


def test_compute_pose_similarity_different() -> None:
    """Test pose similarity with very different poses."""
    landmarks1 = [
        PoseLandmark("left_shoulder", 0.3, 0.2, 0.0, 0.9),
        PoseLandmark("right_shoulder", 0.7, 0.2, 0.0, 0.9),
        PoseLandmark("left_hip", 0.3, 0.8, 0.0, 0.9),
        PoseLandmark("right_hip", 0.7, 0.8, 0.0, 0.9),
    ]

    landmarks2 = [
        PoseLandmark("left_shoulder", 0.5, 0.1, 0.0, 0.9),
        PoseLandmark("right_shoulder", 0.5, 0.1, 0.0, 0.9),
        PoseLandmark("left_hip", 0.4, 0.9, 0.0, 0.9),
        PoseLandmark("right_hip", 0.6, 0.9, 0.0, 0.9),
    ]

    pose1 = PoseFrame(Path("frame1.jpg"), 0.0, landmarks1, 0.9)
    pose2 = PoseFrame(Path("frame2.jpg"), 1.0, landmarks2, 0.9)

    similarity = compute_pose_similarity(pose1, pose2)

    assert 0.0 <= similarity <= 1.0
    # Different poses should have lower similarity
    assert similarity < 0.8


def test_compute_pose_similarity_no_landmarks() -> None:
    """Test pose similarity with missing landmarks."""
    pose1 = PoseFrame(Path("frame1.jpg"), 0.0, [], 0.0)
    pose2 = PoseFrame(Path("frame2.jpg"), 1.0, [], 0.0)

    similarity = compute_pose_similarity(pose1, pose2)

    assert similarity == 0.0


def test_extract_frames_with_motion(tmp_path: Path) -> None:
    """Test motion-based frame extraction."""
    video_path = tmp_path / "test_video.mp4"
    _create_test_video(video_path, frame_count=30, fps=30.0)

    output_root = tmp_path / "output"

    result = extract_frames_with_motion(
        video_path,
        output_root=output_root,
        motion_threshold=2.0,
        similarity_threshold=0.8,
        min_frame_interval=5,
        use_optical_flow=True,
        use_pose_similarity=False,  # Disable pose similarity for faster test
        initial_sampling_rate=0.1,
        write_manifest=True,
        overwrite=False,
    )

    assert result.saved_frames > 0
    assert len(result.frame_paths) == result.saved_frames
    assert all(path.exists() for path in result.frame_paths)
    assert result.manifest_path is not None
    assert result.manifest_path.exists()

    # Check manifest
    import json

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["extraction_method"] == "motion"
    assert manifest["saved_frames"] == result.saved_frames
    assert len(manifest["frames"]) == result.saved_frames


def test_extract_frames_with_motion_low_threshold(tmp_path: Path) -> None:
    """Test motion extraction with very low threshold (should select more frames)."""
    video_path = tmp_path / "test_video.mp4"
    _create_test_video(video_path, frame_count=30, fps=30.0)

    output_root = tmp_path / "output2"

    result = extract_frames_with_motion(
        video_path,
        output_root=output_root,
        motion_threshold=0.1,  # Very low threshold
        similarity_threshold=0.8,
        min_frame_interval=2,  # Smaller interval
        use_optical_flow=True,
        use_pose_similarity=False,
        initial_sampling_rate=0.1,
        write_manifest=True,
        overwrite=False,
    )

    # Should select more frames with lower threshold
    assert result.saved_frames > 0


@pytest.mark.skip(reason="Requires pose estimation - integration test")
def test_extract_frames_with_pose_similarity(tmp_path: Path) -> None:
    """Test motion extraction with pose similarity (requires MediaPipe)."""
    video_path = tmp_path / "test_video.mp4"
    _create_test_video(video_path, frame_count=30, fps=30.0)

    output_root = tmp_path / "output3"

    result = extract_frames_with_motion(
        video_path,
        output_root=output_root,
        motion_threshold=2.0,
        similarity_threshold=0.7,
        min_frame_interval=5,
        use_optical_flow=True,
        use_pose_similarity=True,  # Enable pose similarity
        initial_sampling_rate=0.1,
        write_manifest=True,
        overwrite=False,
    )

    assert result.saved_frames > 0
    assert result.manifest_path is not None

    import json

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["extraction_method"] == "motion_pose"

