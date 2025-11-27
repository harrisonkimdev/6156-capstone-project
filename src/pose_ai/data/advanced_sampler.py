"""Advanced frame extraction using motion detection and pose similarity.

This module provides intelligent frame selection based on:
1. Optical flow motion detection
2. Pose keypoint similarity comparison
3. Minimum interval constraints
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .frame_sampler import FrameExtractionResult
from .video_loader import ensure_directory
from ..pose.estimator import PoseEstimator, PoseFrame

LOGGER = logging.getLogger(__name__)


def compute_optical_flow(frame1: np.ndarray, frame2: np.ndarray, max_size: int = 640) -> tuple[float, np.ndarray]:
    """Compute dense optical flow between two frames.

    Args:
        frame1: First frame (grayscale or BGR)
        frame2: Second frame (grayscale or BGR)
        max_size: Maximum dimension for resizing (to reduce memory usage)

    Returns:
        Tuple of (motion_score, flow_magnitude_array)
    """
    # Convert to grayscale if needed
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1

    if len(frame2.shape) == 3:
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = frame2

    # Resize frames if too large to reduce memory usage and prevent crashes
    h, w = gray1.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray1 = cv2.resize(gray1, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray2 = cv2.resize(gray2, (new_w, new_h), interpolation=cv2.INTER_AREA)
        LOGGER.debug("Resized frames from %dx%d to %dx%d for optical flow", w, h, new_w, new_h)

    # Ensure frames have same shape
    if gray1.shape != gray2.shape:
        h, w = gray1.shape[:2]
        gray2 = cv2.resize(gray2, (w, h), interpolation=cv2.INTER_AREA)

    # Validate frame dimensions
    if gray1.size == 0 or gray2.size == 0:
        LOGGER.warning("Empty frame detected in optical flow computation")
        return 0.0, np.array([])

    # Compute optical flow with error handling
    try:
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
    except cv2.error as e:
        LOGGER.error("OpenCV error in optical flow computation: %s", e)
        return 0.0, np.array([])
    except Exception as e:
        LOGGER.error("Unexpected error in optical flow computation: %s", e)
        return 0.0, np.array([])

    # Compute magnitude
    try:
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        motion_score = float(np.mean(magnitude))
    except Exception as e:
        LOGGER.error("Error computing motion score: %s", e)
        return 0.0, np.array([])

    return motion_score, magnitude


def compute_pose_similarity(pose1: PoseFrame, pose2: PoseFrame) -> float:
    """Compute similarity between two pose frames.

    Args:
        pose1: First pose frame
        pose2: Second pose frame

    Returns:
        Similarity score (0-1, higher = more similar)
    """
    if not pose1.landmarks or not pose2.landmarks:
        return 0.0

    # Match landmarks by name
    landmarks1_dict = {lm.name: lm for lm in pose1.landmarks}
    landmarks2_dict = {lm.name: lm for lm in pose2.landmarks}

    common_names = set(landmarks1_dict.keys()) & set(landmarks2_dict.keys())
    if not common_names:
        return 0.0

    # Compute Euclidean distance for each common landmark
    distances = []
    for name in common_names:
        lm1 = landmarks1_dict[name]
        lm2 = landmarks2_dict[name]

        # Only use visible landmarks
        if lm1.visibility < 0.5 or lm2.visibility < 0.5:
            continue

        # 2D distance (x, y)
        dist = np.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2)
        distances.append(dist)

    if not distances:
        return 0.0

    # Average distance, normalized to [0, 1]
    # Typical pose spans ~0.5-1.0 in normalized coordinates
    avg_distance = np.mean(distances)
    similarity = 1.0 / (1.0 + avg_distance * 2.0)  # Sigmoid-like function

    return float(similarity)


def extract_frames_with_motion(
    video_path: Path | str,
    *,
    output_root: Path | str = Path("data") / "frames",
    motion_threshold: float = 5.0,
    similarity_threshold: float = 0.8,
    min_frame_interval: int = 5,
    use_optical_flow: bool = True,
    use_pose_similarity: bool = True,
    initial_sampling_rate: float = 0.1,  # Sample every 0.1s initially
    write_manifest: bool = True,
    overwrite: bool = False,
) -> FrameExtractionResult:
    """Extract frames based on motion detection and pose similarity.

    Algorithm:
    1. Extract frames at high rate (initial_sampling_rate)
    2. Compute motion scores using optical flow
    3. Run pose estimation on frames with high motion
    4. Compare pose keypoints between consecutive frames
    5. Select frames where pose similarity < threshold (significant pose change)
    6. Apply minimum interval constraint

    Args:
        video_path: Path to video file
        output_root: Directory to save frames
        motion_threshold: Minimum motion score to consider frame
        similarity_threshold: Maximum pose similarity (lower = more diverse)
        min_frame_interval: Minimum frames between selections
        use_optical_flow: Enable/disable optical flow
        use_pose_similarity: Enable/disable pose-based filtering
        initial_sampling_rate: Initial frame sampling rate in seconds
        write_manifest: Write manifest.json file
        overwrite: Overwrite existing frames

    Returns:
        FrameExtractionResult with selected frames
    """
    source_path = Path(video_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Video file not found: {source_path}")

    frames_root = ensure_directory(output_root)
    frame_dir = ensure_directory(frames_root / source_path.stem)

    if overwrite:
        for existing in frame_dir.glob("*.jpg"):
            existing.unlink()

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"CV2 failed to open video file: {source_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    initial_frame_interval = max(int(round(fps * initial_sampling_rate)), 1)

    LOGGER.info(
        "Starting motion-based frame extraction: video=%s, fps=%.2f, initial_rate=%.2fs",
        source_path,
        fps,
        initial_sampling_rate,
    )

    # Step 1: Extract frames at high rate
    temp_frames: list[tuple[int, np.ndarray, float]] = []  # (frame_idx, frame, timestamp)
    frame_idx = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_idx % initial_frame_interval == 0:
            timestamp = frame_idx / fps if fps > 0 else frame_idx * initial_sampling_rate
            temp_frames.append((frame_idx, frame.copy(), timestamp))

        frame_idx += 1

    capture.release()

    if not temp_frames:
        raise ValueError("No frames extracted from video")

    LOGGER.info("Extracted %d candidate frames at high rate", len(temp_frames))

    # Step 2: Compute motion scores
    motion_scores: list[float] = [0.0]  # First frame has no motion
    prev_frame: Optional[np.ndarray] = None

    for idx, (_, frame, _) in enumerate(temp_frames):
        if idx == 0:
            prev_frame = frame
            continue

        if use_optical_flow and prev_frame is not None:
            motion_score, _ = compute_optical_flow(prev_frame, frame)
            motion_scores.append(motion_score)
        else:
            # Simple frame difference
            if prev_frame is not None:
                gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
                gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                diff = cv2.absdiff(gray1, gray2)
                motion_score = float(np.mean(diff))
                motion_scores.append(motion_score)
            else:
                motion_scores.append(0.0)

        prev_frame = frame

    # Step 3: Filter out walking segments (high motion = walking in/out)
    # Select frames with LOW motion (climbing segments)
    # Walking has high motion, climbing has lower motion
    climbing_indices = [
        idx for idx, score in enumerate(motion_scores) if score < motion_threshold
    ]

    if not climbing_indices:
        LOGGER.warning("No frames below motion threshold (all frames appear to be walking). Using all frames.")
        climbing_indices = list(range(len(temp_frames)))
    else:
        LOGGER.info("Found %d climbing frames (low motion, out of %d total)", len(climbing_indices), len(temp_frames))
        LOGGER.info("Excluded %d walking frames (high motion)", len(temp_frames) - len(climbing_indices))

    # Step 4: Pose estimation on climbing frames
    selected_indices: list[int] = []
    pose_frames: list[PoseFrame] = []

    if use_pose_similarity:
        # Save climbing frames temporarily for pose estimation
        temp_frame_dir = frame_dir / "temp_poses"
        temp_frame_dir.mkdir(exist_ok=True)
        temp_frame_paths: list[Path] = []

        for idx in climbing_indices:
            frame_idx, frame, timestamp = temp_frames[idx]
            temp_path = temp_frame_dir / f"temp_{idx:06d}.jpg"
            cv2.imwrite(str(temp_path), frame)
            temp_frame_paths.append(temp_path)

        # Run pose estimation
        try:
            estimator = PoseEstimator()
            pose_frames = estimator.process_paths(
                temp_frame_paths,
                timestamps=[temp_frames[idx][2] for idx in climbing_indices],
            )
            estimator.close()
        except Exception as exc:
            LOGGER.warning("Pose estimation failed: %s. Falling back to motion-only selection.", exc)
            use_pose_similarity = False

        # Clean up temp files
        for temp_path in temp_frame_paths:
            try:
                temp_path.unlink()
            except Exception:
                pass
        try:
            temp_frame_dir.rmdir()
        except Exception:
            pass

    # Step 5: Select frames based on pose similarity
    if use_pose_similarity and pose_frames:
        selected_indices.append(climbing_indices[0])  # Always include first frame

        for i in range(1, len(climbing_indices)):
            current_idx = climbing_indices[i]
            prev_selected_idx = selected_indices[-1]

            # Find corresponding pose frames
            current_pose_idx = i
            prev_pose_idx = climbing_indices.index(prev_selected_idx) if prev_selected_idx in climbing_indices else None

            if prev_pose_idx is not None and current_pose_idx < len(pose_frames) and prev_pose_idx < len(pose_frames):
                similarity = compute_pose_similarity(pose_frames[prev_pose_idx], pose_frames[current_pose_idx])

                # Select if pose is different enough
                if similarity < similarity_threshold:
                    # Check minimum interval
                    if len(selected_indices) == 0 or (current_idx - selected_indices[-1]) >= min_frame_interval:
                        selected_indices.append(current_idx)
    else:
        # Motion-only selection (climbing frames only)
        selected_indices.append(climbing_indices[0])  # Always include first frame

        for idx in climbing_indices[1:]:
            # Check minimum interval
            if len(selected_indices) == 0 or (idx - selected_indices[-1]) >= min_frame_interval:
                selected_indices.append(idx)

    # Ensure we have at least a few frames
    if len(selected_indices) < 3:
        # Add evenly spaced frames
        step = max(1, len(temp_frames) // 3)
        selected_indices = list(range(0, len(temp_frames), step))[:3]

    # Sort selected indices
    selected_indices = sorted(set(selected_indices))

    LOGGER.info("Selected %d frames after filtering", len(selected_indices))

    # Step 6: Save selected frames
    saved_paths: list[Path] = []
    manifest_records: list[dict[str, object]] = []

    for saved_idx, orig_idx in enumerate(selected_indices):
        if orig_idx >= len(temp_frames):
            continue

        _, frame, timestamp = temp_frames[orig_idx]
        frame_filename = f"{source_path.stem}_frame_{saved_idx:04d}.jpg"
        frame_path = frame_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)
        saved_paths.append(frame_path)

        manifest_records.append(
            {
                "frame_index": int(orig_idx),
                "saved_index": int(saved_idx),
                "timestamp_seconds": float(timestamp),
                "relative_path": frame_filename,
                "motion_score": float(motion_scores[orig_idx]) if orig_idx < len(motion_scores) else 0.0,
            }
        )

    manifest_path: Optional[Path] = None
    if write_manifest:
        manifest_path = frame_dir / "manifest.json"
        # Calculate approximate interval from saved frames
        if len(saved_paths) > 1 and fps > 0:
            approx_interval = (temp_frames[selected_indices[-1]][2] - temp_frames[selected_indices[0]][2]) / max(1, len(saved_paths) - 1)
        else:
            approx_interval = initial_sampling_rate
        
        manifest_payload = {
            "video": str(source_path),
            "fps": fps,
            "interval_seconds": approx_interval,
            "extraction_method": "motion_pose" if use_pose_similarity else "motion",
            "motion_threshold": motion_threshold,
            "similarity_threshold": similarity_threshold if use_pose_similarity else None,
            "min_frame_interval": min_frame_interval,
            "total_frames": frame_idx,
            "saved_frames": len(saved_paths),
            "frames": manifest_records,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    LOGGER.info(
        "Motion-based extraction complete: saved=%d frames (directory=%s)",
        len(saved_paths),
        frame_dir,
    )

    return FrameExtractionResult(
        video_path=source_path,
        frame_directory=frame_dir,
        manifest_path=manifest_path,
        fps=fps,
        interval_seconds=initial_sampling_rate,  # Approximate
        total_frames=frame_idx,
        saved_frames=len(saved_paths),
        frame_paths=saved_paths,
    )


__all__ = [
    "compute_optical_flow",
    "compute_pose_similarity",
    "extract_frames_with_motion",
]

