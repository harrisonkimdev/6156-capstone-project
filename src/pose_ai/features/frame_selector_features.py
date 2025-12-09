"""Frame selector feature extraction module.

Extracts features from video frames to train a frame selection model.
Features include motion score, pose keypoint change, brightness, sharpness, and edge density.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class FrameFeatures:
    """Features extracted from a single frame."""
    
    frame_idx: int
    motion_score: float
    pose_keypoint_change: float
    brightness: float
    sharpness: float
    edge_density: float
    timestamp: float
    filename: str


def compute_pose_keypoint_change(
    pose1: Optional[np.ndarray],
    pose2: Optional[np.ndarray]
) -> float:
    """Compute pose keypoint change between two frames.
    
    Args:
        pose1, pose2: (num_keypoints, 2) arrays of [x, y] normalized coordinates
        
    Returns:
        Average Euclidean distance between keypoints (0.0 if pose unavailable)
    """
    if pose1 is None or pose2 is None:
        return 0.0
    
    if pose1.shape != pose2.shape:
        return 0.0
    
    # Compute Euclidean distance for each keypoint
    distances = np.linalg.norm(pose2 - pose1, axis=1)
    return float(np.mean(distances))


def compute_brightness(frame: np.ndarray) -> float:
    """Compute frame brightness (average pixel intensity).
    
    Args:
        frame: BGR image
        
    Returns:
        Average brightness in range [0, 255]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def compute_sharpness(frame: np.ndarray) -> float:
    """Compute frame sharpness using Laplacian variance.
    
    Args:
        frame: BGR image
        
    Returns:
        Laplacian variance (higher = sharper)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_edge_density(frame: np.ndarray) -> float:
    """Compute edge density using Canny edge detection.
    
    Args:
        frame: BGR image
        
    Returns:
        Ratio of edge pixels to total pixels [0, 1]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_ratio = np.sum(edges > 0) / edges.size
    return float(edge_ratio)


def extract_frame_features(
    frame: np.ndarray,
    frame_idx: int,
    prev_frame: Optional[np.ndarray],
    pose_keypoints: Optional[np.ndarray],
    prev_pose_keypoints: Optional[np.ndarray],
    motion_score: float,
    timestamp: float,
    filename: str,
) -> FrameFeatures:
    """Extract all features from a single frame.
    
    Args:
        frame: Current frame (BGR)
        frame_idx: Frame index
        prev_frame: Previous frame for computing changes
        pose_keypoints: Current frame pose keypoints (num_keypoints, 2)
        prev_pose_keypoints: Previous frame pose keypoints
        motion_score: Pre-computed motion score from manifest.json
        timestamp: Frame timestamp in seconds
        filename: Frame filename
        
    Returns:
        FrameFeatures object with all extracted features
    """
    # Compute pose change
    pose_change = compute_pose_keypoint_change(prev_pose_keypoints, pose_keypoints)
    
    # Compute frame quality features
    brightness = compute_brightness(frame)
    sharpness = compute_sharpness(frame)
    edge_density = compute_edge_density(frame)
    
    return FrameFeatures(
        frame_idx=frame_idx,
        motion_score=motion_score,
        pose_keypoint_change=pose_change,
        brightness=brightness,
        sharpness=sharpness,
        edge_density=edge_density,
        timestamp=timestamp,
        filename=filename,
    )


def extract_all_frame_features(
    all_frames_dir: Path,
    manifest_path: Path,
) -> List[FrameFeatures]:
    """Extract features from all frames in the all_frames/ directory.
    
    Args:
        all_frames_dir: Path to all_frames/ directory
        manifest_path: Path to manifest.json (contains motion scores)
        
    Returns:
        List of FrameFeatures for each frame
    """
    LOGGER.info("Extracting features from frames in: %s", all_frames_dir)
    
    # Load manifest for motion scores
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    
    # Get motion scores by frame index
    motion_scores = {}
    if 'frames' in manifest:
        for frame_info in manifest['frames']:
            motion_scores[frame_info['frame_index']] = frame_info.get('motion_score', 0.0)
    
    # Get all frame files sorted by name
    frame_files = sorted(all_frames_dir.glob("*_frame_*.jpg"))
    
    if not frame_files:
        LOGGER.warning("No frames found in: %s", all_frames_dir)
        return []
    
    features_list = []
    prev_frame = None
    prev_pose = None  # TODO: Add pose estimation support
    
    for i, frame_path in enumerate(frame_files):
        # Extract frame index from filename (e.g., IMG_3709_frame_000005.jpg -> 5)
        try:
            frame_idx_str = frame_path.stem.split('_frame_')[1]
            frame_idx = int(frame_idx_str)
        except (IndexError, ValueError):
            LOGGER.warning("Could not parse frame index from: %s", frame_path.name)
            frame_idx = i
        
        # Load frame
        frame = cv2.imread(str(frame_path))
        if frame is None:
            LOGGER.warning("Failed to load frame: %s", frame_path)
            continue
        
        # Get motion score from manifest (default to 0.0 if not found)
        motion_score = motion_scores.get(frame_idx, 0.0)
        
        # Extract timestamp (use frame index * default FPS if not in manifest)
        timestamp = frame_idx / 30.0  # Default 30 FPS
        
        # Extract features
        features = extract_frame_features(
            frame=frame,
            frame_idx=frame_idx,
            prev_frame=prev_frame,
            pose_keypoints=None,  # TODO: Add pose estimation
            prev_pose_keypoints=prev_pose,
            motion_score=motion_score,
            timestamp=timestamp,
            filename=frame_path.name,
        )
        
        features_list.append(features)
        
        # Update previous frame
        prev_frame = frame
    
    LOGGER.info("Extracted features from %d frames", len(features_list))
    return features_list


def features_to_numpy(features_list: List[FrameFeatures]) -> np.ndarray:
    """Convert list of FrameFeatures to numpy array.
    
    Args:
        features_list: List of FrameFeatures
        
    Returns:
        (n_frames, 5) array of features:
            [motion_score, pose_keypoint_change, brightness, sharpness, edge_density]
    """
    feature_matrix = np.zeros((len(features_list), 5), dtype=np.float32)
    
    for i, features in enumerate(features_list):
        feature_matrix[i] = [
            features.motion_score,
            features.pose_keypoint_change,
            features.brightness,
            features.sharpness,
            features.edge_density,
        ]
    
    return feature_matrix
