"""Frame selector feature extraction module.

Extracts pose-based features from video frames:
- Acceleration of limbs (hands, elbows, shoulders)
- Distance from keypoints to holds
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class PoseBasedFrameFeatures:
    """Pose-based features for a single frame."""
    
    frame_idx: int
    timestamp: float
    filename: str
    
    # Acceleration features (6): left/right shoulder, elbow, wrist
    left_shoulder_accel: float
    right_shoulder_accel: float
    left_elbow_accel: float
    right_elbow_accel: float
    left_wrist_accel: float
    right_wrist_accel: float
    
    # Keypoint-to-hold distance features (8): 2 wrists × 4 holds
    left_wrist_to_hold_0: float
    left_wrist_to_hold_1: float
    left_wrist_to_hold_2: float
    left_wrist_to_hold_3: float
    right_wrist_to_hold_0: float
    right_wrist_to_hold_1: float
    right_wrist_to_hold_2: float
    right_wrist_to_hold_3: float
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array (14 dimensions)."""
        return np.array([
            self.left_shoulder_accel,
            self.right_shoulder_accel,
            self.left_elbow_accel,
            self.right_elbow_accel,
            self.left_wrist_accel,
            self.right_wrist_accel,
            self.left_wrist_to_hold_0,
            self.left_wrist_to_hold_1,
            self.left_wrist_to_hold_2,
            self.left_wrist_to_hold_3,
            self.right_wrist_to_hold_0,
            self.right_wrist_to_hold_1,
            self.right_wrist_to_hold_2,
            self.right_wrist_to_hold_3,
        ], dtype=np.float32)


def extract_keypoint_positions(
    pose_landmarks: Optional[Dict[str, Dict[str, float]]]
) -> Dict[str, Tuple[float, float]]:
    """Extract normalized (x, y) positions for key joints.
    
    Args:
        pose_landmarks: Dict mapping landmark name to {x, y, z, visibility}
    
    Returns:
        Dict mapping joint name to (x, y) normalized coordinates
    """
    if not pose_landmarks:
        return {}
    
    positions = {}
    for joint_name in ['left_shoulder', 'right_shoulder', 'left_elbow', 
                       'right_elbow', 'left_wrist', 'right_wrist']:
        if joint_name in pose_landmarks:
            lm = pose_landmarks[joint_name]
            positions[joint_name] = (lm.get('x', 0.0), lm.get('y', 0.0))
    
    return positions


def calculate_velocity(
    pos_prev: Dict[str, Tuple[float, float]],
    pos_curr: Dict[str, Tuple[float, float]],
    fps: float
) -> Dict[str, Tuple[float, float]]:
    """Calculate velocity (position change per second) for each joint.
    
    Args:
        pos_prev: Previous frame joint positions
        pos_curr: Current frame joint positions
        fps: Frames per second
    
    Returns:
        Dict mapping joint name to (vx, vy) velocity
    """
    velocity = {}
    dt = 1.0 / fps
    
    for joint_name in pos_curr:
        if joint_name in pos_prev:
            x_prev, y_prev = pos_prev[joint_name]
            x_curr, y_curr = pos_curr[joint_name]
            vx = (x_curr - x_prev) / dt
            vy = (y_curr - y_prev) / dt
            velocity[joint_name] = (vx, vy)
        else:
            velocity[joint_name] = (0.0, 0.0)
    
    return velocity


def calculate_acceleration(
    vel_prev: Dict[str, Tuple[float, float]],
    vel_curr: Dict[str, Tuple[float, float]],
    fps: float
) -> Dict[str, float]:
    """Calculate acceleration magnitude (velocity change per second).
    
    Args:
        vel_prev: Previous frame velocities
        vel_curr: Current frame velocities
        fps: Frames per second
    
    Returns:
        Dict mapping joint name to acceleration magnitude
    """
    acceleration = {}
    dt = 1.0 / fps
    
    for joint_name in vel_curr:
        if joint_name in vel_prev:
            vx_prev, vy_prev = vel_prev[joint_name]
            vx_curr, vy_curr = vel_curr[joint_name]
            ax = (vx_curr - vx_prev) / dt
            ay = (vy_curr - vy_prev) / dt
            # Acceleration magnitude
            accel_mag = np.sqrt(ax**2 + ay**2)
            acceleration[joint_name] = accel_mag
        else:
            acceleration[joint_name] = 0.0
    
    return acceleration


def calculate_keypoint_to_hold_distances(
    keypoint_pos: Tuple[float, float],
    hold_positions: List[Tuple[float, float]]
) -> List[float]:
    """Calculate Euclidean distances from a keypoint to all holds.
    
    Args:
        keypoint_pos: (x, y) normalized position of keypoint
        hold_positions: List of (x, y) normalized hold positions
    
    Returns:
        List of distances (length 4, padded with 0.0 if fewer holds)
    """
    distances = []
    for hold_pos in hold_positions[:4]:  # Max 4 holds
        dx = keypoint_pos[0] - hold_pos[0]
        dy = keypoint_pos[1] - hold_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        distances.append(distance)
    
    # Pad with 0.0 if fewer than 4 holds
    while len(distances) < 4:
        distances.append(0.0)
    
    return distances


def extract_pose_based_features_from_sequence(
    pose_sequence: List[Dict[str, any]],
    hold_positions: List[Tuple[float, float]],
    fps: float = 30.0
) -> List[PoseBasedFrameFeatures]:
    """Extract pose-based features from a sequence of pose estimations.
    
    Args:
        pose_sequence: List of dicts with 'frame_idx', 'timestamp', 'filename', 'landmarks'
        hold_positions: List of (x, y) normalized hold center positions
        fps: Video frame rate
    
    Returns:
        List of PoseBasedFrameFeatures for each frame
    """
    features_list = []
    
    # Track positions and velocities across frames
    positions_history = []
    velocities_history = []
    
    for frame_idx, pose_data in enumerate(pose_sequence):
        # Extract current frame info
        landmarks = pose_data.get('landmarks', {})
        timestamp = pose_data.get('timestamp', frame_idx / fps)
        filename = pose_data.get('filename', f'frame_{frame_idx:06d}.jpg')
        
        # Extract joint positions
        positions = extract_keypoint_positions(landmarks)
        positions_history.append(positions)
        
        # Calculate velocity (need previous frame)
        if len(positions_history) >= 2:
            velocity = calculate_velocity(positions_history[-2], positions_history[-1], fps)
        else:
            velocity = {k: (0.0, 0.0) for k in positions.keys()}
        velocities_history.append(velocity)
        
        # Calculate acceleration (need previous velocity)
        if len(velocities_history) >= 2:
            acceleration = calculate_acceleration(velocities_history[-2], velocities_history[-1], fps)
        else:
            acceleration = {k: 0.0 for k in velocity.keys()}
        
        # Extract acceleration features
        left_shoulder_accel = acceleration.get('left_shoulder', 0.0)
        right_shoulder_accel = acceleration.get('right_shoulder', 0.0)
        left_elbow_accel = acceleration.get('left_elbow', 0.0)
        right_elbow_accel = acceleration.get('right_elbow', 0.0)
        left_wrist_accel = acceleration.get('left_wrist', 0.0)
        right_wrist_accel = acceleration.get('right_wrist', 0.0)
        
        # Extract keypoint-to-hold distance features
        left_wrist_pos = positions.get('left_wrist', (0.0, 0.0))
        right_wrist_pos = positions.get('right_wrist', (0.0, 0.0))
        
        left_wrist_distances = calculate_keypoint_to_hold_distances(left_wrist_pos, hold_positions)
        right_wrist_distances = calculate_keypoint_to_hold_distances(right_wrist_pos, hold_positions)
        
        # Create feature object
        features = PoseBasedFrameFeatures(
            frame_idx=frame_idx,
            timestamp=timestamp,
            filename=filename,
            left_shoulder_accel=left_shoulder_accel,
            right_shoulder_accel=right_shoulder_accel,
            left_elbow_accel=left_elbow_accel,
            right_elbow_accel=right_elbow_accel,
            left_wrist_accel=left_wrist_accel,
            right_wrist_accel=right_wrist_accel,
            left_wrist_to_hold_0=left_wrist_distances[0],
            left_wrist_to_hold_1=left_wrist_distances[1],
            left_wrist_to_hold_2=left_wrist_distances[2],
            left_wrist_to_hold_3=left_wrist_distances[3],
            right_wrist_to_hold_0=right_wrist_distances[0],
            right_wrist_to_hold_1=right_wrist_distances[1],
            right_wrist_to_hold_2=right_wrist_distances[2],
            right_wrist_to_hold_3=right_wrist_distances[3],
        )
        
        features_list.append(features)
    
    return features_list


def features_to_numpy(features_list: List[PoseBasedFrameFeatures]) -> np.ndarray:
    """Convert list of features to numpy array.
    
    Args:
        features_list: List of PoseBasedFrameFeatures
    
    Returns:
        (n_frames, 14) numpy array
    """
    return np.array([f.to_array() for f in features_list], dtype=np.float32)


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features using Z-score normalization.
    
    Args:
        features: (n_frames, n_features) array
        mean: Optional pre-computed mean (if None, compute from features)
        std: Optional pre-computed std (if None, compute from features)
    
    Returns:
        (normalized_features, mean, std)
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
        std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero
    
    normalized = (features - mean) / std
    return normalized, mean, std


def load_hold_positions(hold_positions_path: Path) -> List[Tuple[float, float]]:
    """Load hold positions from JSON file.
    
    Args:
        hold_positions_path: Path to hold_positions.json
    
    Returns:
        List of (x, y) normalized hold center positions
    """
    if not hold_positions_path.exists():
        LOGGER.warning(f"Hold positions file not found: {hold_positions_path}")
        return []
    
    with open(hold_positions_path, 'r') as f:
        data = json.load(f)
    
    holds = data.get('holds', [])
    positions = []
    
    for hold in holds:
        # Assume format: {"x": 0.5, "y": 0.3} or {"center_x": 0.5, "center_y": 0.3}
        x = hold.get('x') or hold.get('center_x', 0.0)
        y = hold.get('y') or hold.get('center_y', 0.0)
        positions.append((x, y))
    
    return positions
