"""Dataset builder for BiLSTM multitask model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import json

try:
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False
    Dataset = object  # type: ignore
    torch = None  # type: ignore
    np = None  # type: ignore


# Selected joints for feature extraction
SELECTED_JOINTS = [
    "left_wrist", "right_wrist",
    "left_ankle", "right_ankle",
    "left_hip", "right_hip",
    "left_shoulder", "right_shoulder",
]

# Contact limbs
CONTACT_LIMBS = ["left_hand", "right_hand", "left_foot", "right_foot"]


@dataclass(slots=True)
class WindowSample:
    """A single sliding window sample."""
    video_id: str
    start_frame: int
    end_frame: int
    features: "np.ndarray"  # Shape: (T, feature_dim)
    efficiency_label: float
    next_action_label: int  # Class ID


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value, default: int = 0) -> int:
    """Safely convert value to int."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def extract_features_from_row(row: Dict[str, object]) -> List[float]:
    """Extract feature vector from a single feature row.
    
    Features:
    - Joint positions (x, y) for selected joints: 16 features
    - Joint velocities (vx, vy) for selected joints: 16 features
    - Joint accelerations (ax, ay) for selected joints: 16 features
    - COM position and velocity: 4 features
    - Contact states (on/off for each limb): 4 features
    - Support count: 1 feature
    - Body scale: 1 feature
    - Wall distance: 1 feature
    - Efficiency (if available): 1 feature
    
    Total: 60 features
    """
    features = []
    
    # Joint positions (16 features)
    for joint in SELECTED_JOINTS:
        features.append(_safe_float(row.get(f"{joint}_x")))
        features.append(_safe_float(row.get(f"{joint}_y")))
    
    # Joint velocities (16 features)
    for joint in SELECTED_JOINTS:
        features.append(_safe_float(row.get(f"{joint}_vx")))
        features.append(_safe_float(row.get(f"{joint}_vy")))
    
    # Joint accelerations (16 features)
    for joint in SELECTED_JOINTS:
        features.append(_safe_float(row.get(f"{joint}_ax")))
        features.append(_safe_float(row.get(f"{joint}_ay")))
    
    # COM position and velocity (4 features)
    features.append(_safe_float(row.get("com_x")))
    features.append(_safe_float(row.get("com_y")))
    features.append(_safe_float(row.get("com_vx")))
    features.append(_safe_float(row.get("com_vy")))
    
    # Contact states (4 features)
    for limb in CONTACT_LIMBS:
        contact_on = row.get(f"{limb}_contact_on")
        features.append(1.0 if contact_on else 0.0)
    
    # Support count (1 feature)
    support_count = sum(1 for limb in CONTACT_LIMBS if row.get(f"{limb}_contact_on"))
    features.append(float(support_count))
    
    # Body scale (1 feature)
    features.append(_safe_float(row.get("body_scale"), 1.0))
    
    # Wall distance (1 feature)
    features.append(_safe_float(row.get("wall_distance")))
    
    # Efficiency (1 feature)
    features.append(_safe_float(row.get("efficiency")))
    
    return features


def extract_next_action_label(
    rows: List[Dict[str, object]],
    current_idx: int,
    lookahead: int = 5,
) -> int:
    """Extract next-action label from future contact changes.
    
    Labels:
    - 0: no_change (no new contact in lookahead)
    - 1: left_hand
    - 2: right_hand
    - 3: left_foot
    - 4: right_foot
    """
    if current_idx + lookahead >= len(rows):
        return 0  # no_change
    
    current_contacts = {
        limb: rows[current_idx].get(f"{limb}_contact_hold")
        for limb in CONTACT_LIMBS
    }
    
    # Check for first contact change in lookahead window
    for i in range(current_idx + 1, min(current_idx + lookahead + 1, len(rows))):
        for limb_idx, limb in enumerate(CONTACT_LIMBS):
            future_hold = rows[i].get(f"{limb}_contact_hold")
            if future_hold != current_contacts[limb] and future_hold is not None:
                return limb_idx + 1  # 1-4 for limbs
    
    return 0  # no_change


class ClimbingWindowDataset(Dataset):
    """Sliding window dataset for BiLSTM training."""
    
    def __init__(
        self,
        feature_rows: List[Dict[str, object]],
        *,
        window_size: int = 32,
        stride: int = 1,
        video_id: str = "unknown",
        normalize: bool = True,
        lookahead_frames: int = 5,
    ):
        """Initialize dataset.
        
        Args:
            feature_rows: List of feature dictionaries (from JSON export)
            window_size: Number of frames per window (T)
            stride: Stride for sliding window
            video_id: Video identifier
            normalize: Whether to z-score normalize features
            lookahead_frames: Frames to look ahead for next-action labels
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for dataset. Install with: pip install torch")
        
        self.window_size = window_size
        self.stride = stride
        self.video_id = video_id
        self.lookahead_frames = lookahead_frames
        
        # Extract features for all frames
        self.feature_matrix = []
        self.efficiency_labels = []
        
        for row in feature_rows:
            features = extract_features_from_row(row)
            self.feature_matrix.append(features)
            self.efficiency_labels.append(_safe_float(row.get("efficiency")))
        
        self.feature_matrix = np.array(self.feature_matrix, dtype=np.float32)
        self.efficiency_labels = np.array(self.efficiency_labels, dtype=np.float32)
        
        # Normalize features (z-score per feature dimension)
        if normalize and len(self.feature_matrix) > 0:
            self.feature_mean = self.feature_matrix.mean(axis=0)
            self.feature_std = self.feature_matrix.std(axis=0) + 1e-8
            self.feature_matrix = (self.feature_matrix - self.feature_mean) / self.feature_std
        else:
            self.feature_mean = None
            self.feature_std = None
        
        # Store feature rows for next-action label extraction
        self.feature_rows = feature_rows
        
        # Create window indices
        self.window_indices = []
        for i in range(0, len(self.feature_matrix) - window_size + 1, stride):
            self.window_indices.append(i)
    
    def __len__(self) -> int:
        return len(self.window_indices)
    
    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", float, int]:
        """Get a single window sample.
        
        Returns:
            (features, efficiency_label, next_action_label)
            - features: Tensor of shape (T, feature_dim)
            - efficiency_label: float
            - next_action_label: int (0-4)
        """
        start_idx = self.window_indices[idx]
        end_idx = start_idx + self.window_size
        
        # Extract window features
        window_features = self.feature_matrix[start_idx:end_idx]
        
        # Efficiency label: average over window
        window_efficiency = self.efficiency_labels[start_idx:end_idx].mean()
        
        # Next-action label: from the end of window
        next_action_label = extract_next_action_label(
            self.feature_rows,
            end_idx - 1,
            self.lookahead_frames
        )
        
        return (
            torch.from_numpy(window_features),
            float(window_efficiency),
            int(next_action_label),
        )
    
    def get_normalization_params(self) -> Tuple["np.ndarray | None", "np.ndarray | None"]:
        """Get normalization parameters for inference."""
        return self.feature_mean, self.feature_std


def load_features_from_json(json_path: Path) -> List[Dict[str, object]]:
    """Load feature rows from JSON export."""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "features" in data:
        return data["features"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected JSON format in {json_path}")


def create_datasets_from_directory(
    features_dir: Path,
    *,
    window_size: int = 32,
    stride: int = 1,
    train_split: float = 0.7,
    val_split: float = 0.2,
    normalize: bool = True,
) -> Tuple[ClimbingWindowDataset | None, ClimbingWindowDataset | None, ClimbingWindowDataset | None]:
    """Create train/val/test datasets from a directory of feature JSON files.
    
    Args:
        features_dir: Directory containing feature JSON files
        window_size: Window size for sliding windows
        stride: Stride for sliding windows
        train_split: Fraction for training set
        val_split: Fraction for validation set
        normalize: Whether to normalize features
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    
    json_files = sorted(features_dir.glob("*.json"))
    if not json_files:
        return None, None, None
    
    # Combine all feature rows from all videos
    all_rows = []
    for json_path in json_files:
        try:
            rows = load_features_from_json(json_path)
            all_rows.extend(rows)
        except Exception as e:
            print(f"Warning: Failed to load {json_path}: {e}")
    
    if not all_rows:
        return None, None, None
    
    # Split into train/val/test
    total = len(all_rows)
    train_end = int(total * train_split)
    val_end = int(total * (train_split + val_split))
    
    train_rows = all_rows[:train_end]
    val_rows = all_rows[train_end:val_end]
    test_rows = all_rows[val_end:]
    
    # Create datasets
    train_dataset = ClimbingWindowDataset(
        train_rows,
        window_size=window_size,
        stride=stride,
        video_id="train",
        normalize=normalize,
    ) if train_rows else None
    
    val_dataset = ClimbingWindowDataset(
        val_rows,
        window_size=window_size,
        stride=stride,
        video_id="val",
        normalize=False,  # Use train stats
    ) if val_rows else None
    
    test_dataset = ClimbingWindowDataset(
        test_rows,
        window_size=window_size,
        stride=stride,
        video_id="test",
        normalize=False,  # Use train stats
    ) if test_rows else None
    
    # Apply train normalization to val/test
    if train_dataset and train_dataset.feature_mean is not None:
        if val_dataset:
            val_dataset.feature_matrix = (val_dataset.feature_matrix - train_dataset.feature_mean) / train_dataset.feature_std
            val_dataset.feature_mean = train_dataset.feature_mean
            val_dataset.feature_std = train_dataset.feature_std
        if test_dataset:
            test_dataset.feature_matrix = (test_dataset.feature_matrix - train_dataset.feature_mean) / train_dataset.feature_std
            test_dataset.feature_mean = train_dataset.feature_mean
            test_dataset.feature_std = train_dataset.feature_std
    
    return train_dataset, val_dataset, test_dataset


__all__ = [
    "ClimbingWindowDataset",
    "WindowSample",
    "extract_features_from_row",
    "extract_next_action_label",
    "load_features_from_json",
    "create_datasets_from_directory",
]

