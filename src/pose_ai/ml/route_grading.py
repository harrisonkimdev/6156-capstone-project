"""Route difficulty estimation using hold density, wall angle, and move complexity."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pose_ai.recommendation.efficiency import StepEfficiencyResult
from pose_ai.segmentation.steps import StepSegment

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False
    xgb = None  # type: ignore


def _safe_float(value) -> float | None:
    """Safely convert value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def extract_route_features(
    feature_rows: Sequence[Dict[str, object]],
    step_segments: Sequence[StepSegment],
    step_efficiency: Sequence[StepEfficiencyResult],
    holds: Sequence[Dict[str, object]],
    wall_angle: float | None,
) -> Dict[str, float]:
    """Extract route-level features for difficulty prediction.
    
    Args:
        feature_rows: Frame-level feature rows
        step_segments: Step segmentation results
        step_efficiency: Step efficiency scores
        holds: Detected holds with positions and types
        wall_angle: Wall angle in degrees (from IMU or vision)
    
    Returns:
        Dictionary of route features for difficulty prediction
    """
    features: Dict[str, float] = {}
    
    # 1. Hold density
    if holds:
        # Compute bounding box of all holds
        hold_positions = []
        for hold in holds:
            coords = hold.get("coords")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                hold_positions.append((float(coords[0]), float(coords[1])))
        
        if hold_positions:
            xs = [p[0] for p in hold_positions]
            ys = [p[1] for p in hold_positions]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Estimate wall area (normalized coordinates, assume square meters)
            width = max_x - min_x
            height = max_y - min_y
            wall_area = width * height if width > 0 and height > 0 else 1.0
            
            hold_count = len(hold_positions)
            features["hold_density"] = hold_count / (wall_area + 1e-6)
            features["hold_count"] = float(hold_count)
        else:
            features["hold_density"] = 0.0
            features["hold_count"] = 0.0
    else:
        features["hold_density"] = 0.0
        features["hold_count"] = 0.0
    
    # 2. Hold spacing statistics
    if holds and len(holds) > 1:
        hold_positions = []
        for hold in holds:
            coords = hold.get("coords")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                hold_positions.append((float(coords[0]), float(coords[1])))
        
        if len(hold_positions) > 1:
            # Compute pairwise distances (filter by vertical progression)
            distances = []
            for i in range(len(hold_positions)):
                for j in range(i + 1, len(hold_positions)):
                    p1, p2 = hold_positions[i], hold_positions[j]
                    # Only consider upward moves (p2 higher than p1 in normalized coords)
                    if p2[1] < p1[1]:  # Lower y = higher on wall
                        dist = _distance(p1, p2)
                        distances.append(dist)
            
            if distances:
                distances_array = np.array(distances)
                features["hold_spacing_mean"] = float(np.mean(distances_array))
                features["hold_spacing_median"] = float(np.median(distances_array))
                features["hold_spacing_std"] = float(np.std(distances_array))
                features["hold_spacing_min"] = float(np.min(distances_array))
                features["hold_spacing_max"] = float(np.max(distances_array))
            else:
                features["hold_spacing_mean"] = 0.0
                features["hold_spacing_median"] = 0.0
                features["hold_spacing_std"] = 0.0
                features["hold_spacing_min"] = 0.0
                features["hold_spacing_max"] = 0.0
        else:
            features["hold_spacing_mean"] = 0.0
            features["hold_spacing_median"] = 0.0
            features["hold_spacing_std"] = 0.0
            features["hold_spacing_min"] = 0.0
            features["hold_spacing_max"] = 0.0
    else:
        features["hold_spacing_mean"] = 0.0
        features["hold_spacing_median"] = 0.0
        features["hold_spacing_std"] = 0.0
        features["hold_spacing_min"] = 0.0
        features["hold_spacing_max"] = 0.0
    
    # 3. Wall angle
    if wall_angle is not None:
        features["wall_angle"] = float(wall_angle)
    else:
        # Try to extract from feature rows
        wall_angles = []
        for row in feature_rows:
            angle = _safe_float(row.get("wall_angle"))
            if angle is not None:
                wall_angles.append(angle)
        if wall_angles:
            features["wall_angle"] = float(np.mean(wall_angles))
        else:
            features["wall_angle"] = 90.0  # Default vertical
    
    # 4. Move complexity
    features["step_count"] = float(len(step_segments))
    
    if step_efficiency:
        efficiency_scores = [step.score for step in step_efficiency]
        features["avg_efficiency"] = float(np.mean(efficiency_scores))
        features["min_efficiency"] = float(np.min(efficiency_scores))
        features["max_efficiency"] = float(np.max(efficiency_scores))
        features["efficiency_std"] = float(np.std(efficiency_scores))
    else:
        features["avg_efficiency"] = 0.0
        features["min_efficiency"] = 0.0
        features["max_efficiency"] = 0.0
        features["efficiency_std"] = 0.0
    
    # Reach statistics from efficiency components
    reach_penalties = []
    for step in step_efficiency:
        reach_pen = step.components.get("reach_penalty", 0.0)
        if isinstance(reach_pen, (int, float)):
            reach_penalties.append(float(reach_pen))
    
    if reach_penalties:
        features["avg_reach_penalty"] = float(np.mean(reach_penalties))
        features["max_reach_penalty"] = float(np.max(reach_penalties))
    else:
        features["avg_reach_penalty"] = 0.0
        features["max_reach_penalty"] = 0.0
    
    # Contact switch frequency
    if step_segments and feature_rows:
        total_switches = 0
        total_duration = 0.0
        for segment in step_segments:
            segment_frames = feature_rows[segment.start_index:segment.end_index + 1]
            if len(segment_frames) > 1:
                switches = 0
                for i in range(1, len(segment_frames)):
                    prev_row = segment_frames[i - 1]
                    curr_row = segment_frames[i]
                    for limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
                        prev_hold = prev_row.get(f"{limb}_contact_hold")
                        curr_hold = curr_row.get(f"{limb}_contact_hold")
                        if prev_hold != curr_hold and (prev_hold is not None or curr_hold is not None):
                            switches += 1
                total_switches += switches
                total_duration += segment.duration
        features["contact_switches_per_second"] = total_switches / (total_duration + 1e-6)
    else:
        features["contact_switches_per_second"] = 0.0
    
    # 5. Hold type distribution
    hold_types = {
        "jug": 0,
        "crimp": 0,
        "sloper": 0,
        "pinch": 0,
        "foot_only": 0,
        "volume": 0,
        "unknown": 0,
    }
    
    total_typed_holds = 0
    for hold in holds:
        hold_type = hold.get("hold_type") or hold.get("type", "unknown")
        if isinstance(hold_type, str):
            hold_type = hold_type.lower()
            if hold_type in hold_types:
                hold_types[hold_type] += 1
                total_typed_holds += 1
            else:
                hold_types["unknown"] += 1
    
    if total_typed_holds > 0:
        for hold_type, count in hold_types.items():
            features[f"hold_type_ratio_{hold_type}"] = float(count) / total_typed_holds
    else:
        for hold_type in hold_types:
            features[f"hold_type_ratio_{hold_type}"] = 0.0
    
    # 6. Route length (vertical distance)
    if feature_rows:
        com_ys = []
        for row in feature_rows:
            com_y = _safe_float(row.get("com_y"))
            if com_y is not None:
                com_ys.append(com_y)
        
        if len(com_ys) > 1:
            min_y = min(com_ys)
            max_y = max(com_ys)
            # In normalized coords, lower y = higher on wall
            route_length = max_y - min_y
            features["route_length"] = float(route_length)
        else:
            features["route_length"] = 0.0
    else:
        features["route_length"] = 0.0
    
    # 7. Duration
    if feature_rows:
        timestamps = []
        for row in feature_rows:
            ts = _safe_float(row.get("timestamp"))
            if ts is not None:
                timestamps.append(ts)
        
        if len(timestamps) > 1:
            duration = max(timestamps) - min(timestamps)
            features["duration_seconds"] = float(duration)
        else:
            features["duration_seconds"] = 0.0
    else:
        features["duration_seconds"] = 0.0
    
    return features


class RouteDifficultyModel:
    """XGBoost regressor for route difficulty (V0-V10)."""
    
    def __init__(self, model_path: Path | None = None):
        """Initialize model.
        
        Args:
            model_path: Path to trained XGBoost model (.json or .ubj)
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        
        if model_path and model_path.exists():
            self.model = xgb.XGBRegressor()
            self.model.load_model(str(model_path))
            self.feature_names = self.model.get_booster().feature_names
        else:
            # Initialize untrained model
            self.model = xgb.XGBRegressor(
                objective="reg:squarederror",
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42,
            )
            self.feature_names = None
    
    def predict(self, features: Dict[str, float]) -> float:
        """Predict route grade (V0-V10).
        
        Args:
            features: Route features dictionary
        
        Returns:
            Predicted grade (V0-V10)
        """
        # Convert features to array in correct order
        if self.feature_names:
            feature_array = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        else:
            # Use all features in alphabetical order
            feature_array = np.array([[features.get(name, 0.0) for name in sorted(features.keys())]])
        
        prediction = self.model.predict(feature_array)[0]
        # Clamp to valid range
        return max(0.0, min(10.0, float(prediction)))
    
    def predict_with_confidence(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Predict grade with confidence estimate.
        
        Args:
            features: Route features dictionary
        
        Returns:
            (predicted_grade, confidence) where confidence is 0-1
        """
        grade = self.predict(features)
        
        # Simple confidence: based on feature completeness
        # More complete features = higher confidence
        non_zero_features = sum(1 for v in features.values() if v != 0.0)
        total_features = len(features)
        confidence = min(1.0, non_zero_features / max(1, total_features * 0.5))
        
        return grade, confidence


class GymGradeCalibration:
    """Map predicted grade to gym-specific scale."""
    
    # Default V-scale to category mapping
    DEFAULT_MAPPING = {
        (0.0, 2.0): "Beginner",
        (2.0, 4.0): "Intermediate",
        (4.0, 6.0): "Advanced",
        (6.0, 8.0): "Expert",
        (8.0, 10.0): "Elite",
    }
    
    def __init__(self, gym_name: str = "default"):
        """Initialize calibration.
        
        Args:
            gym_name: Gym name for custom mapping (default uses V-scale categories)
        """
        self.gym_name = gym_name
        # TODO: Load gym-specific mapping from config file if needed
        self.mapping = self.DEFAULT_MAPPING
    
    def calibrate(self, predicted_grade: float) -> str:
        """Convert V0-V10 to gym scale.
        
        Args:
            predicted_grade: Predicted grade (V0-V10)
        
        Returns:
            Calibrated grade string
        """
        for (min_grade, max_grade), category in self.mapping.items():
            if min_grade <= predicted_grade < max_grade:
                return f"{category} (V{int(predicted_grade)})"
        
        # Fallback
        if predicted_grade >= 10.0:
            return "Elite (V10+)"
        return f"V{int(predicted_grade)}"


__all__ = [
    "extract_route_features",
    "RouteDifficultyModel",
    "GymGradeCalibration",
]

