"""Hold tracking across video frames using IoU matching and Kalman filtering.

This module provides temporal tracking for hold detections, improving stability
and reducing fragmentation compared to frame-by-frame clustering.

Pipeline:
1. Frame-by-frame detection (from hold_extraction.py)
2. IoU-based matching of detections to existing tracks
3. Kalman filter prediction and update for each track
4. Track management (creation, confirmation, deletion)
5. Final clustering of confirmed tracks

TODO (Future Enhancements):
- Visual feature extraction (ResNet18/34 embeddings) for re-identification
- Multi-hypothesis tracking (MHT) for handling ambiguous associations
- Track splitting/merging logic for holds that separate or combine
- Adaptive IoU threshold based on track confidence and detection quality
- 3D tracking if depth information becomes available
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Sequence

import numpy as np


@dataclass(slots=True)
class HoldTrack:
    """Represents a tracked hold across multiple frames.
    
    Attributes:
        track_id: Unique identifier for this track
        kalman_state: Current Kalman filter state [x, y, vx, vy]
        kalman_covariance: State covariance matrix (4x4)
        bbox: Current bounding box [x_center, y_center, width, height] (normalized)
        label: Hold label (e.g., "hold", "jug", "crimp")
        hold_type: Specific hold type if available
        age: Number of frames since track was created
        hits: Number of times this track was matched with a detection
        misses: Number of consecutive frames without a detection match
        history: List of (frame_idx, x, y, confidence) tuples
    """
    track_id: int
    kalman_state: np.ndarray  # [x, y, vx, vy]
    kalman_covariance: np.ndarray  # 4x4 covariance matrix
    bbox: Tuple[float, float, float, float]  # [x_center, y_center, width, height]
    label: str
    hold_type: str | None = None
    age: int = 0
    hits: int = 0
    misses: int = 0
    history: List[Tuple[int, float, float, float]] = field(default_factory=list)
    
    def get_predicted_bbox(self) -> Tuple[float, float, float, float]:
        """Get predicted bounding box based on Kalman state."""
        x, y = self.kalman_state[0], self.kalman_state[1]
        w, h = self.bbox[2], self.bbox[3]
        return (x, y, w, h)
    
    def is_confirmed(self, min_hits: int = 3) -> bool:
        """Check if track is confirmed (enough detections)."""
        return self.hits >= min_hits
    
    def should_delete(self, max_age: int = 5) -> bool:
        """Check if track should be deleted (too many misses)."""
        return self.misses >= max_age


class KalmanFilter2D:
    """Simple 2D Kalman filter for hold position and velocity tracking.
    
    State: [x, y, vx, vy]
    - x, y: Position (normalized coordinates 0-1)
    - vx, vy: Velocity (change in position per frame)
    
    Motion model: Constant velocity (linear motion)
    - x(t+1) = x(t) + vx(t) * dt
    - y(t+1) = y(t) + vy(t) * dt
    - vx(t+1) = vx(t)
    - vy(t+1) = vy(t)
    
    TODO (Future Enhancements):
    - Adaptive process noise based on detection quality
    - Constant acceleration model for dynamic holds (moving walls, adjustable holds)
    - Outlier rejection using Mahalanobis distance
    - State augmentation with bbox size tracking
    """
    
    def __init__(
        self,
        initial_position: Tuple[float, float],
        process_noise: float = 0.1,
        measurement_noise: float = 0.5,
    ):
        """Initialize Kalman filter.
        
        Args:
            initial_position: Initial (x, y) position
            process_noise: Process noise standard deviation (motion uncertainty)
            measurement_noise: Measurement noise standard deviation (detection uncertainty)
        """
        # State: [x, y, vx, vy]
        self.state = np.array([initial_position[0], initial_position[1], 0.0, 0.0], dtype=np.float32)
        
        # State covariance: Initial high uncertainty in velocity
        self.covariance = np.eye(4, dtype=np.float32)
        self.covariance[2, 2] = 10.0  # High initial velocity uncertainty
        self.covariance[3, 3] = 10.0
        
        # State transition matrix (constant velocity model)
        # x' = x + vx*dt, y' = y + vy*dt, vx' = vx, vy' = vy (dt=1)
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1],  # vy = vy
        ], dtype=np.float32)
        
        # Measurement matrix: We observe position only [x, y]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        
        # Process noise covariance (motion uncertainty)
        # TODO: Make this adaptive based on hold type or detection confidence
        q = process_noise ** 2
        self.Q = np.array([
            [q, 0, 0, 0],
            [0, q, 0, 0],
            [0, 0, q, 0],
            [0, 0, 0, q],
        ], dtype=np.float32)
        
        # Measurement noise covariance (detection uncertainty)
        # TODO: Adapt based on detection confidence score
        r = measurement_noise ** 2
        self.R = np.array([
            [r, 0],
            [0, r],
        ], dtype=np.float32)
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state using motion model.
        
        Returns:
            (predicted_state, predicted_covariance)
        """
        # Predict state: x' = F @ x
        self.state = self.F @ self.state
        
        # Predict covariance: P' = F @ P @ F^T + Q
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        return self.state.copy(), self.covariance.copy()
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Update state with new measurement.
        
        Args:
            measurement: Observed (x, y) position
        
        Returns:
            (updated_state, updated_covariance)
        """
        z = np.array(measurement, dtype=np.float32)
        
        # Innovation: y = z - H @ x
        y = z - self.H @ self.state
        
        # Innovation covariance: S = H @ P @ H^T + R
        S = self.H @ self.covariance @ self.H.T + self.R
        
        # Kalman gain: K = P @ H^T @ S^-1
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K @ y
        self.state = self.state + K @ y
        
        # Update covariance: P = (I - K @ H) @ P
        I = np.eye(4, dtype=np.float32)
        self.covariance = (I - K @ self.H) @ self.covariance
        
        return self.state.copy(), self.covariance.copy()
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position estimate."""
        return float(self.state[0]), float(self.state[1])


class IoUTracker:
    """IoU-based tracker for hold detections across frames.
    
    This tracker uses Intersection over Union (IoU) to match detections
    to existing tracks, combined with Kalman filtering for state estimation.
    
    TODO (Future Enhancements):
    - Visual feature extraction and matching for re-identification after occlusion
    - Multi-hypothesis tracking to handle ambiguous detections
    - Track splitting when a hold is detected as multiple separate holds
    - Track merging when multiple tracks converge to same physical hold
    - Adaptive IoU threshold based on track confidence and age
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        max_age: int = 5,
        min_hits: int = 3,
        process_noise: float = 0.1,
        measurement_noise: float = 0.5,
    ):
        """Initialize tracker.
        
        Args:
            iou_threshold: Minimum IoU for detection-track matching
            max_age: Maximum frames without detection before track deletion
            min_hits: Minimum detections before track is confirmed
            process_noise: Kalman filter process noise
            measurement_noise: Kalman filter measurement noise
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        self.tracks: List[HoldTrack] = []
        self.next_track_id = 0
    
    @staticmethod
    def compute_iou(bbox1: Tuple[float, float, float, float], 
                     bbox2: Tuple[float, float, float, float]) -> float:
        """Compute Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bbox as (x_center, y_center, width, height)
            bbox2: Second bbox as (x_center, y_center, width, height)
        
        Returns:
            IoU score in [0, 1]
        """
        # Convert from center format to corner format
        x1_min = bbox1[0] - bbox1[2] / 2
        y1_min = bbox1[1] - bbox1[3] / 2
        x1_max = bbox1[0] + bbox1[2] / 2
        y1_max = bbox1[1] + bbox1[3] / 2
        
        x2_min = bbox2[0] - bbox2[2] / 2
        y2_min = bbox2[1] - bbox2[3] / 2
        x2_max = bbox2[0] + bbox2[2] / 2
        y2_max = bbox2[1] + bbox2[3] / 2
        
        # Compute intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_width = max(0.0, inter_x_max - inter_x_min)
        inter_height = max(0.0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height
        
        # Compute union
        box1_area = bbox1[2] * bbox1[3]
        box2_area = bbox2[2] * bbox2[3]
        union_area = box1_area + box2_area - inter_area
        
        # Compute IoU
        if union_area <= 0:
            return 0.0
        return inter_area / union_area
    
    def match_detections_to_tracks(
        self,
        detections: List[Tuple[float, float, float, float]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to existing tracks using IoU.
        
        Uses a greedy matching algorithm: for each detection, find the track
        with highest IoU above threshold.
        
        TODO: Implement Hungarian algorithm for optimal assignment
        
        Args:
            detections: List of detection bboxes
        
        Returns:
            (matches, unmatched_detections, unmatched_tracks)
            - matches: List of (detection_idx, track_idx) pairs
            - unmatched_detections: Indices of detections not matched
            - unmatched_tracks: Indices of tracks not matched
        """
        if not detections or not self.tracks:
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self.tracks)))
            return [], unmatched_dets, unmatched_trks
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(self.tracks)), dtype=np.float32)
        for d_idx, det_bbox in enumerate(detections):
            for t_idx, track in enumerate(self.tracks):
                track_bbox = track.get_predicted_bbox()
                iou_matrix[d_idx, t_idx] = self.compute_iou(det_bbox, track_bbox)
        
        # Greedy matching (TODO: use Hungarian algorithm)
        matches = []
        matched_dets = set()
        matched_trks = set()
        
        # Sort by IoU (highest first)
        flat_indices = np.argsort(-iou_matrix.ravel())
        for flat_idx in flat_indices:
            d_idx = flat_idx // len(self.tracks)
            t_idx = flat_idx % len(self.tracks)
            
            if d_idx in matched_dets or t_idx in matched_trks:
                continue
            
            iou = iou_matrix[d_idx, t_idx]
            if iou >= self.iou_threshold:
                matches.append((d_idx, t_idx))
                matched_dets.add(d_idx)
                matched_trks.add(t_idx)
        
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_trks = [i for i in range(len(self.tracks)) if i not in matched_trks]
        
        return matches, unmatched_dets, unmatched_trks
    
    def create_new_track(
        self,
        detection_bbox: Tuple[float, float, float, float],
        label: str,
        hold_type: str | None,
        frame_idx: int,
        confidence: float,
    ) -> HoldTrack:
        """Create a new track from a detection.
        
        Args:
            detection_bbox: Detection bbox (x_center, y_center, width, height)
            label: Hold label
            hold_type: Specific hold type if available
            frame_idx: Current frame index
            confidence: Detection confidence
        
        Returns:
            New HoldTrack object
        """
        # Initialize Kalman filter at detection position
        kalman = KalmanFilter2D(
            initial_position=(detection_bbox[0], detection_bbox[1]),
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
        )
        
        track = HoldTrack(
            track_id=self.next_track_id,
            kalman_state=kalman.state.copy(),
            kalman_covariance=kalman.covariance.copy(),
            bbox=detection_bbox,
            label=label,
            hold_type=hold_type,
            age=1,
            hits=1,
            misses=0,
            history=[(frame_idx, detection_bbox[0], detection_bbox[1], confidence)],
        )
        
        self.next_track_id += 1
        return track
    
    def update_tracks(
        self,
        frame_idx: int,
        detections: List[Tuple[Tuple[float, float, float, float], str, str | None, float]],
    ) -> None:
        """Update tracks with new detections for current frame.
        
        Steps:
        1. Predict all track positions using Kalman filter
        2. Match detections to tracks using IoU
        3. Update matched tracks with Kalman filter
        4. Create new tracks for unmatched detections
        5. Increment age/misses for unmatched tracks
        6. Prune old tracks
        
        Args:
            frame_idx: Current frame index
            detections: List of (bbox, label, hold_type, confidence) tuples
        """
        # Step 1: Predict all tracks
        for track in self.tracks:
            kalman = KalmanFilter2D(
                initial_position=(0, 0),  # Will be overwritten
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
            )
            kalman.state = track.kalman_state
            kalman.covariance = track.kalman_covariance
            
            # Predict
            state, cov = kalman.predict()
            track.kalman_state = state
            track.kalman_covariance = cov
            track.age += 1
        
        # Step 2: Match detections to tracks
        det_bboxes = [det[0] for det in detections]
        matches, unmatched_dets, unmatched_trks = self.match_detections_to_tracks(det_bboxes)
        
        # Step 3: Update matched tracks
        for det_idx, trk_idx in matches:
            track = self.tracks[trk_idx]
            det_bbox, label, hold_type, confidence = detections[det_idx]
            
            # Update Kalman filter
            kalman = KalmanFilter2D(
                initial_position=(0, 0),
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
            )
            kalman.state = track.kalman_state
            kalman.covariance = track.kalman_covariance
            
            state, cov = kalman.update((det_bbox[0], det_bbox[1]))
            track.kalman_state = state
            track.kalman_covariance = cov
            track.bbox = det_bbox
            track.hits += 1
            track.misses = 0
            
            # Update label/type if detection provides more specific info
            if hold_type:
                track.hold_type = hold_type
            if label:
                track.label = label
            
            # Add to history
            track.history.append((frame_idx, det_bbox[0], det_bbox[1], confidence))
        
        # Step 4: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det_bbox, label, hold_type, confidence = detections[det_idx]
            new_track = self.create_new_track(det_bbox, label, hold_type, frame_idx, confidence)
            self.tracks.append(new_track)
        
        # Step 5: Increment misses for unmatched tracks
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].misses += 1
        
        # Step 6: Prune old tracks
        self.prune_tracks()
    
    def prune_tracks(self) -> None:
        """Remove tracks that have too many consecutive misses."""
        self.tracks = [t for t in self.tracks if not t.should_delete(self.max_age)]
    
    def get_confirmed_tracks(self) -> List[HoldTrack]:
        """Get only confirmed tracks (with enough hits).
        
        Returns:
            List of confirmed HoldTrack objects
        """
        return [t for t in self.tracks if t.is_confirmed(self.min_hits)]


__all__ = [
    "HoldTrack",
    "KalmanFilter2D",
    "IoUTracker",
]

