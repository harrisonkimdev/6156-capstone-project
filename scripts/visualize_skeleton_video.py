"""Generate video with skeleton overlay showing load distribution on climber limbs.

This script processes a video frame-by-frame, estimates pose using MediaPipe,
computes load distribution on limbs, and visualizes the skeleton with color-coded
lines representing the load (green=low, yellow=medium, red=high).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add src directory to path for imports
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np

from pose_ai.features.aggregation import compute_center_of_mass
from pose_ai.pose.estimator import PoseEstimator, PoseFrame, PoseLandmark

try:
    from mediapipe.python.solutions.pose import PoseLandmark as MpPoseLandmark, POSE_CONNECTIONS
except ModuleNotFoundError:
    MpPoseLandmark = None
    POSE_CONNECTIONS = []

DEFAULT_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def _get_connections() -> list[tuple[str, str]]:
    """Get pose connections from MediaPipe or use defaults."""
    if MpPoseLandmark is None or not POSE_CONNECTIONS:
        return DEFAULT_CONNECTIONS
    connections = []
    for a_idx, b_idx in POSE_CONNECTIONS:
        connections.append((MpPoseLandmark(a_idx).name.lower(), MpPoseLandmark(b_idx).name.lower()))
    return connections


def _clamp01(value: float) -> float:
    """Clamp value to [0, 1] range."""
    return max(0.0, min(1.0, value))


def _color_from_load(value: float) -> tuple[int, int, int]:
    """Map load 0-1 to green→yellow→red gradient (BGR format for OpenCV)."""
    v = _clamp01(value)
    if v < 0.5:
        # green to yellow
        r = int(2 * v * 255)
        g = 255
    else:
        # yellow to red
        r = 255
        g = int((1 - (v - 0.5) * 2) * 255)
    return (0, g, r)


def _support_points(points: Dict[str, Tuple[int, int]], feat: Optional[Dict] = None) -> list[tuple[int, int]]:
    """Get support points (hands and feet in contact)."""
    keys: list[str] = []
    if feat:
        for limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
            if feat.get(f"{limb}_contact_on") or feat.get(f"{limb}_on_hold"):
                keys.append(limb)
    # Always include feet
    for limb in ("left_foot", "right_foot"):
        if limb not in keys:
            keys.append(limb)
    # Include hands if visible
    for limb in ("left_hand", "right_hand"):
        if limb not in keys and limb in points:
            keys.append(limb)
    return [points[k] for k in keys if k in points]


def _compute_limb_loads(
    points: Dict[str, Tuple[int, int]],
    feat: Optional[Dict] = None,
    width: int = 1,
    height: int = 1,
) -> Dict[str, float]:
    """Estimate limb load from contact + COM projection, then adjust with speed/visibility."""
    default = 0.2
    loads = {k: default for k in ("left_hand", "right_hand", "left_foot", "right_foot")}
    if not feat:
        return loads

    # Base weights: contact flags
    contact_weight = {}
    for limb in loads:
        on_hold = bool(feat.get(f"{limb}_on_hold"))
        contact_on = bool(feat.get(f"{limb}_contact_on"))
        contact_weight[limb] = 1.0 if (on_hold or contact_on) else 0.2

    # Distribute using COM projection distance to supports
    com_x = feat.get("com_x")
    com_y = feat.get("com_y")
    support_pts = _support_points(points, feat)
    if com_x is not None and com_y is not None and support_pts:
        cx = float(com_x) * width
        cy = float(com_y) * height
        weights = []
        keys = []
        for limb, pt in points.items():
            base_name = None
            for l in loads:
                if l.split("_")[0] in limb:
                    base_name = l
                    break
            if base_name is None:
                continue
            dist = math.hypot(cx - pt[0], cy - pt[1]) + 1e-3
            w = 1.0 / dist
            weights.append(w)
            keys.append(base_name)
        total_w = sum(weights) or 1.0
        for k, w in zip(keys, weights):
            loads[k] = 0.4 * (w / total_w) + 0.6 * contact_weight.get(k, 0.2)
    else:
        loads.update({k: max(default, 0.6 * contact_weight.get(k, 0.2)) for k in loads})

    # Dynamic adjustment: speed/visibility
    for limb in loads:
        speed = feat.get(f"{limb}_speed")
        visibility = feat.get(f"{limb}_visibility")
        if speed is not None:
            loads[limb] += min(float(speed) / 2.5, 0.3)
        if visibility is not None:
            loads[limb] = max(loads[limb], float(visibility) * 0.8)
        loads[limb] = _clamp01(loads[limb])

    # Normalize: scale largest load to 1 to avoid over-saturation
    max_load = max(loads.values()) or 1.0
    for limb in loads:
        loads[limb] = _clamp01(loads[limb] / max_load)
    return loads


def _limb_thickness(load: float) -> int:
    """Compute line thickness based on load."""
    return max(2, int(2 + 4 * _clamp01(load)))


def _landmark_to_point(landmark: PoseLandmark, width: int, height: int, min_visibility: float = 0.2) -> Optional[Tuple[int, int]]:
    """Convert normalized landmark to pixel coordinates."""
    if landmark.visibility < min_visibility:
        return None
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return (x, y)


def _extract_feature_row_from_pose(frame: PoseFrame, previous_frame: Optional[PoseFrame] = None, fps: float = 30.0) -> Dict:
    """Extract feature row from pose frame for load computation."""
    feat: Dict = {}
    
    # Compute center of mass
    com = compute_center_of_mass(frame)
    feat["com_x"] = com.get("com_x")
    feat["com_y"] = com.get("com_y")
    
    # Extract limb positions and visibility
    lookup = {lm.name: lm for lm in frame.landmarks}
    
    for limb_name in ("left_hand", "right_hand", "left_foot", "right_foot"):
        landmark_name = {
            "left_hand": "left_wrist",
            "right_hand": "right_wrist",
            "left_foot": "left_ankle",
            "right_foot": "right_ankle",
        }.get(limb_name, limb_name)
        
        landmark = lookup.get(landmark_name)
        if landmark:
            feat[f"{limb_name}_x"] = landmark.x
            feat[f"{limb_name}_y"] = landmark.y
            feat[f"{limb_name}_visibility"] = landmark.visibility
            
            # Compute velocity if previous frame available
            if previous_frame:
                prev_lookup = {lm.name: lm for lm in previous_frame.landmarks}
                prev_landmark = prev_lookup.get(landmark_name)
                if prev_landmark:
                    vx = (landmark.x - prev_landmark.x) * fps
                    vy = (landmark.y - prev_landmark.y) * fps
                    feat[f"{limb_name}_vx"] = vx
                    feat[f"{limb_name}_vy"] = vy
                    feat[f"{limb_name}_speed"] = math.sqrt(vx**2 + vy**2)
        else:
            feat[f"{limb_name}_visibility"] = 0.0
    
    # Simple contact detection: if limb is moving slowly and visible, assume contact
    for limb_name in ("left_hand", "right_hand", "left_foot", "right_foot"):
        speed = feat.get(f"{limb_name}_speed", float("inf"))
        visibility = feat.get(f"{limb_name}_visibility", 0.0)
        # Assume contact if speed is low and visibility is high
        feat[f"{limb_name}_contact_on"] = speed < 0.05 and visibility > 0.5
        feat[f"{limb_name}_on_hold"] = feat[f"{limb_name}_contact_on"]  # Simplified
    
    return feat


def visualize_skeleton_on_video(
    video_path: Path,
    output_path: Path,
    *,
    min_visibility: float = 0.2,
    line_thickness: int = 3,
    point_radius: int = 4,
) -> None:
    """Process video and add skeleton overlay with load visualization.
    
    Args:
        video_path: Input video file path
        output_path: Output video file path
        min_visibility: Minimum landmark visibility to draw
        line_thickness: Base thickness for skeleton lines
        point_radius: Radius for landmark points
    """
    # Initialize pose estimator
    estimator = PoseEstimator(model_complexity="mediapipe-full", smoothing_alpha=0.6)
    estimator.load()
    
    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {video_path.name}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")
    
    # Setup video writer - choose codec based on output file extension
    output_ext = output_path.suffix.lower()
    if output_ext in ('.mov', '.MOV'):
        # Use H.264 codec for MOV files (avc1)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    elif output_ext in ('.mp4', '.MP4'):
        # Use MPEG-4 codec for MP4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        # Default to H.264 for other formats
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_path}")
    
    connections = _get_connections()
    previous_frame: Optional[PoseFrame] = None
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)")
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            result = estimator._engine.process(rgb_frame)
            landmarks = estimator._landmarks_from_result(result)
            
            if landmarks and previous_frame and 0.0 < estimator.smoothing_alpha < 1.0:
                from pose_ai.pose.filters import exponential_smooth
                try:
                    landmarks = exponential_smooth(
                        previous_frame.landmarks,
                        landmarks,
                        estimator.smoothing_alpha
                    )
                except Exception:
                    # If smoothing fails, use original landmarks
                    pass
            
            # Create PoseFrame for feature extraction
            pose_frame = PoseFrame(
                image_path=Path(""),  # Not needed for video processing
                timestamp_seconds=frame_count / fps,
                landmarks=landmarks,
                detection_score=float(np.mean([lm.visibility for lm in landmarks])) if landmarks else 0.0,
            )
            
            # Extract features for load computation
            feat = _extract_feature_row_from_pose(pose_frame, previous_frame, fps)
            
            # Convert landmarks to pixel coordinates
            points: Dict[str, Tuple[int, int]] = {}
            for landmark in landmarks:
                pt = _landmark_to_point(landmark, width, height, min_visibility)
                if pt:
                    points[landmark.name] = pt
            
            # Compute limb loads
            limb_loads = _compute_limb_loads(points, feat, width, height)
            
            # Draw skeleton connections
            for start, end in connections:
                if start in points and end in points:
                    # Determine which limb this connection belongs to
                    limb_key = None
                    for limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
                        if limb.split("_")[0] in start or limb.split("_")[0] in end:
                            limb_key = limb
                            break
                    
                    load_val = limb_loads.get(limb_key, 0.3) if limb_key else 0.3
                    color = _color_from_load(load_val)
                    thickness = max(2, line_thickness + int(2 * load_val))
                    cv2.line(frame, points[start], points[end], color, thickness, lineType=cv2.LINE_AA)
            
            # Draw landmark points
            for name, (x, y) in points.items():
                limb_key = None
                for limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
                    if limb.split("_")[0] in name:
                        limb_key = limb
                        break
                
                load_val = limb_loads.get(limb_key, 0.3) if limb_key else 0.3
                color = _color_from_load(load_val)
                cv2.circle(frame, (x, y), point_radius, color, -1)
            
            # Write frame to output video
            out.write(frame)
            
            previous_frame = pose_frame
    
    finally:
        cap.release()
        out.release()
        estimator.close()
    
    print(f"Completed! Output saved to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate video with skeleton overlay showing load distribution."
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Input video file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video file path (default: input_name_skeleton.MOV)",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.2,
        help="Minimum landmark visibility to draw (default: 0.2)",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=3,
        help="Base thickness for skeleton lines (default: 3)",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=4,
        help="Radius for landmark points (default: 4)",
    )
    return parser


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if args.output:
        output_path = args.output.expanduser().resolve()
    else:
        # Default to MOV format
        output_path = video_path.parent / f"{video_path.stem}_skeleton.MOV"
    
    visualize_skeleton_on_video(
        video_path,
        output_path,
        min_visibility=args.min_visibility,
        line_thickness=args.line_thickness,
        point_radius=args.point_radius,
    )


if __name__ == "__main__":
    main()
