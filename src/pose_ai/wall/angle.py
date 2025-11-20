"""Wall angle estimation utilities.

Approach:
1. Select a representative frame (low motion / early frame) supplied as path.
2. Detect climber / dynamic objects (optional mask) – caller may pass a binary mask.
3. Edge detection (Canny) and Hough line transform to find dominant wall lines.
4. Optionally incorporate hold centers (from YOLO detections) using PCA on their (x, y) to refine angle.
5. Return wall angle in degrees relative to the horizontal axis (0° = perfectly horizontal line; vertical wall ≈ 90°).

The returned payload includes a confidence score based on line count agreement.

NOTE: This is a heuristic first pass; later improvements can add lens distortion correction and RANSAC plane fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np


@dataclass(slots=True)
class WallAngleResult:
    angle_degrees: float | None
    confidence: float
    method: str
    hough_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    pca_angle: float | None = None

    def as_dict(self) -> dict[str, object]:  # pragma: no cover - convenience
        return {
            "angle_degrees": self.angle_degrees,
            "confidence": self.confidence,
            "method": self.method,
            "pca_angle": self.pca_angle,
            "hough_line_count": len(self.hough_lines),
        }


def _compute_angle_from_lines(lines: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]]) -> float | None:
    if not lines:
        return None
    angles: List[float] = []
    for (x1, y1), (x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        # Normalize to [0, 180)
        if angle < 0:
            angle += 180.0
        angles.append(angle)
    if not angles:
        return None
    # Cluster around dominant mode: convert to radians and take circular mean.
    radians = np.radians(angles)
    sin_sum = float(np.sum(np.sin(radians)))
    cos_sum = float(np.sum(np.cos(radians)))
    mean = float(np.degrees(np.arctan2(sin_sum, cos_sum)))
    if mean < 0:
        mean += 180.0
    return mean


def _pca_angle(points: np.ndarray) -> float | None:
    if points.shape[0] < 3:
        return None
    # Center
    centered = points - np.mean(points, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = int(np.argmax(eigvals))
    principal = eigvecs[:, idx]
    angle = float(np.degrees(np.arctan2(principal[1], principal[0])))
    if angle < 0:
        angle += 180.0
    return angle


def estimate_wall_angle(
    image_path: Path | str,
    *,
    hold_centers: Optional[Iterable[Tuple[float, float]]] = None,
    mask: Optional[np.ndarray] = None,
    canny_threshold1: int = 50,
    canny_threshold2: int = 150,
    hough_threshold: int = 120,
    min_line_length: int = 60,
    max_line_gap: int = 10,
) -> WallAngleResult:
    """Estimate wall angle from an image.

    Parameters
    ----------
    image_path: Path | str
        Path to a representative frame image.
    hold_centers: Optional sequence of (x, y) coordinates (normalized or pixel). If provided,
        PCA over these points refines the angle.
    mask: Optional binary mask (same shape as image) to exclude climber or moving objects.

    Returns
    -------
    WallAngleResult
        Angle in degrees relative to horizontal (0° horizontal; 90° vertical wall).
    """
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        return WallAngleResult(angle_degrees=None, confidence=0.0, method="load-error", hough_lines=[])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mask is not None and mask.shape[:2] == gray.shape:
        gray = cv2.bitwise_and(gray, gray, mask=mask.astype(np.uint8))

    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2, L2gradient=True)
    lines_p = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    normalized_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            normalized_lines.append(((int(x1), int(y1)), (int(x2), int(y2))))

    hough_angle = _compute_angle_from_lines(normalized_lines)

    pca_angle_value: float | None = None
    if hold_centers:
        pts = np.array(list(hold_centers), dtype=float)
        # If normalized (0-1), scale to image size for consistency.
        if pts.max() <= 1.2:  # heuristic
            h, w = gray.shape[:2]
            pts[:, 0] *= w
            pts[:, 1] *= h
        pca_angle_value = _pca_angle(pts)

    final_angle: float | None
    method = "hough"
    if hough_angle is not None and pca_angle_value is not None:
        # Blend angles if close; else prefer Hough (structural) but lower confidence.
        diff = abs(hough_angle - pca_angle_value)
        if diff < 15.0:
            final_angle = float((hough_angle + pca_angle_value) / 2.0)
            method = "hough+pca"
            confidence = 0.9
        else:
            final_angle = hough_angle
            confidence = 0.6
    elif hough_angle is not None:
        final_angle = hough_angle
        confidence = 0.7 if len(normalized_lines) >= 5 else 0.5
    else:
        final_angle = pca_angle_value
        confidence = 0.4 if pca_angle_value is not None else 0.0
        method = "pca" if pca_angle_value is not None else "none"

    return WallAngleResult(
        angle_degrees=final_angle,
        confidence=confidence,
        method=method,
        hough_lines=normalized_lines,
        pca_angle=pca_angle_value,
    )


def quaternion_to_euler(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert quaternion to Euler angles (pitch, roll, yaw) in degrees.
    
    Args:
        w, x, y, z: Quaternion components
    
    Returns:
        (pitch, roll, yaw) in degrees
        - pitch: rotation around X axis (-90 to 90 degrees)
        - roll: rotation around Y axis (-180 to 180 degrees)
        - yaw: rotation around Z axis (-180 to 180 degrees)
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    # Convert to degrees
    pitch_deg = np.degrees(pitch)
    roll_deg = np.degrees(roll)
    yaw_deg = np.degrees(yaw)
    
    return pitch_deg, roll_deg, yaw_deg


def compute_wall_angle_from_imu(
    quaternion: list[float] | None = None,
    euler_angles: list[float] | None = None,
) -> WallAngleResult:
    """Compute wall angle from IMU sensor data.
    
    Args:
        quaternion: Device orientation as [w, x, y, z]
        euler_angles: Device orientation as [pitch, roll, yaw] in degrees
    
    Returns:
        WallAngleResult with angle in degrees (0=horizontal, 90=vertical)
    
    Notes:
        - Assumes device is held with screen facing the wall
        - Pitch angle represents wall inclination
        - Positive pitch = wall leaning back (slab/vertical/overhang)
        - If both quaternion and euler_angles provided, quaternion takes priority
    """
    if quaternion is not None:
        if len(quaternion) != 4:
            return WallAngleResult(
                angle_degrees=None,
                confidence=0.0,
                method="imu_error",
                hough_lines=[],
            )
        
        w, x, y, z = quaternion
        pitch, roll, yaw = quaternion_to_euler(w, x, y, z)
    
    elif euler_angles is not None:
        if len(euler_angles) != 3:
            return WallAngleResult(
                angle_degrees=None,
                confidence=0.0,
                method="imu_error",
                hough_lines=[],
            )
        
        pitch, roll, yaw = euler_angles
    
    else:
        return WallAngleResult(
            angle_degrees=None,
            confidence=0.0,
            method="imu_missing",
            hough_lines=[],
        )
    
    # Wall angle is derived from pitch
    # Adjust to climbing context: 0° = horizontal, 90° = vertical, >90° = overhang
    wall_angle = abs(pitch)
    
    # Clamp to reasonable range
    wall_angle = max(0.0, min(180.0, wall_angle))
    
    # IMU sensors are generally very reliable (confidence ~0.95-1.0)
    # Reduce confidence if angles are extreme or unusual
    confidence = 1.0
    if roll > 45 or roll < -45:
        # Device significantly rotated - may not be held properly
        confidence = 0.7
    
    return WallAngleResult(
        angle_degrees=wall_angle,
        confidence=confidence,
        method="imu_sensor",
        hough_lines=[],
        pca_angle=None,
    )


__all__ = ["WallAngleResult", "estimate_wall_angle", "compute_wall_angle_from_imu", "quaternion_to_euler"]
