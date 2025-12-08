"""Hold detection filtering system to remove false positives.

This module provides intelligent filtering to distinguish between:
1. Chalk marks - false positives (small, white, flat, low-texture)
2. Safety holds - gray colored holds for safe descent (gray, often at bottom)
3. Volumes - angular structures (not climbable holds)
4. Valid climbing holds - the actual targets for climbing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class FilterAnalysis:
    """Analysis results for a single detection."""
    detection_id: str
    is_chalk_mark: bool = False
    is_safety_hold: bool = False
    is_volume: bool = False
    confidence: float = 0.0
    reason: str = ""
    
    def is_valid_climbing_hold(self) -> bool:
        """Check if this is a valid climbing hold."""
        return not (self.is_chalk_mark or self.is_safety_hold or self.is_volume)


def analyze_color_properties(roi: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """Analyze HSV and BGR color properties of a region.
    
    Args:
        roi: Region of interest in BGR format
        mask: Optional binary mask to focus on specific pixels
    
    Returns:
        Dict with color statistics (saturation, value, hue, etc.)
    """
    if roi.size == 0:
        return {
            "mean_saturation": 0,
            "mean_value": 0,
            "saturation_std": 0,
            "value_std": 0,
            "bgr_color_variance": 0,
            "is_white": False,
            "is_gray": False,
        }
    
    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Extract pixels
    if mask is not None:
        mask_pixels = hsv[mask > 0]
        bgr_pixels = roi[mask > 0]
    else:
        mask_pixels = hsv.reshape(-1, 3)
        bgr_pixels = roi.reshape(-1, 3)
    
    if len(mask_pixels) == 0:
        return {
            "mean_saturation": 0,
            "mean_value": 0,
            "saturation_std": 0,
            "value_std": 0,
            "bgr_color_variance": 0,
            "is_white": False,
            "is_gray": False,
        }
    
    # HSV analysis
    mean_saturation = float(np.mean(mask_pixels[:, 1]))
    mean_value = float(np.mean(mask_pixels[:, 2]))
    saturation_std = float(np.std(mask_pixels[:, 1]))
    value_std = float(np.std(mask_pixels[:, 2]))
    
    # BGR color variance (how similar RGB channels are)
    bgr_color_variance = float(np.std(np.std(bgr_pixels, axis=0)))
    
    # Color classification
    is_white = mean_saturation < 50 and mean_value > 200
    is_gray = mean_saturation < 50 and 100 < mean_value < 180
    
    return {
        "mean_saturation": mean_saturation,
        "mean_value": mean_value,
        "saturation_std": saturation_std,
        "value_std": value_std,
        "bgr_color_variance": bgr_color_variance,
        "is_white": is_white,
        "is_gray": is_gray,
    }


def analyze_shape_properties(mask: np.ndarray) -> dict:
    """Analyze contour shape properties to distinguish volumes from holds.
    
    Args:
        mask: Binary mask of the object
    
    Returns:
        Dict with shape statistics (solidity, straightness, aspect ratio, etc.)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            "area": 0,
            "solidity": 0,
            "straightness": 0,
            "aspect_ratio": 1.0,
            "num_vertices": 0,
            "contour_perimeter": 0,
        }
    
    cnt = contours[0]
    area = float(cv2.contourArea(cnt))
    
    # Solidity: area / convex hull area
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / (hull_area + 1e-6)
    
    # Straightness: hull perimeter / contour perimeter
    # High = straight edges (volume), Low = curved edges (hold)
    contour_perimeter = float(cv2.arcLength(cnt, True))
    hull_perimeter = float(cv2.arcLength(hull, True))
    straightness = hull_perimeter / (contour_perimeter + 1e-6)
    
    # Aspect ratio from bounding rect
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(max(w, h)) / float(min(w, h) + 1e-6)
    
    # Polygon approximation: how many vertices?
    epsilon = 0.02 * contour_perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    num_vertices = len(approx)
    
    return {
        "area": area,
        "solidity": solidity,
        "straightness": straightness,
        "aspect_ratio": aspect_ratio,
        "num_vertices": num_vertices,
        "contour_perimeter": contour_perimeter,
    }


def analyze_texture_properties(roi: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """Analyze texture to distinguish holds (3D, textured) from chalk marks (flat).
    
    Args:
        roi: Region of interest
        mask: Optional binary mask
    
    Returns:
        Dict with texture statistics (gradient magnitude, variance, etc.)
    """
    if roi.size == 0:
        return {
            "mean_gradient": 0,
            "gradient_std": 0,
            "laplacian_variance": 0,
            "has_texture": False,
        }
    
    # Convert to grayscale
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    # Compute Sobel gradient (edge magnitude)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Compute Laplacian (curvature, good for 3D texture)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    if mask is not None:
        gradient_mag = gradient_mag[mask > 0]
        laplacian = laplacian[mask > 0]
    else:
        gradient_mag = gradient_mag.flatten()
        laplacian = laplacian.flatten()
    
    if len(gradient_mag) == 0:
        return {
            "mean_gradient": 0,
            "gradient_std": 0,
            "laplacian_variance": 0,
            "has_texture": False,
        }
    
    mean_gradient = float(np.mean(gradient_mag))
    gradient_std = float(np.std(gradient_mag))
    laplacian_variance = float(np.var(laplacian))
    
    # Holds have stronger texture than chalk marks
    has_texture = mean_gradient > 5.0 or laplacian_variance > 50.0
    
    return {
        "mean_gradient": mean_gradient,
        "gradient_std": gradient_std,
        "laplacian_variance": laplacian_variance,
        "has_texture": has_texture,
    }


def is_chalk_mark(
    mask: np.ndarray,
    roi: np.ndarray,
    *,
    size_max: float = 1000.0,
    texture_threshold: float = 5.0,
    min_solidity: float = 0.3,
    max_aspect_ratio: float = 5.0,
) -> Tuple[bool, str]:
    """Detect if a detection is a chalk mark (false positive).
    
    Chalk marks have:
    - Small area (< size_max)
    - White color (high V, low S)
    - Flat/low texture (gradient < threshold)
    - Low solidity or high aspect ratio (irregular smudges)
    
    Args:
        mask: Binary mask of the object
        roi: Region of interest in BGR format
        size_max: Maximum area to consider as chalk
        texture_threshold: Minimum gradient magnitude for real texture
        min_solidity: Minimum solidity threshold
        max_aspect_ratio: Maximum aspect ratio for extended marks
    
    Returns:
        Tuple of (is_chalk, reason)
    """
    # Analyze shape
    shape = analyze_shape_properties(mask)
    area = shape["area"]
    solidity = shape["solidity"]
    aspect_ratio = shape["aspect_ratio"]
    
    # Analyze color
    color = analyze_color_properties(roi, mask)
    is_white = color["is_white"]
    
    # Analyze texture
    texture = analyze_texture_properties(roi, mask)
    mean_gradient = texture["mean_gradient"]
    
    # Check chalk characteristics
    reasons = []
    
    # 1. Too small
    if area < 200:
        reasons.append(f"very small area ({area:.0f})")
    
    # 2. White color
    if is_white:
        reasons.append("white color (chalk)")
    else:
        # Not white, so likely not chalk
        return False, "not white"
    
    # 3. Low solidity (irregular smudge)
    if solidity < min_solidity:
        reasons.append(f"low solidity ({solidity:.2f})")
    
    # 4. Very elongated (streak mark)
    if aspect_ratio > max_aspect_ratio:
        reasons.append(f"high aspect ratio ({aspect_ratio:.2f})")
    
    # 5. Flat/low texture (no 3D structure)
    if mean_gradient < texture_threshold:
        reasons.append(f"low texture ({mean_gradient:.2f})")
    
    # Chalk if: white AND (too small OR low solidity OR elongated OR flat)
    is_chalk = is_white and (
        area < 200 or 
        solidity < min_solidity or 
        aspect_ratio > max_aspect_ratio or
        mean_gradient < texture_threshold
    )
    
    reason = "; ".join(reasons) if reasons else "unknown"
    return is_chalk, reason


def is_safety_hold(
    mask: np.ndarray,
    roi: np.ndarray,
    image_shape: Tuple[int, int],
    bbox: Tuple[float, float, float, float],
    *,
    gray_saturation_max: float = 50.0,
    gray_value_range: Tuple[float, float] = (100, 180),
    bgr_variance_max: float = 15.0,
    bottom_threshold: float = 0.85,
) -> Tuple[bool, str]:
    """Detect if a detection is a safety hold (gray, often at bottom).
    
    Safety holds (jugs) have:
    - Gray color (low saturation, medium value)
    - Often located at bottom of wall (safe descent)
    - Usually larger size
    
    Args:
        mask: Binary mask of the object
        roi: Region of interest in BGR format
        image_shape: (height, width) of the image
        bbox: Bounding box (x1, y1, x2, y2) in pixel coordinates
        gray_saturation_max: Maximum saturation for gray
        gray_value_range: Value range for gray (min, max)
        bgr_variance_max: Maximum variance for RGB channels
        bottom_threshold: Consider bottom % of image for safety holds
    
    Returns:
        Tuple of (is_safety_hold, reason)
    """
    # Analyze color
    color = analyze_color_properties(roi, mask)
    mean_saturation = color["mean_saturation"]
    mean_value = color["mean_value"]
    bgr_variance = color["bgr_color_variance"]
    is_gray = color["is_gray"]
    
    # Analyze position
    _, _, _, y2 = bbox
    image_height = image_shape[0]
    is_at_bottom = y2 > image_height * bottom_threshold
    
    # Analyze size
    shape = analyze_shape_properties(mask)
    area = shape["area"]
    is_large = area > 3000
    
    reasons = []
    
    # 1. Gray color
    if is_gray:
        reasons.append(f"gray color (S={mean_saturation:.0f}, V={mean_value:.0f})")
    
    # Alternative: RGB channels are very similar
    if bgr_variance < bgr_variance_max and mean_saturation < gray_saturation_max:
        reasons.append(f"uniform color (BGR std={bgr_variance:.1f})")
    else:
        # Not gray, so likely not safety hold
        return False, "not gray"
    
    # 2. Located at bottom (safe descent position)
    if is_at_bottom:
        reasons.append(f"at bottom ({y2:.0f} > {image_height * bottom_threshold:.0f})")
    
    # 3. Larger size (safety holds tend to be bigger)
    if is_large:
        reasons.append(f"large size ({area:.0f})")
    
    # Safety hold if: gray AND (at bottom OR large)
    is_safety = (is_gray or bgr_variance < bgr_variance_max) and (is_at_bottom or is_large)
    
    reason = "; ".join(reasons) if reasons else "unknown"
    return is_safety, reason


def is_volume(
    mask: np.ndarray,
    roi: np.ndarray,
    *,
    min_vertices: int = 4,
    max_vertices: int = 8,
    straightness_threshold: float = 0.95,
    min_area: float = 5000.0,
    aspect_ratio_range: Tuple[float, float] = (0.4, 2.5),
) -> Tuple[bool, str]:
    """Detect if a detection is a volume (angular structure, not a hold).
    
    Volumes have:
    - Angular/polygonal shape (few vertices, straight edges)
    - Large size
    - Rectangular aspect ratio (not elongated or circular)
    
    Args:
        mask: Binary mask of the object
        roi: Region of interest
        min_vertices: Minimum vertices for polygon
        max_vertices: Maximum vertices for polygon
        straightness_threshold: Minimum straightness ratio
        min_area: Minimum area to be considered a volume
        aspect_ratio_range: Valid aspect ratio range (min, max)
    
    Returns:
        Tuple of (is_volume, reason)
    """
    shape = analyze_shape_properties(mask)
    area = shape["area"]
    straightness = shape["straightness"]
    num_vertices = shape["num_vertices"]
    aspect_ratio = shape["aspect_ratio"]
    
    reasons = []
    
    # 1. Too small to be a volume
    if area < min_area:
        return False, f"too small ({area:.0f} < {min_area})"
    
    reasons.append(f"large area ({area:.0f})")
    
    # 2. Polygonal shape (angular)
    if min_vertices <= num_vertices <= max_vertices:
        reasons.append(f"polygonal ({num_vertices} vertices)")
    else:
        return False, f"not polygonal ({num_vertices} vertices)"
    
    # 3. Straight edges
    if straightness > straightness_threshold:
        reasons.append(f"straight edges ({straightness:.2f})")
    else:
        # Curved edges = hold, not volume
        return False, f"curved edges ({straightness:.2f})"
    
    # 4. Reasonable aspect ratio (not too elongated or circular)
    min_ar, max_ar = aspect_ratio_range
    if min_ar < aspect_ratio < max_ar:
        reasons.append(f"reasonable aspect ratio ({aspect_ratio:.2f})")
    else:
        return False, f"unusual aspect ratio ({aspect_ratio:.2f})"
    
    # Volume if: large AND polygonal AND straight edges AND reasonable shape
    is_vol = (
        area >= min_area and
        min_vertices <= num_vertices <= max_vertices and
        straightness > straightness_threshold
    )
    
    reason = "; ".join(reasons) if reasons else "unknown"
    return is_vol, reason


def filter_single_detection(
    detection_id: str,
    mask: np.ndarray,
    roi: np.ndarray,
    image_shape: Tuple[int, int],
    bbox: Tuple[float, float, float, float],
    *,
    filter_chalk: bool = True,
    filter_safety_holds: bool = True,
    distinguish_volumes: bool = True,
    chalk_config: Optional[dict] = None,
    safety_config: Optional[dict] = None,
    volume_config: Optional[dict] = None,
) -> FilterAnalysis:
    """Analyze a single detection and determine if it should be filtered.
    
    Args:
        detection_id: Unique identifier for the detection
        mask: Binary segmentation mask
        roi: Region of interest in BGR format
        image_shape: (height, width) of the image
        bbox: Bounding box (x1, y1, x2, y2) in normalized or pixel coordinates
        filter_chalk: Enable chalk mark filtering
        filter_safety_holds: Enable safety hold filtering
        distinguish_volumes: Enable volume detection
        chalk_config: Custom config for chalk detection (optional)
        safety_config: Custom config for safety hold detection (optional)
        volume_config: Custom config for volume detection (optional)
    
    Returns:
        FilterAnalysis object with detection results
    """
    analysis = FilterAnalysis(detection_id=detection_id)
    
    # Check for chalk marks
    if filter_chalk:
        chalk_cfg = chalk_config or {}
        is_chalk, chalk_reason = is_chalk_mark(mask, roi, **chalk_cfg)
        if is_chalk:
            analysis.is_chalk_mark = True
            analysis.reason = chalk_reason
            return analysis
    
    # Check for safety holds
    if filter_safety_holds:
        safety_cfg = safety_config or {}
        is_safety, safety_reason = is_safety_hold(mask, roi, image_shape, bbox, **safety_cfg)
        if is_safety:
            analysis.is_safety_hold = True
            analysis.reason = safety_reason
            return analysis
    
    # Check for volumes
    if distinguish_volumes:
        volume_cfg = volume_config or {}
        is_vol, volume_reason = is_volume(mask, roi, **volume_cfg)
        if is_vol:
            analysis.is_volume = True
            analysis.reason = volume_reason
            return analysis
    
    # No filters matched - valid climbing hold
    analysis.reason = "valid climbing hold"
    return analysis


def filter_detections_batch(
    detections_with_masks: Sequence[Tuple[str, np.ndarray, np.ndarray, Tuple[int, int], Tuple[float, float, float, float]]],
    *,
    filter_chalk: bool = True,
    filter_safety_holds: bool = True,
    distinguish_volumes: bool = True,
    chalk_config: Optional[dict] = None,
    safety_config: Optional[dict] = None,
    volume_config: Optional[dict] = None,
) -> Tuple[List[str], List[FilterAnalysis]]:
    """Filter a batch of detections.
    
    Args:
        detections_with_masks: List of (detection_id, mask, roi, image_shape, bbox) tuples
        filter_chalk: Enable chalk mark filtering
        filter_safety_holds: Enable safety hold filtering
        distinguish_volumes: Enable volume detection
        chalk_config: Custom config for chalk detection
        safety_config: Custom config for safety hold detection
        volume_config: Custom config for volume detection
    
    Returns:
        Tuple of (valid_detection_ids, all_analyses)
    """
    valid_ids = []
    all_analyses = []
    
    for detection_id, mask, roi, image_shape, bbox in detections_with_masks:
        analysis = filter_single_detection(
            detection_id,
            mask,
            roi,
            image_shape,
            bbox,
            filter_chalk=filter_chalk,
            filter_safety_holds=filter_safety_holds,
            distinguish_volumes=distinguish_volumes,
            chalk_config=chalk_config,
            safety_config=safety_config,
            volume_config=volume_config,
        )
        
        all_analyses.append(analysis)
        
        if analysis.is_valid_climbing_hold():
            valid_ids.append(detection_id)
        else:
            LOGGER.debug(
                "Filtered detection %s: chalk=%s, safety=%s, volume=%s, reason=%s",
                detection_id,
                analysis.is_chalk_mark,
                analysis.is_safety_hold,
                analysis.is_volume,
                analysis.reason,
            )
    
    filtered_count = len(detections_with_masks) - len(valid_ids)
    LOGGER.info(
        "Filtered %d/%d detections (chalk=%d, safety=%d, volume=%d)",
        filtered_count,
        len(detections_with_masks),
        sum(1 for a in all_analyses if a.is_chalk_mark),
        sum(1 for a in all_analyses if a.is_safety_hold),
        sum(1 for a in all_analyses if a.is_volume),
    )
    
    return valid_ids, all_analyses


__all__ = [
    "FilterAnalysis",
    "analyze_color_properties",
    "analyze_shape_properties",
    "analyze_texture_properties",
    "is_chalk_mark",
    "is_safety_hold",
    "is_volume",
    "filter_single_detection",
    "filter_detections_batch",
]
