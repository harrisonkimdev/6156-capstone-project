"""HSV-based segmentation for hold detection using color masking.

This module provides HSV color-based segmentation for detecting holds in climbing videos,
as an alternative to YOLO-based segmentation.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .yolo_segmentation import SegmentationMask, SegmentationResult

LOGGER = logging.getLogger(__name__)


def make_tolerances(min_val: int, max_val: int, tolerance: int, value: int) -> Tuple[int, int]:
    """Calculate lower and upper bounds for HSV tolerance.

    Args:
        min_val: Minimum possible value
        max_val: Maximum possible value
        tolerance: Tolerance value
        value: Center value

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    out_l = 0
    out_u = 1

    if min_val > max_val:
        min_val, max_val = max_val, min_val

    # Set the upper bound
    if value + tolerance > max_val:
        out_u = max_val
    else:
        out_u = value + tolerance

    # Set the lower bound
    if value - tolerance < min_val:
        out_l = min_val
    else:
        out_l = value - tolerance

    # Ensure lower <= upper
    if out_l > out_u:
        out_l, out_u = out_u, out_l

    return out_l, out_u


def remove_background(
    image: np.ndarray,
    lower_bg: Sequence[int],
    upper_bg: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separates background (whitish-gray) from the foreground.

    Args:
        image: Input BGR image
        lower_bg: Lower HSV bound of background [H, S, V]
        upper_bg: Upper HSV bound of background [H, S, V]

    Returns:
        Tuple of (background_only, foreground_only, bg_mask)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Make background mask
    lower_bg_array = np.array(lower_bg, dtype=np.uint8)
    upper_bg_array = np.array(upper_bg, dtype=np.uint8)
    bg_mask = cv2.inRange(hsv, lower_bg_array, upper_bg_array)

    # Clean up mask (remove noise, fill holes)
    kernel = np.ones((3, 3), np.uint8)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)  # remove specks
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)  # close holes

    # Extract background & foreground
    background_only = cv2.bitwise_and(image, image, mask=bg_mask)
    fg_mask = cv2.bitwise_not(bg_mask)
    foreground_only = cv2.bitwise_and(image, image, mask=fg_mask)

    return background_only, foreground_only, bg_mask


def detect_holds_by_color(
    image: np.ndarray,
    hsv_image: np.ndarray,
    *,
    hue_tol: int = 5,
    sat_tol: int = 50,
    val_tol: int = 40,
    reference_pixel: Tuple[int, int] | None = None,
    reference_hsv: Tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Detect holds using HSV color masking.

    Args:
        image: Input BGR image
        hsv_image: Input HSV image
        hue_tol: Hue tolerance (0-179)
        sat_tol: Saturation tolerance (0-255)
        val_tol: Value tolerance (0-255)
        reference_pixel: Optional (x, y) pixel to use as color reference
        reference_hsv: Optional (H, S, V) to use as color reference

    Returns:
        Binary mask of detected holds
    """
    if reference_pixel is not None:
        x, y = reference_pixel
        pixel_hsv = hsv_image[y, x]
    elif reference_hsv is not None:
        pixel_hsv = np.array(reference_hsv, dtype=np.uint8)
    else:
        # Auto-detect: find dominant color in foreground
        # Use a simple heuristic: find the most common non-background color
        # For now, use a default range that works for common hold colors
        LOGGER.warning("No reference pixel or HSV provided, using default range")
        lower = np.array([0, 50, 50], dtype=np.uint8)
        upper = np.array([179, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv_image, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    # Calculate HSV bounds
    lower_hue, upper_hue = make_tolerances(0, 179, hue_tol, int(pixel_hsv[0]))
    lower_sat, upper_sat = make_tolerances(0, 255, sat_tol, int(pixel_hsv[1]))
    lower_val, upper_val = make_tolerances(0, 255, val_tol, int(pixel_hsv[2]))

    lower = np.array([lower_hue, lower_sat, lower_val], dtype=np.uint8)
    upper = np.array([upper_hue, upper_sat, upper_val], dtype=np.uint8)

    # Create mask
    mask = cv2.inRange(hsv_image, lower, upper)

    # Denoise the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def group_holds(
    mask: np.ndarray,
    *,
    min_area: int = 25,
    max_dist: int = 150,
    start_pixel: Tuple[int, int] | None = None,
) -> List[Tuple[int, int, float]]:
    """Group holds and create a chain from detected blobs.

    Args:
        mask: Binary mask of holds
        min_area: Minimum area for a hold to be considered
        max_dist: Maximum distance between holds in a chain
        start_pixel: Optional (x, y) pixel to start chaining from (default: bottom center)

    Returns:
        List of (center_x, center_y, radius) tuples for detected holds
    """
    # Find contours (white blobs in mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers: List[Tuple[int, int, float]] = []

    # Step 1: detect blobs (holds) and get centers
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y), float(radius))
            centers.append(center)

    return centers


class HsvSegmentationModel:
    """HSV-based segmentation model for hold detection."""

    def __init__(
        self,
        *,
        hue_tolerance: int = 5,
        sat_tolerance: int = 50,
        val_tolerance: int = 40,
        background_lower: Sequence[int] | None = None,
        background_upper: Sequence[int] | None = None,
    ) -> None:
        """Initialize HSV segmentation model.

        Args:
            hue_tolerance: Hue tolerance for color matching (0-179)
            sat_tolerance: Saturation tolerance (0-255)
            val_tolerance: Value tolerance (0-255)
            background_lower: Lower HSV bound for background removal [H, S, V]
            background_upper: Upper HSV bound for background removal [H, S, V]
        """
        self.hue_tol = hue_tolerance
        self.sat_tol = sat_tolerance
        self.val_tol = val_tolerance
        self.background_lower = background_lower or [0, 0, 104]
        self.background_upper = background_upper or [178, 33, 244]

    def segment_frame(
        self,
        image_path: Path | str,
        *,
        conf_threshold: float = 0.25,  # Not used for HSV, kept for API compatibility
        target_classes: Sequence[str] | None = None,
        reference_pixel: Tuple[int, int] | None = None,
        reference_hsv: Tuple[int, int, int] | None = None,
    ) -> SegmentationResult:
        """Segment a single frame using HSV color masking.

        Args:
            image_path: Path to input image
            conf_threshold: Not used (kept for API compatibility)
            target_classes: Target classes to segment (e.g., ["hold"])
            reference_pixel: Optional (x, y) pixel to use as color reference
            reference_hsv: Optional (H, S, V) to use as color reference

        Returns:
            SegmentationResult with detected holds
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Remove background
        bg_only, fg_only, bg_mask = remove_background(
            image,
            self.background_lower,
            self.background_upper,
        )

        # Apply second background removal pass if needed
        # (This matches the original notebook behavior)
        second_lower = [96, 138, 124]
        second_upper = [106, 238, 204]
        _, fg_only, _ = remove_background(fg_only, second_lower, second_upper)

        # Convert to HSV
        hsv_image = cv2.cvtColor(fg_only, cv2.COLOR_BGR2HSV)

        # Detect holds
        hold_mask = detect_holds_by_color(
            fg_only,
            hsv_image,
            hue_tol=self.hue_tol,
            sat_tol=self.sat_tol,
            val_tol=self.val_tol,
            reference_pixel=reference_pixel,
            reference_hsv=reference_hsv,
        )

        # Create segmentation masks
        masks_list: List[SegmentationMask] = []

        # Add hold mask
        if target_classes is None or "hold" in [c.lower() for c in target_classes]:
            # Calculate bounding box from mask
            contours, _ = cv2.findContours(hold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Use largest contour for bbox
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                bbox = (float(x), float(y), float(x + w), float(y + h))
            else:
                bbox = None

            masks_list.append(
                SegmentationMask(
                    class_name="hold",
                    mask=hold_mask,
                    confidence=1.0,  # HSV method doesn't provide confidence scores
                    bbox=bbox,
                )
            )

        # Add wall mask (inverse of foreground)
        if target_classes is None or "wall" in [c.lower() for c in target_classes]:
            wall_mask = cv2.bitwise_not(hold_mask)
            masks_list.append(
                SegmentationMask(
                    class_name="wall",
                    mask=wall_mask,
                    confidence=1.0,
                    bbox=None,
                )
            )

        return SegmentationResult(image_path=image_path, masks=masks_list)

    def batch_segment_frames(
        self,
        image_paths: Sequence[Path | str],
        *,
        conf_threshold: float = 0.25,
        target_classes: Sequence[str] | None = None,
        reference_pixel: Tuple[int, int] | None = None,
        reference_hsv: Tuple[int, int, int] | None = None,
    ) -> List[SegmentationResult]:
        """Segment multiple frames.

        Args:
            image_paths: Sequence of image file paths
            conf_threshold: Not used (kept for API compatibility)
            target_classes: Target classes to segment
            reference_pixel: Optional (x, y) pixel to use as color reference
            reference_hsv: Optional (H, S, V) to use as color reference

        Returns:
            List of SegmentationResult objects
        """
        results: List[SegmentationResult] = []
        for idx, image_path in enumerate(image_paths):
            try:
                result = self.segment_frame(
                    image_path,
                    conf_threshold=conf_threshold,
                    target_classes=target_classes,
                    reference_pixel=reference_pixel,
                    reference_hsv=reference_hsv,
                )
                result.frame_index = idx
                results.append(result)
            except Exception as exc:
                LOGGER.warning("Failed to segment frame %s: %s", image_path, exc)
                results.append(SegmentationResult(image_path=Path(str(image_path)), masks=[], frame_index=idx))
        return results


__all__ = [
    "HsvSegmentationModel",
    "make_tolerances",
    "remove_background",
    "detect_holds_by_color",
    "group_holds",
]

