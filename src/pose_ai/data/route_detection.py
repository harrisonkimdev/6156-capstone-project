"""Utilities for detecting climbing routes from still images."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def make_tolerances(min_value: int, max_value: int, tolerance: int, value: int) -> Tuple[int, int]:
    """Clamp ``value`` ± ``tolerance`` between ``min_value`` and ``max_value``."""
    low = max(min_value, value - tolerance)
    high = min(max_value, value + tolerance)
    if low > high:
        low, high = high, low
    return low, high


@dataclass(slots=True)
class RouteDetectionState:
    """State container for interactive hold detection."""

    working_image: np.ndarray
    hsv_image: np.ndarray
    original_image: np.ndarray
    hue_tol: int = 5
    sat_tol: int = 50
    val_tol: int = 40
    mask: Optional[np.ndarray] = None
    cutout: Optional[np.ndarray] = None
    selected_hsv: Optional[np.ndarray] = None
    selected_point: Optional[Tuple[int, int]] = None
    hold_centers: List[Tuple[int, int]] = field(default_factory=list)


def load_raw_image(image_path: Path | str, scale: float = 1.0) -> RouteDetectionState:
    """Load an image from disk and prepare HSV representation."""
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to open image: {path}")
    if scale != 1.0:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return RouteDetectionState(
        working_image=image.copy(),
        hsv_image=hsv,
        original_image=image.copy(),
    )


def resize_and_convert(image: np.ndarray, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Resize an in-memory image and return BGR + HSV versions."""
    if scale != 1.0:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image, hsv


def remove_background(
    image: np.ndarray,
    lower_bg: Tuple[int, int, int],
    upper_bg: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate background from climber/holds using HSV thresholds."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_bg, dtype=np.uint8), np.array(upper_bg, dtype=np.uint8))
    background_only = cv2.bitwise_and(image, image, mask=mask)
    foreground_mask = cv2.bitwise_not(mask)
    foreground_only = cv2.bitwise_and(image, image, mask=foreground_mask)
    return background_only, foreground_only, mask


def detect_route_click(state: RouteDetectionState, x: int, y: int) -> RouteDetectionState:
    """Apply HSV-based masking after a user clicks a pixel."""
    pixel_hsv = state.hsv_image[y, x]
    state.selected_point = (x, y)
    state.selected_hsv = pixel_hsv

    lower_hue, upper_hue = make_tolerances(0, 179, state.hue_tol, int(pixel_hsv[0]))
    lower_sat, upper_sat = make_tolerances(0, 255, state.sat_tol, int(pixel_hsv[1]))
    lower_val, upper_val = make_tolerances(0, 255, state.val_tol, int(pixel_hsv[2]))

    lower = np.array([lower_hue, lower_sat, lower_val], dtype=np.uint8)
    upper = np.array([upper_hue, upper_sat, upper_val], dtype=np.uint8)

    mask = cv2.inRange(state.hsv_image, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    state.mask = mask
    state.cutout = cv2.bitwise_and(state.working_image, state.working_image, mask=mask)
    return state


def group_holds(
    state: RouteDetectionState,
    *,
    min_area: float = 25.0,
    max_dist: float = 150.0,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Cluster hold contours and connect them in a greedy chain."""
    if state.mask is None:
        raise ValueError("RouteDetectionState.mask is not populated; call detect_route_click first.")

    contours, _ = cv2.findContours(state.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = state.original_image.copy()
    centers: list[Tuple[int, int]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= min_area:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        centers.append(center)
        cv2.circle(output, center, int(radius), (0, 255, 0), 2)
        cv2.putText(
            output,
            str(int(area)),
            (center[0] - 20, center[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    if centers:
        used = set()
        distances = [
            (math.dist((state.selected_point or (center[0], 1000)), center), idx)
            for idx, center in enumerate(centers)
        ]
        distances.sort(key=lambda d: d[0])
        _, current_idx = distances[0]
        used.add(current_idx)

        while True:
            current_center = centers[current_idx]
            remaining = [
                (math.dist(current_center, candidate), candidate_idx)
                for candidate_idx, candidate in enumerate(centers)
                if candidate_idx not in used
            ]
            if not remaining:
                break
            remaining.sort(key=lambda d: d[0])
            distance, next_idx = remaining[0]
            if distance > max_dist:
                break
            cv2.arrowedLine(
                output,
                current_center,
                centers[next_idx],
                (255, 0, 0),
                2,
                tipLength=0.2,
            )
            used.add(next_idx)
            current_idx = next_idx

    state.hold_centers = centers
    return output, centers


__all__ = [
    "RouteDetectionState",
    "detect_route_click",
    "group_holds",
    "load_raw_image",
    "make_tolerances",
    "remove_background",
    "resize_and_convert",
]
