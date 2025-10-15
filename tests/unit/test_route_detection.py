from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pose_ai.data import (
    RouteDetectionState,
    detect_route_click,
    group_holds,
    load_raw_image,
)


def _create_test_image(image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((100, 100, 3), 30, dtype=np.uint8)
    cv2.circle(image, (30, 50), 12, (0, 255, 0), -1)
    cv2.circle(image, (70, 55), 12, (0, 255, 0), -1)
    cv2.imwrite(str(image_path), image)


def test_detect_route_click_and_group(tmp_path: Path) -> None:
    image_path = tmp_path / "frames" / "frame.jpg"
    _create_test_image(image_path)

    state: RouteDetectionState = load_raw_image(image_path)
    state = detect_route_click(state, 30, 50)

    assert state.mask is not None
    assert state.cutout is not None
    assert state.selected_point == (30, 50)
    assert state.selected_hsv is not None

    output, centers = group_holds(state, min_area=50, max_dist=60)
    assert output.shape == state.original_image.shape
    assert centers
    assert any(abs(cx - 30) <= 2 and abs(cy - 50) <= 2 for cx, cy in centers)
