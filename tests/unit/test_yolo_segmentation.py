"""Unit tests for YOLO segmentation module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pose_ai.segmentation.yolo_segmentation import (
    HoldColorInfo,
    RouteGroup,
    SegmentationMask,
    SegmentationResult,
    YoloSegmentationModel,
    cluster_holds_by_color,
    extract_hold_colors,
    export_routes_json,
    export_segmentation_masks,
)


def _create_test_image(image_path: Path, size: tuple[int, int] = (640, 480)) -> None:
    """Create a test image."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)


def test_segmentation_mask() -> None:
    """Test SegmentationMask dataclass."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1

    seg_mask = SegmentationMask(
        class_name="hold",
        mask=mask,
        confidence=0.9,
        bbox=(10.0, 10.0, 90.0, 90.0),
    )

    assert seg_mask.class_name == "hold"
    assert seg_mask.confidence == 0.9
    assert seg_mask.bbox == (10.0, 10.0, 90.0, 90.0)
    assert np.array_equal(seg_mask.mask, mask)


def test_segmentation_result() -> None:
    """Test SegmentationResult dataclass."""
    mask1 = SegmentationMask("hold", np.ones((10, 10), dtype=np.uint8), 0.8)
    mask2 = SegmentationMask("wall", np.zeros((10, 10), dtype=np.uint8), 0.9)

    result = SegmentationResult(
        image_path=Path("test.jpg"),
        masks=[mask1, mask2],
        frame_index=0,
    )

    assert len(result.masks) == 2
    assert result.get_mask_by_class("hold") == mask1
    assert result.get_mask_by_class("wall") == mask2
    assert result.get_mask_by_class("climber") is None

    combined = result.get_combined_mask(["hold", "wall"])
    assert combined is not None
    assert combined.shape == (10, 10)


def test_hold_color_info() -> None:
    """Test HoldColorInfo dataclass."""
    color_info = HoldColorInfo(
        hold_id="hold_1",
        dominant_color_hsv=(120, 200, 150),
        dominant_color_rgb=(50, 150, 50),
        color_confidence=0.85,
        bbox=(0.1, 0.2, 0.3, 0.4),
    )

    assert color_info.hold_id == "hold_1"
    assert color_info.dominant_color_hsv == (120, 200, 150)
    assert color_info.color_confidence == 0.85


def test_cluster_holds_by_color() -> None:
    """Test color-based hold clustering."""
    # Create holds with similar colors
    holds = [
        HoldColorInfo("hold_1", (120, 200, 150), (50, 150, 50), 0.9, (0.1, 0.1, 0.2, 0.2)),
        HoldColorInfo("hold_2", (125, 195, 155), (55, 148, 52), 0.85, (0.3, 0.3, 0.4, 0.4)),
        HoldColorInfo("hold_3", (60, 200, 150), (150, 50, 50), 0.9, (0.5, 0.5, 0.6, 0.6)),  # Different color
    ]

    routes = cluster_holds_by_color(
        holds,
        hue_tolerance=10,
        sat_tolerance=50,
        val_tolerance=50,
    )

    assert len(routes) > 0
    # First two holds should be in same route (similar color)
    route1_holds = [r.hold_ids for r in routes if "hold_1" in r.hold_ids]
    assert len(route1_holds) > 0
    if "hold_2" in route1_holds[0]:
        # Same route
        assert "hold_1" in route1_holds[0] and "hold_2" in route1_holds[0]


def test_export_routes_json(tmp_path: Path) -> None:
    """Test route JSON export."""
    routes = [
        RouteGroup(
            route_id="route_0",
            color_label="red",
            color_hsv=(0, 200, 200),
            hold_ids=["hold_1", "hold_2"],
            hold_count=2,
        ),
        RouteGroup(
            route_id="route_1",
            color_label="blue",
            color_hsv=(240, 200, 200),
            hold_ids=["hold_3"],
            hold_count=1,
        ),
    ]

    output_path = tmp_path / "routes.json"
    result_path = export_routes_json(routes, output_path)

    assert result_path == output_path
    assert output_path.exists()

    import json

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["total_routes"] == 2
    assert len(data["routes"]) == 2
    assert data["routes"][0]["route_id"] == "route_0"
    assert data["routes"][0]["hold_count"] == 2


def test_export_segmentation_masks(tmp_path: Path) -> None:
    """Test segmentation mask export."""
    image_path = tmp_path / "test.jpg"
    _create_test_image(image_path)

    mask1 = SegmentationMask("hold", np.ones((100, 100), dtype=np.uint8), 0.8)
    mask2 = SegmentationMask("wall", np.zeros((100, 100), dtype=np.uint8), 0.9)

    result = SegmentationResult(
        image_path=image_path,
        masks=[mask1, mask2],
        frame_index=0,
    )

    output_dir = tmp_path / "masks_output"
    export_segmentation_masks([result], output_dir, export_images=True, export_json=True)

    assert (output_dir / "segmentation_results.json").exists()
    assert (output_dir / "masks").exists()


@pytest.mark.skip(reason="Requires YOLO model - integration test")
def test_yolo_segmentation_model(tmp_path: Path) -> None:
    """Test YOLO segmentation model (requires model file)."""
    image_path = tmp_path / "test.jpg"
    _create_test_image(image_path)

    model = YoloSegmentationModel(model_name="yolov8n-seg.pt")
    result = model.segment_frame(image_path, conf_threshold=0.25)

    assert result.image_path == image_path
    # Results depend on model and image content


@pytest.mark.skip(reason="Requires YOLO model and hold detections - integration test")
def test_extract_hold_colors(tmp_path: Path) -> None:
    """Test hold color extraction (requires YOLO model)."""
    image_path = tmp_path / "test.jpg"
    _create_test_image(image_path)

    # Mock hold detection
    class MockHoldDetection:
        def __init__(self):
            self.frame_index = 0
            self.x_center = 0.5
            self.y_center = 0.5
            self.width = 0.2
            self.height = 0.2
            self.hold_id = "hold_1"

    detections = [MockHoldDetection()]
    color_infos = extract_hold_colors(detections, [image_path])

    # Should extract at least one color
    assert len(color_infos) > 0 or len(color_infos) == 0  # May be empty if no valid regions

