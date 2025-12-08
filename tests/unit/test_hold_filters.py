"""Unit tests for hold_filters module."""

import pytest
import numpy as np
import cv2
from pathlib import Path

from src.pose_ai.service.hold_filters import (
    analyze_color_properties,
    analyze_shape_properties,
    analyze_texture_properties,
    is_chalk_mark,
    is_safety_hold,
    is_volume,
    filter_single_detection,
    FilterAnalysis,
)


class TestColorAnalysis:
    """Test color property analysis."""
    
    def test_white_color_detection(self):
        """Test detection of white chalk marks."""
        # Create a white region
        roi = np.full((100, 100, 3), [255, 255, 255], dtype=np.uint8)  # BGR white
        
        colors = analyze_color_properties(roi)
        
        assert colors["is_white"] is True
        assert colors["mean_saturation"] < 50
        assert colors["mean_value"] > 200
    
    def test_gray_color_detection(self):
        """Test detection of gray safety holds."""
        # Create a gray region
        gray_value = 150
        roi = np.full((100, 100, 3), [gray_value, gray_value, gray_value], dtype=np.uint8)
        
        colors = analyze_color_properties(roi)
        
        assert colors["is_gray"] is True
        assert colors["mean_saturation"] < 50
        assert 100 < colors["mean_value"] < 180
    
    def test_colored_hold_detection(self):
        """Test detection of colored holds (not white, not gray)."""
        # Create a blue region
        roi = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)  # BGR blue
        
        colors = analyze_color_properties(roi)
        
        assert colors["is_white"] is False
        assert colors["is_gray"] is False
        assert colors["mean_saturation"] > 50  # Colored has saturation


class TestShapeAnalysis:
    """Test shape property analysis."""
    
    def test_small_irregular_shape(self):
        """Test analysis of small, irregular chalk mark."""
        # Create a small, irregular mask
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Draw small irregular shape
        cv2.circle(mask, (100, 100), 5, 255, -1)
        cv2.ellipse(mask, (105, 105), (3, 8), 45, 0, 360, 255, -1)
        
        shape = analyze_shape_properties(mask)
        
        assert shape["area"] < 200
        assert shape["solidity"] < 0.5  # Irregular
    
    def test_large_rectangular_volume(self):
        """Test analysis of large rectangular volume."""
        # Create a large rectangle (like a volume)
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)
        
        shape = analyze_shape_properties(mask)
        
        assert shape["area"] > 5000
        assert shape["num_vertices"] <= 8  # Rectangle = 4 vertices
        assert shape["straightness"] > 0.95  # Straight edges
        assert 0.5 < shape["aspect_ratio"] < 2.0  # Square-ish


class TestTextureAnalysis:
    """Test texture property analysis."""
    
    def test_smooth_chalk_mark(self):
        """Test low texture of chalk mark (flat surface)."""
        # Create a smooth white region
        roi = np.full((100, 100, 3), [255, 255, 255], dtype=np.uint8)
        
        texture = analyze_texture_properties(roi)
        
        assert texture["mean_gradient"] < 5.0
        assert texture["has_texture"] is False
    
    def test_textured_hold(self):
        """Test high texture of real climbing hold."""
        # Create a hold with texture (noise + edges)
        roi = np.full((100, 100, 3), [100, 100, 100], dtype=np.uint8)
        # Add texture with Perlin-like noise
        noise = np.random.randint(0, 50, (100, 100, 3), dtype=np.uint8)
        roi = cv2.add(roi, noise)
        
        texture = analyze_texture_properties(roi)
        
        assert texture["mean_gradient"] > 2.0
        assert texture["laplacian_variance"] > 5.0


class TestChalkDetection:
    """Test chalk mark detection."""
    
    def test_white_small_flat_is_chalk(self):
        """Test that small white flat marks are detected as chalk."""
        # Create chalk-like detection: small, white, flat, irregular
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 8, 255, -1)  # Small
        
        roi = np.full((200, 200, 3), [255, 255, 255], dtype=np.uint8)  # White
        
        is_chalk, reason = is_chalk_mark(mask, roi, size_max=1000)
        
        assert is_chalk is True
        assert "white" in reason.lower() or "small" in reason.lower()
    
    def test_colored_hold_not_chalk(self):
        """Test that colored holds are not detected as chalk."""
        # Create a blue hold
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)
        
        roi = np.full((200, 200, 3), [255, 0, 0], dtype=np.uint8)  # Blue
        
        is_chalk, reason = is_chalk_mark(mask, roi)
        
        assert is_chalk is False


class TestSafetyHoldDetection:
    """Test safety hold detection."""
    
    def test_gray_at_bottom_is_safety(self):
        """Test that gray holds at bottom are detected as safety holds."""
        image_shape = (200, 200)
        
        # Create gray mask at bottom
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (50, 150), (150, 190), 255, -1)  # At bottom
        
        roi = np.full((200, 200, 3), [140, 140, 140], dtype=np.uint8)  # Gray
        bbox = (50, 150, 150, 190)  # (x1, y1, x2, y2)
        
        is_safety, reason = is_safety_hold(
            mask, roi, image_shape, bbox,
            bottom_threshold=0.85
        )
        
        assert is_safety is True
        assert "gray" in reason.lower() or "bottom" in reason.lower()
    
    def test_colored_hold_not_safety(self):
        """Test that colored holds are not detected as safety holds."""
        image_shape = (200, 200)
        
        # Create colored mask
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)
        
        roi = np.full((200, 200, 3), [255, 0, 0], dtype=np.uint8)  # Blue
        bbox = (50, 50, 150, 150)
        
        is_safety, reason = is_safety_hold(mask, roi, image_shape, bbox)
        
        assert is_safety is False


class TestVolumeDetection:
    """Test volume (non-hold structure) detection."""
    
    def test_large_rectangle_is_volume(self):
        """Test that large rectangular structures are detected as volumes."""
        # Create a large rectangle (volume)
        mask = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(mask, (100, 100), (400, 400), 255, -1)
        
        roi = np.random.randint(50, 150, (500, 500, 3), dtype=np.uint8)
        
        is_vol, reason = is_volume(mask, roi, min_area=5000)
        
        assert is_vol is True
        assert "polygonal" in reason.lower() or "large" in reason.lower()
    
    def test_small_hold_not_volume(self):
        """Test that small holds are not detected as volumes."""
        # Create a small mask
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 20, 255, -1)
        
        roi = np.random.randint(50, 150, (200, 200, 3), dtype=np.uint8)
        
        is_vol, reason = is_volume(mask, roi, min_area=5000)
        
        assert is_vol is False


class TestFilterSingleDetection:
    """Test complete filtering of single detections."""
    
    def test_valid_colored_hold_passes_filter(self):
        """Test that colored holds pass all filters."""
        detection_id = "test_hold_1"
        image_shape = (200, 200)
        
        # Create a colored hold mask
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (150, 100), 255, -1)
        
        # Blue colored
        roi = np.full((200, 200, 3), [255, 0, 0], dtype=np.uint8)
        bbox = (50, 50, 150, 100)
        
        analysis = filter_single_detection(
            detection_id, mask, roi, image_shape, bbox,
            filter_chalk=True,
            filter_safety_holds=True,
            distinguish_volumes=True,
        )
        
        assert analysis.is_valid_climbing_hold() is True
        assert analysis.is_chalk_mark is False
        assert analysis.is_safety_hold is False
        assert analysis.is_volume is False
    
    def test_chalk_mark_filtered(self):
        """Test that chalk marks are filtered out."""
        detection_id = "chalk_1"
        image_shape = (200, 200)
        
        # Create white chalk mark
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 5, 255, -1)
        
        roi = np.full((200, 200, 3), [255, 255, 255], dtype=np.uint8)
        bbox = (95, 95, 105, 105)
        
        analysis = filter_single_detection(
            detection_id, mask, roi, image_shape, bbox,
            filter_chalk=True,
        )
        
        assert analysis.is_chalk_mark is True
        assert analysis.is_valid_climbing_hold() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
