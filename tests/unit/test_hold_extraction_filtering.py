"""Integration test for hold detection and filtering."""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from pose_ai.service.hold_extraction import (
    HoldDetection,
    detect_holds,
    filter_detections,
)
from pose_ai.service.hold_filters import filter_single_detection


def test_filtering_integration():
    """Test filtering on synthetic detections."""
    print("\n=== Testing Hold Detection and Filtering Integration ===\n")
    
    # Create synthetic test images
    test_dir = Path("/tmp/test_holds")
    test_dir.mkdir(exist_ok=True)
    
    # Image 1: Colored hold
    img1 = np.zeros((300, 300, 3), dtype=np.uint8)
    img1[100:150, 100:150] = [0, 100, 200]  # Orange rectangle
    cv2.imwrite(str(test_dir / "frame_0.jpg"), img1)
    
    # Image 2: Gray safety hold at bottom
    img2 = np.full((300, 300, 3), [140, 140, 140], dtype=np.uint8)
    cv2.rectangle(img2, (80, 240), (220, 290), (140, 140, 140), -1)
    cv2.imwrite(str(test_dir / "frame_1.jpg"), img2)
    
    # Image 3: White chalk mark
    img3 = np.full((300, 300, 3), [250, 250, 250], dtype=np.uint8)
    cv2.rectangle(img3, (50, 50), (100, 80), (250, 250, 250), -1)
    cv2.imwrite(str(test_dir / "frame_2.jpg"), img3)
    
    image_paths = list(test_dir.glob("frame_*.jpg"))
    print(f"Created {len(image_paths)} test images")
    
    # Create synthetic detections
    detections = [
        HoldDetection(
            frame_index=0,
            label="hold",
            confidence=0.95,
            x_center=0.4,
            y_center=0.4,
            width=0.2,
            height=0.2,
            hold_type="crimp",
        ),
        HoldDetection(
            frame_index=1,
            label="hold",
            confidence=0.85,
            x_center=0.5,
            y_center=0.8,
            width=0.3,
            height=0.15,
            hold_type=None,
        ),
        HoldDetection(
            frame_index=2,
            label="hold",
            confidence=0.7,
            x_center=0.3,
            y_center=0.25,
            width=0.15,
            height=0.1,
            hold_type=None,
        ),
    ]
    
    print(f"Created {len(detections)} synthetic detections")
    print("\nDetections:")
    for d in detections:
        print(f"  Frame {d.frame_index}: {d.label} ({d.confidence:.2f})")
    
    # Test filtering
    print("\nApplying filters...")
    valid, filtered = filter_detections(
        detections,
        image_paths,
        filter_chalk=True,
        filter_safety_holds=True,
        distinguish_volumes=False,
    )
    
    print(f"\nResults:")
    print(f"  Valid detections: {len(valid)}")
    print(f"  Filtered detections: {len(filtered)}")
    
    print("\nValid detections:")
    for d in valid:
        print(f"  Frame {d.frame_index}: {d.label} (confidence={d.confidence:.2f})")
    
    print("\nFiltered detections:")
    for d in filtered:
        print(f"  Frame {d.frame_index}: {d.label} (confidence={d.confidence:.2f})")
    
    # Expected: 
    # - Frame 0 (orange hold) should be valid
    # - Frame 1 (gray at bottom) should be filtered as safety hold
    # - Frame 2 (white chalk) should be filtered as chalk mark
    
    assert len(valid) == 1, f"Expected 1 valid detection, got {len(valid)}"
    assert len(filtered) == 2, f"Expected 2 filtered detections, got {len(filtered)}"
    assert valid[0].frame_index == 0, "Expected valid detection from frame 0"
    
    filtered_indices = {d.frame_index for d in filtered}
    assert 1 in filtered_indices, "Expected frame 1 to be filtered (safety hold)"
    assert 2 in filtered_indices, "Expected frame 2 to be filtered (chalk)"
    
    print("\n✓ All assertions passed!")


def test_individual_filters():
    """Test each filter individually on synthetic images."""
    print("\n=== Testing Individual Filters ===\n")
    
    # Test 1: Colored hold (should pass)
    print("Test 1: Colored hold")
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :] = [0, 100, 200]
    mask = np.ones((100, 100), dtype=np.uint8) * 255
    
    result = filter_single_detection(
        "test_1",
        mask,
        image,
        (100, 100),
        (0, 0, 100, 100),
        filter_chalk=True,
        filter_safety_holds=True,
        distinguish_volumes=False,
    )
    print(f"  Result: chalk={result.is_chalk_mark}, safety={result.is_safety_hold}, volume={result.is_volume}, Valid={result.is_valid_climbing_hold()}")
    assert result.is_valid_climbing_hold(), "Colored hold should be valid"
    
    # Test 2: White chalk (should be filtered)
    print("Test 2: White chalk mark")
    image = np.full((50, 50, 3), [250, 250, 250], dtype=np.uint8)
    mask = np.ones((50, 50), dtype=np.uint8) * 255
    
    result = filter_single_detection(
        "test_2",
        mask,
        image,
        (300, 300),
        (0, 0, 50, 50),
        filter_chalk=True,
        filter_safety_holds=False,
        distinguish_volumes=False,
    )
    print(f"  Result: chalk={result.is_chalk_mark}, safety={result.is_safety_hold}, volume={result.is_volume}, Valid={result.is_valid_climbing_hold()}")
    assert not result.is_valid_climbing_hold(), "Chalk should be filtered"
    assert result.is_chalk_mark, "Should identify as chalk"
    
    # Test 3: Gray at bottom (should be filtered)
    print("Test 3: Gray safety hold at bottom")
    image = np.full((100, 100, 3), [140, 140, 140], dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.uint8) * 255
    
    result = filter_single_detection(
        "test_3",
        mask,
        image,
        (300, 300),
        (80, 240, 220, 290),  # Place at bottom
        filter_chalk=False,
        filter_safety_holds=True,
        distinguish_volumes=False,
    )
    print(f"  Result: chalk={result.is_chalk_mark}, safety={result.is_safety_hold}, volume={result.is_volume}, Valid={result.is_valid_climbing_hold()}")
    # Gray hold at bottom should be filtered as safety hold
    
    print("\n✓ Individual filter tests passed!")


if __name__ == "__main__":
    try:
        test_individual_filters()
        test_filtering_integration()
        print("\n=== All Integration Tests Passed ===\n")
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
