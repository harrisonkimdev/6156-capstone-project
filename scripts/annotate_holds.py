#!/usr/bin/env python3
"""Interactive hold annotation tool for creating YOLO training datasets.

Usage:
    python scripts/annotate_holds.py <image_dir> --output data/holds_training
    
Controls:
    - Click and drag to draw bounding box
    - Number keys 1-6 to select hold type
    - 'd' to delete last annotation
    - 's' to save and move to next image
    - 'q' to quit
    - 'n' to skip to next image without saving
"""

import argparse
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: OpenCV is required. Install with: pip install opencv-python")
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pose_ai.service.hold_annotation import (
    HoldAnnotation,
    HOLD_TYPES,
    HOLD_TYPE_TO_ID,
    save_yolo_annotations,
    load_yolo_annotations,
)


class HoldAnnotator:
    """Interactive hold annotation tool."""
    
    def __init__(self, image_paths: list[Path], output_root: Path):
        self.image_paths = image_paths
        self.output_root = output_root
        self.current_idx = 0
        
        self.image = None
        self.display_image = None
        self.annotations = []
        self.current_hold_type = "crimp"  # Default
        
        # Drawing state
        self.drawing = False
        self.start_point = None
        self.end_point = None
        
        # Colors for each hold type (BGR)
        self.colors = {
            "crimp": (0, 0, 255),      # Red
            "sloper": (0, 255, 0),     # Green
            "jug": (255, 0, 0),        # Blue
            "pinch": (0, 255, 255),    # Yellow
            "foot_only": (255, 0, 255),# Magenta
            "volume": (255, 255, 0),   # Cyan
        }
        
        # Ensure output directories exist
        (output_root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    
    def load_image(self, idx: int) -> bool:
        """Load image and existing annotations."""
        if idx < 0 or idx >= len(self.image_paths):
            return False
        
        self.current_idx = idx
        img_path = self.image_paths[idx]
        self.image = cv2.imread(str(img_path))
        
        if self.image is None:
            print(f"Error: Could not load {img_path}")
            return False
        
        # Load existing annotations if they exist
        label_name = img_path.stem + ".txt"
        label_path = self.output_root / "labels" / "train" / label_name
        
        if label_path.exists():
            self.annotations = load_yolo_annotations(label_path)
        else:
            self.annotations = []
        
        self.display_image = self.image.copy()
        return True
    
    def draw_annotations(self):
        """Draw all annotations on display image."""
        self.display_image = self.image.copy()
        h, w = self.image.shape[:2]
        
        for ann in self.annotations:
            # Convert normalized coords to pixel coords
            x_center = int(ann.x_center * w)
            y_center = int(ann.y_center * h)
            box_w = int(ann.width * w)
            box_h = int(ann.height * h)
            
            x1 = x_center - box_w // 2
            y1 = y_center - box_h // 2
            x2 = x_center + box_w // 2
            y2 = y_center + box_h // 2
            
            color = self.colors.get(ann.hold_type, (128, 128, 128))
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = ann.hold_type
            cv2.putText(
                self.display_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
    
    def draw_info(self):
        """Draw info panel on image."""
        h, w = self.display_image.shape[:2]
        
        # Draw semi-transparent panel
        panel_h = 120
        overlay = self.display_image.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, self.display_image, 0.5, 0, self.display_image)
        
        # Draw text
        y_offset = 20
        cv2.putText(
            self.display_image,
            f"Image {self.current_idx + 1}/{len(self.image_paths)}: {self.image_paths[self.current_idx].name}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        y_offset += 25
        cv2.putText(
            self.display_image,
            f"Current Type: {self.current_hold_type}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.colors[self.current_hold_type],
            2
        )
        
        y_offset += 25
        cv2.putText(
            self.display_image,
            "1:crimp 2:sloper 3:jug 4:pinch 5:foot_only 6:volume",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        
        y_offset += 20
        cv2.putText(
            self.display_image,
            "s:save  d:delete  n:next  q:quit",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                # Draw current box
                temp_img = self.display_image.copy()
                cv2.rectangle(
                    temp_img,
                    self.start_point,
                    self.end_point,
                    self.colors[self.current_hold_type],
                    2
                )
                self.draw_info()
                cv2.imshow("Annotate Holds", temp_img)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.end_point = (x, y)
                
                # Create annotation
                h, w = self.image.shape[:2]
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                
                # Only add if box has reasonable size
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    ann = HoldAnnotation.from_bbox(
                        self.current_hold_type,
                        x1, y1, x2, y2,
                        w, h
                    )
                    self.annotations.append(ann)
                
                self.draw_annotations()
                self.draw_info()
                cv2.imshow("Annotate Holds", self.display_image)
    
    def save_annotations(self):
        """Save annotations to file."""
        img_path = self.image_paths[self.current_idx]
        
        # Copy image to output
        output_img_path = self.output_root / "images" / "train" / img_path.name
        cv2.imwrite(str(output_img_path), self.image)
        
        # Save annotations
        label_name = img_path.stem + ".txt"
        label_path = self.output_root / "labels" / "train" / label_name
        save_yolo_annotations(self.annotations, label_path)
        
        print(f"Saved: {len(self.annotations)} annotations")
    
    def run(self):
        """Run the annotation tool."""
        if not self.load_image(0):
            print("No images to annotate")
            return
        
        cv2.namedWindow("Annotate Holds")
        cv2.setMouseCallback("Annotate Holds", self.mouse_callback)
        
        self.draw_annotations()
        self.draw_info()
        cv2.imshow("Annotate Holds", self.display_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Select hold type
            if key == ord('1'):
                self.current_hold_type = "crimp"
                self.draw_info()
                cv2.imshow("Annotate Holds", self.display_image)
            elif key == ord('2'):
                self.current_hold_type = "sloper"
                self.draw_info()
                cv2.imshow("Annotate Holds", self.display_image)
            elif key == ord('3'):
                self.current_hold_type = "jug"
                self.draw_info()
                cv2.imshow("Annotate Holds", self.display_image)
            elif key == ord('4'):
                self.current_hold_type = "pinch"
                self.draw_info()
                cv2.imshow("Annotate Holds", self.display_image)
            elif key == ord('5'):
                self.current_hold_type = "foot_only"
                self.draw_info()
                cv2.imshow("Annotate Holds", self.display_image)
            elif key == ord('6'):
                self.current_hold_type = "volume"
                self.draw_info()
                cv2.imshow("Annotate Holds", self.display_image)
            
            # Delete last annotation
            elif key == ord('d'):
                if self.annotations:
                    self.annotations.pop()
                    self.draw_annotations()
                    self.draw_info()
                    cv2.imshow("Annotate Holds", self.display_image)
            
            # Save and next
            elif key == ord('s'):
                self.save_annotations()
                if self.current_idx + 1 < len(self.image_paths):
                    if self.load_image(self.current_idx + 1):
                        self.draw_annotations()
                        self.draw_info()
                        cv2.imshow("Annotate Holds", self.display_image)
                else:
                    print("All images annotated!")
                    break
            
            # Next without saving
            elif key == ord('n'):
                if self.current_idx + 1 < len(self.image_paths):
                    if self.load_image(self.current_idx + 1):
                        self.draw_annotations()
                        self.draw_info()
                        cv2.imshow("Annotate Holds", self.display_image)
                else:
                    print("All images annotated!")
                    break
            
            # Quit
            elif key == ord('q'):
                print("Quitting...")
                break
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive hold annotation tool")
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing images to annotate"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/holds_training"),
        help="Output directory for dataset (default: data/holds_training)"
    )
    parser.add_argument(
        "--pattern",
        default="*.jpg",
        help="File pattern to match (default: *.jpg)"
    )
    
    args = parser.parse_args()
    
    # Find images
    image_paths = sorted(args.image_dir.glob(args.pattern))
    if not image_paths:
        print(f"No images found in {args.image_dir} matching {args.pattern}")
        return
    
    print(f"Found {len(image_paths)} images to annotate")
    
    # Initialize dataset structure
    from pose_ai.service.hold_annotation import initialize_dataset_structure
    initialize_dataset_structure(args.output)
    print(f"Initialized dataset structure at {args.output}")
    
    # Run annotator
    annotator = HoldAnnotator(image_paths, args.output)
    annotator.run()
    
    # Print stats
    from pose_ai.service.hold_annotation import get_annotation_stats
    stats = get_annotation_stats(args.output)
    print("\nAnnotation Statistics:")
    for split, counts in stats.items():
        total = sum(counts.values())
        if total > 0:
            print(f"\n{split.upper()}:")
            for hold_type, count in counts.items():
                print(f"  {hold_type}: {count}")


if __name__ == "__main__":
    main()

