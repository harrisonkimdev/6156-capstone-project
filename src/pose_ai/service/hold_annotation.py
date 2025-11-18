"""Hold annotation utilities for creating YOLO-format training datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import json


# Hold type classes (YOLO class IDs)
HOLD_TYPES = {
    0: "crimp",
    1: "sloper",
    2: "jug",
    3: "pinch",
    4: "foot_only",
    5: "volume",
}

HOLD_TYPE_TO_ID = {v: k for k, v in HOLD_TYPES.items()}


@dataclass(slots=True)
class HoldAnnotation:
    """Single hold annotation in YOLO format."""
    class_id: int
    x_center: float  # normalized 0-1
    y_center: float  # normalized 0-1
    width: float     # normalized 0-1
    height: float    # normalized 0-1
    confidence: float = 1.0
    
    @property
    def hold_type(self) -> str:
        return HOLD_TYPES.get(self.class_id, "unknown")
    
    def to_yolo_line(self) -> str:
        """Convert to YOLO format line."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"
    
    @classmethod
    def from_yolo_line(cls, line: str) -> HoldAnnotation:
        """Parse YOLO format line."""
        parts = line.strip().split()
        return cls(
            class_id=int(parts[0]),
            x_center=float(parts[1]),
            y_center=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4]),
        )
    
    @classmethod
    def from_bbox(
        cls,
        hold_type: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        img_width: int,
        img_height: int,
    ) -> HoldAnnotation:
        """Create annotation from pixel bbox coordinates."""
        class_id = HOLD_TYPE_TO_ID.get(hold_type, 0)
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        return cls(
            class_id=class_id,
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
        )


def save_yolo_annotations(annotations: List[HoldAnnotation], output_path: Path) -> None:
    """Save annotations to YOLO format text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ann in annotations:
            f.write(ann.to_yolo_line() + "\n")


def load_yolo_annotations(label_path: Path) -> List[HoldAnnotation]:
    """Load annotations from YOLO format text file."""
    if not label_path.exists():
        return []
    annotations = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(HoldAnnotation.from_yolo_line(line))
    return annotations


def create_dataset_yaml(
    dataset_root: Path,
    train_dir: str = "images/train",
    val_dir: str = "images/val",
    test_dir: str | None = "images/test",
) -> Path:
    """Create YOLO dataset.yaml configuration file."""
    yaml_content = {
        "path": str(dataset_root.absolute()),
        "train": train_dir,
        "val": val_dir,
        "names": HOLD_TYPES,
    }
    if test_dir:
        yaml_content["test"] = test_dir
    
    yaml_path = dataset_root / "dataset.yaml"
    
    # Write YAML manually (avoid pyyaml dependency)
    with open(yaml_path, "w") as f:
        f.write(f"# Hold Type Classification Dataset\n")
        f.write(f"path: {dataset_root.absolute()}\n")
        f.write(f"train: {train_dir}\n")
        f.write(f"val: {val_dir}\n")
        if test_dir:
            f.write(f"test: {test_dir}\n")
        f.write(f"\n# Classes\n")
        f.write(f"names:\n")
        for class_id, name in HOLD_TYPES.items():
            f.write(f"  {class_id}: {name}\n")
    
    return yaml_path


def initialize_dataset_structure(dataset_root: Path) -> None:
    """Create directory structure for YOLO training dataset."""
    dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "images" / "test",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
        dataset_root / "labels" / "test",
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataset.yaml
    create_dataset_yaml(dataset_root)
    
    # Create README
    readme_path = dataset_root / "README.md"
    with open(readme_path, "w") as f:
        f.write("# Hold Type Classification Dataset\n\n")
        f.write("## Classes\n\n")
        for class_id, name in HOLD_TYPES.items():
            f.write(f"- {class_id}: {name}\n")
        f.write("\n## Dataset Structure\n\n")
        f.write("```\n")
        f.write("holds_training/\n")
        f.write("  ├── images/\n")
        f.write("  │   ├── train/    # Training images\n")
        f.write("  │   ├── val/      # Validation images\n")
        f.write("  │   └── test/     # Test images\n")
        f.write("  ├── labels/\n")
        f.write("  │   ├── train/    # Training labels (YOLO format)\n")
        f.write("  │   ├── val/      # Validation labels\n")
        f.write("  │   └── test/     # Test labels\n")
        f.write("  ├── dataset.yaml  # YOLO dataset configuration\n")
        f.write("  └── README.md     # This file\n")
        f.write("```\n\n")
        f.write("## YOLO Format\n\n")
        f.write("Each label file contains one line per object:\n")
        f.write("`<class_id> <x_center> <y_center> <width> <height>`\n\n")
        f.write("All coordinates are normalized to [0, 1].\n\n")
        f.write("## Annotation Tool\n\n")
        f.write("Use `python scripts/annotate_holds.py` to annotate images.\n")


def get_annotation_stats(dataset_root: Path) -> dict:
    """Get statistics about annotations in the dataset."""
    stats = {
        "train": {name: 0 for name in HOLD_TYPES.values()},
        "val": {name: 0 for name in HOLD_TYPES.values()},
        "test": {name: 0 for name in HOLD_TYPES.values()},
    }
    
    for split in ["train", "val", "test"]:
        labels_dir = dataset_root / "labels" / split
        if not labels_dir.exists():
            continue
        
        for label_file in labels_dir.glob("*.txt"):
            annotations = load_yolo_annotations(label_file)
            for ann in annotations:
                hold_type = ann.hold_type
                if hold_type in stats[split]:
                    stats[split][hold_type] += 1
    
    return stats


__all__ = [
    "HoldAnnotation",
    "HOLD_TYPES",
    "HOLD_TYPE_TO_ID",
    "save_yolo_annotations",
    "load_yolo_annotations",
    "create_dataset_yaml",
    "initialize_dataset_structure",
    "get_annotation_stats",
]

