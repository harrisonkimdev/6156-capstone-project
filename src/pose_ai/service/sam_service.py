"""SAM service for hold detection and labeling.

This module provides a high-level service interface for using SAM
in the hold labeling workflow.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..segmentation.sam_annotator import (
    SamAnnotator,
    SamSegment,
    LabeledSegment,
    segment_to_yolo_format,
)

LOGGER = logging.getLogger(__name__)

# Hold type to YOLO class ID mapping (from data/holds_training/dataset.yaml)
HOLD_TYPE_TO_CLASS_ID = {
    "crimp": 0,
    "sloper": 1,
    "jug": 2,
    "pinch": 3,
    "foot_only": 4,
    "volume": 5,
}

CLASS_ID_TO_HOLD_TYPE = {v: k for k, v in HOLD_TYPE_TO_CLASS_ID.items()}


class SamService:
    """Service for managing SAM-based hold segmentation and labeling."""
    
    def __init__(
        self,
        sam_checkpoint: Optional[Path] = None,
        device: str = "cpu",
        cache_dir: Optional[Path] = None,
        edge_filter_enabled: bool = True,
        edge_strength_thresh: Optional[float] = None,
    ):
        """Initialize SAM service.
        
        Args:
            sam_checkpoint: Path to SAM model checkpoint
            device: Device to run inference on (cpu, cuda, mps)
            cache_dir: Directory to cache segment data
            edge_filter_enabled: Whether to filter out low-edge (chalk-like) segments
            edge_strength_thresh: Optional absolute threshold for mean edge magnitude
        """
        self.sam_checkpoint = sam_checkpoint
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.edge_filter_enabled = edge_filter_enabled
        self.edge_strength_thresh = edge_strength_thresh
        
        self.annotator: Optional[SamAnnotator] = None
        self._segments_cache: Dict[str, List[SamSegment]] = {}
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def initialize(self, checkpoint_path: Optional[Path] = None) -> None:
        """Initialize the SAM model.
        
        Args:
            checkpoint_path: Path to SAM checkpoint (overrides __init__ path)
        """
        checkpoint = checkpoint_path or self.sam_checkpoint
        if checkpoint is None:
            raise ValueError("SAM checkpoint path must be provided")
        
        self.annotator = SamAnnotator(
            model_type="vit_b",  # Default to base model for speed
            checkpoint_path=checkpoint,
            device=self.device,
            edge_filter_enabled=self.edge_filter_enabled,
            edge_strength_thresh=self.edge_strength_thresh,
        )
        self.annotator.initialize()
        LOGGER.info(
            "SamService initialized with checkpoint: %s (edge_filter=%s)",
            checkpoint,
            self.edge_filter_enabled,
        )
    
    def is_ready(self) -> bool:
        """Check if service is ready to use."""
        return self.annotator is not None and self.annotator.is_initialized()
    
    def segment_frame(
        self,
        image_path: Path | str,
        use_cache: bool = True,
    ) -> List[SamSegment]:
        """Generate segments for a single frame.
        
        Args:
            image_path: Path to image file
            use_cache: Whether to use cached segments if available
            
        Returns:
            List of SamSegment objects
        """
        if not self.is_ready():
            raise RuntimeError("SamService not initialized. Call initialize() first.")
        
        image_path = Path(image_path)
        cache_key = str(image_path)
        
        # Check cache
        if use_cache and cache_key in self._segments_cache:
            LOGGER.debug("Using cached segments for %s", image_path.name)
            return self._segments_cache[cache_key]
        
        # Check file cache
        if use_cache and self.cache_dir:
            cache_file = self.cache_dir / f"{image_path.stem}_segments.json"
            if cache_file.exists():
                segments = self._load_segments_from_cache(cache_file, image_path)
                if segments:
                    self._segments_cache[cache_key] = segments
                    return segments
        
        # Generate new segments
        segments = self.annotator.segment_image(image_path)
        
        # Update cache
        self._segments_cache[cache_key] = segments
        if self.cache_dir:
            self._save_segments_to_cache(segments, image_path)
        
        return segments
    
    def segment_frames(
        self,
        image_paths: List[Path | str],
        use_cache: bool = True,
    ) -> List[List[SamSegment]]:
        """Generate segments for multiple frames.
        
        Args:
            image_paths: List of image paths
            use_cache: Whether to use cached segments if available
            
        Returns:
            List of segment lists (one per image)
        """
        all_segments = []
        for img_path in image_paths:
            segments = self.segment_frame(img_path, use_cache=use_cache)
            all_segments.append(segments)
        return all_segments
    
    def _save_segments_to_cache(
        self,
        segments: List[SamSegment],
        image_path: Path,
    ) -> None:
        """Save segments to cache file (without masks to save space)."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{image_path.stem}_segments.json"
        data = {
            "image_path": str(image_path),
            "segments": [seg.to_dict() for seg in segments],
        }
        
        try:
            cache_file.write_text(json.dumps(data, indent=2))
            LOGGER.debug("Saved segments cache to %s", cache_file)
        except Exception as exc:
            LOGGER.warning("Failed to save segments cache: %s", exc)
    
    def _load_segments_from_cache(
        self,
        cache_file: Path,
        image_path: Path,
    ) -> Optional[List[SamSegment]]:
        """Load segments from cache file (masks are regenerated)."""
        try:
            data = json.loads(cache_file.read_text())
            segments = []
            
            # Load image to regenerate masks
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            
            for seg_data in data["segments"]:
                # Note: Masks are not stored in cache, so we create dummy masks
                # In practice, for labeling UI we only need bbox info
                segment = SamSegment(
                    segment_id=seg_data["segment_id"],
                    mask=np.zeros(image.shape, dtype=bool),  # Dummy mask
                    bbox=tuple(seg_data["bbox"]),
                    area=seg_data["area"],
                    predicted_iou=seg_data["predicted_iou"],
                    stability_score=seg_data["stability_score"],
                )
                segments.append(segment)
            
            LOGGER.debug("Loaded segments from cache: %s", cache_file)
            return segments
            
        except Exception as exc:
            LOGGER.warning("Failed to load segments cache: %s", exc)
            return None
    
    def export_to_yolo(
        self,
        labeled_segments: List[LabeledSegment],
        image_path: Path,
        output_dir: Path,
        split: str = "train",
    ) -> None:
        """Export labeled segments to YOLO format.
        
        Args:
            labeled_segments: List of segments with hold type labels
            image_path: Path to source image
            output_dir: Output directory (should be data/holds_training/)
            split: Dataset split (train, val, test)
        """
        output_dir = Path(output_dir)
        images_dir = output_dir / "images" / split
        labels_dir = output_dir / "labels" / split
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        img_height, img_width = image.shape[:2]
        dst_image = images_dir / image_path.name
        cv2.imwrite(str(dst_image), image)
        
        # Generate YOLO labels
        label_lines = []
        for segment in labeled_segments:
            if not segment.is_hold or segment.hold_type is None:
                continue
            
            if segment.hold_type not in HOLD_TYPE_TO_CLASS_ID:
                LOGGER.warning("Unknown hold type: %s", segment.hold_type)
                continue
            
            class_id = HOLD_TYPE_TO_CLASS_ID[segment.hold_type]
            yolo_line = segment_to_yolo_format(segment, class_id, img_width, img_height)
            label_lines.append(yolo_line)
        
        # Save labels
        label_file = labels_dir / f"{image_path.stem}.txt"
        label_file.write_text("\n".join(label_lines) + "\n")
        
        LOGGER.info(
            "Exported %d holds from %s to %s split",
            len(label_lines),
            image_path.name,
            split,
        )
    
    def clear_cache(self) -> None:
        """Clear in-memory and file cache."""
        self._segments_cache.clear()
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*_segments.json"):
                cache_file.unlink()
            LOGGER.info("Cleared SAM segments cache")
    
    def close(self) -> None:
        """Clean up resources."""
        if self.annotator:
            self.annotator.close()
            self.annotator = None
        self._segments_cache.clear()
        LOGGER.info("SamService closed")


__all__ = [
    "SamService",
    "HOLD_TYPE_TO_CLASS_ID",
    "CLASS_ID_TO_HOLD_TYPE",
]
