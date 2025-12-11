"""SAM (Segment Anything Model) based automatic segmentation for hold detection.

This module provides automatic mask generation using Meta's Segment Anything Model,
converting masks to YOLO-compatible bounding boxes for hold labeling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class SamSegment:
    """A segment detected by SAM."""
    
    segment_id: str
    mask: np.ndarray  # Binary mask (H, W)
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    area: int
    predicted_iou: float
    stability_score: float
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (without mask for transport)."""
        return {
            "segment_id": self.segment_id,
            "bbox": self.bbox,
            "area": int(self.area),
            "predicted_iou": float(self.predicted_iou),
            "stability_score": float(self.stability_score),
        }
    
    def get_center(self) -> Tuple[float, float]:
        """Get the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def to_normalized_bbox(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert pixel bbox to normalized YOLO format (x_center, y_center, width, height)."""
        x1, y1, x2, y2 = self.bbox
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        return (x_center, y_center, width, height)


@dataclass
class LabeledSegment(SamSegment):
    """A SAM segment with user-assigned hold type label."""
    
    hold_type: Optional[str] = None  # crimp, sloper, jug, pinch, foot_only, volume
    is_hold: bool = False
    user_confirmed: bool = False
    
    def to_dict(self) -> dict:
        """Include label information in dict."""
        d = super().to_dict()
        d.update({
            "hold_type": self.hold_type,
            "is_hold": self.is_hold,
            "user_confirmed": self.user_confirmed,
        })
        return d


class SamAnnotator:
    """Wrapper for SAM model to generate hold segmentation masks.
    
    This class loads the SAM model and provides automatic mask generation
    for images, converting the output to labeled segments.
    """
    
    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_path: Optional[Path] = None,
        device: str = "cpu",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.90,
        min_mask_region_area: int = 50,
        edge_filter_enabled: bool = True,
        edge_strength_thresh: Optional[float] = None,
        edge_kernel_size: int = 3,
    ):
        """Initialize SAM annotator.
        
        Args:
            model_type: SAM model variant (vit_b, vit_l, vit_h)
            checkpoint_path: Path to SAM checkpoint file. If None, must be provided later.
            device: Device to run inference on (cpu, cuda, mps)
            points_per_side: Number of points per side for automatic mask generation
            pred_iou_thresh: Predicted IoU threshold for filtering masks
            stability_score_thresh: Stability score threshold for filtering masks
            min_mask_region_area: Minimum mask area in pixels
            edge_filter_enabled: Whether to filter out low-edge (chalk-like) segments
            edge_strength_thresh: Optional absolute threshold for mean edge magnitude
            edge_kernel_size: Kernel size for edge detector (Sobel)
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        self.edge_filter_enabled = edge_filter_enabled
        self.edge_strength_thresh = edge_strength_thresh
        self.edge_kernel_size = edge_kernel_size
        
        self.sam = None
        self.mask_generator = None
        self._initialized = False
        
        LOGGER.info(
            "Created SamAnnotator: model=%s, device=%s, points_per_side=%d",
            model_type,
            device,
            points_per_side,
        )
    
    def initialize(self, checkpoint_path: Optional[Path] = None) -> None:
        """Load SAM model and initialize mask generator.
        
        Args:
            checkpoint_path: Path to SAM checkpoint. Uses self.checkpoint_path if None.
        """
        if self._initialized:
            LOGGER.warning("SamAnnotator already initialized")
            return
        
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError as exc:
            raise ImportError(
                "segment-anything not installed. Run: pip install segment-anything"
            ) from exc
        
        checkpoint = checkpoint_path or self.checkpoint_path
        if checkpoint is None:
            raise ValueError(
                "checkpoint_path must be provided either in __init__ or initialize()"
            )
        
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        
        LOGGER.info("Loading SAM model from %s", checkpoint)
        
        try:
            self.sam = sam_model_registry[self.model_type](checkpoint=str(checkpoint))
            self.sam.to(device=self.device)
            
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                min_mask_region_area=self.min_mask_region_area,
            )
            
            self._initialized = True
            LOGGER.info("SAM model loaded successfully")
            
        except Exception as exc:
            LOGGER.error("Failed to load SAM model: %s", exc)
            raise
    
    def is_initialized(self) -> bool:
        """Check if SAM model is loaded and ready."""
        return self._initialized
    
    def segment_image(self, image_path: Path | str) -> List[SamSegment]:
        """Generate automatic segmentation masks for an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of SamSegment objects detected in the image
        """
        if not self._initialized:
            raise RuntimeError("SamAnnotator not initialized. Call initialize() first.")
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image in RGB format (SAM expects RGB)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        LOGGER.info("Running SAM inference on %s", image_path.name)
        
        try:
            masks = self.mask_generator.generate(image_rgb)
        except Exception as exc:
            LOGGER.error("SAM inference failed: %s", exc)
            raise
        
        num_masks_before_filter = len(masks)
        LOGGER.info("SAM generated %d initial masks for %s", num_masks_before_filter, image_path.name)
        
        # Get image dimensions for filtering
        img_height, img_width = image_rgb.shape[:2]
        img_area = img_height * img_width
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        # Global edge map for dynamic threshold
        grad_x_full = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=self.edge_kernel_size)
        grad_y_full = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=self.edge_kernel_size)
        edge_mag_full = cv2.magnitude(grad_x_full, grad_y_full)
        global_edge_mean = float(edge_mag_full.mean()) if edge_mag_full.size else 0.0
        # Dynamic default threshold: conservative (keep most, drop very low-edge regions)
        dynamic_edge_thresh = global_edge_mean * 0.8
        edge_thresh = (
            float(self.edge_strength_thresh)
            if self.edge_strength_thresh is not None
            else dynamic_edge_thresh
        )
        
        if self.edge_filter_enabled:
            LOGGER.info(
                "Edge filter enabled: threshold=%.3f (global_mean=%.3f, dynamic=%.3f)",
                edge_thresh,
                global_edge_mean,
                dynamic_edge_thresh,
            )
        
        # Convert SAM output to SamSegment objects with filtering
        segments = []
        edge_filtered_count = 0
        for idx, mask_data in enumerate(masks):
            segment_id = f"{image_path.stem}_seg_{idx:04d}"
            
            # Extract mask and bbox
            mask = mask_data["segmentation"]  # Binary mask (H, W)
            bbox = mask_data["bbox"]  # [x, y, w, h]
            
            # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
            x, y, w, h = bbox
            bbox_xyxy = (float(x), float(y), float(x + w), float(y + h))
            
            area = int(mask_data["area"])
            
            # Filter out segments that are likely walls or large backgrounds
            # 1. Too large (> 10% of image) - likely wall or background
            if area > img_area * 0.1:
                continue
            
            # 2. Extremely elongated (likely wall edges/boundaries)
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if aspect_ratio > 10:  # Very elongated
                continue
            
            # 3. Spans large portion of image width or height (likely wall boundary)
            if w > img_width * 0.6 or h > img_height * 0.6:
                continue
            
            # 4. Check for wall-like boundaries (very rectangular with large perimeter)
            perimeter = 2 * (w + h)
            bbox_area = w * h
            # If bounding box is much larger than actual mask area, it's likely a thin boundary
            if bbox_area > 0 and area / bbox_area < 0.3 and perimeter > min(img_width, img_height) * 0.5:
                continue
            
            # 5. Edge-strength filter to drop chalk-like smudges (weak boundaries)
            if self.edge_filter_enabled and edge_thresh > 0:
                mask_bool = mask.astype(bool)
                if mask_bool.any():
                    # Compute mean edge magnitude within mask
                    masked_edges = edge_mag_full[mask_bool]
                    mean_edge = float(masked_edges.mean()) if masked_edges.size else 0.0
                    if mean_edge < edge_thresh:
                        edge_filtered_count += 1
                        LOGGER.debug(
                            "Dropping segment %s for low edge strength (%.3f < %.3f)",
                            segment_id,
                            mean_edge,
                            edge_thresh,
                        )
                        continue
            
            segment = SamSegment(
                segment_id=segment_id,
                mask=mask,
                bbox=bbox_xyxy,
                area=area,
                predicted_iou=float(mask_data["predicted_iou"]),
                stability_score=float(mask_data["stability_score"]),
            )
            segments.append(segment)
        
        num_segments_after = len(segments)
        filter_summary = f"size/geometry: {num_masks_before_filter - num_segments_after - edge_filtered_count}"
        if self.edge_filter_enabled:
            filter_summary += f", edge: {edge_filtered_count}"
        
        LOGGER.info(
            "Generated %d segments (filtered from %d: %s) for %s",
            num_segments_after,
            num_masks_before_filter,
            filter_summary,
            image_path.name,
        )
        return segments
    
    def segment_batch(
        self,
        image_paths: List[Path | str],
    ) -> List[List[SamSegment]]:
        """Generate segmentation masks for multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of segment lists (one per image)
        """
        if not self._initialized:
            raise RuntimeError("SamAnnotator not initialized. Call initialize() first.")
        
        all_segments = []
        for img_path in image_paths:
            try:
                segments = self.segment_image(img_path)
                all_segments.append(segments)
            except Exception as exc:
                LOGGER.error("Failed to segment %s: %s", img_path, exc)
                all_segments.append([])
        
        total_segments = sum(len(segs) for segs in all_segments)
        LOGGER.info(
            "Batch segmentation complete: %d images, %d total segments",
            len(image_paths),
            total_segments,
        )
        
        return all_segments
    
    def close(self) -> None:
        """Clean up resources."""
        if self.sam is not None:
            # Move model to CPU to free GPU memory
            self.sam.to(device="cpu")
            del self.sam
            del self.mask_generator
            self.sam = None
            self.mask_generator = None
            self._initialized = False
            LOGGER.info("SAM model unloaded")


def visualize_segments(
    image_path: Path | str,
    segments: List[SamSegment],
    output_path: Optional[Path | str] = None,
    show_bbox: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    """Visualize SAM segments on the image.
    
    Args:
        image_path: Path to original image
        segments: List of SamSegment objects
        output_path: Optional path to save visualization
        show_bbox: Whether to draw bounding boxes
        show_labels: Whether to show segment IDs
        
    Returns:
        Image with segment overlays (BGR format)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    overlay = image.copy()
    
    # Create colormap for segments
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(segments), 3), dtype=np.uint8)
    
    for idx, segment in enumerate(segments):
        color = tuple(map(int, colors[idx]))
        
        # Apply mask overlay
        mask_color = np.zeros_like(image)
        mask_color[segment.mask] = color
        overlay = cv2.addWeighted(overlay, 1.0, mask_color, 0.3, 0)
        
        if show_bbox:
            # Draw bounding box
            x1, y1, x2, y2 = map(int, segment.bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        if show_labels:
            # Draw segment ID
            x1, y1 = map(int, segment.bbox[:2])
            label = f"#{idx}"
            cv2.putText(
                overlay,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
    
    if output_path:
        cv2.imwrite(str(output_path), overlay)
        LOGGER.info("Saved visualization to %s", output_path)
    
    return overlay


def segment_to_yolo_format(
    segment: SamSegment,
    class_id: int,
    img_width: int,
    img_height: int,
) -> str:
    """Convert a labeled segment to YOLO format line.
    
    Args:
        segment: SamSegment with bbox
        class_id: YOLO class ID (0-based)
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        YOLO format string: "class_id x_center y_center width height"
    """
    x_center, y_center, width, height = segment.to_normalized_bbox(img_width, img_height)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


__all__ = [
    "SamAnnotator",
    "SamSegment",
    "LabeledSegment",
    "visualize_segments",
    "segment_to_yolo_format",
]
