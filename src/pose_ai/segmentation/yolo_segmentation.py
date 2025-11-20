"""YOLO-based segmentation for wall, holds, and climber separation.

This module provides pixel-level segmentation using YOLO segmentation models,
with support for color-based route grouping.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO as UltralyticsYOLO  # type: ignore
except ModuleNotFoundError:
    UltralyticsYOLO = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SegmentationMask:
    """Single class mask from segmentation."""

    class_name: str
    mask: np.ndarray  # Binary mask (H, W)
    confidence: float
    bbox: Tuple[float, float, float, float] | None = None  # (x1, y1, x2, y2)


@dataclass(slots=True)
class SegmentationResult:
    """Segmentation results for a single frame."""

    image_path: Path
    masks: List[SegmentationMask]
    frame_index: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def get_mask_by_class(self, class_name: str) -> Optional[SegmentationMask]:
        """Get mask for a specific class."""
        for mask in self.masks:
            if mask.class_name.lower() == class_name.lower():
                return mask
        return None

    def get_combined_mask(self, class_names: Sequence[str]) -> Optional[np.ndarray]:
        """Get combined binary mask for multiple classes."""
        masks_to_combine = [
            m.mask for m in self.masks if m.class_name.lower() in [c.lower() for c in class_names]
        ]
        if not masks_to_combine:
            return None
        combined = np.zeros_like(masks_to_combine[0], dtype=np.uint8)
        for mask in masks_to_combine:
            combined = np.logical_or(combined, mask).astype(np.uint8)
        return combined


@dataclass(slots=True)
class HoldColorInfo:
    """Color information for a hold."""

    hold_id: str
    dominant_color_hsv: Tuple[int, int, int]  # (H, S, V)
    dominant_color_rgb: Tuple[int, int, int]  # (R, G, B)
    color_confidence: float  # How dominant the color is (0-1)
    bbox: Tuple[float, float, float, float]  # Normalized (x1, y1, x2, y2)


@dataclass(slots=True)
class RouteGroup:
    """A group of holds belonging to the same route/problem (same color)."""

    route_id: str
    color_label: str  # Human-readable color name
    color_hsv: Tuple[int, int, int]
    hold_ids: List[str]
    hold_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "route_id": self.route_id,
            "color_label": self.color_label,
            "color_hsv": list(self.color_hsv),
            "hold_ids": self.hold_ids,
            "hold_count": self.hold_count,
        }


class YoloSegmentationModel:
    """Wrapper for YOLO segmentation model."""

    def __init__(
        self,
        model_name: str = "yolov8n-seg.pt",
        device: str | None = None,
        imgsz: int = 640,
    ) -> None:
        if UltralyticsYOLO is None:
            raise ModuleNotFoundError(
                "ultralytics is required for YOLO segmentation. Install with `pip install ultralytics`."
            )
        self.model_name = model_name
        self.device = device
        self.imgsz = imgsz
        self._model: Optional[object] = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._model = UltralyticsYOLO(self.model_name)

    def segment_frame(
        self,
        image_path: Path | str,
        *,
        conf_threshold: float = 0.25,
        target_classes: Sequence[str] | None = None,
    ) -> SegmentationResult:
        """Segment a single frame."""
        self._ensure_model()
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self._model.predict(  # type: ignore[call-arg]
            source=str(image_path),
            device=self.device,
            imgsz=self.imgsz,
            conf=conf_threshold,
            verbose=False,
        )

        if not results:
            return SegmentationResult(image_path=image_path, masks=[])

        result = results[0]
        masks_list: List[SegmentationMask] = []

        # Check if segmentation masks are available
        if not hasattr(result, "masks") or result.masks is None:
            LOGGER.warning("No segmentation masks found in YOLO result. Model may not support segmentation.")
            return SegmentationResult(image_path=image_path, masks=[])

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return SegmentationResult(image_path=image_path, masks=[])

        xyxy = getattr(boxes, "xyxy", None)
        cls = getattr(boxes, "cls", None)
        conf = getattr(boxes, "conf", None)
        names = getattr(result, "names", {})

        # Extract masks
        masks = result.masks
        if hasattr(masks, "data"):
            mask_data = masks.data
        elif hasattr(masks, "cpu"):
            mask_data = masks.cpu().numpy()
        else:
            mask_data = np.array(masks) if isinstance(masks, (list, np.ndarray)) else None

        if mask_data is None:
            return SegmentationResult(image_path=image_path, masks=[])

        # Get image dimensions for mask resizing
        orig_shape = getattr(result, "orig_shape", None)
        if orig_shape:
            img_h, img_w = orig_shape[:2]
        else:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            img_h, img_w = img.shape[:2]

        # Process each detection
        if hasattr(xyxy, "cpu"):
            xyxy_list = xyxy.cpu().tolist()
            cls_list = cls.cpu().tolist() if cls is not None else []
            conf_list = conf.cpu().tolist() if conf is not None else []
        else:
            xyxy_list = xyxy.tolist() if hasattr(xyxy, "tolist") else []
            cls_list = cls.tolist() if hasattr(cls, "tolist") else []
            conf_list = conf.tolist() if hasattr(conf, "tolist") else []

        num_detections = len(xyxy_list)
        if isinstance(mask_data, np.ndarray) and len(mask_data.shape) == 3:
            num_masks = mask_data.shape[0]
        else:
            num_masks = 0

        for idx in range(min(num_detections, num_masks)):
            class_idx = int(cls_list[idx]) if idx < len(cls_list) else -1
            class_name = str(names.get(class_idx, class_idx)).lower()
            confidence = float(conf_list[idx]) if idx < len(conf_list) else 0.0

            # Filter by target classes if specified
            if target_classes and class_name not in [c.lower() for c in target_classes]:
                continue

            # Get mask for this detection
            if len(mask_data.shape) == 3:
                mask_tensor = mask_data[idx]
            else:
                continue

            # Resize mask to original image size
            if hasattr(mask_tensor, "cpu"):
                mask_array = mask_tensor.cpu().numpy()
            else:
                mask_array = np.array(mask_tensor)

            if mask_array.shape != (img_h, img_w):
                mask_resized = cv2.resize(mask_array.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
            else:
                mask_binary = (mask_array > 0.5).astype(np.uint8)

            bbox = tuple(float(v) for v in xyxy_list[idx][:4]) if idx < len(xyxy_list) else None

            masks_list.append(
                SegmentationMask(
                    class_name=class_name,
                    mask=mask_binary,
                    confidence=confidence,
                    bbox=bbox,
                )
            )

        return SegmentationResult(image_path=image_path, masks=masks_list)

    def batch_segment_frames(
        self,
        image_paths: Sequence[Path | str],
        *,
        conf_threshold: float = 0.25,
        target_classes: Sequence[str] | None = None,
    ) -> List[SegmentationResult]:
        """Segment multiple frames."""
        results: List[SegmentationResult] = []
        for idx, image_path in enumerate(image_paths):
            try:
                result = self.segment_frame(
                    image_path,
                    conf_threshold=conf_threshold,
                    target_classes=target_classes,
                )
                result.frame_index = idx
                results.append(result)
            except Exception as exc:
                LOGGER.warning("Failed to segment frame %s: %s", image_path, exc)
                results.append(SegmentationResult(image_path=Path(str(image_path)), masks=[], frame_index=idx))
        return results


def extract_hold_colors(
    hold_detections: Sequence[object],
    image_paths: Sequence[Path | str],
    *,
    segmentation_results: List[SegmentationResult] | None = None,
) -> List[HoldColorInfo]:
    """Extract dominant color from each hold region.

    Args:
        hold_detections: List of HoldDetection objects with bbox information
        image_paths: Paths to images corresponding to detections
        segmentation_results: Optional pre-computed segmentation results

    Returns:
        List of HoldColorInfo objects
    """
    color_infos: List[HoldColorInfo] = []

    # Group detections by frame
    detections_by_frame: dict[int, List[Tuple[int, object]]] = {}
    for idx, det in enumerate(hold_detections):
        frame_idx = getattr(det, "frame_index", 0)
        if frame_idx not in detections_by_frame:
            detections_by_frame[frame_idx] = []
        detections_by_frame[frame_idx].append((idx, det))

    for frame_idx, detections in detections_by_frame.items():
        if frame_idx >= len(image_paths):
            continue

        image_path = Path(str(image_paths[frame_idx]))
        if not image_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        img_h, img_w = image.shape[:2]
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Get segmentation mask for holds if available
        hold_mask: np.ndarray | None = None
        if segmentation_results and frame_idx < len(segmentation_results):
            seg_result = segmentation_results[frame_idx]
            hold_mask_obj = seg_result.get_mask_by_class("hold")
            if hold_mask_obj:
                hold_mask = hold_mask_obj.mask

        for det_idx, det in detections:
            # Get bounding box
            x_center = getattr(det, "x_center", 0.5)
            y_center = getattr(det, "y_center", 0.5)
            width = getattr(det, "width", 0.1)
            height = getattr(det, "height", 0.1)

            # Convert normalized coords to pixel coords
            x1 = int((x_center - width / 2) * img_w)
            y1 = int((y_center - height / 2) * img_h)
            x2 = int((x_center + width / 2) * img_w)
            y2 = int((y_center + height / 2) * img_h)

            # Clamp to image bounds
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # Extract region
            region_hsv = hsv_image[y1:y2, x1:x2]

            # Apply hold mask if available
            if hold_mask is not None:
                region_mask = hold_mask[y1:y2, x1:x2]
                if region_mask.size > 0:
                    masked_region = region_hsv[region_mask > 0]
                    if masked_region.size > 0:
                        region_hsv = masked_region.reshape(-1, 3)

            if region_hsv.size == 0:
                continue

            # Compute dominant color using K-means
            try:
                from sklearn.cluster import KMeans  # type: ignore

                pixels = region_hsv.reshape(-1, 3)
                if len(pixels) < 3:
                    # Fallback: use mean
                    dominant_hsv = tuple(int(v) for v in pixels.mean(axis=0))
                else:
                    kmeans = KMeans(n_clusters=min(3, len(pixels)), random_state=42, n_init=10)
                    kmeans.fit(pixels)
                    dominant_hsv = tuple(int(v) for v in kmeans.cluster_centers_[0])

                # Convert to RGB for reference
                hsv_array = np.uint8([[dominant_hsv]])
                rgb_array = cv2.cvtColor(hsv_array, cv2.COLOR_HSV2RGB)
                dominant_rgb = tuple(int(v) for v in rgb_array[0, 0])

                # Color confidence: ratio of pixels in dominant cluster
                if len(pixels) > 3:
                    labels = kmeans.labels_
                    dominant_count = np.sum(labels == 0)
                    color_confidence = dominant_count / len(labels)
                else:
                    color_confidence = 1.0

                hold_id = getattr(det, "hold_id", f"hold_{det_idx}")
                color_infos.append(
                    HoldColorInfo(
                        hold_id=str(hold_id),
                        dominant_color_hsv=dominant_hsv,
                        dominant_color_rgb=dominant_rgb,
                        color_confidence=float(color_confidence),
                        bbox=(x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h),
                    )
                )
            except ImportError:
                # Fallback: use mean color
                dominant_hsv = tuple(int(v) for v in pixels.mean(axis=0))
                hsv_array = np.uint8([[dominant_hsv]])
                rgb_array = cv2.cvtColor(hsv_array, cv2.COLOR_HSV2RGB)
                dominant_rgb = tuple(int(v) for v in rgb_array[0, 0])
                hold_id = getattr(det, "hold_id", f"hold_{det_idx}")
                color_infos.append(
                    HoldColorInfo(
                        hold_id=str(hold_id),
                        dominant_color_hsv=dominant_hsv,
                        dominant_color_rgb=dominant_rgb,
                        color_confidence=0.5,
                        bbox=(x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h),
                    )
                )

    return color_infos


def cluster_holds_by_color(
    holds_with_colors: Sequence[HoldColorInfo],
    *,
    hue_tolerance: int = 10,
    sat_tolerance: int = 50,
    val_tolerance: int = 50,
) -> List[RouteGroup]:
    """Group holds by similar color to identify routes/problems.

    Args:
        holds_with_colors: List of HoldColorInfo objects
        hue_tolerance: Maximum hue difference for same route (0-179)
        sat_tolerance: Maximum saturation difference (0-255)
        val_tolerance: Maximum value difference (0-255)

    Returns:
        List of RouteGroup objects
    """
    if not holds_with_colors:
        return []

    # Use DBSCAN-like clustering in HSV space
    try:
        from sklearn.cluster import DBSCAN  # type: ignore

        # Normalize HSV for clustering (hue is circular, so we need special handling)
        features = []
        for hold in holds_with_colors:
            h, s, v = hold.dominant_color_hsv
            # Normalize: hue in [0, 1], sat and val in [0, 1]
            # For hue, use sin/cos to handle circularity
            h_rad = np.radians(h * 2)  # Scale to 0-360 degrees
            features.append([np.sin(h_rad), np.cos(h_rad), s / 255.0, v / 255.0])

        features_array = np.array(features)
        # Adjust eps based on tolerances
        eps = np.sqrt(
            (np.sin(np.radians(hue_tolerance * 2)) ** 2 + np.cos(np.radians(hue_tolerance * 2)) ** 2)
            + (sat_tolerance / 255.0) ** 2
            + (val_tolerance / 255.0) ** 2
        )

        clustering = DBSCAN(eps=eps, min_samples=1).fit(features_array)
        cluster_ids = clustering.labels_

    except ImportError:
        # Fallback: simple greedy clustering
        cluster_ids = np.zeros(len(holds_with_colors), dtype=int)
        current_cluster = 0
        for i, hold1 in enumerate(holds_with_colors):
            if cluster_ids[i] != 0:  # Already assigned
                continue
            cluster_ids[i] = current_cluster
            h1, s1, v1 = hold1.dominant_color_hsv
            for j in range(i + 1, len(holds_with_colors)):
                if cluster_ids[j] != 0:
                    continue
                h2, s2, v2 = holds_with_colors[j].dominant_color_hsv
                # Check hue (circular)
                h_diff = min(abs(h1 - h2), 179 - abs(h1 - h2))
                if h_diff <= hue_tolerance and abs(s1 - s2) <= sat_tolerance and abs(v1 - v2) <= val_tolerance:
                    cluster_ids[j] = current_cluster
            current_cluster += 1

    # Group holds by cluster
    routes: List[RouteGroup] = []
    unique_clusters = sorted(set(cluster_ids))
    color_names = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "white", "black", "gray"]

    for cluster_id in unique_clusters:
        if cluster_id < 0:  # Noise points
            continue

        cluster_holds = [holds_with_colors[i] for i in range(len(holds_with_colors)) if cluster_ids[i] == cluster_id]
        if not cluster_holds:
            continue

        # Get average color
        avg_h = int(np.mean([h.dominant_color_hsv[0] for h in cluster_holds]))
        avg_s = int(np.mean([h.dominant_color_hsv[1] for h in cluster_holds]))
        avg_v = int(np.mean([h.dominant_color_hsv[2] for h in cluster_holds]))

        # Assign color label
        color_label = color_names[cluster_id % len(color_names)] if cluster_id < len(color_names) else f"color_{cluster_id}"

        routes.append(
            RouteGroup(
                route_id=f"route_{cluster_id}",
                color_label=color_label,
                color_hsv=(avg_h, avg_s, avg_v),
                hold_ids=[h.hold_id for h in cluster_holds],
                hold_count=len(cluster_holds),
            )
        )

    return routes


def export_segmentation_masks(
    segmentation_results: Sequence[SegmentationResult],
    output_dir: Path | str,
    *,
    export_images: bool = True,
    export_json: bool = True,
) -> Path:
    """Export segmentation masks to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = output_dir / "masks"
    if export_images:
        masks_dir.mkdir(exist_ok=True)

    results_data = []
    for result in segmentation_results:
        frame_data = {
            "image_path": str(result.image_path),
            "frame_index": result.frame_index,
            "masks": [],
        }
        for mask in result.masks:
            mask_data = {
                "class_name": mask.class_name,
                "confidence": mask.confidence,
            }
            if mask.bbox:
                mask_data["bbox"] = list(mask.bbox)

            if export_images:
                mask_filename = f"{result.image_path.stem}_{mask.class_name}_mask.png"
                mask_path = masks_dir / mask_filename
                cv2.imwrite(str(mask_path), mask.mask * 255)
                mask_data["mask_path"] = str(mask_path)

            frame_data["masks"].append(mask_data)
        results_data.append(frame_data)

    if export_json:
        json_path = output_dir / "segmentation_results.json"
        with json_path.open("w") as f:
            json.dump(results_data, f, indent=2)

    return output_dir


def export_routes_json(routes: Sequence[RouteGroup], output_path: Path | str) -> Path:
    """Export route groupings to JSON file."""
    output_path = Path(output_path)
    routes_data = {
        "routes": [route.as_dict() for route in routes],
        "total_routes": len(routes),
    }
    with output_path.open("w") as f:
        json.dump(routes_data, f, indent=2)
    return output_path


__all__ = [
    "SegmentationMask",
    "SegmentationResult",
    "HoldColorInfo",
    "RouteGroup",
    "YoloSegmentationModel",
    "extract_hold_colors",
    "cluster_holds_by_color",
    "export_segmentation_masks",
    "export_routes_json",
]

