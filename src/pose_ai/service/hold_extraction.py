"""YOLO-based hold extraction and clustering utilities.

Pipeline:
1. Run YOLO detections on a set of frame image paths.
2. Filter detections to hold-related labels (configurable).
3. Normalize bbox centers to [0,1] coordinates.
4. Cluster centers across frames to derive stable route-level hold IDs.

Clustering strategy: DBSCAN (density-based) for robustness to variable frame counts;
fallback to k-means when DBSCAN finds no clusters.

NOTE: This is an initial heuristic. Further improvements may incorporate temporal
association, IoU tracking, and visual descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import json
import math
import numpy as np

try:  # Optional dependency (ultralytics)
    from ultralytics import YOLO as UltralyticsYOLO  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    UltralyticsYOLO = None  # type: ignore[assignment]


@dataclass(slots=True)
class HoldDetection:
    frame_index: int
    label: str
    confidence: float
    x_center: float  # normalized 0-1
    y_center: float  # normalized 0-1
    width: float     # normalized 0-1
    height: float    # normalized 0-1
    hold_type: str | None = None  # Specific type: crimp, sloper, jug, pinch, foot_only, volume


@dataclass(slots=True)
class ClusteredHold:
    hold_id: str
    label: str
    x: float
    y: float
    radius: float
    detections: int
    avg_confidence: float
    hold_type: str | None = None  # Specific type: crimp, sloper, jug, pinch, foot_only, volume
    type_confidence: float | None = None  # Confidence of type prediction

    def as_dict(self) -> dict[str, object]:  # pragma: no cover - convenience
        result = {
            "hold_id": self.hold_id,
            "label": self.label,
            "coords": [self.x, self.y],
            "radius": self.radius,
            "detections": self.detections,
            "avg_confidence": self.avg_confidence,
            "normalized": True,
        }
        if self.hold_type is not None:
            result["hold_type"] = self.hold_type
        if self.type_confidence is not None:
            result["type_confidence"] = self.type_confidence
        return result


def _ensure_model(model_name: str) -> object:
    if UltralyticsYOLO is None:
        raise ModuleNotFoundError("ultralytics is required for hold extraction. Install with `pip install ultralytics`.")
    return UltralyticsYOLO(model_name)


def detect_holds(
    image_paths: Sequence[Path],
    *,
    model_name: str = "yolov8n.pt",
    device: str | None = None,
    imgsz: int = 640,
    hold_labels: Sequence[str] = ("hold", "foot_hold", "volume", "jug", "crimp", "sloper", "pinch"),
) -> List[HoldDetection]:
    model = _ensure_model(model_name)
    if not image_paths:
        return []
    results = model.predict(source=[str(p) for p in image_paths], device=device, imgsz=imgsz, stream=False, verbose=False)  # type: ignore[call-arg]
    detections: List[HoldDetection] = []
    for frame_idx, result in enumerate(results):
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = getattr(boxes, "xyxy", None)
        cls = getattr(boxes, "cls", None)
        conf = getattr(boxes, "conf", None)
        names = getattr(result, "names", {})
        if any(getattr(obj, "cpu", None) for obj in (xyxy, cls, conf)):
            xyxy_list = xyxy.cpu().tolist()
            cls_list = cls.cpu().tolist()
            conf_list = conf.cpu().tolist()
        else:
            xyxy_list = xyxy.tolist() if hasattr(xyxy, "tolist") else []
            cls_list = cls.tolist() if hasattr(cls, "tolist") else []
            conf_list = conf.tolist() if hasattr(conf, "tolist") else []
        for box_idx, bbox in enumerate(xyxy_list):
            class_idx = int(cls_list[box_idx]) if box_idx < len(cls_list) else -1
            label = str(names.get(class_idx, class_idx)).lower()
            confidence = float(conf_list[box_idx]) if box_idx < len(conf_list) else 0.0
            if label not in hold_labels:
                continue
            x1, y1, x2, y2 = (float(v) for v in bbox[:4])
            width = max(1e-6, x2 - x1)
            height = max(1e-6, y2 - y1)
            # Normalize by image size (assuming result.orig_shape)
            h, w = getattr(result, "orig_shape", (None, None))
            if h and w:
                x_center = (x1 + width / 2.0) / w
                y_center = (y1 + height / 2.0) / h
                width_n = width / w
                height_n = height / h
            else:
                # Fallback: treat coordinates as already normalized
                x_center = (x1 + width / 2.0)
                y_center = (y1 + height / 2.0)
                width_n = width
                height_n = height
            # Determine hold_type if label is specific type
            hold_type_labels = ("crimp", "sloper", "jug", "pinch", "foot_only", "volume")
            hold_type = label if label in hold_type_labels else None
            
            detections.append(
                HoldDetection(
                    frame_index=frame_idx,
                    label=label,
                    confidence=confidence,
                    x_center=x_center,
                    y_center=y_center,
                    width=width_n,
                    height=height_n,
                    hold_type=hold_type,
                )
            )
    return detections


def cluster_holds(
    detections: Sequence[HoldDetection],
    *,
    eps: float = 0.03,
    min_samples: int = 3,
) -> List[ClusteredHold]:
    if not detections:
        return []
    points = np.array([[d.x_center, d.y_center] for d in detections], dtype=float)
    labels = np.array([d.label for d in detections], dtype=object)
    confidences = np.array([d.confidence for d in detections], dtype=float)
    hold_types = np.array([d.hold_type if d.hold_type else "" for d in detections], dtype=object)

    try:
        from sklearn.cluster import DBSCAN  # type: ignore
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        cluster_ids = clustering.labels_
    except Exception:  # pragma: no cover - fallback
        cluster_ids = np.full(points.shape[0], -1, dtype=int)

    unique_ids = [cid for cid in sorted(set(int(c) for c in cluster_ids)) if cid >= 0]
    clustered: List[ClusteredHold] = []
    if not unique_ids:
        # Fallback: treat each detection as unique hold (could later merge via k-means)
        for idx, d in enumerate(detections):
            clustered.append(
                ClusteredHold(
                    hold_id=f"hold_{idx}",
                    label=d.label,
                    x=d.x_center,
                    y=d.y_center,
                    radius=max(d.width, d.height) / 2.0,
                    detections=1,
                    avg_confidence=d.confidence,
                    hold_type=d.hold_type,
                    type_confidence=d.confidence if d.hold_type else None,
                )
            )
        return clustered

    for cid in unique_ids:
        mask = cluster_ids == cid
        cluster_pts = points[mask]
        cluster_labels = labels[mask]
        cluster_conf = confidences[mask]
        cluster_types = hold_types[mask]
        x_mean = float(cluster_pts[:, 0].mean())
        y_mean = float(cluster_pts[:, 1].mean())
        # Radius: mean distance to centroid + small buffer
        dists = np.linalg.norm(cluster_pts - np.array([[x_mean, y_mean]]), axis=1)
        radius = float(dists.mean() + 0.01)
        # Dominant label
        lbl_values, counts = np.unique(cluster_labels, return_counts=True)
        dominant_label = str(lbl_values[int(np.argmax(counts))])
        # Dominant hold_type (if any)
        non_empty_types = cluster_types[cluster_types != ""]
        dominant_type = None
        type_conf = None
        if len(non_empty_types) > 0:
            type_values, type_counts = np.unique(non_empty_types, return_counts=True)
            dominant_type = str(type_values[int(np.argmax(type_counts))])
            # Average confidence of detections with this type
            type_mask = cluster_types == dominant_type
            type_conf = float(cluster_conf[type_mask].mean()) if type_mask.any() else None
        clustered.append(
            ClusteredHold(
                hold_id=f"hold_{cid}",
                label=dominant_label,
                x=x_mean,
                y=y_mean,
                radius=radius,
                detections=int(mask.sum()),
                avg_confidence=float(cluster_conf.mean()),
                hold_type=dominant_type,
                type_confidence=type_conf,
            )
        )
    return clustered


def track_holds(
    detections: Sequence[HoldDetection],
    *,
    iou_threshold: float = 0.5,
    max_age: int = 5,
    min_hits: int = 3,
    process_noise: float = 0.1,
    measurement_noise: float = 0.5,
) -> List["HoldTrack"]:
    """Track holds across frames using IoU matching and Kalman filtering.
    
    This provides temporal tracking with better stability than clustering alone.
    Holds are tracked frame-by-frame with:
    - IoU matching: Associate detections to existing tracks
    - Kalman filtering: Predict and update hold positions
    - Track management: Create, confirm, and delete tracks
    
    Args:
        detections: All hold detections from detect_holds()
        iou_threshold: Minimum IoU for detection-track matching
        max_age: Maximum frames without detection before track deletion
        min_hits: Minimum detections before track is confirmed
        process_noise: Kalman filter process noise
        measurement_noise: Kalman filter measurement noise
    
    Returns:
        List of confirmed HoldTrack objects
    """
    from .hold_tracking import IoUTracker
    
    if not detections:
        return []
    
    # Group detections by frame
    detections_by_frame: dict[int, List[HoldDetection]] = {}
    for det in detections:
        if det.frame_index not in detections_by_frame:
            detections_by_frame[det.frame_index] = []
        detections_by_frame[det.frame_index].append(det)
    
    # Initialize tracker
    tracker = IoUTracker(
        iou_threshold=iou_threshold,
        max_age=max_age,
        min_hits=min_hits,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
    )
    
    # Process frames in order
    for frame_idx in sorted(detections_by_frame.keys()):
        frame_dets = detections_by_frame[frame_idx]
        
        # Convert detections to tracker format: (bbox, label, hold_type, confidence)
        frame_dets_formatted = [
            (
                (det.x_center, det.y_center, det.width, det.height),
                det.label,
                det.hold_type,
                det.confidence,
            )
            for det in frame_dets
        ]
        
        # Update tracker with frame detections
        tracker.update_tracks(frame_idx, frame_dets_formatted)
    
    # Return confirmed tracks only
    return tracker.get_confirmed_tracks()


def cluster_tracks(
    tracks: Sequence["HoldTrack"],
    *,
    eps: float = 0.03,
    min_samples: int = 1,
) -> List[ClusteredHold]:
    """Cluster tracked holds to derive final route-level hold positions.
    
    This is similar to cluster_holds() but operates on track centroids
    using the track history for more stable position estimates.
    
    Args:
        tracks: List of HoldTrack objects from track_holds()
        eps: DBSCAN epsilon (spatial distance threshold)
        min_samples: DBSCAN minimum samples per cluster
    
    Returns:
        List of ClusteredHold objects
    """
    if not tracks:
        return []
    
    # Extract track centroids (using Kalman filtered positions)
    points = []
    track_info = []
    
    for track in tracks:
        # Use Kalman state for position (more stable than raw detections)
        x = float(track.kalman_state[0])
        y = float(track.kalman_state[1])
        points.append([x, y])
        
        # Average confidence from history
        if track.history:
            avg_conf = sum(h[3] for h in track.history) / len(track.history)
        else:
            avg_conf = 0.5
        
        track_info.append({
            "label": track.label,
            "hold_type": track.hold_type,
            "confidence": avg_conf,
            "hits": track.hits,
            "bbox": track.bbox,
        })
    
    points_array = np.array(points, dtype=float)
    
    # Cluster using DBSCAN
    try:
        from sklearn.cluster import DBSCAN  # type: ignore
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_array)
        cluster_ids = clustering.labels_
    except Exception:  # pragma: no cover - fallback
        cluster_ids = np.arange(len(points_array))
    
    unique_ids = [cid for cid in sorted(set(int(c) for c in cluster_ids)) if cid >= 0]
    clustered: List[ClusteredHold] = []
    
    # Handle unclustered tracks (noise points)
    if not unique_ids or -1 in cluster_ids:
        for idx, (point, info) in enumerate(zip(points, track_info)):
            if cluster_ids[idx] == -1 or not unique_ids:
                clustered.append(
                    ClusteredHold(
                        hold_id=f"track_{idx}",
                        label=info["label"],
                        x=point[0],
                        y=point[1],
                        radius=max(info["bbox"][2], info["bbox"][3]) / 2.0,
                        detections=info["hits"],
                        avg_confidence=info["confidence"],
                        hold_type=info["hold_type"],
                        type_confidence=info["confidence"] if info["hold_type"] else None,
                    )
                )
    
    # Process clusters
    for cid in unique_ids:
        mask = cluster_ids == cid
        cluster_pts = points_array[mask]
        cluster_info = [track_info[i] for i in range(len(track_info)) if mask[i]]
        
        # Cluster centroid
        x_mean = float(cluster_pts[:, 0].mean())
        y_mean = float(cluster_pts[:, 1].mean())
        
        # Radius: mean distance to centroid + buffer
        dists = np.linalg.norm(cluster_pts - np.array([[x_mean, y_mean]]), axis=1)
        radius = float(dists.mean() + 0.01)
        
        # Aggregate cluster properties
        labels = [info["label"] for info in cluster_info]
        hold_types = [info["hold_type"] for info in cluster_info if info["hold_type"]]
        confidences = [info["confidence"] for info in cluster_info]
        total_hits = sum(info["hits"] for info in cluster_info)
        
        # Dominant label
        from collections import Counter
        label_counts = Counter(labels)
        dominant_label = label_counts.most_common(1)[0][0]
        
        # Dominant hold type
        dominant_type = None
        type_conf = None
        if hold_types:
            type_counts = Counter(hold_types)
            dominant_type = type_counts.most_common(1)[0][0]
            # Average confidence of tracks with this type
            type_confs = [info["confidence"] for info in cluster_info if info["hold_type"] == dominant_type]
            type_conf = sum(type_confs) / len(type_confs) if type_confs else None
        
        clustered.append(
            ClusteredHold(
                hold_id=f"track_cluster_{cid}",
                label=dominant_label,
                x=x_mean,
                y=y_mean,
                radius=radius,
                detections=total_hits,
                avg_confidence=sum(confidences) / len(confidences),
                hold_type=dominant_type,
                type_confidence=type_conf,
            )
        )
    
    return clustered


def export_holds_json(
    clustered: Sequence[ClusteredHold],
    *,
    output_path: Path | str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {hold.hold_id: hold.as_dict() for hold in clustered}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def extract_and_cluster_holds(
    image_paths: Sequence[Path],
    *,
    model_name: str = "yolov8n.pt",
    device: str | None = None,
    hold_labels: Sequence[str] = ("hold", "foot_hold", "volume", "jug", "crimp", "sloper", "pinch"),
    eps: float = 0.03,
    min_samples: int = 3,
    use_tracking: bool = True,
    iou_threshold: float = 0.5,
    max_age: int = 5,
    min_hits: int = 3,
) -> List[ClusteredHold]:
    """Extract and cluster holds from video frames.
    
    Args:
        image_paths: Sequence of image file paths
        model_name: YOLO model name
        device: Device for YOLO inference
        hold_labels: Labels to filter for holds
        eps: DBSCAN epsilon for clustering
        min_samples: DBSCAN minimum samples
        use_tracking: If True, use temporal tracking (IoU + Kalman). If False, use old DBSCAN-only method.
        iou_threshold: IoU threshold for tracking (only used if use_tracking=True)
        max_age: Max frames without detection for tracking (only used if use_tracking=True)
        min_hits: Min detections for confirmed track (only used if use_tracking=True)
    
    Returns:
        List of ClusteredHold objects
    """
    detections = detect_holds(image_paths, model_name=model_name, device=device, hold_labels=hold_labels)
    
    if use_tracking:
        # New method: temporal tracking with IoU + Kalman
        tracks = track_holds(
            detections,
            iou_threshold=iou_threshold,
            max_age=max_age,
            min_hits=min_hits,
        )
        return cluster_tracks(tracks, eps=eps, min_samples=min_samples)
    else:
        # Old method: DBSCAN clustering only (backward compatibility)
        return cluster_holds(detections, eps=eps, min_samples=min_samples)


__all__ = [
    "HoldDetection",
    "ClusteredHold",
    "detect_holds",
    "cluster_holds",
    "track_holds",
    "cluster_tracks",
    "export_holds_json",
    "extract_and_cluster_holds",
]
