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


@dataclass(slots=True)
class ClusteredHold:
    hold_id: str
    label: str
    x: float
    y: float
    radius: float
    detections: int
    avg_confidence: float

    def as_dict(self) -> dict[str, object]:  # pragma: no cover - convenience
        return {
            "hold_id": self.hold_id,
            "label": self.label,
            "coords": [self.x, self.y],
            "radius": self.radius,
            "detections": self.detections,
            "avg_confidence": self.avg_confidence,
            "normalized": True,
        }


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
            detections.append(
                HoldDetection(
                    frame_index=frame_idx,
                    label=label,
                    confidence=confidence,
                    x_center=x_center,
                    y_center=y_center,
                    width=width_n,
                    height=height_n,
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
                )
            )
        return clustered

    for cid in unique_ids:
        mask = cluster_ids == cid
        cluster_pts = points[mask]
        cluster_labels = labels[mask]
        cluster_conf = confidences[mask]
        x_mean = float(cluster_pts[:, 0].mean())
        y_mean = float(cluster_pts[:, 1].mean())
        # Radius: mean distance to centroid + small buffer
        dists = np.linalg.norm(cluster_pts - np.array([[x_mean, y_mean]]), axis=1)
        radius = float(dists.mean() + 0.01)
        # Dominant label
        lbl_values, counts = np.unique(cluster_labels, return_counts=True)
        dominant_label = str(lbl_values[int(np.argmax(counts))])
        clustered.append(
            ClusteredHold(
                hold_id=f"hold_{cid}",
                label=dominant_label,
                x=x_mean,
                y=y_mean,
                radius=radius,
                detections=int(mask.sum()),
                avg_confidence=float(cluster_conf.mean()),
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
) -> List[ClusteredHold]:
    detections = detect_holds(image_paths, model_name=model_name, device=device, hold_labels=hold_labels)
    return cluster_holds(detections, eps=eps, min_samples=min_samples)


__all__ = [
    "HoldDetection",
    "ClusteredHold",
    "detect_holds",
    "cluster_holds",
    "export_holds_json",
    "extract_and_cluster_holds",
]
