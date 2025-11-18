"""YOLO-powered helpers for selecting informative still frames."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

LOGGER = logging.getLogger(__name__)

try:  # Optional dependency; we fail gracefully when unavailable.
    from ultralytics import YOLO as UltralyticsYOLO  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised via fallback logic.
    UltralyticsYOLO = None  # type: ignore[assignment]
except Exception as exc:  # pragma: no cover - defensive import guard.
    LOGGER.warning("Failed to import ultralytics YOLO backend: %s", exc)
    UltralyticsYOLO = None  # type: ignore[assignment]


@dataclass(slots=True)
class Detection:
    """Normalized detection metadata returned by a YOLO model."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    class_id: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "class_id": self.class_id,
        }


@dataclass(slots=True)
class YoloSelectionResult:
    """Summary of how many frames were kept after YOLO filtering."""

    manifest_path: Path
    total_frames: int
    selected_frames: int
    skipped_reason: str | None = None


def _normalize_label(label: str, label_map: dict[str, str]) -> str:
    normalized = label.lower()
    return label_map.get(normalized, normalized)


def _coerce_detection(obj, label_map: dict[str, str]) -> Detection:
    if isinstance(obj, Detection):
        if obj.label == obj.label.lower():
            return obj
        return Detection(
            label=_normalize_label(obj.label, label_map),
            confidence=obj.confidence,
            bbox=obj.bbox,
            class_id=obj.class_id,
        )
    label = _normalize_label(str(getattr(obj, "label", getattr(obj, "name", "unknown"))), label_map)
    confidence = float(getattr(obj, "confidence", getattr(obj, "score", 0.0)))
    bbox = getattr(obj, "bbox", None) or getattr(obj, "xyxy", None)
    if bbox is None and hasattr(obj, "bbox_xyxy"):
        bbox = obj.bbox_xyxy
    if bbox is None:
        bbox = (0.0, 0.0, 0.0, 0.0)
    if isinstance(bbox, (list, tuple)):
        bbox_values = tuple(float(v) for v in bbox[:4])
    else:
        bbox_values = (0.0, 0.0, 0.0, 0.0)
    class_id = getattr(obj, "class_id", getattr(obj, "class_index", None))
    if class_id is not None:
        try:
            class_id = int(class_id)
        except (TypeError, ValueError):
            class_id = None
    return Detection(label=label, confidence=confidence, bbox=bbox_values, class_id=class_id)


class UltralyticsYoloSelector:
    """Thin wrapper around ``ultralytics.YOLO`` with lazy loading."""

    def __init__(
        self,
        *,
        model_name: str = "yolov8n.pt",
        device: str | None = None,
        imgsz: int = 640,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.imgsz = imgsz
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if UltralyticsYOLO is None:
            raise ModuleNotFoundError(
                "The `ultralytics` package is required for YOLO filtering. "
                "Install it via `pip install ultralytics`."
            )
        self._model = UltralyticsYOLO(self.model_name)

    def detect_many(self, image_paths: Sequence[Path]) -> List[List[Detection]]:
        self._ensure_model()
        if not image_paths:
            return []
        results = self._model.predict(  # type: ignore[call-arg]
            source=[str(path) for path in image_paths],
            device=self.device,
            imgsz=self.imgsz,
            stream=False,
            verbose=False,
        )
        detections: list[list[Detection]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                detections.append([])
                continue
            xyxy = getattr(boxes, "xyxy", None)
            cls = getattr(boxes, "cls", None)
            conf = getattr(boxes, "conf", None)
            xyxy_list = xyxy.cpu().tolist() if hasattr(xyxy, "cpu") else (xyxy.tolist() if hasattr(xyxy, "tolist") else [])
            cls_list = cls.cpu().tolist() if hasattr(cls, "cpu") else (cls.tolist() if hasattr(cls, "tolist") else [])
            conf_list = conf.cpu().tolist() if hasattr(conf, "cpu") else (conf.tolist() if hasattr(conf, "tolist") else [])
            names = getattr(result, "names", None) or {}
            frame_detections: list[Detection] = []
            for idx, bbox in enumerate(xyxy_list):
                class_idx = int(cls_list[idx]) if idx < len(cls_list) else None
                confidence = float(conf_list[idx]) if idx < len(conf_list) else 0.0
                label = names.get(class_idx, str(class_idx)) if isinstance(names, dict) else str(class_idx)
                frame_detections.append(
                    Detection(
                        label=str(label).lower(),
                        confidence=confidence,
                        bbox=tuple(float(v) for v in bbox[:4]),
                        class_id=class_idx,
                    )
                )
            detections.append(frame_detections)
        return detections


def annotate_manifest_with_yolo(
    manifest_path: Path | str,
    *,
    enabled: bool = True,
    selector: UltralyticsYoloSelector | None = None,
    min_confidence: float = 0.35,
    required_labels: Sequence[str] | None = None,
    target_labels: Sequence[str] | None = None,
    max_frames: int | None = None,
    label_map: dict[str, str] | None = None,
) -> YoloSelectionResult:
    """Filter frames inside ``manifest_path`` using YOLO detections."""

    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames: list[dict[str, object]] = list(payload.get("frames", []))
    total_frames = len(frames)

    if not enabled:
        return YoloSelectionResult(manifest_path, total_frames, total_frames, skipped_reason="disabled")
    if total_frames == 0:
        return YoloSelectionResult(manifest_path, total_frames, 0, skipped_reason="empty-manifest")

    selector_instance = selector
    if selector_instance is None:
        selector_instance = UltralyticsYoloSelector()

    lower_label_map = {key.lower(): value.lower() for key, value in (label_map or {}).items()}
    normalized_required = {label.lower() for label in (required_labels or [])}
    normalized_targets = {label.lower() for label in (target_labels or [])}

    frame_dir = manifest_path.parent
    image_paths = [frame_dir / str(entry.get("relative_path", "")) for entry in frames]

    try:
        if hasattr(selector_instance, "detect_many"):
            detections_per_frame = selector_instance.detect_many(image_paths)  # type: ignore[arg-type]
        else:  # pragma: no cover - convenience fallback
            detections_per_frame = [
                selector_instance.detect(path)  # type: ignore[attr-defined]
                for path in image_paths
            ]
    except ModuleNotFoundError:
        LOGGER.warning("YOLO dependency missing; manifest %s left untouched.", manifest_path)
        return YoloSelectionResult(manifest_path, total_frames, total_frames, skipped_reason="missing-dependency")
    except Exception as exc:  # pragma: no cover - defensive catch-all
        LOGGER.warning("YOLO filtering failed for %s: %s", manifest_path, exc)
        return YoloSelectionResult(manifest_path, total_frames, total_frames, skipped_reason="inference-error")

    if len(detections_per_frame) < len(frames):
        detections_per_frame.extend([[] for _ in range(len(frames) - len(detections_per_frame))])
    elif len(detections_per_frame) > len(frames):  # pragma: no cover - defensive
        detections_per_frame = detections_per_frame[: len(frames)]

    selected_entries: list[dict[str, object]] = []

    for entry, detections in zip(frames, detections_per_frame):
        normalized_detections = [
            _coerce_detection(det, lower_label_map) for det in detections
        ]
        target_matches = [
            det for det in normalized_detections if not normalized_targets or det.label in normalized_targets
        ]
        max_score = max((det.confidence for det in target_matches), default=0.0)
        has_required = (
            not normalized_required
            or any(det.label in normalized_required and det.confidence >= min_confidence for det in normalized_detections)
        )
        is_selected = has_required and (max_score >= min_confidence or not normalized_targets)
        reason = "meets-threshold" if is_selected else "below-threshold"

        entry_copy = dict(entry)
        entry_metadata = dict(entry_copy.get("metadata") or {})
        entry_metadata["yolo"] = {
            "selected": bool(is_selected),
            "score": float(max_score),
            "min_confidence": float(min_confidence),
            "required_labels": sorted(normalized_required),
            "target_labels": sorted(normalized_targets),
            "detections": [det.as_dict() for det in normalized_detections],
            "reason": reason,
        }
        entry_copy["metadata"] = entry_metadata
        if is_selected:
            selected_entries.append(entry_copy)

    if not selected_entries:
        LOGGER.warning(
            "YOLO filter kept 0/%d frames for %s; manifest left unchanged.",
            total_frames,
            manifest_path,
        )
        return YoloSelectionResult(manifest_path, total_frames, total_frames, skipped_reason="no-selected-frames")

    if max_frames is not None and max_frames > 0 and len(selected_entries) > max_frames:
        selected_entries = sorted(
            selected_entries,
            key=lambda row: float(row.get("metadata", {}).get("yolo", {}).get("score", 0.0)),
            reverse=True,
        )[:max_frames]
        selected_entries = sorted(selected_entries, key=lambda row: int(row.get("frame_index", 0)))

    for saved_idx, entry in enumerate(selected_entries):
        entry["saved_index"] = saved_idx

    payload["frames"] = selected_entries
    payload["saved_frames"] = len(selected_entries)
    payload["yolo_filter"] = {
        "model": getattr(selector_instance, "model_name", "unknown"),
        "min_confidence": min_confidence,
        "required_labels": sorted(normalized_required),
        "target_labels": sorted(normalized_targets),
        "total_frames": total_frames,
        "retained_frames": len(selected_entries),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return YoloSelectionResult(manifest_path, total_frames, len(selected_entries))


__all__ = [
    "Detection",
    "UltralyticsYoloSelector",
    "YoloSelectionResult",
    "annotate_manifest_with_yolo",
]
