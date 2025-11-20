"""Pose estimation utilities for climbing pose analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import cv2
import numpy as np

from .filters import exponential_smooth

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PoseLandmark:
    """Normalized landmark coordinates."""

    name: str
    x: float
    y: float
    z: float
    visibility: float


@dataclass(slots=True)
class PoseFrame:
    """Pose metadata associated with a single frame."""

    image_path: Path
    timestamp_seconds: float
    landmarks: List[PoseLandmark]
    detection_score: float


class PoseEstimator:
    """Run pose inference via MediaPipe or injected backends."""

    def __init__(
        self,
        *,
        engine_factory: Callable[[], object] | None = None,
        model_complexity: str = "mediapipe-full",
        landmark_names: Optional[Sequence[str]] = None,
        smoothing_alpha: float = 0.6,
    ) -> None:
        self._engine_factory = engine_factory
        self.model_complexity = model_complexity
        self._engine: Optional[object] = None
        self._landmark_names = list(landmark_names) if landmark_names else None
        self.smoothing_alpha = float(smoothing_alpha)

    # --------------------------------------------------------------------- #
    # Engine lifecycle
    # --------------------------------------------------------------------- #
    def _create_mediapipe_engine(self) -> object:
        try:
            import mediapipe as mp
        except ModuleNotFoundError as exc:  # pragma: no cover - requires optional dependency
            raise ModuleNotFoundError(
                "mediapipe is required for PoseEstimator. Install it with `pip install mediapipe` "
                "or inject a custom engine via `engine_factory`."
            ) from exc

        model_complexity = 2 if self.model_complexity in {"mediapipe-full", "full"} else 1
        self._landmark_names = [lm.name.lower() for lm in mp.solutions.pose.PoseLandmark]
        return mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def load(self) -> None:
        """Initialise the underlying pose engine."""
        if self._engine is not None:
            return
        factory = self._engine_factory or self._create_mediapipe_engine
        self._engine = factory()

    def close(self) -> None:
        """Release engine resources."""
        if self._engine is None:
            return
        close_fn = getattr(self._engine, "close", None)
        if callable(close_fn):
            close_fn()
        self._engine = None

    # --------------------------------------------------------------------- #
    # Inference helpers
    # --------------------------------------------------------------------- #
    def _ensure_landmark_names(self, count: int) -> List[str]:
        if self._landmark_names is None or len(self._landmark_names) < count:
            self._landmark_names = [f"landmark_{idx}" for idx in range(count)]
        return self._landmark_names

    def _landmarks_from_result(self, result) -> List[PoseLandmark]:
        pose_landmarks = getattr(result, "pose_landmarks", None)
        if pose_landmarks is None:
            return []
        raw = getattr(pose_landmarks, "landmark", None)
        if not raw:
            return []

        names = self._ensure_landmark_names(len(raw))
        converted: list[PoseLandmark] = []
        for idx, landmark in enumerate(raw):
            converted.append(
                PoseLandmark(
                    name=names[idx],
                    x=float(landmark.x),
                    y=float(landmark.y),
                    z=float(landmark.z),
                    visibility=float(getattr(landmark, "visibility", 0.0)),
                )
            )
        return converted

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def process_paths(
        self,
        image_paths: Sequence[Path],
        *,
        timestamps: Optional[Sequence[float]] = None,
    ) -> List[PoseFrame]:
        """Run pose inference on image paths."""
        self.load()
        if self._engine is None:
            raise RuntimeError("Pose engine failed to load.")

        results: list[PoseFrame] = []
        previous_landmarks: Optional[List[PoseLandmark]] = None

        for idx, path in enumerate(image_paths):
            timestamp = timestamps[idx] if timestamps and idx < len(timestamps) else float(idx)
            image = cv2.imread(str(path))
            if image is None:
                LOGGER.warning("Unable to read image %s; skipping pose detection.", path)
                results.append(PoseFrame(path, timestamp, [], 0.0))
                previous_landmarks = None
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            process_fn = getattr(self._engine, "process", None)
            if not callable(process_fn):
                raise AttributeError("Pose engine does not expose a callable `process` method.")

            result = process_fn(rgb)
            landmarks = self._landmarks_from_result(result)

            if landmarks and previous_landmarks and 0.0 < self.smoothing_alpha < 1.0:
                landmarks = exponential_smooth(previous_landmarks, landmarks, self.smoothing_alpha)

            detection_score = float(np.mean([lm.visibility for lm in landmarks])) if landmarks else 0.0
            results.append(PoseFrame(path, timestamp, landmarks, detection_score))
            previous_landmarks = landmarks if landmarks else None

        return results


__all__ = ["PoseEstimator", "PoseFrame", "PoseLandmark"]
