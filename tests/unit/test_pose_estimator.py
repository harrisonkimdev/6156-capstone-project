from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pose_ai.pose import PoseEstimator


class _DummyResult:
    def __init__(self, coords):
        self.pose_landmarks = SimpleNamespace(
            landmark=[
                SimpleNamespace(x=x, y=y, z=z, visibility=vis) for (x, y, z, vis) in coords
            ]
        )


class _DummyEngine:
    def __init__(self, sequences):
        self._sequences = sequences
        self._idx = 0
        self.closed = False

    def process(self, image):  # pragma: no cover - trivial proxy
        try:
            coords = self._sequences[self._idx]
        except IndexError:
            coords = []
        self._idx += 1
        return _DummyResult(coords)

    def close(self) -> None:
        self.closed = True


def test_pose_estimator_applies_smoothing(monkeypatch, tmp_path: Path) -> None:
    image_paths = []
    for idx in range(2):
        path = tmp_path / f"{idx}.jpg"
        path.write_bytes(b"fake")
        image_paths.append(path)

    fake_cv2 = SimpleNamespace(
        imread=lambda path: np.zeros((4, 4, 3), dtype=np.uint8),
        cvtColor=lambda image, _: image,
        COLOR_BGR2RGB=1,
    )
    monkeypatch.setattr("pose_ai.pose.estimator.cv2", fake_cv2, raising=False)

    sequences = [
        [(0.1, 0.2, 0.0, 0.9), (0.2, 0.3, 0.0, 0.8)],
        [(0.3, 0.5, 0.0, 0.95), (0.4, 0.6, 0.0, 0.85)],
    ]
    engine = _DummyEngine(sequences)
    estimator = PoseEstimator(
        engine_factory=lambda: engine,
        landmark_names=["hip", "shoulder"],
        smoothing_alpha=0.5,
    )

    frames = estimator.process_paths(image_paths, timestamps=[0.0, 0.6])
    estimator.close()

    assert len(frames) == 2
    assert frames[0].landmarks[0].name == "hip"
    assert frames[0].landmarks[0].x == pytest.approx(0.1)
    # Smoothed value should lie between previous (0.1) and current (0.3)
    assert frames[1].landmarks[0].x == pytest.approx(0.2)
    assert engine.closed is True


def test_pose_estimator_requires_mediapipe(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise ModuleNotFoundError("mediapipe missing")

    monkeypatch.setattr(PoseEstimator, "_create_mediapipe_engine", _raise, raising=False)
    estimator = PoseEstimator()
    estimator._engine_factory = None
    with pytest.raises(ModuleNotFoundError):
        estimator.load()
