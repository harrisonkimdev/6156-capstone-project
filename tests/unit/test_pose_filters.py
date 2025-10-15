from __future__ import annotations

import pytest

from pose_ai.pose.estimator import PoseLandmark
from pose_ai.pose.filters import exponential_smooth


def test_exponential_smooth_blends_sequences():
    prev = [
        PoseLandmark(name="hip", x=0.2, y=0.3, z=0.0, visibility=0.7),
        PoseLandmark(name="shoulder", x=0.4, y=0.5, z=0.0, visibility=0.8),
    ]
    curr = [
        PoseLandmark(name="hip", x=0.4, y=0.6, z=0.0, visibility=0.9),
        PoseLandmark(name="shoulder", x=0.6, y=0.7, z=0.0, visibility=0.95),
    ]
    smoothed = exponential_smooth(prev, curr, alpha=0.5)
    assert smoothed[0].x == pytest.approx(0.3)
    assert smoothed[1].visibility == pytest.approx(0.875)
