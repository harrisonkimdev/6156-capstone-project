"""Smoothing utilities for pose landmark sequences."""

from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imports only for type hints
    from .estimator import PoseLandmark


def exponential_smooth(
    previous: Sequence["PoseLandmark"] | None,
    current: Sequence["PoseLandmark"],
    alpha: float,
) -> list["PoseLandmark"]:
    """Apply exponential moving average between landmark sequences."""
    from .estimator import PoseLandmark

    if not current:
        return []
    if previous is None or len(previous) != len(current) or not (0.0 < alpha < 1.0):
        return list(current)

    beta = 1.0 - alpha
    smoothed: list[PoseLandmark] = []
    for prev, curr in zip(previous, current):
        smoothed.append(
            PoseLandmark(
                name=curr.name,
                x=alpha * curr.x + beta * prev.x,
                y=alpha * curr.y + beta * prev.y,
                z=alpha * curr.z + beta * prev.z,
                visibility=alpha * curr.visibility + beta * prev.visibility,
            )
        )
    return smoothed


__all__ = ["exponential_smooth"]
