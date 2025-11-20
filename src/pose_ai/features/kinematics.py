"""Temporal derivative utilities for pose feature rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Sequence

import math


@dataclass(slots=True)
class DerivativeConfig:
    fps: float = 25.0
    tracked_fields: Sequence[str] = ("com", "left_hand", "right_hand", "left_foot", "right_foot")


def _ensure_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:  # pragma: no cover - defensive
        return None


def _diff(value_now: float | None, value_prev: float | None, fps: float) -> float | None:
    if value_now is None or value_prev is None:
        return None
    return (value_now - value_prev) * fps


def append_temporal_derivatives(
    feature_rows: Sequence[MutableMapping[str, object]],
    config: DerivativeConfig | None = None,
) -> Sequence[MutableMapping[str, object]]:
    """Compute velocity/acceleration/jerk for selected fields in-place."""
    if not feature_rows:
        return feature_rows

    cfg = config or DerivativeConfig()
    fps = float(cfg.fps or 25.0)
    prev_values: Dict[str, Dict[str, float | None]] = {}
    prev_velocity: Dict[str, Dict[str, float | None]] = {}

    for row in feature_rows:
        for field in cfg.tracked_fields:
            x_key = f"{field}_x"
            y_key = f"{field}_y"
            prev = prev_values.setdefault(field, {"x": None, "y": None})
            prev_v = prev_velocity.setdefault(field, {"x": None, "y": None})

            x = _ensure_float(row.get(x_key))
            y = _ensure_float(row.get(y_key))

            vx = _diff(x, prev["x"], fps)
            vy = _diff(y, prev["y"], fps)
            if vx is not None:
                row[f"{field}_vx"] = vx
            if vy is not None:
                row[f"{field}_vy"] = vy
            if vx is not None and vy is not None:
                row[f"{field}_speed"] = math.sqrt(vx**2 + vy**2)

            ax = _diff(vx, prev_v["x"], fps) if vx is not None and prev_v["x"] is not None else None
            ay = _diff(vy, prev_v["y"], fps) if vy is not None and prev_v["y"] is not None else None
            if ax is not None:
                row[f"{field}_ax"] = ax
            if ay is not None:
                row[f"{field}_ay"] = ay
            if ax is not None and ay is not None:
                row[f"{field}_accel"] = math.sqrt(ax**2 + ay**2)

            # jerk derived from acceleration deltas if enough history
            jerk_x_key = f"{field}_jerk_x"
            jerk_y_key = f"{field}_jerk_y"
            if ax is not None and "ax_prev" in prev_v:
                jerk_x = _diff(ax, prev_v["ax_prev"], fps)
                if jerk_x is not None:
                    row[jerk_x_key] = jerk_x
            if ay is not None and "ay_prev" in prev_v:
                jerk_y = _diff(ay, prev_v["ay_prev"], fps)
                if jerk_y is not None:
                    row[jerk_y_key] = jerk_y
            if row.get(jerk_x_key) is not None and row.get(jerk_y_key) is not None:
                jx = float(row[jerk_x_key])
                jy = float(row[jerk_y_key])
                row[f"{field}_jerk"] = math.sqrt(jx**2 + jy**2)

            prev["x"], prev["y"] = x, y
            prev_v["x"], prev_v["y"] = vx, vy
            prev_v["ax_prev"] = ax
            prev_v["ay_prev"] = ay

    return feature_rows


__all__ = ["append_temporal_derivatives", "DerivativeConfig"]
