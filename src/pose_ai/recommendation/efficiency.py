"""Step-level efficiency scoring and next-action heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import math

from pose_ai.segmentation.steps import StepSegment


Point = Tuple[float, float]


@dataclass(slots=True)
class EfficiencyWeights:
    stability: float = 0.35
    path: float = 0.25
    support: float = 0.20
    wall: float = 0.10
    smoothness: float = 0.07
    reach: float = 0.03


@dataclass(slots=True)
class StepEfficiencyResult:
    step_id: int
    score: float
    components: Dict[str, float]
    start_time: float
    end_time: float

    def as_dict(self) -> Dict[str, float | int]:
        payload = {
            "step_id": self.step_id,
            "score": self.score,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
        payload.update(self.components)
        return payload


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:  # pragma: no cover
        return None


def _collect_support_points(row: MutableMapping[str, object]) -> List[Point]:
    points: List[Point] = []
    for limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
        if not row.get(f"{limb}_contact_on"):
            continue
        hx = _safe_float(row.get(f"{limb}_target_x"))
        hy = _safe_float(row.get(f"{limb}_target_y"))
        if hx is not None and hy is not None:
            points.append((hx, hy))
        else:
            lx = _safe_float(row.get(f"{limb}_x"))
            ly = _safe_float(row.get(f"{limb}_y"))
            if lx is not None and ly is not None:
                points.append((lx, ly))
    return points


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        cond = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-6) + x1)
        if cond:
            inside = not inside
    return inside


def _distance_point_to_segment(point: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = point
    abx = bx - ax
    aby = by - ay
    if abs(abx) < 1e-9 and abs(aby) < 1e-9:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby)))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def _distance_to_polygon(point: Point, polygon: Sequence[Point]) -> float:
    if not polygon:
        return float("inf")
    if len(polygon) == 1:
        px, py = polygon[0]
        return math.sqrt((point[0] - px) ** 2 + (point[1] - py) ** 2)
    if _point_in_polygon(point, polygon):
        return 0.0
    distances = [
        _distance_point_to_segment(point, polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    ]
    return min(distances)


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


class StepEfficiencyComputer:
    def __init__(
        self,
        *,
        weights: EfficiencyWeights | None = None,
        reach_threshold: float = 1.1,
        wall_reference: float = 0.0,
        stability_alpha: float = 4.0,
    ) -> None:
        self.weights = weights or EfficiencyWeights()
        self.reach_threshold = reach_threshold
        self.wall_reference = wall_reference
        self.stability_alpha = stability_alpha

    def score_step(
        self,
        step_frames: Sequence[MutableMapping[str, object]],
        segment: StepSegment,
    ) -> StepEfficiencyResult:
        if not step_frames:
            return StepEfficiencyResult(
                step_id=segment.step_id,
                score=0.0,
                components={},
                start_time=segment.start_time,
                end_time=segment.end_time,
            )

        stability_scores: List[float] = []
        support_penalties: List[float] = []
        wall_penalties: List[float] = []
        jerk_penalties: List[float] = []
        reach_penalties: List[float] = []

        prev_row: MutableMapping[str, object] | None = None
        switch_events: List[int] = []

        for row in step_frames:
            body_scale = _safe_float(row.get("body_scale")) or 1.0
            com_point = (_safe_float(row.get("com_x")), _safe_float(row.get("com_y")))
            support_points = _collect_support_points(row)
            if None not in com_point and support_points:
                distance = _distance_to_polygon(com_point, support_points)
                stability = math.exp(-self.stability_alpha * (distance / ((body_scale or 1.0) + 1e-6)))
            else:
                stability = 0.0
            stability_scores.append(stability)

            support_count = len(support_points)
            support_penalty = 1.0 if support_count < 2 else 0.0
            if prev_row is not None:
                switches = 0
                for limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
                    if (
                        row.get(f"{limb}_contact_hold") != prev_row.get(f"{limb}_contact_hold")
                        or row.get(f"{limb}_contact_type") != prev_row.get(f"{limb}_contact_type")
                    ):
                        switches += 1
                switch_events.append(switches)
                support_penalty += 0.1 * switches
            support_penalties.append(support_penalty)

            com_perp = _safe_float(row.get("com_perp_wall"))
            wall_penalty = max(0.0, com_perp - self.wall_reference) if com_perp is not None else 0.0
            wall_penalties.append(wall_penalty)

            com_jerk = _safe_float(row.get("com_jerk"))
            jerk_penalties.append((com_jerk or 0.0) / ((body_scale or 1.0) + 1e-6))

            hip_x = _safe_float(row.get("hip_center_x"))
            hip_y = _safe_float(row.get("hip_center_y"))
            reach_norm = 0.0
            if hip_x is not None and hip_y is not None:
                for limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
                    lx = _safe_float(row.get(f"{limb}_x"))
                    ly = _safe_float(row.get(f"{limb}_y"))
                    if lx is None or ly is None:
                        continue
                    dist = math.sqrt((lx - hip_x) ** 2 + (ly - hip_y) ** 2)
                    if body_scale:
                        dist /= body_scale
                    reach_norm = max(reach_norm, dist)
            reach_penalties.append(max(0.0, reach_norm - self.reach_threshold))
            prev_row = row

        com_points = [
            (row.get("com_x"), row.get("com_y"))
            for row in step_frames
            if row.get("com_x") is not None and row.get("com_y") is not None
        ]
        if len(com_points) >= 2:
            net_disp = math.sqrt(
                (float(com_points[-1][0]) - float(com_points[0][0])) ** 2
                + (float(com_points[-1][1]) - float(com_points[0][1])) ** 2
            )
            path_length = 0.0
            for idx in range(1, len(com_points)):
                prev_point = com_points[idx - 1]
                cur_point = com_points[idx]
                path_length += math.sqrt(
                    (float(cur_point[0]) - float(prev_point[0])) ** 2
                    + (float(cur_point[1]) - float(prev_point[1])) ** 2
                )
            path_eff = net_disp / (path_length + 1e-6)
        else:
            path_eff = 0.0

        weights = self.weights
        components = {
            "stability": _mean(stability_scores),
            "path_efficiency": path_eff,
            "support_penalty": _mean(support_penalties),
            "wall_penalty": _mean(wall_penalties),
            "jerk_penalty": _mean(jerk_penalties),
            "reach_penalty": _mean(reach_penalties),
        }

        score = (
            weights.stability * components["stability"]
            + weights.path * components["path_efficiency"]
            - (
                weights.support * components["support_penalty"]
                + weights.wall * components["wall_penalty"]
                + weights.smoothness * components["jerk_penalty"]
                + weights.reach * components["reach_penalty"]
            )
        )

        return StepEfficiencyResult(
            step_id=segment.step_id,
            score=score,
            components=components,
            start_time=segment.start_time,
            end_time=segment.end_time,
        )


def score_steps(
    feature_rows: Sequence[MutableMapping[str, object]],
    segments: Sequence[StepSegment],
    *,
    computer: Optional[StepEfficiencyComputer] = None,
) -> List[StepEfficiencyResult]:
    comp = computer or StepEfficiencyComputer()
    results: List[StepEfficiencyResult] = []
    for segment in segments:
        step_frames = feature_rows[segment.start_index : segment.end_index + 1]
        results.append(comp.score_step(step_frames, segment))
    return results


def suggest_next_actions(
    current_row: MutableMapping[str, object],
    holds: Sequence[Dict[str, object]],
    *,
    top_k: int = 3,
) -> List[Dict[str, object]]:
    """Suggest next holds prioritising COM proximity and novelty."""
    com_x = _safe_float(current_row.get("com_x"))
    com_y = _safe_float(current_row.get("com_y"))
    if com_x is None or com_y is None:
        return []
    used = {
        str(current_row.get(f"{limb}_contact_hold"))
        for limb in ("left_hand", "right_hand", "left_foot", "right_foot")
        if current_row.get(f"{limb}_contact_hold")
    }
    ranked: List[Tuple[float, Dict[str, object]]] = []
    for hold in holds:
        coords = hold.get("coords")
        if not isinstance(coords, (list, tuple)) or len(coords) != 2:
            continue
        hx, hy = float(coords[0]), float(coords[1])
        dist = math.sqrt((hx - com_x) ** 2 + (hy - com_y) ** 2)
        dist_norm = min(dist, 1.0)
        novel = 0.0 if str(hold.get("hold_id") or hold.get("name")) in used else 1.0
        score = 0.7 * (1.0 - dist_norm) + 0.3 * novel
        ranked.append((score, hold))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [hold for _, hold in ranked[:top_k]]


__all__ = [
    "EfficiencyWeights",
    "StepEfficiencyComputer",
    "StepEfficiencyResult",
    "score_steps",
    "suggest_next_actions",
]
