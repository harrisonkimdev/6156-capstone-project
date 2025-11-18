"""Rule-based next-move planner with efficiency simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, MutableMapping, Optional, Sequence, Tuple

import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    HAS_NUMPY = False
    np = None  # type: ignore


Point = Tuple[float, float]


@dataclass(slots=True)
class MoveCandidate:
    """Represents a candidate next move."""
    limb: str
    hold_id: str
    hold_position: Point
    simulated_efficiency: float
    efficiency_delta: float
    reasoning: str
    constraint_violations: List[str]


@dataclass(slots=True)
class PlannerConfig:
    """Configuration for the rule-based planner."""
    k_candidates: int = 10
    upward_bias: float = 0.3
    min_support_count: int = 2
    max_reach_ratio: float = 1.2
    com_polygon_tolerance: float = 0.15
    stability_alpha: float = 4.0


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:  # pragma: no cover
        return None


def _distance(p1: Point, p2: Point) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray casting algorithm for point-in-polygon test."""
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


def _distance_to_polygon(point: Point, polygon: Sequence[Point]) -> float:
    """Distance from point to polygon."""
    if not polygon:
        return float("inf")
    if len(polygon) == 1:
        return _distance(point, polygon[0])
    if _point_in_polygon(point, polygon):
        return 0.0
    
    min_dist = float("inf")
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        dist = _distance_point_to_segment(point, a, b)
        min_dist = min(min_dist, dist)
    return min_dist


def _distance_point_to_segment(point: Point, a: Point, b: Point) -> float:
    """Distance from point to line segment."""
    ax, ay = a
    bx, by = b
    px, py = point
    abx = bx - ax
    aby = by - ay
    if abs(abx) < 1e-9 and abs(aby) < 1e-9:
        return _distance(point, a)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby)))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def _compute_stability_score(com: Point, support_points: List[Point], body_scale: float, alpha: float = 4.0) -> float:
    """Compute support polygon stability score."""
    if not support_points:
        return 0.0
    distance = _distance_to_polygon(com, support_points)
    return math.exp(-alpha * (distance / (body_scale + 1e-6)))


def _simulate_efficiency(
    current_row: MutableMapping[str, object],
    limb: str,
    hold_position: Point,
    config: PlannerConfig,
) -> Tuple[float, List[str]]:
    """Simulate efficiency score for a candidate move."""
    violations: List[str] = []
    
    # Extract current state
    com_x = _safe_float(current_row.get("com_x"))
    com_y = _safe_float(current_row.get("com_y"))
    body_scale = _safe_float(current_row.get("body_scale")) or 1.0
    hip_x = _safe_float(current_row.get("hip_center_x"))
    hip_y = _safe_float(current_row.get("hip_center_y"))
    
    if com_x is None or com_y is None:
        violations.append("missing_com")
        return 0.0, violations
    
    com = (com_x, com_y)
    
    # Build simulated support set
    support_points: List[Point] = []
    for other_limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
        if other_limb == limb:
            # Use the new hold position
            support_points.append(hold_position)
        else:
            # Use existing contact if present
            if current_row.get(f"{other_limb}_contact_on"):
                tx = _safe_float(current_row.get(f"{other_limb}_target_x"))
                ty = _safe_float(current_row.get(f"{other_limb}_target_y"))
                if tx is not None and ty is not None:
                    support_points.append((tx, ty))
                else:
                    lx = _safe_float(current_row.get(f"{other_limb}_x"))
                    ly = _safe_float(current_row.get(f"{other_limb}_y"))
                    if lx is not None and ly is not None:
                        support_points.append((lx, ly))
    
    # Constraint checks
    support_count = len(support_points)
    if support_count < config.min_support_count:
        violations.append(f"low_support_count_{support_count}")
    
    # Check reach limit
    if hip_x is not None and hip_y is not None:
        reach = _distance((hip_x, hip_y), hold_position) / body_scale
        if reach > config.max_reach_ratio:
            violations.append(f"reach_exceeded_{reach:.2f}")
    
    # Check COM inside or near polygon
    if support_points:
        dist_to_poly = _distance_to_polygon(com, support_points)
        normalized_dist = dist_to_poly / body_scale
        if normalized_dist > config.com_polygon_tolerance:
            violations.append(f"com_outside_polygon_{normalized_dist:.2f}")
    
    # Compute simulated stability score
    stability = _compute_stability_score(com, support_points, body_scale, config.stability_alpha)
    
    # Simple efficiency estimate (stability weighted)
    efficiency = stability * 0.5 + (1.0 - len(violations) * 0.1)
    
    return max(0.0, efficiency), violations


class NextMovePlanner:
    """Rule-based planner for next move recommendations."""
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
    
    def plan_next_move(
        self,
        current_row: MutableMapping[str, object],
        holds: Sequence[Dict[str, object]],
        *,
        top_k: int = 3,
    ) -> List[MoveCandidate]:
        """Generate ranked next-move candidates with efficiency simulation."""
        if not holds:
            return []
        
        com_x = _safe_float(current_row.get("com_x"))
        com_y = _safe_float(current_row.get("com_y"))
        body_scale = _safe_float(current_row.get("body_scale")) or 1.0
        
        if com_x is None or com_y is None:
            return []
        
        com = (com_x, com_y)
        
        # Identify currently used holds
        used_holds = set()
        for limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
            hold_id = current_row.get(f"{limb}_contact_hold")
            if hold_id:
                used_holds.add(str(hold_id))
        
        # Sample candidate holds
        candidates: List[Tuple[str, str, Point, float]] = []  # (limb, hold_id, position, score)
        
        for hold in holds:
            hold_id = str(hold.get("hold_id") or hold.get("name", ""))
            if hold_id in used_holds:
                continue
            
            coords = hold.get("coords")
            if not isinstance(coords, (list, tuple)) or len(coords) < 2:
                continue
            
            hx, hy = float(coords[0]), float(coords[1])
            hold_pos = (hx, hy)
            
            # Distance-based sampling score
            dist = _distance(com, hold_pos)
            dist_score = 1.0 / (dist / body_scale + 1e-6)
            
            # Upward bias (lower y = higher on wall in normalized coords)
            upward_score = 0.0
            if hy < com_y:
                upward_score = self.config.upward_bias * (com_y - hy)
            
            sample_score = dist_score + upward_score
            
            # Determine which limb would use this hold (heuristic: hands for higher holds, feet for lower)
            if hy < com_y - 0.1 * body_scale:
                # Higher hold - prefer hands
                for limb in ("left_hand", "right_hand"):
                    if not current_row.get(f"{limb}_contact_on"):
                        candidates.append((limb, hold_id, hold_pos, sample_score))
                        break
                else:
                    # Both hands occupied, pick closest
                    lh_x = _safe_float(current_row.get("left_hand_x")) or com_x
                    lh_y = _safe_float(current_row.get("left_hand_y")) or com_y
                    rh_x = _safe_float(current_row.get("right_hand_x")) or com_x
                    rh_y = _safe_float(current_row.get("right_hand_y")) or com_y
                    if _distance(hold_pos, (lh_x, lh_y)) < _distance(hold_pos, (rh_x, rh_y)):
                        candidates.append(("left_hand", hold_id, hold_pos, sample_score))
                    else:
                        candidates.append(("right_hand", hold_id, hold_pos, sample_score))
            else:
                # Lower hold - prefer feet
                for limb in ("left_foot", "right_foot"):
                    if not current_row.get(f"{limb}_contact_on"):
                        candidates.append((limb, hold_id, hold_pos, sample_score))
                        break
                else:
                    lf_x = _safe_float(current_row.get("left_foot_x")) or com_x
                    lf_y = _safe_float(current_row.get("left_foot_y")) or com_y
                    rf_x = _safe_float(current_row.get("right_foot_x")) or com_x
                    rf_y = _safe_float(current_row.get("right_foot_y")) or com_y
                    if _distance(hold_pos, (lf_x, lf_y)) < _distance(hold_pos, (rf_x, rf_y)):
                        candidates.append(("left_foot", hold_id, hold_pos, sample_score))
                    else:
                        candidates.append(("right_foot", hold_id, hold_pos, sample_score))
        
        # Sort by sample score and take top K
        candidates.sort(key=lambda x: x[3], reverse=True)
        candidates = candidates[:self.config.k_candidates]
        
        # Simulate efficiency for each candidate
        move_candidates: List[MoveCandidate] = []
        for limb, hold_id, hold_pos, _ in candidates:
            sim_eff, violations = _simulate_efficiency(current_row, limb, hold_pos, self.config)
            
            # Current efficiency (simplified)
            current_eff = 0.5  # Placeholder
            delta_eff = sim_eff - current_eff
            
            reasoning_parts = []
            if delta_eff > 0:
                reasoning_parts.append(f"Improves efficiency by {delta_eff:.2f}")
            if hold_pos[1] < com_y:
                reasoning_parts.append("Upward progression")
            if not violations:
                reasoning_parts.append("No constraint violations")
            
            reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Candidate move"
            
            move_candidates.append(
                MoveCandidate(
                    limb=limb,
                    hold_id=hold_id,
                    hold_position=hold_pos,
                    simulated_efficiency=sim_eff,
                    efficiency_delta=delta_eff,
                    reasoning=reasoning,
                    constraint_violations=violations,
                )
            )
        
        # Filter out moves with critical violations and rank by efficiency
        valid_candidates = [c for c in move_candidates if not any("exceeded" in v or "low_support" in v for v in c.constraint_violations)]
        if not valid_candidates:
            valid_candidates = move_candidates  # Fall back to all if none are valid
        
        valid_candidates.sort(key=lambda c: c.simulated_efficiency, reverse=True)
        
        return valid_candidates[:top_k]


__all__ = ["NextMovePlanner", "MoveCandidate", "PlannerConfig"]

