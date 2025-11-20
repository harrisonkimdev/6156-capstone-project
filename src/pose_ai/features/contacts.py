"""Contact inference utilities with hysteresis and technique tagging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, MutableMapping, Optional, Sequence

import math


LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")


@dataclass(slots=True)
class ContactParams:
    r_on: float = 0.22
    r_off: float = 0.28
    v_hold: float = 0.03
    min_on_frames: int = 3
    smear_distance: float = 0.25
    smear_z_epsilon: float = 0.03
    fps: float = 25.0
    limbs: Sequence[str] = LIMBS


@dataclass(slots=True)
class LimbContactState:
    on: bool = False
    hold_id: Optional[str] = None
    frames_in_state: int = 0
    pending_on: int = 0
    contact_type: str = "none"  # "hold" | "smear" | "none"


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:  # pragma: no cover - defensive
        return None


def _speed(row: MutableMapping[str, object], limb: str) -> float:
    vx = _safe_float(row.get(f"{limb}_vx"))
    vy = _safe_float(row.get(f"{limb}_vy"))
    if vx is None or vy is None:
        return float("inf")
    return math.sqrt(vx**2 + vy**2)


def _detect_smear(row: MutableMapping[str, object], limb: str, params: ContactParams, body_scale: float) -> bool:
    if "foot" not in limb:
        return False
    distance = _safe_float(row.get(f"{limb}_distance"))
    if distance is None:
        return False
    if distance <= params.smear_distance * body_scale:
        return False
    z = _safe_float(row.get(f"{limb}_z"))
    if z is None:
        return False
    return abs(z) <= params.smear_z_epsilon and _speed(row, limb) <= params.v_hold * body_scale


def _update_row(row: MutableMapping[str, object], limb: str, state: LimbContactState) -> None:
    row[f"{limb}_contact_on"] = int(state.on)
    row[f"{limb}_contact_type"] = state.contact_type
    row[f"{limb}_contact_hold"] = state.hold_id


def apply_contact_filter(
    feature_rows: Sequence[MutableMapping[str, object]],
    params: Optional[ContactParams] = None,
) -> Sequence[MutableMapping[str, object]]:
    """Infer contact states with hysteresis and smear detection."""
    if not feature_rows:
        return feature_rows

    cfg = params or ContactParams()
    limb_states: Dict[str, LimbContactState] = {limb: LimbContactState() for limb in cfg.limbs}

    for row in feature_rows:
        body_scale = _safe_float(row.get("body_scale")) or 1.0
        for limb, state in limb_states.items():
            distance = _safe_float(row.get(f"{limb}_distance"))
            hold_id = row.get(f"{limb}_target")
            speed = _speed(row, limb)

            # smear detection only for feet
            is_smear = _detect_smear(row, limb, cfg, body_scale)

            should_activate = (
                distance is not None
                and distance <= cfg.r_on * body_scale
                and speed <= cfg.v_hold * body_scale
            )
            should_deactivate = (
                distance is not None
                and distance >= cfg.r_off * body_scale
            )

            if state.on:
                state.frames_in_state += 1
                if state.frames_in_state >= cfg.min_on_frames and should_deactivate and not is_smear:
                    state.on = False
                    state.hold_id = None
                    state.contact_type = "none"
                    state.frames_in_state = 0
                    state.pending_on = 0
                else:
                    # maintain hold id preference
                    if hold_id:
                        state.hold_id = hold_id
                    state.contact_type = "smear" if is_smear else "hold"
            else:
                if should_activate or is_smear:
                    state.pending_on += 1
                    if state.pending_on >= cfg.min_on_frames:
                        state.on = True
                        state.hold_id = hold_id if not is_smear else None
                        state.contact_type = "smear" if is_smear else "hold"
                        state.frames_in_state = 0
                        state.pending_on = 0
                else:
                    state.pending_on = 0
                    state.contact_type = "none"
                    state.hold_id = None
            _update_row(row, limb, state)
    return feature_rows


def detect_techniques(row: MutableMapping[str, object]) -> Dict[str, float]:
    """Very lightweight heuristics for technique tagging."""
    features: Dict[str, float] = {}
    bicycle = 0.0
    if (
        row.get("left_foot_contact_on")
        and row.get("right_foot_contact_on")
        and row.get("left_foot_contact_hold")
        == row.get("right_foot_contact_hold")
    ):
        # penalize high relative foot speed, reward opposing directions
        lf_vx = _safe_float(row.get("left_foot_vx")) or 0.0
        rf_vx = _safe_float(row.get("right_foot_vx")) or 0.0
        bicycle = max(0.0, 1.0 - abs(lf_vx + rf_vx))
    features["technique_bicycle"] = bicycle

    # back-flag heuristic: free leg moves opposite direction of hips
    hip_x = _safe_float(row.get("hip_center_x"))
    lf_x = _safe_float(row.get("left_foot_x"))
    rf_x = _safe_float(row.get("right_foot_x"))
    back_flag = 0.0
    if hip_x is not None and lf_x is not None and rf_x is not None:
        if lf_x < hip_x and rf_x < hip_x:
            back_flag = 0.5
        elif lf_x > hip_x and rf_x > hip_x:
            back_flag = 0.5
    features["technique_back_flag"] = back_flag

    # drop-knee heuristic: knee and ankle separation vs hip line
    lknee = _safe_float(row.get("left_knee_x"))
    rknee = _safe_float(row.get("right_knee_x"))
    drop_knee = 0.0
    if lf_x is not None and lknee is not None and hip_x is not None:
        drop_knee = max(drop_knee, min(1.0, abs(lf_x - lknee) / (abs(hip_x - lf_x) + 1e-6)))
    if rf_x is not None and rknee is not None and hip_x is not None:
        drop_knee = max(drop_knee, min(1.0, abs(rf_x - rknee) / (abs(hip_x - rf_x) + 1e-6)))
    features["technique_drop_knee"] = drop_knee
    row.update(features)
    return features


def annotate_techniques(feature_rows: Sequence[MutableMapping[str, object]]) -> None:
    for row in feature_rows:
        detect_techniques(row)


__all__ = [
    "apply_contact_filter",
    "ContactParams",
    "annotate_techniques",
    "detect_techniques",
]
