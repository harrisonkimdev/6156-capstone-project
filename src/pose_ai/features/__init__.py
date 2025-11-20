"""Feature engineering helpers for pose analysis."""

from .aggregation import (
    HoldDefinition,
    analyze_hold_relationship,
    compute_center_of_mass,
    compute_joint_angles,
    compute_segment_inclination,
    pose_to_feature_row,
    summarize_features,
)
from .config import CENTER_OF_MASS_WEIGHTS, CONTACT_LANDMARKS, JOINT_DEFINITIONS

__all__ = [
    "analyze_hold_relationship",
    "compute_center_of_mass",
    "compute_joint_angles",
    "compute_segment_inclination",
    "pose_to_feature_row",
    "summarize_features",
    "HoldDefinition",
    "CENTER_OF_MASS_WEIGHTS",
    "CONTACT_LANDMARKS",
    "JOINT_DEFINITIONS",
]
