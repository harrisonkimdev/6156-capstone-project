# Climber Analysis API / Overlay Spec (Draft)

This document outlines how the server produces analysis payloads and how the client renders overlays per mode. The server always returns the full set of metrics; the client selects what to draw based on the chosen mode.

## Common Concepts
- **Frame-level**: Per-frame coordinates/metrics used for overlays.
- **Segment-level**: Metrics summarised over a range of frames (e.g., dyno preparation, rest).
- **Tags**: Issues or points of interest with reasons and suggested fixes.
- **Suggested pose**: Optional dotted skeleton to show a recommended posture for comparison.

## Frame Schema (example)
```json
{
  "frame_index": 55,
  "timestamp": 12.34,
  "image_url": "https://.../IMG_3571_frame_0055.jpg",
  "pose": {
    "landmarks": [
      { "name": "left_wrist", "x": 0.38, "y": 0.61, "visibility": 0.75 }
    ]
  },
  "center_of_mass": { "x": 0.47, "y": 0.52 },
  "support_polygon": {
    "points": [ [0.20, 0.90], [0.80, 0.88], [0.60, 0.40] ],
    "inside": true
  },
  "limb_load": {
    "left_arm": 0.72,
    "right_arm": 0.45,
    "left_leg": 0.30,
    "right_leg": 0.55
  },
  "limb_direction": {
    "left_arm": { "vx": -0.1, "vy": -0.2 },
    "right_arm": { "vx": 0.1, "vy": -0.05 }
  },
  "core_rotation": {
    "angle_deg": 35.0,
    "load_score": 0.68
  },
  "motion_style": {
    "dynamic_score": 0.85,
    "is_dyno": true
  },
  "events": [ "left_hand_release", "dyno_takeoff" ]
}
```

## Segment Schema (example)
```json
{
  "segment_id": "seg_3",
  "start_frame": 50,
  "end_frame": 80,
  "label": "dyno_preparation",
  "metrics": {
    "avg_limb_load": { "left_arm": 0.78, "right_arm": 0.40, "left_leg": 0.25, "right_leg": 0.58 },
    "com_horizontal_std": 0.12,
    "core_load_mean": 0.65,
    "regrip_count": 3
  }
}
```

## Tag Schema (example)
```json
{
  "tag_id": "tag_17",
  "type": "left_arm_overuse",
  "severity": "warning",
  "anchor_frame": 60,
  "frame_range": [55, 70],
  "title": "Left arm overuse",
  "reason": "Left arm load exceeded 70% for more than 3 seconds.",
  "suggested_fix": "Step the left foot up one hold and initiate the next move with the right hand to offload the left arm."
}
```

## Suggested Pose (optional)
```json
{
  "tag_id": "tag_17",
  "suggested_pose": {
    "frame_like": 60,
    "landmarks": [
      { "name": "left_foot", "x": 0.35, "y": 0.88 }
    ]
  }
}
```

## Overlay Modes
- **Mode A: Load**  
  - Draw: four limb skeleton lines colored by load (fixed thickness), COM crosshair/box, support polygon.  
  - Panel: limb load bars and values.
- **Mode B: Balance**  
  - Draw: COM crosshair/box, support polygon colored/thickened by COM deviation, recent COM trail, arrow from COM to support centroid. Limbs are neutral/minimal.  
  - Panel: COM horizontal deviation graph/values.
- **Mode C: Dynamics**  
  - Draw: COM trail, hand/foot velocity arrows (from limb velocities), torso tilt bar (core load proxy). Dyno-specific highlighting can be added on top.  
  - Panel: dynamic score, core load.
- **Mode D: Strategy**  
  - Draw: regrip counts near holds, tag icons, optional suggested pose dotted skeleton.  
  - Panel: segment efficiency/retry table, tag list.

## Implementation Notes
- The server computes all fields; the client renders a subset depending on mode.
- Initial load/direction values are heuristic and can be replaced as force estimation improves.
- Current load approximation mixes **contact flags + COM projection distance + limb speed/visibility** and normalizes to 0–1. It can be swapped out later if GRF/force sensors are available.
