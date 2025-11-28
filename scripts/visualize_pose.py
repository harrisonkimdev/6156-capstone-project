"""Generate pose visualization overlays from pose_results.json."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2

from pose_ai.features.aggregation import compute_center_of_mass
from pose_ai.pose.estimator import PoseFrame, PoseLandmark as PoseLandmarkData

try:  # optional dependency for connection definitions
    from mediapipe.python.solutions.pose import PoseLandmark as MpPoseLandmark, POSE_CONNECTIONS
except ModuleNotFoundError:  # fallback if mediapipe not installed
    MpPoseLandmark = None
    POSE_CONNECTIONS = []


DEFAULT_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize pose landmarks on extracted frames.")
    parser.add_argument("pose_results", type=Path, help="Path to pose_results.json")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Directory to write visualized images (defaults to frame directory).",
    )
    parser.add_argument(
        "--include-missing", action="store_true",
        help="Include frames even when detection score/visibility is low.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.3,
        help="Minimum frame detection score required for visualization (default: 0.3).",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.2,
        help="Minimum landmark visibility to draw a point/connection (default: 0.2).",
    )
    parser.add_argument(
        "--draw-com",
        action="store_true",
        help="Overlay center-of-mass crosshair and bounding box.",
    )
    parser.add_argument(
        "--com-box-scale",
        type=float,
        default=0.3,
        help="Fraction of frame width used to size the COM bounding box when --draw-com is set (default: 0.3).",
    )
    parser.add_argument(
        "--mode",
        choices=["load", "balance", "dynamics", "strategy"],
        default="load",
        help="Select overlay mode (load/balance/dynamics/strategy). Default: load.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Optional pose_features.json path. Used for load/support estimation when provided.",
    )
    return parser


def _get_connections():
    if MpPoseLandmark is None or not POSE_CONNECTIONS:
        return DEFAULT_CONNECTIONS
    connections = []
    for a_idx, b_idx in POSE_CONNECTIONS:
        connections.append((MpPoseLandmark(a_idx).name.lower(), MpPoseLandmark(b_idx).name.lower()))
    return connections


def visualize_pose_results(
    pose_results_path: Path,
    output_dir: Path | None = None,
    *,
    include_missing: bool = False,
    min_score: float = 0.3,
    min_visibility: float = 0.2,
    draw_com: bool = False,
    com_box_scale: float = 0.3,
    mode: str = "load",
    features_path: Path | None = None,
) -> int:
    def _load_features(path: Path) -> list[dict]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return payload if isinstance(payload, list) else payload.get("rows", [])

    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _color_from_load(value: float) -> tuple[int, int, int]:
        """Map load 0-1 to green→yellow→red gradient."""
        v = _clamp01(value)
        if v < 0.5:
            # green to yellow
            r = int(2 * v * 255)
            g = 255
        else:
            # yellow to red
            r = 255
            g = int((1 - (v - 0.5) * 2) * 255)
        return (0, g, r)

    def _support_points(points: dict[str, tuple[int, int]], feat: dict | None) -> list[tuple[int, int]]:
        keys: list[str] = []
        if feat:
            for limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
                if feat.get(f"{limb}_contact_on") or feat.get(f"{limb}_on_hold"):
                    keys.append(limb)
        # Always include feet
        for limb in ("left_foot", "right_foot"):
            if limb not in keys:
                keys.append(limb)
        # Include hands if visible
        for limb in ("left_hand", "right_hand"):
            if limb not in keys and limb in points:
                keys.append(limb)
        return [points[k] for k in keys if k in points]

    def _compute_limb_loads(points: dict[str, tuple[int, int]], feat: dict | None, width: int, height: int) -> dict[str, float]:
        """Estimate limb load from contact + COM projection, then adjust with speed/visibility."""
        default = 0.2
        loads = {k: default for k in ("left_hand", "right_hand", "left_foot", "right_foot")}
        if not feat:
            return loads

        # Base weights: contact flags
        contact_weight = {}
        for limb in loads:
            on_hold = bool(feat.get(f"{limb}_on_hold"))
            contact_on = bool(feat.get(f"{limb}_contact_on"))
            contact_weight[limb] = 1.0 if (on_hold or contact_on) else 0.2

        # Distribute using COM projection distance to supports
        com_x = feat.get("com_x")
        com_y = feat.get("com_y")
        support_pts = _support_points(points, feat)
        if com_x is not None and com_y is not None and support_pts:
            cx = float(com_x) * width
            cy = float(com_y) * height
            weights = []
            keys = []
            for limb, pt in points.items():
                base_name = None
                for l in loads:
                    if l.split("_")[0] in limb:
                        base_name = l
                        break
                if base_name is None:
                    continue
                dist = math.hypot(cx - pt[0], cy - pt[1]) + 1e-3
                w = 1.0 / dist
                weights.append(w)
                keys.append(base_name)
            total_w = sum(weights) or 1.0
            for k, w in zip(keys, weights):
                loads[k] = 0.4 * (w / total_w) + 0.6 * contact_weight.get(k, 0.2)
        else:
            loads.update({k: max(default, 0.6 * contact_weight.get(k, 0.2)) for k in loads})

        # Dynamic adjustment: speed/visibility
        for limb in loads:
            speed = feat.get(f"{limb}_speed")
            visibility = feat.get(f"{limb}_visibility")
            if speed is not None:
                loads[limb] += min(float(speed) / 2.5, 0.3)
            if visibility is not None:
                loads[limb] = max(loads[limb], float(visibility) * 0.8)
            loads[limb] = _clamp01(loads[limb])

        # Normalize: scale largest load to 1 to avoid over-saturation
        max_load = max(loads.values()) or 1.0
        for limb in loads:
            loads[limb] = _clamp01(loads[limb] / max_load)
        return loads

    def _limb_thickness(load: float) -> int:
        return max(2, int(2 + 4 * _clamp01(load)))

    features = _load_features(features_path) if features_path else []

    payload = json.loads(pose_results_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    count = 0
    connections = _get_connections()

    for idx, frame in enumerate(frames):
        image_path = Path(frame["image_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        landmarks = frame.get("landmarks", [])
        detection_score = float(frame.get("detection_score", 0.0))
        if not include_missing and detection_score < min_score:
            continue
        if not landmarks and not include_missing:
            continue

        points = {}
        pose_landmarks: list[PoseLandmarkData] = []
        for landmark in landmarks:
            x = int(landmark["x"] * width)
            y = int(landmark["y"] * height)
            if landmark.get("visibility", 1.0) < min_visibility:
                continue
            points[landmark["name"]] = (x, y)
            pose_landmarks.append(
                PoseLandmarkData(
                    name=landmark["name"],
                    x=float(landmark["x"]),
                    y=float(landmark["y"]),
                    z=float(landmark.get("z", 0.0)),
                    visibility=float(landmark.get("visibility", 0.0)),
                )
            )

        feat = features[idx] if idx < len(features) else None

        limb_loads = _compute_limb_loads(points, feat, width, height)

        for start, end in connections:
            if start in points and end in points:
                # Use limb name of endpoints to decide color/thickness
                limb_key = None
                for limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
                    if limb.split("_")[0] in start or limb.split("_")[0] in end:
                        limb_key = limb
                        break
                load_val = limb_loads.get(limb_key, 0.3)
                if mode == "balance":
                    color = (170, 170, 170)
                    thickness = _limb_thickness(load_val)
                else:
                    color = _color_from_load(load_val)
                    thickness = 2
                cv2.line(image, points[start], points[end], color, thickness)
        for name, (x, y) in points.items():
            if mode == "balance":
                color = (200, 200, 200)
                cv2.circle(image, (x, y), 4, color, -1)
            else:
                limb_key = None
                for limb in ("left_hand", "right_hand", "left_foot", "right_foot"):
                    if limb.split("_")[0] in name:
                        limb_key = limb
                        break
                color = _color_from_load(limb_loads.get(limb_key, 0.3))
                cv2.circle(image, (x, y), 4, color, -1)

        if draw_com and pose_landmarks:
            pose_frame = PoseFrame(
                image_path=image_path,
                timestamp_seconds=float(frame.get("timestamp_seconds", 0.0)),
                landmarks=pose_landmarks,
                detection_score=detection_score,
            )
            com = compute_center_of_mass(pose_frame)
            com_x, com_y = com.get("com_x"), com.get("com_y")
            if com_x is not None and com_y is not None:
                cx = int(com_x * width)
                cy = int(com_y * height)
                cross_half = max(8, int(min(width, height) * 0.02))
                com_color = (0, 0, 255) if mode != "balance" else (0, 180, 180)
                cv2.line(image, (cx - cross_half, cy), (cx + cross_half, cy), com_color, 2)
                cv2.line(image, (cx, cy - cross_half), (cx, cy + cross_half), com_color, 2)
                half_box = max(10, int(width * float(com_box_scale) * 0.5))
                cv2.rectangle(
                    image,
                    (cx - half_box, cy - half_box),
                    (cx + half_box, cy + half_box),
                    (0, 255, 255) if mode != "balance" else (0, 200, 200),
                    2,
                )

        # Support polygon
        support_pts = _support_points(points, feat)
        if len(support_pts) >= 3:
            import numpy as np

            pts_array = np.array(support_pts, dtype=np.int32).reshape((-1, 1, 2))
            hull = cv2.convexHull(pts_array)
            hull_pts = [(int(p[0][0]), int(p[0][1])) for p in hull]
            color = (0, 200, 255) if mode != "balance" else (0, 255, 0)
            cv2.polylines(image, [np.array(hull_pts, dtype=np.int32)], True, color, 2)
        elif len(support_pts) == 2:
            color = (0, 200, 255) if mode != "balance" else (0, 255, 0)
            cv2.line(image, support_pts[0], support_pts[1], color, 2)

        target_dir = output_dir or image_path.parent / "visualized"
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{image_path.stem}_viz{image_path.suffix}"
        cv2.imwrite(str(out_path), image)
        count += 1
    return count


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    processed = visualize_pose_results(
        args.pose_results,
        args.output,
        include_missing=args.include_missing,
        min_score=args.min_score,
        min_visibility=args.min_visibility,
        draw_com=args.draw_com,
        com_box_scale=args.com_box_scale,
        mode=args.mode,
        features_path=args.features,
    )
    print(f"Saved {processed} annotated frames")


if __name__ == "__main__":
    main()
