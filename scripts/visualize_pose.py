"""Generate pose visualization overlays from pose_results.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

try:  # optional dependency for connection definitions
    from mediapipe.python.solutions.pose import PoseLandmark, POSE_CONNECTIONS
except ModuleNotFoundError:  # fallback if mediapipe not installed
    PoseLandmark = None
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
    return parser


def _get_connections():
    if PoseLandmark is None or not POSE_CONNECTIONS:
        return DEFAULT_CONNECTIONS
    connections = []
    for a_idx, b_idx in POSE_CONNECTIONS:
        connections.append((PoseLandmark(a_idx).name.lower(), PoseLandmark(b_idx).name.lower()))
    return connections


def visualize_pose_results(
    pose_results_path: Path,
    output_dir: Path | None = None,
    *,
    include_missing: bool = False,
    min_score: float = 0.3,
    min_visibility: float = 0.2,
) -> int:
    payload = json.loads(pose_results_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    count = 0
    connections = _get_connections()

    for frame in frames:
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
        for landmark in landmarks:
            x = int(landmark["x"] * width)
            y = int(landmark["y"] * height)
            if landmark.get("visibility", 1.0) < min_visibility:
                continue
            points[landmark["name"]] = (x, y)
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

        for start, end in connections:
            if start in points and end in points:
                cv2.line(image, points[start], points[end], (255, 0, 0), 2)

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
    )
    print(f"Saved {processed} annotated frames")


if __name__ == "__main__":
    main()
