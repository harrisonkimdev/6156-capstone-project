"""CLI to run pose estimation on extracted frame sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pose_ai.pose import PoseEstimator
from pose_ai.service import estimate_poses_for_directory, estimate_poses_from_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MediaPipe pose estimation on frame manifests.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path, help="Path to a manifest.json file.")
    group.add_argument("--frames-root", type=Path, help="Directory containing extracted frame folders.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print pose results as JSON instead of a textual summary.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable writing pose_results.json files alongside frames.",
    )
    return parser


def frames_to_dict(frames):
    return [
        {
            "image_path": str(frame.image_path),
            "timestamp_seconds": frame.timestamp_seconds,
            "detection_score": frame.detection_score,
            "landmarks": [
                {
                    "name": landmark.name,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility,
                }
                for landmark in frame.landmarks
            ],
        }
        for frame in frames
    ]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    estimator = PoseEstimator()
    try:
        if args.manifest:
            frames = estimate_poses_from_manifest(
                args.manifest,
                estimator=estimator,
                save_json=not args.no_save,
            )
            if args.json:
                print(json.dumps(frames_to_dict(frames), indent=2))
            else:
                print(f"Processed {len(frames)} frames from {args.manifest}")
        else:
            results = estimate_poses_for_directory(
                args.frames_root,
                estimator=estimator,
                save_json=not args.no_save,
            )
            if args.json:
                payload = {manifest: frames_to_dict(frames) for manifest, frames in results.items()}
                print(json.dumps(payload, indent=2))
            else:
                for manifest, frames in results.items():
                    print(f"{manifest}: {len(frames)} frames")
    except ModuleNotFoundError as exc:
        parser.error(
            f"{exc}. Ensure mediapipe is installed in your environment (e.g. `pip install mediapipe`)."
        )
    finally:
        estimator.close()


if __name__ == "__main__":
    main()
