"""End-to-end pipeline: extract frames, run pose estimation, features, segments, visualize."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pose_ai.service import (  # type: ignore  # pylint: disable=wrong-import-position
    estimate_poses_from_manifest,
    export_features_for_manifest,
    generate_segment_report,
)
from extract_frames import extract_frames_every_n_seconds, iter_video_files  # type: ignore
from visualize_pose import visualize_pose_results  # type: ignore

# NOTE: For simplicity we call into script helpers directly for pose/feature export
# and reuse service APIs for intermediate steps.

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run entire pose analysis pipeline")
    parser.add_argument("video_dir", type=Path, help="Directory containing videos (.mp4, etc.)")
    parser.add_argument("--out", type=Path, default=Path("data/frames"), help="Frame output directory")
    parser.add_argument("--interval", type=float, default=1.0, help="Extraction interval (seconds)")
    parser.add_argument("--skip-visuals", action="store_true", help="Skip visualization step")
    return parser


def extract_frames(video_dir: Path, out_dir: Path, interval: float) -> list[Path]:
    manifests = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for video_file in iter_video_files(video_dir):
        result = extract_frames_every_n_seconds(video_file, output_root=out_dir, interval_seconds=interval)
        manifests.append(result.frame_directory / "manifest.json")
    return manifests


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    manifests = extract_frames(args.video_dir, args.out, args.interval)
    for manifest in manifests:
        print(f"Processing manifest {manifest}")
        estimate_poses_from_manifest(manifest)
        export_features_for_manifest(manifest)
        generate_segment_report(manifest)
        if not args.skip_visuals:
            frame_dir = manifest.parent
            visualize_pose_results(frame_dir / "pose_results.json")
    print("Pipeline completed.")


if __name__ == "__main__":
    main()
