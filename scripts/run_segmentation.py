"""CLI to run rule-based segmentation over extracted frame manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pose_ai.service import segment_video_from_manifest, segment_videos_under_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Produce rest/movement segments from frame manifests.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path, help="Path to a manifest.json file.")
    group.add_argument("--frames-root", type=Path, help="Directory containing subfolders with manifest.json files.")
    parser.add_argument("--json", action="store_true", help="Print segmentation results as JSON.")
    return parser


def _segment_to_dict(segment):
    return {
        "start_time": segment.start_time,
        "end_time": segment.end_time,
        "label": segment.label,
        "duration": segment.duration,
        "frame_indices": segment.frame_indices,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.manifest:
        segments = segment_video_from_manifest(args.manifest)
        if args.json:
            print(json.dumps([_segment_to_dict(seg) for seg in segments], indent=2))
        else:
            print(f"Segments for {args.manifest}:")
            for seg in segments:
                print(
                    f"- {seg.label:9s} {seg.start_time:5.2f}s → {seg.end_time:5.2f}s "
                    f"(duration {seg.duration:4.2f}s, frames {seg.frame_indices})"
                )
    else:
        results = segment_videos_under_directory(args.frames_root)
        if args.json:
            payload = {
                manifest: [_segment_to_dict(seg) for seg in segments]
                for manifest, segments in results.items()
            }
            print(json.dumps(payload, indent=2))
        else:
            for manifest, segments in results.items():
                print(f"Segments for {manifest}:")
                for seg in segments:
                    print(
                        f"  - {seg.label:9s} {seg.start_time:5.2f}s → {seg.end_time:5.2f}s "
                        f"(duration {seg.duration:4.2f}s, frames {seg.frame_indices})"
                    )


if __name__ == "__main__":
    main()
