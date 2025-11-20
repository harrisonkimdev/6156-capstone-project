"""CLI to generate segment-level metrics."""

from __future__ import annotations

import argparse

from pose_ai.service import generate_segment_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export segment metrics (COM, joints, contacts)")
    parser.add_argument("manifest", type=str, help="Path to manifest.json")
    parser.add_argument("--holds", type=str, help="Optional holds JSON path", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    metrics = generate_segment_report(args.manifest, holds_path=Path(args.holds) if args.holds else None)
    print(f"Saved {len(metrics)} segments")


if __name__ == "__main__":
    main()
