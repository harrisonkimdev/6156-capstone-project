"""CLI to export pose-derived features from manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pose_ai.service import export_features_for_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export pose feature rows from pose_results.json.")
    parser.add_argument("manifest", type=Path, help="Path to manifest.json")
    parser.add_argument(
        "--holds",
        type=Path,
        help="Optional JSON describing holds (name -> coords, normalized, etc).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (defaults to manifest directory).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_path = export_features_for_manifest(
        args.manifest,
        holds_path=args.holds,
        output_root=args.out,
    )
    print(f"Feature rows saved to {output_path}")


if __name__ == "__main__":
    main()
