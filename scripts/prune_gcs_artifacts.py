"""Purge aged artifacts from Google Cloud Storage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pose_ai.cloud.gcs import get_gcs_manager  # type: ignore  # pylint: disable=wrong-import-position


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Delete aged GCS artifacts.")
    parser.add_argument("--kind", choices=["videos", "frames", "models"], default="videos", help="Artifact type to prune.")
    parser.add_argument(
        "--bucket",
        help="Override bucket name (defaults to configured bucket for the selected kind).",
    )
    parser.add_argument(
        "--prefix",
        help="Override object prefix (defaults to configured prefix for the selected kind).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Delete objects older than this many days.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of objects to delete in a single run.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    manager = get_gcs_manager()
    if manager is None:
        parser.error("GCS is not configured. Set the GCS_* environment variables before running this script.")

    bucket_map = {
        "videos": manager.config.video_bucket,
        "frames": manager.config.frame_bucket,
        "models": manager.config.model_bucket,
    }
    prefix_map = {
        "videos": manager.config.raw_prefix,
        "frames": manager.config.frame_prefix,
        "models": manager.config.model_prefix,
    }

    bucket = args.bucket or bucket_map[args.kind]
    if not bucket:
        parser.error(f"No bucket configured for artifact kind '{args.kind}'.")
    prefix = args.prefix or prefix_map[args.kind]
    deleted = manager.prune_objects(
        bucket_name=bucket,
        prefix=prefix.rstrip("/") + "/",
        older_than_days=args.days,
        limit=args.limit,
    )
    print(f"Deleted {len(deleted)} objects from {bucket}/{prefix} older than {args.days} days.")


if __name__ == "__main__":
    main()
