"""CLI for extracting frame sequences from climbing videos."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pose_ai.data import FrameExtractionResult, extract_frames_every_n_seconds, iter_video_files


LOGGER = logging.getLogger("pose_ai.scripts.extract_frames")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def extract_from_directory(
    source_dir: Path,
    *,
    output_root: Path,
    interval_seconds: float,
    recursive: bool,
    overwrite: bool,
    write_manifest: bool,
) -> list[FrameExtractionResult]:
    results: list[FrameExtractionResult] = []
    for video_path in iter_video_files(source_dir, recursive=recursive):
        LOGGER.info("Processing %s", video_path)
        result = extract_frames_every_n_seconds(
            video_path,
            interval_seconds=interval_seconds,
            output_root=output_root,
            write_manifest=write_manifest,
            overwrite=overwrite,
        )
        LOGGER.info(
            "Saved %d frames for %s",
            result.saved_frames,
            video_path.name,
        )
        results.append(result)
    if not results:
        LOGGER.warning("No video files found in %s", source_dir)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract frame sequences from climbing videos.",
    )
    parser.add_argument(
        "video_dir",
        type=Path,
        help="Directory containing source video files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "frames",
        help="Directory where frame folders will be stored.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between captured frames (default: 1.0).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for videos recursively.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted frames.",
    )
    parser.add_argument(
        "--no-manifest",
        dest="write_manifest",
        action="store_false",
        help="Disable writing manifest.json files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)
    extract_from_directory(
        args.video_dir,
        output_root=args.output,
        interval_seconds=args.interval,
        recursive=args.recursive,
        overwrite=args.overwrite,
        write_manifest=args.write_manifest,
    )


if __name__ == "__main__":
    main()
