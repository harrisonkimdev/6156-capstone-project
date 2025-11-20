"""CLI for extracting frame sequences from climbing videos."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pose_ai.data import (
    FrameExtractionResult,
    extract_frames_every_n_seconds,
    extract_frames_with_motion,
    iter_video_files,
)


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
    method: str = "interval",
    motion_threshold: float = 5.0,
    similarity_threshold: float = 0.8,
    min_frame_interval: int = 5,
    use_optical_flow: bool = True,
    use_pose_similarity: bool = True,
    initial_sampling_rate: float = 0.1,
    segmentation: bool = False,
    seg_method: str = "yolo",
    seg_model: str = "yolov8n-seg.pt",
    hsv_hue_tolerance: int = 5,
    hsv_sat_tolerance: int = 50,
    hsv_val_tolerance: int = 40,
) -> list[FrameExtractionResult]:
    results: list[FrameExtractionResult] = []
    for video_path in iter_video_files(source_dir, recursive=recursive):
        LOGGER.info("Processing %s (method: %s)", video_path, method)
        
        if method in ("motion", "motion_pose"):
            result = extract_frames_with_motion(
                video_path,
                output_root=output_root,
                motion_threshold=motion_threshold,
                similarity_threshold=similarity_threshold,
                min_frame_interval=min_frame_interval,
                use_optical_flow=use_optical_flow,
                use_pose_similarity=use_pose_similarity and method == "motion_pose",
                initial_sampling_rate=initial_sampling_rate,
                write_manifest=write_manifest,
                overwrite=overwrite,
            )
        else:
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
        
        # Run segmentation if enabled
        if segmentation and result.frame_directory:
            try:
                from pose_ai.segmentation import (  # type: ignore
                    HsvSegmentationModel,
                    YoloSegmentationModel,
                    export_segmentation_masks,
                )
                
                image_paths = sorted([p for p in result.frame_directory.glob("*.jpg")])
                if image_paths:
                    if seg_method.lower() == "hsv":
                        LOGGER.info("Running HSV segmentation...")
                        seg_model_instance = HsvSegmentationModel(
                            hue_tolerance=hsv_hue_tolerance,
                            sat_tolerance=hsv_sat_tolerance,
                            val_tolerance=hsv_val_tolerance,
                        )
                        seg_results = seg_model_instance.batch_segment_frames(
                            image_paths,
                            target_classes=["wall", "hold"],
                        )
                    else:  # yolo
                        LOGGER.info("Running YOLO segmentation...")
                        seg_model_instance = YoloSegmentationModel(model_name=seg_model)
                        seg_results = seg_model_instance.batch_segment_frames(image_paths, conf_threshold=0.25)
                    export_segmentation_masks(seg_results, result.frame_directory, export_images=True, export_json=True)
                    LOGGER.info("Segmentation masks exported to %s", result.frame_directory / "masks")
            except Exception as exc:
                LOGGER.warning("Segmentation failed: %s", exc)
        
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
        help="Seconds between captured frames (default: 1.0, used for 'interval' method).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["interval", "motion", "motion_pose"],
        default="interval",
        help="Frame extraction method: 'interval' (time-based), 'motion' (motion-based), 'motion_pose' (motion + pose similarity)",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=5.0,
        help="Minimum motion score for motion-based extraction (default: 5.0).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Maximum pose similarity for motion_pose method (default: 0.8, lower = more diverse).",
    )
    parser.add_argument(
        "--min-frame-interval",
        type=int,
        default=5,
        help="Minimum frames between selections (default: 5).",
    )
    parser.add_argument(
        "--no-optical-flow",
        dest="use_optical_flow",
        action="store_false",
        help="Disable optical flow for motion detection.",
    )
    parser.add_argument(
        "--no-pose-similarity",
        dest="use_pose_similarity",
        action="store_false",
        help="Disable pose similarity filtering.",
    )
    parser.add_argument(
        "--initial-sampling-rate",
        type=float,
        default=0.1,
        help="Initial frame sampling rate in seconds for motion extraction (default: 0.1).",
    )
    parser.add_argument(
        "--segmentation",
        action="store_true",
        help="Enable segmentation after frame extraction.",
    )
    parser.add_argument(
        "--seg-method",
        type=str,
        choices=["yolo", "hsv"],
        default="yolo",
        help="Segmentation method: 'yolo' or 'hsv' (default: yolo).",
    )
    parser.add_argument(
        "--seg-model",
        type=str,
        default="yolov8n-seg.pt",
        help="YOLO segmentation model name (default: yolov8n-seg.pt). Only used for 'yolo' method.",
    )
    parser.add_argument(
        "--hsv-hue-tolerance",
        type=int,
        default=5,
        help="HSV method: Hue tolerance for hold detection (default: 5).",
    )
    parser.add_argument(
        "--hsv-sat-tolerance",
        type=int,
        default=50,
        help="HSV method: Saturation tolerance for hold detection (default: 50).",
    )
    parser.add_argument(
        "--hsv-val-tolerance",
        type=int,
        default=40,
        help="HSV method: Value tolerance for hold detection (default: 40).",
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
        method=args.method,
        motion_threshold=args.motion_threshold,
        similarity_threshold=args.similarity_threshold,
        min_frame_interval=args.min_frame_interval,
        use_optical_flow=args.use_optical_flow,
        use_pose_similarity=args.use_pose_similarity,
        initial_sampling_rate=args.initial_sampling_rate,
        segmentation=args.segmentation,
        seg_method=args.seg_method,
        seg_model=args.seg_model,
        hsv_hue_tolerance=args.hsv_hue_tolerance,
        hsv_sat_tolerance=args.hsv_sat_tolerance,
        hsv_val_tolerance=args.hsv_val_tolerance,
    )


if __name__ == "__main__":
    main()
