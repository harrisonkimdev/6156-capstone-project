"""Pipeline execution helpers used by the web UI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pose_ai.data import extract_frames_every_n_seconds, iter_video_files  # type: ignore  # pylint: disable=wrong-import-position
from pose_ai.service import (  # type: ignore  # pylint: disable=wrong-import-position
    estimate_poses_from_manifest,
    export_features_for_manifest,
    generate_segment_report,
)

MAX_POSE_SAMPLES = 10
MAX_VISUALS = 20


try:  # optional visualization step
    from scripts.visualize_pose import visualize_pose_results  # type: ignore  # pylint: disable=wrong-import-position
except ModuleNotFoundError:  # pragma: no cover
    visualize_pose_results = None  # type: ignore

from webapp.jobs import PipelineJob


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_repo_url(path: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(ROOT_DIR)
    except ValueError:
        return None
    return f"/repo/{relative.as_posix()}"


def run_pipeline_stage(
    *,
    video_dir: Path,
    output_dir: Path,
    interval: float,
    skip_visuals: bool,
    log: Callable[[str], None],
) -> Tuple[List[str], List[dict[str, str]], List[dict[str, object]]]:
    video_dir = video_dir.expanduser().resolve()
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    output_dir = _ensure_directory(output_dir.expanduser().resolve())

    manifests: list[Path] = []
    for index, video_path in enumerate(iter_video_files(video_dir)):
        log(f"[{index + 1}] Extracting frames from {video_path.name}")
        result = extract_frames_every_n_seconds(
            video_path,
            interval_seconds=interval,
            output_root=output_dir,
            write_manifest=True,
            overwrite=False,
        )
        if result.manifest_path is None:
            raise RuntimeError(f"Manifest not written for {video_path}")
        log(f"Saved {result.saved_frames} frames to {result.frame_directory}")
        manifests.append(result.manifest_path)

    if not manifests:
        log("No videos found; nothing to process.")
        return [], [], []

    visualization_items: list[dict[str, str]] = []
    pose_samples: list[dict[str, object]] = []

    for manifest_path in manifests:
        log(f"Estimating poses for {manifest_path}")
        estimate_poses_from_manifest(manifest_path)

        log("Exporting pose-derived features")
        export_features_for_manifest(manifest_path)

        log("Aggregating segment metrics")
        generate_segment_report(manifest_path)

        pose_results = manifest_path.parent / "pose_results.json"
        if pose_results.exists() and len(pose_samples) < MAX_POSE_SAMPLES:
            payload = json.loads(pose_results.read_text(encoding="utf-8"))
            for frame in payload.get("frames", []):
                if len(pose_samples) >= MAX_POSE_SAMPLES:
                    break
                image_value = frame.get("image_path")
                image_path = Path(image_value) if image_value else None
                sample: dict[str, object] = {
                    "timestamp_seconds": float(frame.get("timestamp_seconds", 0.0)),
                    "detection_score": float(frame.get("detection_score", 0.0)),
                }
                if image_path:
                    sample["image_path"] = str(image_path)
                    sample["image_name"] = image_path.name
                    url = _to_repo_url(image_path)
                    if url is not None:
                        sample["image_url"] = url
                pose_samples.append(
                    sample
                )

        if skip_visuals:
            continue

        if visualize_pose_results is None:
            log("Visualization skipped (mediapipe/cv2 dependencies not available).")
            continue
        try:
            count = visualize_pose_results(pose_results)
        except Exception as exc:  # pragma: no cover
            log(f"Visualization failed for {pose_results.name}: {exc}")
            continue

        viz_dir = pose_results.parent / "visualized"
        log(f"Generated {count} visualized frames in {viz_dir}")
        for image in sorted(viz_dir.glob("*"))[:MAX_VISUALS]:
            url = _to_repo_url(image)
            if url is None:
                continue
            visualization_items.append(
                {
                    "url": url,
                    "label": image.name,
                }
            )

    return [str(path) for path in manifests], visualization_items, pose_samples


def execute_job(job: PipelineJob) -> None:
    job.start()
    job.log("Pipeline execution started")
    try:
        manifests, visualizations, pose_samples = run_pipeline_stage(
            video_dir=Path(job.video_dir),
            output_dir=Path(job.output_dir),
            interval=job.interval,
            skip_visuals=job.skip_visuals,
            log=job.log,
        )
    except Exception as exc:  # pragma: no cover
        job.log(f"Pipeline failed: {exc}")
        job.fail(exc)
        return

    job.complete(manifests, visualizations, pose_samples)
    job.log("Pipeline completed successfully")
