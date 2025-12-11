"""Pipeline execution helpers used by the web UI."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pose_ai.data import (  # type: ignore  # pylint: disable=wrong-import-position
    extract_frames_with_motion,
    iter_video_files,
)
from pose_ai.service import (  # type: ignore  # pylint: disable=wrong-import-position
    estimate_poses_from_manifest,
    export_features_for_manifest,
    generate_segment_report,
    annotate_manifest_with_yolo,
    UltralyticsYoloSelector,
)
from pose_ai.service.hold_extraction import extract_and_cluster_holds, export_holds_json  # type: ignore
from pose_ai.cloud.gcs import get_gcs_manager

MAX_POSE_SAMPLES = 10
MAX_VISUALS = 20


try:  # optional visualization step
    from scripts.visualize_pose import visualize_pose_results  # type: ignore  # pylint: disable=wrong-import-position
except ModuleNotFoundError:  # pragma: no cover
    visualize_pose_results = None  # type: ignore

from webapp.jobs import PipelineJob
GCS_MANAGER = get_gcs_manager()


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_repo_url(path: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(ROOT_DIR)
    except ValueError:
        return None
    return f"/repo/{relative.as_posix()}"


def _upload_raw_video(job: PipelineJob, video_path: Path, log: Callable[[str], None]) -> None:
    """Upload raw video to GCS. Required operation - will raise exception on failure."""
    uri = GCS_MANAGER.upload_raw_video(video_path, upload_id=job.id)
    job.add_artifact("raw_videos", uri)
    log(f"Uploaded raw video to {uri}")


def _upload_frame_directory(job: PipelineJob, frame_dir: Path, log: Callable[[str], None]) -> None:
    """Upload frame directory to GCS. Required operation - will raise exception on failure."""
    uri = GCS_MANAGER.upload_frame_directory(frame_dir, job_id=job.id)
    job.add_artifact("frames", uri)
    log(f"Uploaded frames to {uri}")


def apply_frame_selector_model(
    manifest_path: Path,
    frame_selector_model_path: Path,
    threshold: float = 0.5,
    log: Callable[[str], None] = print,
) -> Path:
    """Apply BiLSTM model to select key frames and filter manifest.
    
    Args:
        manifest_path: Path to original manifest.json
        frame_selector_model_path: Path to trained BiLSTM model
        threshold: Key frame prediction threshold
        log: Logging function
    
    Returns:
        Filtered manifest path (may differ from original)
    """
    try:
        from pose_ai.service.frame_selector_service import predict_key_frames
        
        workflow_dir = manifest_path.parent
        
        # Check all_frames/ directory
        all_frames_dir = workflow_dir / 'all_frames'
        if not all_frames_dir.exists():
            log("Warning: all_frames/ directory not found, skipping frame selector")
            return manifest_path
        
        # Predict key frames using BiLSTM
        log("Applying BiLSTM frame selector model...")
        results = predict_key_frames(
            workflow_dir=workflow_dir,
            model_path=frame_selector_model_path,
            threshold=threshold,
            fps=30.0,  # TODO: Extract from manifest or pass as parameter
        )
        
        # Get frame names from selected_frames/ directory
        selected_frames_dir = workflow_dir / 'selected_frames'
        if not selected_frames_dir.exists():
            log("Warning: selected_frames/ directory not found after prediction")
            return manifest_path
        
        selected_frame_files = {f.name for f in selected_frames_dir.glob("*.jpg")}
        log(f"BiLSTM selected {len(selected_frame_files)} key frames")
        
        # Update manifest.json: filter only frames in selected_frames
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
        
        original_frame_count = len(manifest_data.get('frames', []))
        
        # Filter only frames in selected_frames
        filtered_frames = [
            frame for frame in manifest_data.get('frames', [])
            if Path(frame['image_path']).name in selected_frame_files
        ]
        
        if not filtered_frames:
            log("Warning: No frames matched after filtering, using original frames")
            return manifest_path
        
        manifest_data['frames'] = filtered_frames
        
        # Save updated manifest
        updated_manifest_path = workflow_dir / 'manifest_filtered.json'
        with open(updated_manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        log(f"Filtered manifest: {len(filtered_frames)} key frames (from {original_frame_count} total)")
        return updated_manifest_path
        
    except Exception as exc:
        log(f"Frame selector model failed: {exc}, using original frames")
        return manifest_path


def run_pipeline_stage(
    *,
    job: PipelineJob,
    video_dir: Path,
    output_dir: Path,
    yolo_options: dict[str, object] | None = None,
    segmentation_options: dict[str, object] | None = None,
    frame_extraction_options: dict[str, object] | None = None,
    log: Callable[[str], None],
) -> Tuple[List[str], List[dict[str, str]], List[dict[str, object]]]:
    video_dir = video_dir.expanduser().resolve()
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    output_dir = _ensure_directory(output_dir.expanduser().resolve())

    yolo_config = yolo_options or {}
    yolo_enabled = bool(yolo_config.get("enabled", True))
    yolo_selector: UltralyticsYoloSelector | None = None
    if yolo_enabled:
        yolo_selector = UltralyticsYoloSelector(
            model_name=str(yolo_config.get("model_name") or "yolov8n.pt"),
            device=yolo_config.get("device"),
            imgsz=int(yolo_config.get("imgsz") or 640),
        )

    # Determine frame extraction method
    extraction_config = frame_extraction_options or {}
    extraction_method = str(extraction_config.get("method", "motion")).lower()

    manifests: list[Path] = []
    for index, video_path in enumerate(iter_video_files(video_dir)):
        log(f"[{index + 1}] Extracting frames from {video_path.name} (method: {extraction_method})")
        
        # Use advanced motion-based extraction
        # save_all_frames=True: Save all frames to all_frames/ for BiLSTM analysis
        result = extract_frames_with_motion(
            video_path,
            output_root=output_dir,
            motion_threshold=float(extraction_config.get("motion_threshold", 5.0)),
            similarity_threshold=float(extraction_config.get("similarity_threshold", 0.8)),
            min_frame_interval=int(extraction_config.get("min_frame_interval", 5)),
            use_optical_flow=bool(extraction_config.get("use_optical_flow", True)),
            use_pose_similarity=bool(extraction_config.get("use_pose_similarity", True)) and extraction_method == "motion_pose",
            initial_sampling_rate=float(extraction_config.get("initial_sampling_rate", 0.1)),
            write_manifest=True,
            overwrite=False,
            save_all_frames=True,  # Production: Save all frames for BiLSTM frame selector
        )
        
        if result.manifest_path is None:
            raise RuntimeError(f"Manifest not written for {video_path}")
        log(f"Saved {result.saved_frames} frames to {result.frame_directory}")
        _upload_raw_video(job, result.video_path, log)
        manifest_path = result.manifest_path
        
        # Run segmentation if enabled
        seg_config = segmentation_options or {}
        if bool(seg_config.get("enabled", False)):
            try:
                from pose_ai.segmentation.yolo_segmentation import (  # type: ignore
                    YoloSegmentationModel,
                    export_segmentation_masks,
                    extract_hold_colors,
                    cluster_holds_by_color,
                    export_routes_json,
                )
                
                log("Running YOLO segmentation...")
                frame_dir = manifest_path.parent
                image_paths = sorted([p for p in frame_dir.glob("*.jpg")])
                
                if image_paths:
                    seg_model = YoloSegmentationModel(
                        model_name=str(seg_config.get("model_name", "yolov8n-seg.pt")),
                        device=yolo_config.get("device") if yolo_config else None,
                        imgsz=int(yolo_config.get("imgsz", 640)) if yolo_config else 640,
                    )
                    
                    seg_results = seg_model.batch_segment_frames(
                        image_paths,
                        conf_threshold=float(yolo_config.get("min_confidence", 0.25)) if yolo_config else 0.25,
                        target_classes=["wall", "hold", "climber", "person"],
                    )
                    
                    if bool(seg_config.get("export_masks", True)):
                        export_segmentation_masks(seg_results, frame_dir, export_images=True, export_json=True)
                        log(f"Exported segmentation masks to {frame_dir / 'masks'}")
                    
                    # Color-based route grouping if enabled
                    if bool(seg_config.get("group_by_color", True)):
                        try:
                            from pose_ai.service.hold_extraction import detect_holds  # type: ignore
                            
                            # Detect holds first
                            hold_detections = detect_holds(
                                image_paths,
                                model_name=str(yolo_config.get("model_name", "yolov8n.pt")) if yolo_config else "yolov8n.pt",
                                device=yolo_config.get("device") if yolo_config else None,
                            )
                            
                            if hold_detections:
                                # Extract colors
                                color_infos = extract_hold_colors(hold_detections, image_paths, segmentation_results=seg_results)
                                
                                if color_infos:
                                    # Cluster by color
                                    routes = cluster_holds_by_color(
                                        color_infos,
                                        hue_tolerance=int(seg_config.get("hue_tolerance", 10)),
                                        sat_tolerance=int(seg_config.get("sat_tolerance", 50)),
                                        val_tolerance=int(seg_config.get("val_tolerance", 50)),
                                    )
                                    
                                    if routes:
                                        routes_path = export_routes_json(routes, frame_dir / "routes.json")
                                        job.add_artifact("routes", _to_repo_url(routes_path) or str(routes_path))
                                        log(f"Grouped {len(routes)} routes by color → {routes_path.name}")
                        except Exception as exc:
                            log(f"Color-based route grouping failed: {exc}")
            except Exception as exc:
                log(f"Segmentation failed: {exc}")
        
        if yolo_enabled:
            selection = annotate_manifest_with_yolo(
                manifest_path,
                enabled=True,
                selector=yolo_selector,
                min_confidence=float(yolo_config.get("min_confidence") or 0.35),
                required_labels=yolo_config.get("required_labels"),
                target_labels=yolo_config.get("target_labels"),
                max_frames=int(yolo_config["max_frames"]) if yolo_config.get("max_frames") else None,
                label_map=yolo_config.get("label_map") or {},
            )
            if selection.skipped_reason:
                log(
                    f"YOLO filtering skipped ({selection.skipped_reason}); "
                    f"kept {selection.selected_frames}/{selection.total_frames} frames"
                )
            else:
                log(
                    f"YOLO filtered frames: {selection.selected_frames}/{selection.total_frames} retained"
                )
            manifest_path = selection.manifest_path
        manifests.append(manifest_path)

    if not manifests:
        log("No videos found; nothing to process.")
        return [], [], []

    # Apply BiLSTM frame selector (if production model is configured)
    production_frame_selector_model = os.getenv("PRODUCTION_FRAME_SELECTOR_MODEL")
    production_frame_selector_threshold = float(os.getenv("PRODUCTION_FRAME_SELECTOR_THRESHOLD", "0.5"))
    
    if production_frame_selector_model and Path(production_frame_selector_model).exists():
        log(f"BiLSTM frame selector model found: {production_frame_selector_model}")
        filtered_manifests = []
        for manifest_path in manifests:
            filtered_manifest = apply_frame_selector_model(
                manifest_path,
                Path(production_frame_selector_model),
                threshold=production_frame_selector_threshold,
                log=log,
            )
            filtered_manifests.append(filtered_manifest)
        manifests = filtered_manifests
    else:
        if production_frame_selector_model:
            log(f"Warning: Frame selector model not found at {production_frame_selector_model}, using all frames")
        else:
            log("No BiLSTM frame selector model configured, using motion-based frames")

    visualization_items: list[dict[str, str]] = []
    pose_samples: list[dict[str, object]] = []

    for manifest_path in manifests:
        log(f"Estimating poses for {manifest_path}")
        estimate_poses_from_manifest(manifest_path)

        # Hold extraction + clustering (single pass per manifest)
        try:
            frame_dir = manifest_path.parent
            image_paths = [p for p in frame_dir.glob("*.jpg")]
            if image_paths:
                # Check production YOLO model path (configurable via environment variable)
                production_yolo_model = os.getenv(
                    "PRODUCTION_YOLO_MODEL",
                    str(ROOT_DIR / "runs" / "hold_type" / "train" / "weights" / "best.pt")
                )
                if Path(production_yolo_model).exists():
                    yolo_model_name = production_yolo_model
                    log(f"Using production YOLO model: {production_yolo_model}")
                else:
                    yolo_model_name = str(yolo_config.get("model_name") or "yolov8n.pt")
                    log(f"Production YOLO model not found at {production_yolo_model}, using default: {yolo_model_name}")
                
                log(f"Extracting holds (model {yolo_model_name})")
                clustered = extract_and_cluster_holds(
                    image_paths,
                    model_name=yolo_model_name,
                    device=yolo_config.get("device"),
                )
                if clustered:
                    holds_path = export_holds_json(clustered, output_path=frame_dir / "holds.json")
                    job.add_artifact("holds", _to_repo_url(holds_path) or str(holds_path))
                    log(f"Clustered {len(clustered)} holds → {holds_path.name}")
                else:
                    holds_path = None
                    log("No holds detected; skipping holds.json export")
            else:
                holds_path = None
                log("No JPEG frames found for hold extraction.")
        except Exception as exc:  # pragma: no cover - defensive
            log(f"Hold extraction failed: {exc}")
            holds_path = None

        log("Exporting pose-derived features")
        # Extract IMU and climber parameters from job metadata
        metadata = job.metadata or {}
        imu_quaternion = metadata.get("imu_quaternion")
        imu_euler_angles = metadata.get("imu_euler_angles")
        climber_height = metadata.get("climber_height")
        climber_wingspan = metadata.get("climber_wingspan")
        climber_flexibility = metadata.get("climber_flexibility")
        
        export_features_for_manifest(
            manifest_path,
            holds_path=holds_path,
            auto_wall_angle=True,
            imu_quaternion=imu_quaternion,
            imu_euler_angles=imu_euler_angles,
            climber_height=climber_height,
            climber_wingspan=climber_wingspan,
            climber_flexibility=climber_flexibility,
        )

        log("Aggregating segment metrics")
        generate_segment_report(manifest_path)
        _upload_frame_directory(job, manifest_path.parent, log)

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
        # Extract options from job metadata or use defaults
        segmentation_options = getattr(job, "segmentation_options", None) or job.metadata.get("segmentation_options") if job.metadata else None
        frame_extraction_options = getattr(job, "frame_extraction_options", None) or job.metadata.get("frame_extraction_options") if job.metadata else None
        
        manifests, visualizations, pose_samples = run_pipeline_stage(
            job=job,
            video_dir=Path(job.video_dir),
            output_dir=Path(job.output_dir),
            yolo_options=job.yolo_options,
            segmentation_options=segmentation_options,
            frame_extraction_options=frame_extraction_options,
            log=job.log,
        )
    except Exception as exc:  # pragma: no cover
        job.log(f"Pipeline failed: {exc}")
        job.fail(exc)
        return

    job.complete(manifests, visualizations, pose_samples)
    job.log("Pipeline completed successfully")
