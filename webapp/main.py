"""FastAPI application exposing a minimal web UI for the BetaMove pipeline."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

from pose_ai.cloud.gcs import get_gcs_manager
from .jobs import JobManager
from .pipeline_runner import execute_job
from .training_jobs import TrainManager
from .labeling_sessions import LabelingManager, SessionStatus
from .yolo_training_jobs import YoloTrainManager, execute_yolo_training
from pose_ai.recommendation.efficiency import suggest_next_actions
from pose_ai.service.sam_service import SamService

app = FastAPI(title="BetaMove Pipeline UI")

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/repo", StaticFiles(directory=str(ROOT_DIR)), name="repo")

job_manager = JobManager()
UPLOAD_ROOT = ROOT_DIR / "data" / "uploads"
train_manager = TrainManager()
LABELING_SESSIONS_DIR = ROOT_DIR / "data" / "labeling_sessions"
labeling_manager = LabelingManager(LABELING_SESSIONS_DIR)
yolo_train_manager = YoloTrainManager()
LOGGER = logging.getLogger(__name__)
GCS_MANAGER = get_gcs_manager()


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_uploaded_file(video: UploadFile) -> tuple[Path, str]:
    """Save uploaded video file locally and upload to GCS.
    
    Raises:
        HTTPException: If file is not provided.
        Exception: If GCS upload fails.
    """
    if video is None:
        raise HTTPException(status_code=400, detail="No file provided")
    target_dir = _ensure_directory(UPLOAD_ROOT / uuid4().hex)
    file_path = target_dir / video.filename
    with file_path.open("wb") as destination:
        video.file.seek(0)
        shutil.copyfileobj(video.file, destination)
    video.file.close()
    # GCS upload is required - will raise exception on failure
    gcs_uri = GCS_MANAGER.upload_raw_video(
        file_path,
        upload_id=target_dir.name,
        metadata={"filename": video.filename},
    )
    return target_dir, gcs_uri


def _default_required_labels() -> list[str]:
    return ["climber", "person"]


def _default_target_labels() -> list[str]:
    return ["climber", "person", "hold", "wall"]


class MediaMetadata(BaseModel):
    route_name: str | None = Field(None, description="Route or boulder name")
    gym_location: str | None = Field(None, description="Venue or wall identifier")
    camera_orientation: str | None = Field(None, description="portrait/landscape/other")
    notes: str | None = Field(None, description="Freeform notes supplied by the climber")
    tags: list[str] = Field(default_factory=list, description="Optional list of search tags")
    
    # IMU/Gyroscope sensor data (raw data from mobile device)
    # Provide either quaternion or euler angles
    imu_quaternion: list[float] | None = Field(None, description="Device orientation as quaternion [w, x, y, z]", min_length=4, max_length=4)
    imu_euler_angles: list[float] | None = Field(None, description="Device orientation as Euler angles [pitch, roll, yaw] in degrees", min_length=3, max_length=3)
    imu_timestamp: float | None = Field(None, description="IMU reading timestamp (unix milliseconds)")
    
    # Climber physical parameters (optional, for personalized recommendations)
    climber_height: float | None = Field(None, description="Climber height in cm", gt=0, le=250)
    climber_wingspan: float | None = Field(None, description="Climber wingspan (fingertip to fingertip) in cm", gt=0, le=300)
    climber_flexibility: float | None = Field(None, description="Flexibility score: 0=low, 0.5=average, 1=high", ge=0, le=1.0)


class YoloOptions(BaseModel):
    enabled: bool = Field(True, description="Toggle YOLO still selection")
    model_name: str = Field("yolov8n.pt", description="Model weights passed to ultralytics")
    min_confidence: float = Field(0.35, ge=0.0, le=1.0)
    required_labels: list[str] = Field(default_factory=_default_required_labels)
    target_labels: list[str] = Field(default_factory=_default_target_labels)
    max_frames: int | None = Field(None, gt=0)
    imgsz: int = Field(640, ge=128, le=1536)
    device: str | None = Field(None, description="Optional torch device string")


class SegmentationOptions(BaseModel):
    enabled: bool = Field(False, description="Enable YOLO segmentation")
    method: str = Field("yolo", description="Segmentation method: 'yolo' or 'none'")
    model_name: str = Field("yolov8n-seg.pt", description="YOLO segmentation model name")
    export_masks: bool = Field(True, description="Export segmentation masks as images")
    group_by_color: bool = Field(True, description="Group holds by color to identify routes")
    hue_tolerance: int = Field(10, ge=0, le=90, description="Hue tolerance for color grouping")
    sat_tolerance: int = Field(50, ge=0, le=255, description="Saturation tolerance for color grouping")
    val_tolerance: int = Field(50, ge=0, le=255, description="Value tolerance for color grouping")


class FrameExtractionOptions(BaseModel):
    method: str = Field("motion", description="Extraction method: 'motion' or 'motion_pose'")
    motion_threshold: float = Field(5.0, ge=0.0, description="Minimum motion score for motion-based extraction")
    similarity_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Maximum pose similarity (lower = more diverse)")
    min_frame_interval: int = Field(5, ge=1, description="Minimum frames between selections")
    use_optical_flow: bool = Field(True, description="Use optical flow for motion detection")
    use_pose_similarity: bool = Field(True, description="Use pose similarity for frame selection")
    initial_sampling_rate: float = Field(0.1, gt=0, description="Initial frame sampling rate in seconds for motion extraction")


class PipelineRequest(BaseModel):
    video_dir: str = Field(..., description="Directory containing source videos")
    output_dir: str = Field("data/frames", description="Directory to write pipeline artifacts")
    source_uri: str | None = Field(None, description="Optional GCS URI referencing the uploaded video")
    metadata: MediaMetadata | None = Field(None, description="Optional capture metadata")
    yolo: YoloOptions = Field(default_factory=YoloOptions)
    segmentation: SegmentationOptions = Field(default_factory=SegmentationOptions)
    frame_extraction: FrameExtractionOptions = Field(default_factory=FrameExtractionOptions)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    jobs = [job.as_dict() for job in job_manager.list_jobs()]
    return templates.TemplateResponse("index.html", {"request": request, "jobs": jobs})


@app.get("/training", response_class=HTMLResponse)
async def training_page(request: Request) -> HTMLResponse:
    jobs = [job.as_dict() for job in train_manager.list()]
    return templates.TemplateResponse("training.html", {"request": request, "jobs": jobs})


@app.get("/grading", response_class=HTMLResponse)
async def grading_page(request: Request) -> HTMLResponse:
    """Route difficulty grading page."""
    from webapp.jobs import JobStatus
    jobs = [
        job.as_dict() for job in job_manager.list_jobs()
        if job.status == JobStatus.COMPLETED
    ]
    return templates.TemplateResponse("grading.html", {"request": request, "jobs": jobs})


@app.get("/api/jobs")
async def list_jobs() -> list[dict[str, object]]:
    return [job.as_dict() for job in job_manager.list_jobs()]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, object]:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    payload = job.as_dict()
    payload["logs"] = job.log_lines()
    return payload


@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(job_id: str) -> dict[str, object]:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"id": job.id, "logs": job.log_lines()}


@app.post("/api/jobs", status_code=201)
async def create_job(payload: PipelineRequest, background_tasks: BackgroundTasks) -> dict[str, object]:
    metadata_payload = payload.metadata.dict() if payload.metadata else {}
    if payload.source_uri:
        metadata_payload = dict(metadata_payload)
        metadata_payload["source_uri"] = payload.source_uri
    # Add segmentation and frame extraction options to metadata
    metadata_payload["segmentation_options"] = payload.segmentation.dict()
    metadata_payload["frame_extraction_options"] = payload.frame_extraction.dict()
    metadata_payload = metadata_payload or None
    job = job_manager.create_job(
        video_dir=payload.video_dir,
        output_dir=payload.output_dir,
        metadata=metadata_payload,
        yolo_options=payload.yolo.dict(),
    )
    background_tasks.add_task(execute_job, job)
    return job.as_dict()


@app.get("/api/videos")
async def list_videos() -> dict[str, list[str]]:
    root = Path("BetaMove") / "videos"
    if not root.exists():
        return {"videos": []}
    videos = sorted(str(path) for path in root.glob("**/*.mp4"))
    return {"videos": videos}


@app.post("/api/upload", response_class=JSONResponse)
async def upload_video(video: UploadFile = File(...)) -> dict[str, object]:
    saved_dir, gcs_uri = _save_uploaded_file(video)
    payload = {"video_dir": str(saved_dir), "gcs_uri": gcs_uri}
    return payload


async def run_frame_selector_inference(upload_id: str, video_name: str) -> None:
    """Run frame selector model inference if model exists and save predictions to selected_frames/."""
    import sys
    SRC_DIR = ROOT_DIR / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    
    try:
        from pose_ai.service.frame_selector_service import predict_key_frames
        
        workflow_dir = ROOT_DIR / "data" / "workflow_frames" / upload_id / video_name
        
        # Check if model checkpoint exists
        model_dir = workflow_dir / "frame_selector_output" / "checkpoints"
        if not model_dir.exists():
            LOGGER.info(f"No frame selector model found for {video_name}, skipping inference")
            return
        
        LOGGER.info(f"Running frame selector inference for {video_name}")
        
        # Run prediction and save to selected_frames/
        results = predict_key_frames(
            workflow_dir=workflow_dir,
            output_dir=workflow_dir / "frame_selector_output",
        )
        
        LOGGER.info(f"Frame selector inference complete: {results}")
        
    except ImportError as e:
        LOGGER.debug(f"Frame selector service not available: {e}")
    except Exception as e:
        LOGGER.warning(f"Frame selector inference failed: {e}", exc_info=True)


@app.post("/api/workflow/extract-frames", response_class=JSONResponse)
async def workflow_extract_frames(
    video: UploadFile = File(...),
) -> dict[str, object]:
    """Extract frames from uploaded video for workflow (accepts file upload)."""
    import sys
    from pathlib import Path
    
    SRC_DIR = ROOT_DIR / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    
    from pose_ai.data import extract_frames_with_motion
    
    # Save uploaded video
    upload_id = uuid4().hex
    upload_dir = _ensure_directory(UPLOAD_ROOT / upload_id)
    video_path = upload_dir / video.filename
    
    try:
        print(f"[FRAME EXTRACTION] Starting frame extraction for video: {video.filename}")
        print(f"[FRAME EXTRACTION] Upload ID: {upload_id}")
        
        with video_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)
        
        print(f"[FRAME EXTRACTION] Video saved to: {video_path}")
        
        # Extract frames - use workflow_frames root, let extract_frames_with_motion create video folder
        output_root = _ensure_directory(ROOT_DIR / "data" / "workflow_frames" / upload_id)
        print(f"[FRAME EXTRACTION] Output root: {output_root}")
        
        result = extract_frames_with_motion(
            video_path,
            output_root=output_root,
            motion_threshold=5.0,  # Default value
            similarity_threshold=0.8,  # Default value
            write_manifest=True,
            overwrite=True,
            save_all_frames=True,
            use_pose_similarity=True,
        )
        
        print(f"[FRAME EXTRACTION] Extraction complete! Saved {result.saved_frames} frames")
        print(f"[FRAME EXTRACTION] Frame directory: {result.frame_directory}")
        
        # Run inference if model exists
        try:
            await run_frame_selector_inference(
                upload_id=upload_id,
                video_name=Path(video.filename).stem,
            )
        except Exception as e:
            LOGGER.warning(f"Frame selector inference failed (model may not exist yet): {e}")
            # Don't raise - inference is optional
        
        return {
            "status": "success",
            "frame_directory": str(result.frame_directory),
            "frame_count": result.saved_frames,
            "manifest_path": str(result.manifest_path) if result.manifest_path else None,
        }
    except Exception as e:
        LOGGER.exception("Workflow frame extraction failed")
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {str(e)}")
    finally:
        video.file.close()


@app.get("/api/workflow/extract-test-video", response_class=JSONResponse)
async def extract_test_video() -> dict[str, object]:
    """Extract frames from hardcoded test video using motion detection (for development)."""
    import sys
    
    SRC_DIR = ROOT_DIR / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    
    from pose_ai.data import extract_frames_with_motion
    
    # Hardcoded test video path (original .mov file now works!)
    test_video_path = ROOT_DIR / "data" / "test_video" / "IMG_3571.mov"
    
    if not test_video_path.exists():
        raise HTTPException(status_code=404, detail=f"Test video not found: {test_video_path}")
    
    try:
        print(f"[TEST FRAME EXTRACTION] Using motion detection on: {test_video_path}")
        
        # Use fixed output directory for test video
        upload_id = "test_video_frames"
        output_root = _ensure_directory(ROOT_DIR / "data" / "workflow_frames" / upload_id)
        print(f"[TEST FRAME EXTRACTION] Output root: {output_root}")
        print(f"[TEST FRAME EXTRACTION] Motion threshold: {motion_threshold}, Similarity threshold: {similarity_threshold}")
        result = extract_frames_with_motion(
            test_video_path,
            output_root=output_root,
            motion_threshold=5.0,  # Default value
            similarity_threshold=0.8,  # Default value
            write_manifest=True,
            overwrite=True,
            save_all_frames=True,
            use_pose_similarity=True,
        )
        
        print(f"[TEST FRAME EXTRACTION] Extraction complete! Saved {result.saved_frames} frames")
        print(f"[TEST FRAME EXTRACTION] Frame directory: {result.frame_directory}")
        
        return {
            "status": "success",
            "frame_directory": str(result.frame_directory),
            "frame_count": result.saved_frames,
            "manifest_path": str(result.manifest_path) if result.manifest_path else None,
            "video_path": str(test_video_path),
        }
    except Exception as e:
        LOGGER.exception("Test video frame extraction failed")
        print(f"[TEST FRAME EXTRACTION] ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {str(e)}")


@app.get("/api/workflow/use-test-frames", response_class=JSONResponse)
async def use_test_frames() -> dict[str, object]:
    """Use pre-extracted test frames (instant, no processing needed - dev mode)."""
    test_frame_dir = ROOT_DIR / "data" / "test_frames" / "IMG_3571"
    
    if not test_frame_dir.exists():
        raise HTTPException(status_code=404, detail=f"Test frames not found: {test_frame_dir}")
    
    # Count frames
    frame_files = list(test_frame_dir.glob("IMG_3571_frame_*.jpg"))
    
    LOGGER.info("Using pre-extracted test frames: %s (%d frames)", test_frame_dir, len(frame_files))
    
    return {
        "status": "success",
        "frame_directory": str(test_frame_dir),
        "frame_count": len(frame_files),
        "manifest_path": str(test_frame_dir / "manifest.json"),
        "test_mode": True,
        "message": "Using pre-extracted test frames (instant!)",
    }


# ========== Frame Selection Learning Endpoints ==========

@app.get("/api/workflow/frames/{upload_id}/{video_name}", response_class=JSONResponse)
async def get_frames_for_selection(upload_id: str, video_name: str) -> dict[str, object]:
    """Get all frames from all_frames/ directory for manual selection."""
    workflow_dir = ROOT_DIR / "data" / "workflow_frames" / upload_id / video_name
    all_frames_dir = workflow_dir / "all_frames"
    human_selected_dir = workflow_dir / "human_selected_frames"
    
    if not all_frames_dir.exists():
        raise HTTPException(status_code=404, detail=f"all_frames directory not found: {all_frames_dir}")
    
    # Get all frame files
    frame_files = sorted(all_frames_dir.glob(f"{video_name}_frame_*.jpg"))
    
    # Get already selected frames
    selected_files = set()
    if human_selected_dir.exists():
        selected_files = {f.name for f in human_selected_dir.glob(f"{video_name}_frame_*.jpg")}
    
    frames = []
    for frame_path in frame_files:
        frames.append({
            "filename": frame_path.name,
            "path": f"/repo/data/workflow_frames/{upload_id}/{video_name}/all_frames/{frame_path.name}",
            "selected": frame_path.name in selected_files,
        })
    
    return {
        "status": "success",
        "upload_id": upload_id,
        "video_name": video_name,
        "total_frames": len(frames),
        "selected_count": len(selected_files),
        "frames": frames,
    }


@app.post("/api/workflow/frames/{upload_id}/{video_name}/select", response_class=JSONResponse)
async def select_frame(upload_id: str, video_name: str, frame_name: str) -> dict[str, object]:
    """Copy a frame from all_frames/ to human_selected_frames/."""
    workflow_dir = ROOT_DIR / "data" / "workflow_frames" / upload_id / video_name
    all_frames_dir = workflow_dir / "all_frames"
    human_selected_dir = workflow_dir / "human_selected_frames"
    
    source_path = all_frames_dir / frame_name
    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"Frame not found: {frame_name}")
    
    # Ensure human_selected_frames/ directory exists
    human_selected_dir.mkdir(parents=True, exist_ok=True)
    
    dest_path = human_selected_dir / frame_name
    
    # Copy file
    shutil.copy2(source_path, dest_path)
    
    LOGGER.info("Selected frame: %s -> %s", source_path, dest_path)
    
    return {
        "status": "success",
        "message": f"Frame {frame_name} selected",
        "frame_name": frame_name,
    }


@app.delete("/api/workflow/frames/{upload_id}/{video_name}/select/{frame_name}", response_class=JSONResponse)
async def deselect_frame(upload_id: str, video_name: str, frame_name: str) -> dict[str, object]:
    """Remove a frame from human_selected_frames/."""
    workflow_dir = ROOT_DIR / "data" / "workflow_frames" / upload_id / video_name
    human_selected_dir = workflow_dir / "human_selected_frames"
    
    frame_path = human_selected_dir / frame_name
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail=f"Frame not found in human_selected_frames: {frame_name}")
    
    frame_path.unlink()
    
    LOGGER.info("Deselected frame: %s", frame_path)
    
    return {
        "status": "success",
        "message": f"Frame {frame_name} deselected",
        "frame_name": frame_name,
    }


@app.post("/api/workflow/save-to-training-pool", response_class=JSONResponse)
async def save_to_training_pool(request: Request) -> dict[str, object]:
    """Save selected frames from current video to training pool."""
    try:
        data = await request.json()
        upload_id = data.get("upload_id")
        video_name = data.get("video_name")
        
        if not upload_id or not video_name:
            raise HTTPException(status_code=400, detail="Missing upload_id or video_name")
        
        workflow_dir = ROOT_DIR / "data" / "workflow_frames" / upload_id / video_name
        human_selected_dir = workflow_dir / "human_selected_frames"
        
        if not workflow_dir.exists():
            raise HTTPException(status_code=404, detail="Workflow directory not found")
        
        if not human_selected_dir.exists() or len(list(human_selected_dir.glob("*.jpg"))) == 0:
            raise HTTPException(status_code=400, detail="No frames selected. Please select at least one frame.")
        
        # Create training pool directory structure
        training_pool_dir = ROOT_DIR / "data" / "training_pool"
        training_pool_dir.mkdir(parents=True, exist_ok=True)
        
        # Use upload_id + video_name as unique identifier
        pool_video_dir = training_pool_dir / f"{upload_id}_{video_name}"
        
        # Copy workflow directory to training pool
        if pool_video_dir.exists():
            # If already exists, remove and replace
            shutil.rmtree(pool_video_dir)
        
        shutil.copytree(workflow_dir, pool_video_dir, dirs_exist_ok=True)
        
        LOGGER.info(f"Saved video '{video_name}' to training pool: {pool_video_dir}")
        
        # Count total videos and frames in pool
        pool_videos = [d for d in training_pool_dir.iterdir() if d.is_dir()]
        total_frames = 0
        for video_dir in pool_videos:
            selected_dir = video_dir / "human_selected_frames"
            if selected_dir.exists():
                total_frames += len(list(selected_dir.glob("*.jpg")))
        
        return {
            "status": "success",
            "message": f"Saved to training pool",
            "total_videos": len(pool_videos),
            "total_frames": total_frames,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Failed to save to training pool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workflow/training-pool-info", response_class=JSONResponse)
async def get_training_pool_info() -> dict[str, int]:
    """Get training pool statistics."""
    try:
        training_pool_dir = ROOT_DIR / "data" / "training_pool"
        
        if not training_pool_dir.exists():
            return {"video_count": 0, "frame_count": 0}
        
        pool_videos = [d for d in training_pool_dir.iterdir() if d.is_dir()]
        total_frames = 0
        
        for video_dir in pool_videos:
            selected_dir = video_dir / "human_selected_frames"
            if selected_dir.exists():
                total_frames += len(list(selected_dir.glob("*.jpg")))
        
        return {
            "video_count": len(pool_videos),
            "frame_count": total_frames,
        }
        
    except Exception as e:
        LOGGER.error(f"Failed to get training pool info: {e}")
        return {"video_count": 0, "frame_count": 0}


@app.post("/api/workflow/train-frame-selector", response_class=JSONResponse)
async def train_frame_selector(request: Request) -> dict[str, object]:
    """Train frame selector model using manually selected frames from current video."""
    try:
        import sys
        SRC_DIR = ROOT_DIR / "src"
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        
        from pose_ai.service.frame_selector_service import train_frame_selector_pipeline
        from pose_ai.ml.frame_selector_trainer import TrainerConfig
        
        data = await request.json()
        upload_id = data.get("upload_id")
        video_name = data.get("video_name")
        fps = float(data.get("fps", 30.0))
        
        if not upload_id or not video_name:
            raise HTTPException(status_code=400, detail="Missing upload_id or video_name")
        
        workflow_dir = ROOT_DIR / "data" / "workflow_frames" / upload_id / video_name
        human_selected_dir = workflow_dir / "human_selected_frames"
        
        if not human_selected_dir.exists():
            raise HTTPException(status_code=404, detail="No human_selected_frames directory found")
        
        # Count selected frames
        selected_frames = list(human_selected_dir.glob("*.jpg"))
        if len(selected_frames) == 0:
            raise HTTPException(status_code=400, detail="No frames selected. Please select at least one frame.")
        
        LOGGER.info("Starting frame selector training with %d selected frames", len(selected_frames))
        
        # Run training pipeline
        output_dir = workflow_dir / "frame_selector_output"
        config = TrainerConfig(
            epochs=50,
            batch_size=8,
            patience=10,
            pos_weight=10.0,
            save_dir=output_dir / "checkpoints",
            device='cpu',  # Use CPU for now
        )
        
        results = train_frame_selector_pipeline(
            workflow_dir=workflow_dir,
            output_dir=output_dir,
            fps=fps,
            config=config,
        )
        
        LOGGER.info("Frame selector training completed successfully")
        
        return {
            "status": "success",
            "message": f"Training completed with {len(selected_frames)} key frames",
            "results": results,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception("Failed to train frame selector")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/api/system/clear", response_class=JSONResponse)
async def clear_data() -> dict[str, object]:
    """Clear all uploads and workflow frames data."""
    try:
        cleared_dirs = []
        
        # Clear uploads directory
        if UPLOAD_ROOT.exists():
            for item in UPLOAD_ROOT.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            cleared_dirs.append("uploads")
            LOGGER.info("Cleared uploads directory: %s", UPLOAD_ROOT)
        
        # Clear workflow_frames directory
        workflow_frames_dir = ROOT_DIR / "data" / "workflow_frames"
        if workflow_frames_dir.exists():
            for item in workflow_frames_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            cleared_dirs.append("workflow_frames")
            LOGGER.info("Cleared workflow_frames directory: %s", workflow_frames_dir)
        
        return {
            "status": "success",
            "message": f"Cleared: {', '.join(cleared_dirs)}",
            "cleared_directories": cleared_dirs,
        }
    except Exception as e:
        LOGGER.exception("Failed to clear data")
        raise HTTPException(status_code=500, detail=f"Failed to clear data: {str(e)}")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/train/upload", response_class=JSONResponse)
async def upload_features(file: UploadFile = File(...)) -> dict[str, object]:
    root = _ensure_directory(UPLOAD_ROOT / "features")
    dest = root / file.filename
    with dest.open("wb") as f:
        file.file.seek(0)
        shutil.copyfileobj(file.file, f)
    file.file.close()
    return {"features_path": str(dest)}


class TrainRequest(BaseModel):
    features_path: str
    task: str = Field("classification", description="classification or regression")
    label_column: str = Field("detection_score")
    label_threshold: float | None = 0.6
    test_size: float = 0.2
    random_state: int = 42
    model_out: str = "models/xgb_pose.json"
    upload_to_gcs: bool = Field(False, description="Upload trained model to GCS (default: False)")


@app.post("/api/train", response_class=JSONResponse)
async def start_training(payload: TrainRequest, background_tasks: BackgroundTasks) -> dict[str, object]:
    params = payload.dict()
    features_path = params.pop("features_path")
    job = train_manager.create(features_path=features_path, params=params)
    from webapp.training_runner import execute_training  # lazy import
    background_tasks.add_task(execute_training, job)
    return job.as_dict()


@app.get("/api/train/jobs")
async def list_train_jobs() -> list[dict[str, object]]:
    return [job.as_dict() for job in train_manager.list()]


@app.get("/api/train/jobs/{job_id}")
async def get_train_job(job_id: str) -> dict[str, object]:
    job = train_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.as_dict()


@app.post("/api/jobs/{job_id}/clear")
async def clear_job(job_id: str) -> dict[str, object]:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    # Clear logs, visualizations, and pose samples (non-destructive to files)
    job.clear(include_logs=True, include_visuals=True, include_samples=True)
    payload = job.as_dict()
    payload["logs"] = job.log_lines()
    return payload


@app.get("/api/jobs/{job_id}/analysis")
async def job_analysis(job_id: str) -> dict[str, object]:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.manifests:
        raise HTTPException(status_code=400, detail="Job has no manifests yet")
    # Use first manifest directory for analysis artifacts.
    frame_dir = Path(job.manifests[0]).parent
    features_path = frame_dir / "pose_features.json"
    if not features_path.exists():
        raise HTTPException(status_code=400, detail="Features not found for job")
    import json
    feature_rows = json.loads(features_path.read_text(encoding="utf-8"))
    holds_path = frame_dir / "holds.json"
    holds_payload: dict[str, dict] = {}
    if holds_path.exists():
        holds_payload = json.loads(holds_path.read_text(encoding="utf-8"))
    next_holds = []
    if feature_rows and holds_payload:
        next_holds = suggest_next_actions(feature_rows[-1], list(holds_payload.values()), top_k=3)
    return {
        "job_id": job.id,
        "next_holds": next_holds,
        "holds_count": len(holds_payload),
        "wall_angle": feature_rows[-1].get("wall_angle") if feature_rows else None,
    }
@app.get("/api/jobs/{job_id}/ml_predictions")
def get_ml_predictions(job_id: str):
    """Get ML model predictions for a job.
    
    Returns efficiency scores and next-action predictions from BiLSTM model.
    Requires trained model at models/checkpoints/bilstm_multitask.pt
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # Check if model exists
    model_path = ROOT_DIR / "models" / "checkpoints" / "bilstm_multitask.pt"
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail="ML model not available. Train model first using scripts/train_bilstm.py"
        )
    
    # Load features
    import json
    features_path = ROOT_DIR / "data" / "uploads" / job_id / "features.json"
    if not features_path.exists():
        raise HTTPException(status_code=404, detail="Features not found")
    
    try:
        # Initialize inference engine (auto-detects BiLSTM or Transformer)
        from pose_ai.ml.inference import ModelInference
        
        norm_path = ROOT_DIR / "models" / "checkpoints" / "normalization.npz"
        inference = ModelInference(
            model_path=model_path,
            normalization_path=norm_path if norm_path.exists() else None,
            device="cpu",  # Use CPU for web service
            window_size=32,
        )
        
        # Run inference
        results = inference.predict_from_json(features_path, stride=5)
        
        # Format results
        predictions = [
            {
                "frame_index": r.frame_index,
                "efficiency_score": r.efficiency_score,
                "next_action": r.next_action_name,
                "next_action_probs": r.next_action_probs,
            }
            for r in results
        ]
        
        return {
            "job_id": job_id,
            "num_predictions": len(predictions),
            "predictions": predictions,
        }
    
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="PyTorch not installed. Install with: pip install torch"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/api/jobs/{job_id}/route_grade")
async def get_route_grade(job_id: str):
    """Get predicted route difficulty grade (V0-V10).
    
    Requires completed pipeline job with features, segments, efficiency, and holds.
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # Find job output directory
    # Jobs are stored in data/uploads/{job_id} or output_dir from job
    job_output_dir = ROOT_DIR / "data" / "uploads" / job_id
    if not job_output_dir.exists():
        # Try output_dir from job metadata
        output_dir_str = job.output_dir if hasattr(job, "output_dir") else None
        if output_dir_str:
            job_output_dir = Path(output_dir_str)
        else:
            raise HTTPException(status_code=404, detail="Job output directory not found")
    
    # Load required files
    import json
    
    features_path = job_output_dir / "pose_features.json"
    efficiency_path = job_output_dir / "step_efficiency.json"
    holds_path = job_output_dir / "holds.json"
    
    if not features_path.exists():
        raise HTTPException(status_code=404, detail="pose_features.json not found")
    
    try:
        # Load feature rows
        with open(features_path, "r") as f:
            feature_rows = json.load(f)
        
        # Load step efficiency
        step_efficiency = []
        step_segments = []
        if efficiency_path.exists():
            with open(efficiency_path, "r") as f:
                efficiency_data = json.load(f)
                from pose_ai.recommendation.efficiency import StepEfficiencyResult
                from pose_ai.segmentation.steps import StepSegment
                
                for i, eff_dict in enumerate(efficiency_data):
                    step_segments.append(StepSegment(
                        step_id=eff_dict.get("step_id", i),
                        start_index=0,
                        end_index=0,
                        start_time=eff_dict.get("start_time", 0.0),
                        end_time=eff_dict.get("end_time", 0.0),
                        label=eff_dict.get("label", "unknown"),
                    ))
                    step_efficiency.append(StepEfficiencyResult(
                        step_id=eff_dict.get("step_id", i),
                        score=eff_dict.get("score", 0.0),
                        components=eff_dict.get("components", {}),
                        start_time=eff_dict.get("start_time", 0.0),
                        end_time=eff_dict.get("end_time", 0.0),
                    ))
        
        # Load holds
        holds = []
        if holds_path.exists():
            with open(holds_path, "r") as f:
                holds_data = json.load(f)
                if isinstance(holds_data, dict):
                    holds = list(holds_data.values())
                elif isinstance(holds_data, list):
                    holds = holds_data
        
        # Extract wall angle
        wall_angle = None
        for row in feature_rows:
            angle = row.get("wall_angle")
            if angle is not None:
                wall_angle = float(angle)
                break
        
        # Extract route features
        from pose_ai.ml.route_grading import extract_route_features, RouteDifficultyModel, GymGradeCalibration
        
        route_features = extract_route_features(
            feature_rows=feature_rows,
            step_segments=step_segments,
            step_efficiency=step_efficiency,
            holds=holds,
            wall_angle=wall_angle,
        )
        
        # Predict grade
        model_path = ROOT_DIR / "models" / "route_grader.json"
        if model_path.exists():
            model = RouteDifficultyModel(model_path=model_path)
            grade, confidence = model.predict_with_confidence(route_features)
            
            # Calibrate to gym scale
            calibration = GymGradeCalibration()
            calibrated_grade = calibration.calibrate(grade)
        else:
            # No model available - return features only
            grade = None
            confidence = 0.0
            calibrated_grade = None
        
        return {
            "job_id": job_id,
            "grade": grade,
            "confidence": confidence,
            "calibrated_grade": calibrated_grade,
            "features": route_features,
        }
    
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Required packages not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route grading failed: {str(e)}")


# Testing API endpoints for individual step execution
TEST_OUTPUT_ROOT = ROOT_DIR / "data" / "test_outputs"


class TestExtractFramesRequest(BaseModel):
    video_file: str = Field(..., description="Path to video file or upload ID")
    output_dir: str | None = Field(None, description="Output directory (auto-generated if not provided)")
    motion_threshold: float = Field(5.0, ge=0.0, description="Minimum motion score for motion-based extraction")
    similarity_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Maximum pose similarity")


@app.post("/api/test/extract-frames", response_class=JSONResponse)
async def test_extract_frames(payload: TestExtractFramesRequest) -> dict[str, object]:
    """Test frame extraction step independently."""
    import sys
    from pathlib import Path
    
    SRC_DIR = ROOT_DIR / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    
    from pose_ai.data import extract_frames_with_motion
    
    video_path = Path(payload.video_file)
    if not video_path.exists():
        # Try as upload ID
        upload_path = UPLOAD_ROOT / payload.video_file
        if upload_path.exists():
            # Find video file in upload directory
            video_files = list(upload_path.glob("*.mp4")) + list(upload_path.glob("*.avi")) + list(upload_path.glob("*.mov"))
            if video_files:
                video_path = video_files[0]
            else:
                raise HTTPException(status_code=404, detail=f"Video file not found: {payload.video_file}")
        else:
            raise HTTPException(status_code=404, detail=f"Video file not found: {payload.video_file}")
    
    output_dir = Path(payload.output_dir) if payload.output_dir else _ensure_directory(TEST_OUTPUT_ROOT / uuid4().hex)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = extract_frames_with_motion(
            video_path,
            output_root=output_dir,
            motion_threshold=payload.motion_threshold,
            similarity_threshold=payload.similarity_threshold,
            write_manifest=True,
            overwrite=False,
        )
        
        manifest_data = None
        if result.manifest_path and result.manifest_path.exists():
            import json
            manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        
        return {
            "status": "success",
            "output_dir": str(output_dir),
            "manifest_path": str(result.manifest_path) if result.manifest_path else None,
            "saved_frames": result.saved_frames,
            "manifest": manifest_data,
        }
    except Exception as e:
        LOGGER.exception("Frame extraction failed")
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {str(e)}")


class TestDetectHoldsRequest(BaseModel):
    frame_dir: str = Field(..., description="Directory containing frame images")
    model_name: str = Field("yolov8n.pt", description="YOLO model name")
    device: str | None = Field(None, description="Device (cpu/cuda:0)")
    eps: float = Field(0.03, description="DBSCAN epsilon")
    min_samples: int = Field(3, description="DBSCAN min_samples")
    use_tracking: bool = Field(True, description="Use temporal tracking")


@app.post("/api/test/detect-holds", response_class=JSONResponse)
async def test_detect_holds(payload: TestDetectHoldsRequest) -> dict[str, object]:
    """Test hold detection step independently."""
    import sys
    from pathlib import Path
    
    SRC_DIR = ROOT_DIR / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    
    from pose_ai.service.hold_extraction import extract_and_cluster_holds, export_holds_json
    
    frame_dir = Path(payload.frame_dir)
    if not frame_dir.exists():
        raise HTTPException(status_code=404, detail=f"Frame directory not found: {payload.frame_dir}")
    
    image_paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
    if not image_paths:
        raise HTTPException(status_code=400, detail=f"No image files found in {payload.frame_dir}")
    
    try:
        clustered = extract_and_cluster_holds(
            image_paths,
            model_name=payload.model_name,
            device=payload.device,
            eps=payload.eps,
            min_samples=payload.min_samples,
            use_tracking=payload.use_tracking,
        )
        
        output_path = frame_dir / "holds.json"
        if clustered:
            export_holds_json(clustered, output_path=output_path)
            holds_data = {hold.hold_id: hold.as_dict() for hold in clustered}
        else:
            holds_data = {}
        
        return {
            "status": "success",
            "holds_path": str(output_path),
            "holds_count": len(clustered),
            "holds": holds_data,
        }
    except Exception as e:
        LOGGER.exception("Hold detection failed")
        raise HTTPException(status_code=500, detail=f"Hold detection failed: {str(e)}")


class TestEstimateWallAngleRequest(BaseModel):
    image_path: str = Field(..., description="Path to representative frame image")
    hold_centers: list[list[float]] | None = Field(None, description="Optional hold centers [[x, y], ...]")


@app.post("/api/test/estimate-wall-angle", response_class=JSONResponse)
async def test_estimate_wall_angle(payload: TestEstimateWallAngleRequest) -> dict[str, object]:
    """Test wall angle estimation step independently."""
    import sys
    from pathlib import Path
    
    SRC_DIR = ROOT_DIR / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    
    from pose_ai.wall.angle import estimate_wall_angle
    
    image_path = Path(payload.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found: {payload.image_path}")
    
    hold_centers = None
    if payload.hold_centers:
        hold_centers = [tuple(center) for center in payload.hold_centers]
    
    try:
        result = estimate_wall_angle(image_path, hold_centers=hold_centers)
        return {
            "status": "success",
            "result": result.as_dict(),
        }
    except Exception as e:
        LOGGER.exception("Wall angle estimation failed")
        raise HTTPException(status_code=500, detail=f"Wall angle estimation failed: {str(e)}")


class TestEstimatePoseRequest(BaseModel):
    manifest_path: str = Field(..., description="Path to manifest.json")


@app.post("/api/test/estimate-pose", response_class=JSONResponse)
async def test_estimate_pose(payload: TestEstimatePoseRequest) -> dict[str, object]:
    """Test pose estimation step independently."""
    import sys
    from pathlib import Path
    
    SRC_DIR = ROOT_DIR / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    
    from pose_ai.service.pose_service import estimate_poses_from_manifest
    
    manifest_path = Path(payload.manifest_path)
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Manifest file not found: {payload.manifest_path}")
    
    try:
        frames = estimate_poses_from_manifest(manifest_path, save_json=True)
        
        pose_results_path = manifest_path.parent / "pose_results.json"
        pose_data = None
        if pose_results_path.exists():
            import json
            pose_data = json.loads(pose_results_path.read_text(encoding="utf-8"))
        
        return {
            "status": "success",
            "pose_results_path": str(pose_results_path),
            "frames_processed": len(frames),
            "pose_results": pose_data,
        }
    except Exception as e:
        LOGGER.exception("Pose estimation failed")
        raise HTTPException(status_code=500, detail=f"Pose estimation failed: {str(e)}")


class TestExtractFeaturesRequest(BaseModel):
    manifest_path: str = Field(..., description="Path to manifest.json")
    holds_path: str | None = Field(None, description="Optional path to holds.json")
    auto_wall_angle: bool = Field(True, description="Auto-estimate wall angle")
    climber_height: float | None = Field(None, description="Climber height in cm")
    climber_wingspan: float | None = Field(None, description="Climber wingspan in cm")
    climber_flexibility: float | None = Field(None, description="Climber flexibility (0-1)")


@app.post("/api/test/extract-features", response_class=JSONResponse)
async def test_extract_features(payload: TestExtractFeaturesRequest) -> dict[str, object]:
    """Test feature extraction step independently (includes segmentation and efficiency)."""
    import sys
    from pathlib import Path
    
    SRC_DIR = ROOT_DIR / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    
    from pose_ai.service.feature_service import export_features_for_manifest
    
    manifest_path = Path(payload.manifest_path)
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Manifest file not found: {payload.manifest_path}")
    
    holds_path = Path(payload.holds_path) if payload.holds_path else None
    if holds_path and not holds_path.exists():
        raise HTTPException(status_code=404, detail=f"Holds file not found: {payload.holds_path}")
    
    try:
        output_path = export_features_for_manifest(
            manifest_path,
            holds_path=holds_path,
            auto_wall_angle=payload.auto_wall_angle,
            climber_height=payload.climber_height,
            climber_wingspan=payload.climber_wingspan,
            climber_flexibility=payload.climber_flexibility,
        )
        
        features_data = None
        efficiency_data = None
        if output_path.exists():
            import json
            features_data = json.loads(output_path.read_text(encoding="utf-8"))
        
        efficiency_path = output_path.parent / "step_efficiency.json"
        if efficiency_path.exists():
            import json
            efficiency_data = json.loads(efficiency_path.read_text(encoding="utf-8"))
        
        return {
            "status": "success",
            "features_path": str(output_path),
            "efficiency_path": str(efficiency_path) if efficiency_path.exists() else None,
            "features_count": len(features_data) if features_data else 0,
            "efficiency_count": len(efficiency_data) if efficiency_data else 0,
            "sample_feature": features_data[0] if features_data else None,
            "sample_efficiency": efficiency_data[0] if efficiency_data else None,
        }
    except Exception as e:
        LOGGER.exception("Feature extraction failed")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")


@app.get("/api/test/validate/{step_name}", response_class=JSONResponse)
async def validate_step_output(step_name: str) -> dict[str, object]:
    """Validate output of a specific step."""
    # This is a placeholder - can be expanded with actual validation logic
    valid_steps = [
        "extract-frames",
        "detect-holds",
        "estimate-wall-angle",
        "estimate-pose",
        "extract-features",
        "segment",
        "compute-efficiency",
    ]
    
    if step_name not in valid_steps:
        raise HTTPException(status_code=400, detail=f"Invalid step name: {step_name}. Valid steps: {valid_steps}")
    
    return {
        "step": step_name,
        "status": "validation_not_implemented",
        "message": "Step validation logic to be implemented",
    }


@app.get("/testing", response_class=HTMLResponse)
async def testing_page(request: Request) -> HTMLResponse:
    """Testing page for individual step execution."""
    return templates.TemplateResponse("testing.html", {"request": request})


# ============================================================================
# Hold Labeling Endpoints
# ============================================================================

@app.get("/workflow", response_class=HTMLResponse)
async def workflow_page(request: Request):
    """Render integrated workflow page."""
    return templates.TemplateResponse("workflow.html", {"request": request})


@app.get("/labeling", response_class=HTMLResponse)
async def labeling_page(request: Request):
    """Render hold labeling UI page."""
    return templates.TemplateResponse("labeling.html", {"request": request})


class CreateLabelingSessionRequest(BaseModel):
    """Request to create a new labeling session."""
    
    name: str = Field(..., description="Session name")
    frame_dir: str = Field(..., description="Directory containing frames")
    sam_checkpoint: str | None = Field(default=None, description="Path to SAM checkpoint")
    use_sam: bool = Field(default=True, description="Whether to run SAM segmentation")


@app.post("/api/labeling/sessions")
async def create_labeling_session(
    request: CreateLabelingSessionRequest,
    background_tasks: BackgroundTasks,
):
    """Create a new labeling session and optionally run SAM segmentation.
    
    Returns:
        Session ID and metadata
    """
    try:
        frame_dir = Path(request.frame_dir)
        if not frame_dir.exists():
            raise HTTPException(status_code=404, detail=f"Frame directory not found: {frame_dir}")
        
        # Create session
        session = labeling_manager.create_session(
            name=request.name,
            frame_dir=frame_dir,
            sam_checkpoint=request.sam_checkpoint,
        )
        
        print(f"Created labeling session: {session.id} (frames: {frame_dir})")
        
        # Run SAM segmentation in background if requested
        if request.use_sam:
            print(f"Starting SAM background task for session {session.id}...")
            background_tasks.add_task(
                run_sam_segmentation,
                session_id=session.id,
                frame_dir=frame_dir,
                sam_checkpoint=request.sam_checkpoint,
            )
            print(f"SAM background task queued successfully")
        else:
            print(f"Warning: SAM disabled for session {session.id}")
        
        return {
            "session_id": session.id,
            "name": session.name,
            "frame_count": len(list(frame_dir.glob("*.jpg"))),
            "status": session.status.value,
        }
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception("Failed to create labeling session")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


async def run_sam_segmentation(
    session_id: str,
    frame_dir: Path,
    sam_checkpoint: str | None,
):
    """Run SAM segmentation on all frames in the session (background task)."""
    print(f"\n{'='*60}")
    print(f"🚀 SAM BACKGROUND TASK STARTED")
    print(f"Session ID: {session_id}")
    print(f"Frame directory: {frame_dir}")
    print(f"SAM checkpoint: {sam_checkpoint}")
    print(f"{'='*60}\n")
    
    session = labeling_manager.get_session(session_id)
    if not session:
        LOGGER.error("Session not found: %s", session_id)
        return
    
    try:
        # Use default checkpoint if none provided
        if not sam_checkpoint:
            default_checkpoint = ROOT_DIR / "models" / "sam_vit_b_01ec64.pth"
            print(f"Looking for default checkpoint: {default_checkpoint}")
            if default_checkpoint.exists():
                sam_checkpoint = str(default_checkpoint)
                print(f"Using default SAM checkpoint: {sam_checkpoint}")
            else:
                print(f"No SAM checkpoint found at: {default_checkpoint}")
                print(f"Warning: Please download: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P models/")
                return
        
        # Initialize SAM service
        print(f"Initializing SAM service with checkpoint: {sam_checkpoint}")
        sam_service = SamService(
            sam_checkpoint=sam_checkpoint,
            device="cpu",  # Use CPU by default
            cache_dir=session.get_session_dir(LABELING_SESSIONS_DIR) / "sam_cache",
        )
        print(f"Calling sam_service.initialize()...")
        sam_service.initialize(Path(sam_checkpoint))
        print(f"SAM service initialized successfully!")
        
        # Get all image files
        all_images = sorted(frame_dir.glob("*.jpg"))
        
        # Demo mode: Process only first frame for quick testing
        demo_frame_indices = [0]
        
        print(f"Running SAM on demo frames {demo_frame_indices} (out of {len(all_images)} total) for session {session_id}")
        
        # Process specific frames
        for frame_index in demo_frame_indices:
            if frame_index >= len(all_images):
                print(f"  Warning: Skipping frame {frame_index} (out of range)")
                continue
            
            img_path = all_images[frame_index]
            print(f"  Processing frame {frame_index}: {img_path.name}")
            segments = sam_service.segment_frame(img_path, use_cache=True)
            segment_dicts = [seg.to_dict() for seg in segments]
            
            # Update segments for existing frame
            session.update_frame_segments(frame_index, segment_dicts)
            print(f"     Found {len(segments)} segments, updated frame index {frame_index}")
        
        session.status = SessionStatus.IN_PROGRESS
        labeling_manager.update_session(session)
        
        sam_service.close()
        
        print(f"SAM segmentation complete for session {session_id}!")
        
    except Exception as exc:
        print(f"SAM segmentation FAILED for session {session_id}")
        print(f"Error: {exc}")
        import traceback
        traceback.print_exc()
        session.status = SessionStatus.FAILED
        labeling_manager.update_session(session)


@app.get("/api/labeling/sessions/{session_id}")
async def get_labeling_session(session_id: str):
    """Get labeling session details."""
    session = labeling_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.id,
        "name": session.name,
        "status": session.status.value,
        "frames": [str(f) for f in session.frames],
        "progress": session.get_progress(),
        "created_at": session.created_at,
    }


@app.get("/api/labeling/sessions/{session_id}/frames/{frame_index}")
async def get_frame_data(session_id: str, frame_index: int):
    """Get frame data with segments."""
    session = labeling_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if frame_index < 0 or frame_index >= len(session.frames):
        raise HTTPException(status_code=404, detail="Frame index out of range")
    
    frame_labels = session.get_frame_labels(frame_index)
    if not frame_labels:
        raise HTTPException(status_code=404, detail="Frame labels not found")
    
    frame_path = session.frames[frame_index]
    
    # Convert segments dict to list for frontend
    segments = []
    for seg_id, seg_data in frame_labels.segments.items():
        segments.append({
            "segment_id": seg_id,
            **seg_data,
        })
    
    return {
        "frame_index": frame_index,
        "frame_path": str(frame_path),
        "image_url": f"/repo/{frame_path.relative_to(ROOT_DIR)}",
        "segments": segments,
    }


class UpdateLabelRequest(BaseModel):
    """Request to update a segment label."""
    
    frame_index: int
    segment_id: str
    hold_type: str | None
    is_hold: bool


@app.post("/api/labeling/sessions/{session_id}/labels")
async def update_segment_label(session_id: str, request: UpdateLabelRequest):
    """Update label for a specific segment."""
    session = labeling_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session.update_segment_label(
            frame_index=request.frame_index,
            segment_id=request.segment_id,
            hold_type=request.hold_type,
            is_hold=request.is_hold,
        )
        
        labeling_manager.update_session(session)
        
        return {
            "status": "ok",
            "progress": session.get_progress(),
        }
        
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/labeling/sessions/{session_id}/save")
async def save_session_labels(session_id: str):
    """Manually save session labels (auto-saved on update, but allows explicit save)."""
    session = labeling_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    labeling_manager.update_session(session)
    
    return {
        "status": "saved",
        "labeled_segments": session.labeled_segments,
    }


@app.post("/api/labeling/sessions/{session_id}/export")
async def export_to_yolo_dataset(session_id: str):
    """Export labeled segments to YOLO training dataset."""
    session = labeling_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get labeled segments
    labeled_segments = session.get_labeled_segments()
    if not labeled_segments:
        raise HTTPException(status_code=400, detail="No labeled segments to export")
    
    # Group by frame
    from collections import defaultdict
    frames_dict = defaultdict(list)
    for seg in labeled_segments:
        frames_dict[seg["frame_path"]].append(seg)
    
    # Export to YOLO format
    from pose_ai.segmentation.sam_annotator import LabeledSegment, segment_to_yolo_format
    from pose_ai.service.sam_service import HOLD_TYPE_TO_CLASS_ID
    import cv2
    
    dataset_dir = ROOT_DIR / "data" / "holds_training"
    train_split = 0.7
    val_split = 0.15
    # test_split = 0.15 (remaining)
    
    exported_count = 0
    frame_paths = list(frames_dict.keys())
    
    for idx, frame_path in enumerate(frame_paths):
        # Determine split
        if idx < len(frame_paths) * train_split:
            split = "train"
        elif idx < len(frame_paths) * (train_split + val_split):
            split = "val"
        else:
            split = "test"
        
        # Load image to get dimensions
        img = cv2.imread(frame_path)
        if img is None:
            LOGGER.warning("Failed to load image: %s", frame_path)
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Create output directories
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy image
        dst_image = images_dir / Path(frame_path).name
        shutil.copy(frame_path, dst_image)
        
        # Generate YOLO labels
        label_lines = []
        for seg_data in frames_dict[frame_path]:
            hold_type = seg_data["hold_type"]
            if hold_type not in HOLD_TYPE_TO_CLASS_ID:
                continue
            
            class_id = HOLD_TYPE_TO_CLASS_ID[hold_type]
            x1, y1, x2, y2 = seg_data["bbox"]
            
            # Convert to YOLO format (normalized)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Save labels
        if label_lines:
            label_file = labels_dir / f"{Path(frame_path).stem}.txt"
            label_file.write_text("\n".join(label_lines) + "\n")
            exported_count += 1
    
    # Mark session as exported
    session.mark_exported()
    labeling_manager.update_session(session)
    
    LOGGER.info("Exported %d frames to YOLO dataset from session %s", exported_count, session_id)
    
    return {
        "status": "exported",
        "exported_count": exported_count,
        "dataset_dir": str(dataset_dir),
        "splits": {
            "train": int(exported_count * train_split),
            "val": int(exported_count * val_split),
            "test": exported_count - int(exported_count * train_split) - int(exported_count * val_split),
        },
    }


@app.get("/api/labeling/sessions")
async def list_labeling_sessions():
    """List all labeling sessions."""
    sessions = labeling_manager.list_sessions()
    return {
        "sessions": [
            {
                "id": s.id,
                "name": s.name,
                "status": s.status.value,
                "frame_count": len(s.frames),
                "labeled_segments": s.labeled_segments,
                "created_at": s.created_at,
            }
            for s in sessions
        ],
        "stats": labeling_manager.get_stats(),
    }


# ============================================================================
# YOLO Training Endpoints
# ============================================================================

class YoloTrainRequest(BaseModel):
    """Request to start YOLO training."""
    
    dataset_yaml: str = Field(..., description="Path to dataset.yaml")
    model: str = Field("yolov8n.pt", description="YOLO model")
    epochs: int = Field(100, ge=1, le=1000)
    batch: int = Field(16, ge=1, le=64)
    imgsz: int = Field(640, ge=320, le=1280)
    upload_to_gcs: bool = Field(False, description="Upload trained model to GCS")


@app.post("/api/yolo/train")
async def start_yolo_training(
    request: YoloTrainRequest,
    background_tasks: BackgroundTasks,
):
    """Start YOLO hold detection training job.
    
    Returns:
        Job ID and metadata
    """
    dataset_yaml = Path(request.dataset_yaml)
    if not dataset_yaml.exists():
        raise HTTPException(status_code=404, detail=f"Dataset config not found: {dataset_yaml}")
    
    # Create training job
    job = yolo_train_manager.create_job(
        dataset_yaml=dataset_yaml,
        model=request.model,
        epochs=request.epochs,
        batch=request.batch,
        imgsz=request.imgsz,
    )
    
    # Start training in background
    background_tasks.add_task(
        execute_yolo_training,
        job=job,
        upload_to_gcs=request.upload_to_gcs,
    )
    
    return {
        "job_id": job.id,
        "status": job.status.value,
        "dataset": str(dataset_yaml),
        "epochs": request.epochs,
    }


@app.get("/api/yolo/train/jobs")
async def list_yolo_training_jobs():
    """List all YOLO training jobs."""
    jobs = yolo_train_manager.list_jobs()
    return {
        "jobs": [job.to_dict() for job in jobs],
        "active_jobs": len(yolo_train_manager.get_active_jobs()),
    }


@app.get("/api/yolo/train/jobs/{job_id}")
async def get_yolo_training_job(job_id: str):
    """Get YOLO training job details."""
    job = yolo_train_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job.to_dict()


@app.get("/api/yolo/train/jobs/{job_id}/logs")
async def get_yolo_training_logs(job_id: str):
    """Get training job logs."""
    job = yolo_train_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "logs": job.get_logs(),
    }


@app.post("/api/yolo/train/jobs/{job_id}/upload")
async def upload_trained_model_to_gcs(job_id: str):
    """Upload a completed training job's model to GCS.
    
    This endpoint allows manual upload after training is complete.
    """
    job = yolo_train_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed")
    
    if not job.best_model_path or not job.best_model_path.exists():
        raise HTTPException(status_code=404, detail="Trained model not found")
    
    if job.gcs_uri:
        return {
            "status": "already_uploaded",
            "gcs_uri": job.gcs_uri,
        }
    
    try:
        job.log("Uploading model to GCS...")
        gcs_uri = GCS_MANAGER.upload_model(job.best_model_path, job_id=job.id)
        job.gcs_uri = gcs_uri
        job.log(f"Model uploaded to: {gcs_uri}")
        
        return {
            "status": "uploaded",
            "gcs_uri": gcs_uri,
        }
        
    except Exception as exc:
        LOGGER.error("Failed to upload model to GCS: %s", exc)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")


# ============================================================================
# Middleware
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Emit simple structured logs for every incoming HTTP request."""
    LOGGER.info(">> %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("!! %s %s failed: %s", request.method, request.url.path, exc)
        raise
    LOGGER.info("<< %s %s %s", request.method, request.url.path, response.status_code)
    return response
