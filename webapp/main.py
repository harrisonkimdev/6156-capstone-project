"""FastAPI application exposing a minimal web UI for the BetaMove pipeline."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from pose_ai.cloud.gcs import get_gcs_manager
from webapp.jobs import JobManager
from webapp.pipeline_runner import execute_job
from webapp.training_jobs import TrainManager
from pose_ai.recommendation.efficiency import suggest_next_actions

app = FastAPI(title="BetaMove Pipeline UI")

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/repo", StaticFiles(directory=str(ROOT_DIR)), name="repo")

job_manager = JobManager()
UPLOAD_ROOT = ROOT_DIR / "data" / "uploads"
train_manager = TrainManager()
LOGGER = logging.getLogger(__name__)
GCS_MANAGER = get_gcs_manager()


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_uploaded_file(video: UploadFile) -> tuple[Path, str | None]:
    if video is None:
        raise HTTPException(status_code=400, detail="No file provided")
    target_dir = _ensure_directory(UPLOAD_ROOT / uuid4().hex)
    file_path = target_dir / video.filename
    with file_path.open("wb") as destination:
        video.file.seek(0)
        shutil.copyfileobj(video.file, destination)
    video.file.close()
    gcs_uri: str | None = None
    if GCS_MANAGER is not None:
        try:
            gcs_uri = GCS_MANAGER.upload_raw_video(
                file_path,
                upload_id=target_dir.name,
                metadata={"filename": video.filename},
            )
        except Exception as exc:  # pragma: no cover - depends on GCS credentials
            LOGGER.warning("Failed to upload raw video to GCS: %s", exc)
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


class YoloOptions(BaseModel):
    enabled: bool = Field(True, description="Toggle YOLO still selection")
    model_name: str = Field("yolov8n.pt", description="Model weights passed to ultralytics")
    min_confidence: float = Field(0.35, ge=0.0, le=1.0)
    required_labels: list[str] = Field(default_factory=_default_required_labels)
    target_labels: list[str] = Field(default_factory=_default_target_labels)
    max_frames: int | None = Field(None, gt=0)
    imgsz: int = Field(640, ge=128, le=1536)
    device: str | None = Field(None, description="Optional torch device string")


class PipelineRequest(BaseModel):
    video_dir: str = Field(..., description="Directory containing source videos")
    output_dir: str = Field("data/frames", description="Directory to write pipeline artifacts")
    interval: float = Field(1.0, gt=0, description="Frame extraction interval in seconds")
    skip_visuals: bool = Field(False, description="Disable OpenCV visual overlays")
    source_uri: str | None = Field(None, description="Optional GCS URI referencing the uploaded video")
    metadata: MediaMetadata | None = Field(None, description="Optional capture metadata")
    yolo: YoloOptions = Field(default_factory=YoloOptions)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    jobs = [job.as_dict() for job in job_manager.list_jobs()]
    return templates.TemplateResponse("index.html", {"request": request, "jobs": jobs})


@app.get("/training", response_class=HTMLResponse)
async def training_page(request: Request) -> HTMLResponse:
    jobs = [job.as_dict() for job in train_manager.list()]
    return templates.TemplateResponse("training.html", {"request": request, "jobs": jobs})


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
    metadata_payload = metadata_payload or None
    job = job_manager.create_job(
        video_dir=payload.video_dir,
        output_dir=payload.output_dir,
        interval=payload.interval,
        skip_visuals=payload.skip_visuals,
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
    payload = {"video_dir": str(saved_dir)}
    if gcs_uri:
        payload["gcs_uri"] = gcs_uri
    return payload


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
