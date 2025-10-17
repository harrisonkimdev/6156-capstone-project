"""FastAPI application exposing a minimal web UI for the BetaMove pipeline."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import shutil

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from webapp.jobs import JobManager
from webapp.pipeline_runner import execute_job
from webapp.training_jobs import TrainManager

app = FastAPI(title="BetaMove Pipeline UI")

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/repo", StaticFiles(directory=str(ROOT_DIR)), name="repo")

job_manager = JobManager()
UPLOAD_ROOT = ROOT_DIR / "data" / "uploads"
train_manager = TrainManager()


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_uploaded_file(video: UploadFile) -> Path:
    if video is None:
        raise HTTPException(status_code=400, detail="No file provided")
    target_dir = _ensure_directory(UPLOAD_ROOT / uuid4().hex)
    file_path = target_dir / video.filename
    with file_path.open("wb") as destination:
        video.file.seek(0)
        shutil.copyfileobj(video.file, destination)
    video.file.close()
    return target_dir


class PipelineRequest(BaseModel):
    video_dir: str = Field(..., description="Directory containing source videos")
    output_dir: str = Field("data/frames", description="Directory to write pipeline artifacts")
    interval: float = Field(1.0, gt=0, description="Frame extraction interval in seconds")
    skip_visuals: bool = Field(False, description="Disable OpenCV visual overlays")


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
    job = job_manager.create_job(
        video_dir=payload.video_dir,
        output_dir=payload.output_dir,
        interval=payload.interval,
        skip_visuals=payload.skip_visuals,
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
    saved_dir = _save_uploaded_file(video)
    return {"video_dir": str(saved_dir)}


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
