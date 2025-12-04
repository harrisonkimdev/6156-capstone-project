"""YOLO hold detection training jobs management."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

LOGGER = logging.getLogger(__name__)


class YoloTrainStatus(str, Enum):
    """Status of a YOLO training job."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class YoloTrainJob:
    """YOLO training job metadata."""
    
    id: str
    dataset_yaml: Path
    model: str  # e.g., "yolov8n.pt"
    epochs: int
    batch: int
    imgsz: int
    status: YoloTrainStatus = YoloTrainStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    best_model_path: Optional[Path] = None
    gcs_uri: Optional[str] = None
    metrics: Dict = field(default_factory=dict)
    _log: List[str] = field(default_factory=list)
    
    def start(self) -> None:
        """Mark job as started."""
        self.status = YoloTrainStatus.RUNNING
        self.started_at = datetime.now().isoformat()
        self.log("Training job started")
    
    def complete(
        self,
        metrics: Dict,
        best_model_path: Path,
        gcs_uri: Optional[str] = None,
    ) -> None:
        """Mark job as completed."""
        self.status = YoloTrainStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()
        self.metrics = metrics
        self.best_model_path = best_model_path
        self.gcs_uri = gcs_uri
        self.log(f"Training completed: {metrics}")
    
    def fail(self, exc: Exception) -> None:
        """Mark job as failed."""
        self.status = YoloTrainStatus.FAILED
        self.completed_at = datetime.now().isoformat()
        self.log(f"Training failed: {exc}")
    
    def log(self, message: str) -> None:
        """Add log message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log.append(f"[{timestamp}] {message}")
        LOGGER.info("[Job %s] %s", self.id, message)
    
    def get_logs(self) -> List[str]:
        """Get all log messages."""
        return self._log.copy()
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "id": self.id,
            "dataset_yaml": str(self.dataset_yaml),
            "model": self.model,
            "epochs": self.epochs,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "best_model_path": str(self.best_model_path) if self.best_model_path else None,
            "gcs_uri": self.gcs_uri,
            "metrics": self.metrics,
        }


class YoloTrainManager:
    """Manager for YOLO training jobs."""
    
    def __init__(self):
        """Initialize manager."""
        self._jobs: Dict[str, YoloTrainJob] = {}
        LOGGER.info("YoloTrainManager initialized")
    
    def create_job(
        self,
        dataset_yaml: Path,
        model: str = "yolov8n.pt",
        epochs: int = 100,
        batch: int = 16,
        imgsz: int = 640,
    ) -> YoloTrainJob:
        """Create a new training job.
        
        Args:
            dataset_yaml: Path to YOLO dataset configuration
            model: YOLO model to use (e.g., "yolov8n.pt")
            epochs: Number of training epochs
            batch: Batch size
            imgsz: Image size
            
        Returns:
            YoloTrainJob instance
        """
        job_id = uuid4().hex[:12]
        job = YoloTrainJob(
            id=job_id,
            dataset_yaml=Path(dataset_yaml),
            model=model,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
        )
        
        self._jobs[job_id] = job
        LOGGER.info("Created YOLO training job: %s", job_id)
        return job
    
    def get_job(self, job_id: str) -> Optional[YoloTrainJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)
    
    def list_jobs(self) -> List[YoloTrainJob]:
        """List all jobs."""
        return list(self._jobs.values())
    
    def get_active_jobs(self) -> List[YoloTrainJob]:
        """Get jobs that are running or pending."""
        return [
            job for job in self._jobs.values()
            if job.status in (YoloTrainStatus.PENDING, YoloTrainStatus.RUNNING)
        ]


def execute_yolo_training(job: YoloTrainJob, upload_to_gcs: bool = False) -> None:
    """Execute YOLO training job (to be run in background task).
    
    Args:
        job: YoloTrainJob instance
        upload_to_gcs: Whether to upload trained model to GCS
    """
    job.start()
    
    try:
        # Use ultralytics YOLO API
        from ultralytics import YOLO
        
        job.log(f"Loading model: {job.model}")
        model = YOLO(job.model)
        
        job.log(f"Starting training: {job.epochs} epochs, batch={job.batch}")
        
        # Train
        results = model.train(
            data=str(job.dataset_yaml),
            epochs=job.epochs,
            batch=job.batch,
            imgsz=job.imgsz,
            project="runs/hold_type",
            name="train",
            verbose=True,
        )
        
        # Get best model path
        best_model_path = Path("runs/hold_type/train/weights/best.pt")
        if not best_model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {best_model_path}")
        
        job.log(f"Training completed, best model: {best_model_path}")
        
        # Extract metrics
        metrics = {}
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
        
        # Upload to GCS if requested
        gcs_uri = None
        if upload_to_gcs:
            try:
                from pose_ai.cloud.gcs import get_gcs_manager
                gcs_manager = get_gcs_manager()
                
                job.log("Uploading model to GCS...")
                gcs_uri = gcs_manager.upload_model(best_model_path, job_id=job.id)
                job.log(f"Model uploaded to: {gcs_uri}")
                
            except Exception as exc:
                job.log(f"Failed to upload to GCS: {exc}")
                LOGGER.error("GCS upload failed: %s", exc)
        
        job.complete(
            metrics=metrics,
            best_model_path=best_model_path,
            gcs_uri=gcs_uri,
        )
        
    except Exception as exc:
        LOGGER.error("YOLO training failed for job %s: %s", job.id, exc)
        job.fail(exc)


__all__ = [
    "YoloTrainJob",
    "YoloTrainManager",
    "YoloTrainStatus",
    "execute_yolo_training",
]
