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
    gcs_uri: Optional[str] = None  # Model weights URI (if uploaded)
    metadata_uri: Optional[str] = None  # Metadata URI (always uploaded)
    training_data_uri: Optional[str] = None  # Training data URI (if uploaded)
    drive_model_id: Optional[str] = None  # Google Drive model ID (if uploaded)
    drive_metadata_id: Optional[str] = None  # Google Drive metadata ID (if uploaded)
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
        metadata_uri: Optional[str] = None,
        training_data_uri: Optional[str] = None,
        drive_model_id: Optional[str] = None,
        drive_metadata_id: Optional[str] = None,
    ) -> None:
        """Mark job as completed."""
        self.status = YoloTrainStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()
        self.metrics = metrics
        self.best_model_path = best_model_path
        self.gcs_uri = gcs_uri
        self.metadata_uri = metadata_uri
        self.training_data_uri = training_data_uri
        self.drive_model_id = drive_model_id
        self.drive_metadata_id = drive_metadata_id
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
            "metadata_uri": self.metadata_uri,
            "training_data_uri": self.training_data_uri,
            "drive_model_id": self.drive_model_id,
            "drive_metadata_id": self.drive_metadata_id,
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


def execute_yolo_training(
    job: YoloTrainJob,
    upload_to_gcs: bool = False,
    upload_training_data: bool = False,
) -> None:
    """Execute YOLO training job (to be run in background task).
    
    Args:
        job: YoloTrainJob instance.
        upload_to_gcs: Whether to upload trained model weights to storage.
        upload_training_data: Whether to upload training data to storage.
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
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
        
        # Prepare metadata
        metadata = {
            "job_id": job.id,
            "model_type": "yolo",
            "base_model": job.model,
            "hyperparameters": {
                "epochs": job.epochs,
                "batch": job.batch,
                "imgsz": job.imgsz,
            },
            "metrics": metrics,
            "dataset_yaml": str(job.dataset_yaml),
            "best_model_path": str(best_model_path),
            "created_at": job.created_at,
            "completed_at": datetime.now().isoformat(),
        }
        
        # Upload to storage using unified storage manager
        metadata_uri = None
        gcs_uri = None
        training_data_uri = None
        drive_model_id = None
        drive_metadata_id = None
        
        try:
            from pose_ai.cloud.storage import get_storage_manager
            storage = get_storage_manager()
            
            # Always upload metadata
            job.log("Uploading model metadata to storage...")
            metadata_result = storage.upload_model_metadata(metadata, job_id=job.id)
            metadata_uri = metadata_result.gcs_uri
            drive_metadata_id = metadata_result.drive_id
            if metadata_uri:
                job.log(f"Metadata uploaded to GCS: {metadata_uri}")
            if drive_metadata_id:
                job.log(f"Metadata uploaded to Google Drive: {drive_metadata_id}")
            
            # Optionally upload model weights
            if upload_to_gcs:
                job.log("Uploading model weights to storage...")
                model_result = storage.upload_model(best_model_path, job_id=job.id)
                gcs_uri = model_result.gcs_uri
                drive_model_id = model_result.drive_id
                if gcs_uri:
                    job.log(f"Model uploaded to GCS: {gcs_uri}")
                if drive_model_id:
                    job.log(f"Model uploaded to Google Drive: {drive_model_id}")
            
            # Optionally upload training data
            if upload_training_data and job.dataset_yaml.parent.exists():
                job.log("Uploading training data to storage...")
                data_result = storage.upload_training_data(
                    job.dataset_yaml.parent, job_id=job.id, data_type="dataset"
                )
                training_data_uri = data_result.gcs_uri
                if training_data_uri:
                    job.log(f"Training data uploaded to GCS: {training_data_uri}")
                if data_result.drive_id:
                    job.log(f"Training data uploaded to Google Drive: {data_result.drive_id}")
            
        except Exception as exc:
            job.log(f"Storage upload failed: {exc}")
            LOGGER.error("Storage upload failed: %s", exc)
        
        job.complete(
            metrics=metrics,
            best_model_path=best_model_path,
            gcs_uri=gcs_uri,
            metadata_uri=metadata_uri,
            training_data_uri=training_data_uri,
            drive_model_id=drive_model_id,
            drive_metadata_id=drive_metadata_id,
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
