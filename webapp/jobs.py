"""In-memory job tracking for web-triggered pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Dict, Iterable, List, Optional
from uuid import uuid4


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class PipelineJob:
    id: str
    video_dir: str
    output_dir: str
    interval: float
    skip_visuals: bool
    metadata: dict[str, object] | None = None
    yolo_options: dict[str, object] | None = None
    artifacts: Dict[str, List[str]] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=_utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    manifests: List[str] = field(default_factory=list)
    visualizations: List[dict[str, str]] = field(default_factory=list)
    pose_samples: List[dict[str, object]] = field(default_factory=list)
    error: Optional[str] = None
    _log: List[str] = field(default_factory=list, repr=False)
    _lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def _timestamp(self) -> str:
        return _utcnow().strftime("%H:%M:%S")

    def log(self, message: str) -> None:
        with self._lock:
            self._log.append(f"[{self._timestamp()}] {message}")

    def start(self) -> None:
        with self._lock:
            self.status = JobStatus.RUNNING
            self.started_at = _utcnow()

    def complete(
        self,
        manifests: Iterable[str],
        visualizations: Iterable[dict[str, str]],
        pose_samples: Iterable[dict[str, object]],
    ) -> None:
        with self._lock:
            self.status = JobStatus.COMPLETED
            self.finished_at = _utcnow()
            self.manifests = list(manifests)
            self.visualizations = list(visualizations)
            self.pose_samples = list(pose_samples)

    def fail(self, exc: Exception) -> None:
        with self._lock:
            self.status = JobStatus.FAILED
            self.finished_at = _utcnow()
            self.error = str(exc)

    def as_dict(self) -> Dict[str, object]:
        with self._lock:
            return {
                "id": self.id,
                "video_dir": self.video_dir,
                "output_dir": self.output_dir,
                "interval": self.interval,
                "skip_visuals": self.skip_visuals,
                "metadata": self.metadata or {},
                "yolo_options": self.yolo_options or {},
                "artifacts": self.artifacts,
                "status": self.status.value,
                "created_at": self.created_at.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "finished_at": self.finished_at.isoformat() if self.finished_at else None,
                "manifests": self.manifests,
                "visualizations": self.visualizations,
                "pose_samples": self.pose_samples,
                "error": self.error,
            }

    def log_lines(self) -> List[str]:
        with self._lock:
            return list(self._log)

    def clear(self, include_logs: bool = True, include_visuals: bool = True, include_samples: bool = True) -> None:
        """Clear in-memory artifacts for this job (non-destructive; does not delete files)."""
        with self._lock:
            if include_logs:
                self._log = []
            if include_visuals:
                self.visualizations = []
            if include_samples:
                self.pose_samples = []

    def add_artifact(self, kind: str, uri: str) -> None:
        with self._lock:
            entries = self.artifacts.setdefault(kind, [])
            entries.append(uri)


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, PipelineJob] = {}
        self._lock = RLock()

    def create_job(
        self,
        *,
        video_dir: str,
        output_dir: str,
        interval: float,
        skip_visuals: bool,
        metadata: dict[str, object] | None = None,
        yolo_options: dict[str, object] | None = None,
    ) -> PipelineJob:
        job = PipelineJob(
            id=uuid4().hex,
            video_dir=video_dir,
            output_dir=output_dir,
            interval=interval,
            skip_visuals=skip_visuals,
            metadata=metadata,
            yolo_options=yolo_options,
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> Optional[PipelineJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[PipelineJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)


__all__ = ["JobManager", "JobStatus", "PipelineJob"]
