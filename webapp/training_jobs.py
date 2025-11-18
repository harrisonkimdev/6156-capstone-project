from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Dict, List, Optional
from uuid import uuid4


class TrainStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TrainJob:
    id: str
    features_path: str
    params: Dict[str, object]
    status: TrainStatus = TrainStatus.PENDING
    created_at: datetime = field(default_factory=_utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    metrics: Dict[str, object] = field(default_factory=dict)
    model_path: Optional[str] = None
    model_uri: Optional[str] = None
    error: Optional[str] = None
    _log: List[str] = field(default_factory=list, repr=False)
    _lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def start(self) -> None:
        with self._lock:
            self.status = TrainStatus.RUNNING
            self.started_at = _utcnow()

    def log(self, message: str) -> None:
        with self._lock:
            self._log.append(message)

    def complete(self, *, metrics: Dict[str, object], model_path: str, model_uri: Optional[str] = None) -> None:
        with self._lock:
            self.status = TrainStatus.COMPLETED
            self.finished_at = _utcnow()
            self.metrics = metrics
            self.model_path = model_path
            self.model_uri = model_uri

    def fail(self, exc: Exception) -> None:
        with self._lock:
            self.status = TrainStatus.FAILED
            self.finished_at = _utcnow()
            self.error = str(exc)

    def as_dict(self) -> Dict[str, object]:
        with self._lock:
            return {
                "id": self.id,
                "features_path": self.features_path,
                "params": self.params,
                "status": self.status.value,
                "created_at": self.created_at.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "finished_at": self.finished_at.isoformat() if self.finished_at else None,
                "metrics": self.metrics,
                "model_path": self.model_path,
                "model_uri": self.model_uri,
                "error": self.error,
                "logs": list(self._log),
            }


class TrainManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, TrainJob] = {}
        self._lock = RLock()

    def create(self, *, features_path: str, params: Dict[str, object]) -> TrainJob:
        job = TrainJob(id=uuid4().hex, features_path=features_path, params=params)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[TrainJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[TrainJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
