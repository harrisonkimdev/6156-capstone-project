"""Google Cloud Storage helpers for uploading pipeline artifacts."""

from __future__ import annotations

import logging
import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

LOGGER = logging.getLogger(__name__)

try:  # Optional dependency; defer failure until actually used.
    from google.cloud import storage  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - environment without google-cloud-storage.
    storage = None  # type: ignore[assignment]


@dataclass(slots=True)
class GCSConfig:
    """Runtime configuration for Cloud Storage usage."""

    project: str | None = None
    credentials_path: str | None = None
    video_bucket: str | None = None
    frame_bucket: str | None = None
    model_bucket: str | None = None
    raw_prefix: str = "videos/raw"
    frame_prefix: str = "videos/frames"
    model_prefix: str = "models"

    def any_bucket_configured(self) -> bool:
        return any([self.video_bucket, self.frame_bucket, self.model_bucket])


def build_config_from_env() -> GCSConfig:
    """Populate :class:`GCSConfig` from standard environment variables."""
    return GCSConfig(
        project=os.getenv("GCS_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT"),
        credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        video_bucket=os.getenv("GCS_VIDEO_BUCKET"),
        frame_bucket=os.getenv("GCS_FRAME_BUCKET"),
        model_bucket=os.getenv("GCS_MODEL_BUCKET"),
        raw_prefix=os.getenv("GCS_VIDEO_PREFIX", "videos/raw"),
        frame_prefix=os.getenv("GCS_FRAME_PREFIX", "videos/frames"),
        model_prefix=os.getenv("GCS_MODEL_PREFIX", "models"),
    )


class CloudStorageManager:
    """Encapsulates uploads, downloads, and retention operations for GCS."""

    def __init__(self, config: GCSConfig, *, client=None) -> None:
        self.config = config
        self._client = client

    # ------------------------------------------------------------------ #
    # Client helpers
    # ------------------------------------------------------------------ #
    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if storage is None:  # pragma: no cover - depends on optional install.
            raise ModuleNotFoundError(
                "google-cloud-storage is required. Install it via `pip install google-cloud-storage`."
            )
        if self.config.credentials_path:
            self._client = storage.Client.from_service_account_json(
                self.config.credentials_path,
                project=self.config.project,
            )
        else:
            self._client = storage.Client(project=self.config.project)
        return self._client

    def _bucket(self, name: str):
        client = self._ensure_client()
        bucket = client.bucket(name)
        return bucket

    # ------------------------------------------------------------------ #
    # Upload helpers
    # ------------------------------------------------------------------ #
    def upload_file(
        self,
        local_path: Path | str,
        *,
        bucket_name: str,
        object_name: str,
        content_type: str | None = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> str:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {path}")
        bucket = self._bucket(bucket_name)
        blob = bucket.blob(object_name)
        guessed_type = content_type or mimetypes.guess_type(path.name)[0]
        blob.metadata = metadata
        blob.upload_from_filename(str(path), content_type=guessed_type)
        uri = f"gs://{bucket_name}/{object_name}"
        LOGGER.debug("Uploaded %s -> %s", path, uri)
        return uri

    def upload_directory(
        self,
        directory: Path | str,
        *,
        bucket_name: str,
        prefix: str,
        include_hidden: bool = False,
    ) -> str:
        root = Path(directory)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {root}")
        for file_path in sorted(root.rglob("*")):
            if file_path.is_dir():
                continue
            if not include_hidden and any(part.startswith(".") for part in file_path.relative_to(root).parts):
                continue
            relative = file_path.relative_to(root).as_posix()
            object_name = f"{prefix.rstrip('/')}/{relative}"
            self.upload_file(file_path, bucket_name=bucket_name, object_name=object_name)
        return f"gs://{bucket_name}/{prefix.rstrip('/')}"

    def upload_raw_video(
        self,
        video_path: Path | str,
        *,
        upload_id: str,
        metadata: Optional[dict[str, str]] = None,
    ) -> str | None:
        bucket = self.config.video_bucket
        if not bucket:
            return None
        prefix = f"{self.config.raw_prefix.rstrip('/')}/{upload_id}"
        object_name = f"{prefix}/{Path(video_path).name}"
        return self.upload_file(video_path, bucket_name=bucket, object_name=object_name, metadata=metadata)

    def upload_frame_directory(self, frame_dir: Path | str, *, job_id: str) -> str | None:
        bucket = self.config.frame_bucket
        if not bucket:
            return None
        folder_name = Path(frame_dir).name
        prefix = f"{self.config.frame_prefix.rstrip('/')}/{job_id}/{folder_name}"
        return self.upload_directory(frame_dir, bucket_name=bucket, prefix=prefix)

    def upload_model(self, model_path: Path | str, *, job_id: str | None = None) -> str | None:
        bucket = self.config.model_bucket
        if not bucket:
            return None
        suffix = Path(model_path).name
        prefix = self.config.model_prefix.rstrip("/")
        if job_id:
            object_name = f"{prefix}/{job_id}/{suffix}"
        else:
            object_name = f"{prefix}/{suffix}"
        return self.upload_file(model_path, bucket_name=bucket, object_name=object_name)

    # ------------------------------------------------------------------ #
    # Retention helpers
    # ------------------------------------------------------------------ #
    def prune_objects(
        self,
        *,
        bucket_name: str,
        prefix: str,
        older_than_days: int,
        limit: Optional[int] = None,
    ) -> List[str]:
        if older_than_days <= 0:
            raise ValueError("older_than_days must be positive.")
        client = self._ensure_client()
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        deleted: list[str] = []
        iterator = client.list_blobs(bucket_name, prefix=prefix)
        for blob in iterator:
            created = getattr(blob, "time_created", None)
            if created is None or created >= cutoff:
                continue
            uri = f"gs://{bucket_name}/{blob.name}"
            blob.delete()
            deleted.append(uri)
            LOGGER.info("Deleted %s (created %s)", uri, created)
            if limit and len(deleted) >= limit:
                break
        return deleted


_CACHED_MANAGER: CloudStorageManager | None = None


def get_gcs_manager(force_refresh: bool = False) -> CloudStorageManager | None:
    """Return a cached :class:`CloudStorageManager` if any bucket is configured."""
    global _CACHED_MANAGER  # pylint: disable=global-statement
    if not force_refresh and _CACHED_MANAGER is not None:
        return _CACHED_MANAGER
    config = build_config_from_env()
    if not config.any_bucket_configured():
        _CACHED_MANAGER = None
        return None
    _CACHED_MANAGER = CloudStorageManager(config)
    return _CACHED_MANAGER


__all__ = [
    "CloudStorageManager",
    "GCSConfig",
    "build_config_from_env",
    "get_gcs_manager",
]
