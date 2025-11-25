"""Tests for the CloudStorageManager GCS helper."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pose_ai.cloud.gcs import CloudStorageManager, GCSConfig, build_config_from_env, get_gcs_manager


class DummyBlob:
    def __init__(self, name: str) -> None:
        self.name = name
        self.upload_calls: list[dict[str, object]] = []

    def upload_from_filename(self, filename: str, *, content_type: str | None = None) -> None:
        self.upload_calls.append({"filename": filename, "content_type": content_type})

    def delete(self) -> None:
        self.upload_calls.append({"deleted": True})


class DummyBucket:
    def __init__(self, name: str) -> None:
        self.name = name
        self.blobs: dict[str, DummyBlob] = {}

    def blob(self, object_name: str) -> DummyBlob:
        blob = self.blobs.setdefault(object_name, DummyBlob(object_name))
        return blob


class DummyClient:
    def __init__(self) -> None:
        self.buckets: dict[str, DummyBucket] = {}

    def bucket(self, name: str) -> DummyBucket:
        return self.buckets.setdefault(name, DummyBucket(name))

    def list_blobs(self, bucket_name: str, prefix: str):
        bucket = self.buckets.get(bucket_name, DummyBucket(bucket_name))
        for blob in bucket.blobs.values():
            if blob.name.startswith(prefix):
                blob.time_created = None  # pragma: no cover - retention not tested here
                yield blob


def test_build_config_from_env(monkeypatch):
    monkeypatch.setenv("GCS_PROJECT", "test-project")
    monkeypatch.setenv("GCS_VIDEO_BUCKET", "raw-videos")
    monkeypatch.setenv("GCS_FRAME_BUCKET", "frame-bucket")
    monkeypatch.setenv("GCS_MODEL_BUCKET", "model-bucket")
    monkeypatch.setenv("GCS_VIDEO_PREFIX", "custom/raw")
    config = build_config_from_env()
    assert config.project == "test-project"
    assert config.video_bucket == "raw-videos"
    assert config.frame_bucket == "frame-bucket"
    assert config.model_bucket == "model-bucket"
    assert config.raw_prefix == "custom/raw"


def test_upload_file_uses_dummy_client(tmp_path):
    config = GCSConfig(video_bucket="videos")
    client = DummyClient()
    manager = CloudStorageManager(config, client=client)
    local_file = tmp_path / "sample.mp4"
    local_file.write_bytes(b"data")
    uri = manager.upload_file(local_file, bucket_name="videos", object_name="raw/video.mp4", content_type="video/mp4")
    assert uri == "gs://videos/raw/video.mp4"
    bucket = client.buckets["videos"]
    blob = bucket.blobs["raw/video.mp4"]
    assert blob.upload_calls[0]["filename"].endswith("sample.mp4")


def test_get_gcs_manager_raises_when_unconfigured(monkeypatch):
    """Test that get_gcs_manager raises ValueError when required env vars are missing."""
    for key in list(os.environ):
        if key.startswith("GCS_") or key.startswith("GOOGLE_CLOUD") or key == "GCLOUD_PROJECT":
            monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="GCS is required but missing environment variables"):
        get_gcs_manager(force_refresh=True)


def test_build_config_from_env_raises_when_missing_required(monkeypatch):
    """Test that build_config_from_env raises ValueError when required vars are missing."""
    # Clear all GCS-related env vars
    for key in list(os.environ):
        if key.startswith("GCS_") or key.startswith("GOOGLE_CLOUD") or key == "GCLOUD_PROJECT":
            monkeypatch.delenv(key, raising=False)
    
    with pytest.raises(ValueError, match="GCS is required but missing environment variables"):
        build_config_from_env()
    
    # Test with only project missing
    monkeypatch.setenv("GCS_VIDEO_BUCKET", "videos")
    monkeypatch.setenv("GCS_FRAME_BUCKET", "frames")
    monkeypatch.setenv("GCS_MODEL_BUCKET", "models")
    with pytest.raises(ValueError, match="GCS_PROJECT"):
        build_config_from_env()
