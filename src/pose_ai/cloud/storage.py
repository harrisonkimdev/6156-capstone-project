"""Unified storage manager for GCS and Google Drive.

Provides a single interface to upload/download artifacts to either or both storage backends.
The backend selection is controlled by the STORAGE_BACKEND environment variable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .gcs import CloudStorageManager, get_gcs_manager
from .drive import GoogleDriveManager, get_drive_manager, is_drive_enabled

LOGGER = logging.getLogger(__name__)


class StorageBackend(str, Enum):
    """Storage backend selection."""
    
    GCS = "gcs"
    DRIVE = "drive"
    BOTH = "both"


@dataclass
class UploadResult:
    """Result of an upload operation."""
    
    gcs_uri: str | None = None
    drive_id: str | None = None
    
    def to_dict(self) -> Dict[str, str | None]:
        """Convert to dictionary."""
        return {
            "gcs_uri": self.gcs_uri,
            "drive_id": self.drive_id,
        }


def get_storage_backend() -> StorageBackend:
    """Get the configured storage backend from environment variables.
    
    Environment variable:
        STORAGE_BACKEND: One of "gcs", "drive", or "both" (default: "gcs").
    
    Returns:
        StorageBackend enum value.
    """
    backend = os.getenv("STORAGE_BACKEND", "gcs").lower()
    
    if backend == "drive":
        return StorageBackend.DRIVE
    elif backend == "both":
        return StorageBackend.BOTH
    else:
        return StorageBackend.GCS


class UnifiedStorageManager:
    """Unified storage manager that handles both GCS and Google Drive.
    
    Provides a single interface to upload/download artifacts to either or both
    storage backends based on configuration.
    """

    def __init__(
        self,
        gcs_manager: CloudStorageManager | None = None,
        drive_manager: GoogleDriveManager | None = None,
        backend: StorageBackend | None = None,
    ) -> None:
        """Initialize the unified storage manager.
        
        Args:
            gcs_manager: Optional pre-configured GCS manager.
            drive_manager: Optional pre-configured Google Drive manager.
            backend: Optional storage backend override.
        """
        self._gcs_manager = gcs_manager
        self._drive_manager = drive_manager
        self._backend = backend or get_storage_backend()

    @property
    def gcs(self) -> CloudStorageManager | None:
        """Get the GCS manager (lazy initialization)."""
        if self._gcs_manager is None and self._should_use_gcs():
            try:
                self._gcs_manager = get_gcs_manager()
            except ValueError as e:
                LOGGER.warning("GCS not available: %s", e)
        return self._gcs_manager

    @property
    def drive(self) -> GoogleDriveManager | None:
        """Get the Google Drive manager (lazy initialization)."""
        if self._drive_manager is None and self._should_use_drive():
            self._drive_manager = get_drive_manager()
        return self._drive_manager

    def _should_use_gcs(self) -> bool:
        """Check if GCS should be used."""
        return self._backend in (StorageBackend.GCS, StorageBackend.BOTH)

    def _should_use_drive(self) -> bool:
        """Check if Google Drive should be used."""
        return self._backend in (StorageBackend.DRIVE, StorageBackend.BOTH) and is_drive_enabled()

    # ------------------------------------------------------------------ #
    # Model operations
    # ------------------------------------------------------------------ #
    def upload_model(
        self,
        model_path: Path | str,
        *,
        job_id: str | None = None,
    ) -> UploadResult:
        """Upload a model file to configured storage backends.
        
        Args:
            model_path: Path to the model file.
            job_id: Optional job ID for organizing uploads.
            
        Returns:
            UploadResult with URIs/IDs for each backend.
        """
        result = UploadResult()
        
        if self._should_use_gcs() and self.gcs:
            try:
                result.gcs_uri = self.gcs.upload_model(model_path, job_id=job_id)
                LOGGER.info("Uploaded model to GCS: %s", result.gcs_uri)
            except Exception as e:
                LOGGER.error("Failed to upload model to GCS: %s", e)
        
        if self._should_use_drive() and self.drive:
            try:
                result.drive_id = self.drive.upload_model(model_path, job_id=job_id)
                LOGGER.info("Uploaded model to Google Drive: %s", result.drive_id)
            except Exception as e:
                LOGGER.error("Failed to upload model to Google Drive: %s", e)
        
        return result

    def upload_model_metadata(
        self,
        metadata: dict,
        *,
        job_id: str | None = None,
        filename: str = "model_metadata.json",
    ) -> UploadResult:
        """Upload model metadata to configured storage backends.
        
        Args:
            metadata: Dictionary containing model metadata.
            job_id: Optional job ID for organizing uploads.
            filename: Name of the metadata file.
            
        Returns:
            UploadResult with URIs/IDs for each backend.
        """
        result = UploadResult()
        
        if self._should_use_gcs() and self.gcs:
            try:
                result.gcs_uri = self.gcs.upload_model_metadata(metadata, job_id=job_id, filename=filename)
                LOGGER.info("Uploaded model metadata to GCS: %s", result.gcs_uri)
            except Exception as e:
                LOGGER.error("Failed to upload model metadata to GCS: %s", e)
        
        if self._should_use_drive() and self.drive:
            try:
                result.drive_id = self.drive.upload_model_metadata(metadata, job_id=job_id, filename=filename)
                LOGGER.info("Uploaded model metadata to Google Drive: %s", result.drive_id)
            except Exception as e:
                LOGGER.error("Failed to upload model metadata to Google Drive: %s", e)
        
        return result

    # ------------------------------------------------------------------ #
    # Training data operations
    # ------------------------------------------------------------------ #
    def upload_training_data(
        self,
        data_path: Path | str,
        *,
        job_id: str | None = None,
        data_type: str = "training_data",
    ) -> UploadResult:
        """Upload training data to configured storage backends.
        
        Args:
            data_path: Path to training data file or directory.
            job_id: Optional job ID for organizing uploads.
            data_type: Type of training data (e.g., "features", "dataset", "labels").
            
        Returns:
            UploadResult with URIs/IDs for each backend.
        """
        result = UploadResult()
        
        if self._should_use_gcs() and self.gcs:
            try:
                result.gcs_uri = self.gcs.upload_training_data(data_path, job_id=job_id, data_type=data_type)
                LOGGER.info("Uploaded training data to GCS: %s", result.gcs_uri)
            except Exception as e:
                LOGGER.error("Failed to upload training data to GCS: %s", e)
        
        if self._should_use_drive() and self.drive:
            try:
                result.drive_id = self.drive.upload_training_data(data_path, job_id=job_id, data_type=data_type)
                LOGGER.info("Uploaded training data to Google Drive: %s", result.drive_id)
            except Exception as e:
                LOGGER.error("Failed to upload training data to Google Drive: %s", e)
        
        return result

    # ------------------------------------------------------------------ #
    # Convenience methods
    # ------------------------------------------------------------------ #
    def upload_training_artifacts(
        self,
        *,
        job_id: str,
        metadata: dict,
        model_path: Path | str | None = None,
        training_data_path: Path | str | None = None,
        data_type: str = "dataset",
        upload_weights: bool = False,
        upload_data: bool = False,
    ) -> Dict[str, UploadResult]:
        """Upload all training artifacts in one call.
        
        This is a convenience method that uploads metadata (always), and optionally
        uploads model weights and training data.
        
        Args:
            job_id: Job ID for organizing uploads.
            metadata: Model metadata dictionary.
            model_path: Optional path to model weights file.
            training_data_path: Optional path to training data.
            data_type: Type of training data.
            upload_weights: Whether to upload model weights.
            upload_data: Whether to upload training data.
            
        Returns:
            Dictionary with results for each artifact type.
        """
        results: Dict[str, UploadResult] = {}
        
        # Always upload metadata
        results["metadata"] = self.upload_model_metadata(metadata, job_id=job_id)
        
        # Optionally upload model weights
        if upload_weights and model_path:
            results["model"] = self.upload_model(model_path, job_id=job_id)
        
        # Optionally upload training data
        if upload_data and training_data_path:
            results["training_data"] = self.upload_training_data(
                training_data_path, job_id=job_id, data_type=data_type
            )
        
        return results

    def get_backend_status(self) -> Dict[str, bool]:
        """Get the status of each storage backend.
        
        Returns:
            Dictionary with backend availability status.
        """
        return {
            "gcs_enabled": self._should_use_gcs(),
            "gcs_available": self.gcs is not None,
            "drive_enabled": self._should_use_drive(),
            "drive_available": self.drive is not None,
            "backend": self._backend.value,
        }


# ------------------------------------------------------------------ #
# Module-level helpers
# ------------------------------------------------------------------ #
_CACHED_MANAGER: UnifiedStorageManager | None = None


def get_storage_manager(force_refresh: bool = False) -> UnifiedStorageManager:
    """Return a cached :class:`UnifiedStorageManager`.
    
    Args:
        force_refresh: Whether to force re-initialization.
        
    Returns:
        UnifiedStorageManager instance.
    """
    global _CACHED_MANAGER
    
    if not force_refresh and _CACHED_MANAGER is not None:
        return _CACHED_MANAGER
    
    _CACHED_MANAGER = UnifiedStorageManager()
    return _CACHED_MANAGER


__all__ = [
    "StorageBackend",
    "UnifiedStorageManager",
    "UploadResult",
    "get_storage_backend",
    "get_storage_manager",
]
