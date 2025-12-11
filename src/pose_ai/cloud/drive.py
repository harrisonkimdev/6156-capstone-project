"""Google Drive helpers for uploading and downloading training artifacts.

Supports two authentication methods:
1. OAuth 2.0 (for Colab - user authentication)
2. Service Account (for local automation)
"""

from __future__ import annotations

import io
import json
import logging
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)

# Optional dependencies - will be checked at runtime
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

    HAS_DRIVE_API = True
except ImportError:
    HAS_DRIVE_API = False


@dataclass(slots=True)
class DriveConfig:
    """Runtime configuration for Google Drive usage."""

    enabled: bool = False
    root_folder_id: str | None = None
    service_account_path: str | None = None
    # Folder structure within the root folder
    models_folder: str = "models"
    training_data_folder: str = "training_data"


def build_drive_config_from_env() -> DriveConfig:
    """Populate :class:`DriveConfig` from standard environment variables.
    
    Environment variables:
        GOOGLE_DRIVE_ENABLED: Whether Google Drive integration is enabled.
        GOOGLE_DRIVE_ROOT_FOLDER_ID: The root folder ID for storage.
        GOOGLE_DRIVE_SERVICE_ACCOUNT_PATH: Path to service account JSON file.
    
    Returns:
        DriveConfig instance.
    """
    enabled = os.getenv("GOOGLE_DRIVE_ENABLED", "false").lower() in ("true", "1", "yes")
    
    return DriveConfig(
        enabled=enabled,
        root_folder_id=os.getenv("GOOGLE_DRIVE_ROOT_FOLDER_ID"),
        service_account_path=os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_PATH"),
        models_folder=os.getenv("GOOGLE_DRIVE_MODELS_FOLDER", "models"),
        training_data_folder=os.getenv("GOOGLE_DRIVE_TRAINING_DATA_FOLDER", "training_data"),
    )


class GoogleDriveManager:
    """Encapsulates uploads, downloads, and folder operations for Google Drive."""

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    def __init__(self, config: DriveConfig, *, service=None) -> None:
        """Initialize the Google Drive manager.
        
        Args:
            config: DriveConfig instance with configuration.
            service: Optional pre-built Google Drive service (for testing).
        """
        self.config = config
        self._service = service
        self._folder_cache: Dict[str, str] = {}  # path -> folder_id

    def _ensure_dependencies(self) -> None:
        """Check that required dependencies are installed."""
        if not HAS_DRIVE_API:
            raise ModuleNotFoundError(
                "Google Drive API dependencies are required. "
                "Install them via: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )

    def _ensure_service(self):
        """Ensure the Google Drive service is initialized."""
        if self._service is not None:
            return self._service
        
        self._ensure_dependencies()
        
        if not self.config.enabled:
            raise ValueError("Google Drive integration is not enabled. Set GOOGLE_DRIVE_ENABLED=true")
        
        if self.config.service_account_path:
            # Use service account for authentication
            creds = service_account.Credentials.from_service_account_file(
                self.config.service_account_path,
                scopes=self.SCOPES,
            )
            self._service = build("drive", "v3", credentials=creds)
        else:
            raise ValueError(
                "Google Drive service account path is required for local usage. "
                "Set GOOGLE_DRIVE_SERVICE_ACCOUNT_PATH environment variable."
            )
        
        return self._service

    # ------------------------------------------------------------------ #
    # Folder helpers
    # ------------------------------------------------------------------ #
    def _get_or_create_folder(self, folder_name: str, parent_id: str | None = None) -> str:
        """Get or create a folder in Google Drive.
        
        Args:
            folder_name: Name of the folder to get/create.
            parent_id: Parent folder ID (None for root).
            
        Returns:
            Folder ID.
        """
        service = self._ensure_service()
        
        # Check cache first
        cache_key = f"{parent_id or 'root'}/{folder_name}"
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]
        
        # Search for existing folder
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        
        results = service.files().list(
            q=query,
            spaces="drive",
            fields="files(id, name)",
        ).execute()
        
        files = results.get("files", [])
        if files:
            folder_id = files[0]["id"]
        else:
            # Create new folder
            file_metadata = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
            }
            if parent_id:
                file_metadata["parents"] = [parent_id]
            
            folder = service.files().create(
                body=file_metadata,
                fields="id",
            ).execute()
            folder_id = folder.get("id")
            LOGGER.info("Created folder '%s' with ID: %s", folder_name, folder_id)
        
        self._folder_cache[cache_key] = folder_id
        return folder_id

    def _ensure_folder_path(self, path: str) -> str:
        """Ensure a folder path exists, creating folders as needed.
        
        Args:
            path: Folder path like "models/job123/weights".
            
        Returns:
            ID of the final folder.
        """
        current_parent = self.config.root_folder_id
        
        for folder_name in path.split("/"):
            if not folder_name:
                continue
            current_parent = self._get_or_create_folder(folder_name, current_parent)
        
        return current_parent

    # ------------------------------------------------------------------ #
    # Upload helpers
    # ------------------------------------------------------------------ #
    def upload_file(
        self,
        local_path: Path | str,
        *,
        remote_path: str,
        content_type: str | None = None,
    ) -> str:
        """Upload a file to Google Drive.
        
        Args:
            local_path: Path to local file.
            remote_path: Remote path like "models/job123/model.pt".
            content_type: Optional MIME type.
            
        Returns:
            Google Drive file ID.
        """
        service = self._ensure_service()
        path = Path(local_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {path}")
        
        # Get or create parent folder
        folder_path = "/".join(remote_path.split("/")[:-1])
        file_name = remote_path.split("/")[-1]
        parent_id = self._ensure_folder_path(folder_path) if folder_path else self.config.root_folder_id
        
        # Determine content type
        guessed_type = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        
        # Upload file
        file_metadata = {
            "name": file_name,
        }
        if parent_id:
            file_metadata["parents"] = [parent_id]
        
        media = MediaFileUpload(str(path), mimetype=guessed_type)
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id",
        ).execute()
        
        file_id = file.get("id")
        LOGGER.debug("Uploaded %s -> Google Drive (ID: %s)", path, file_id)
        return file_id

    def upload_directory(
        self,
        directory: Path | str,
        *,
        remote_path: str,
        include_hidden: bool = False,
    ) -> str:
        """Upload a directory to Google Drive.
        
        Args:
            directory: Path to local directory.
            remote_path: Remote path like "training_data/job123/dataset".
            include_hidden: Whether to include hidden files.
            
        Returns:
            Google Drive folder ID.
        """
        root = Path(directory)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {root}")
        
        folder_id = self._ensure_folder_path(remote_path)
        
        for file_path in sorted(root.rglob("*")):
            if file_path.is_dir():
                continue
            if not include_hidden and any(part.startswith(".") for part in file_path.relative_to(root).parts):
                continue
            relative = file_path.relative_to(root).as_posix()
            file_remote_path = f"{remote_path}/{relative}"
            self.upload_file(file_path, remote_path=file_remote_path)
        
        return folder_id

    def upload_model(
        self,
        model_path: Path | str,
        *,
        job_id: str | None = None,
    ) -> str:
        """Upload a model file to Google Drive.
        
        Args:
            model_path: Path to model file.
            job_id: Optional job ID for organizing uploads.
            
        Returns:
            Google Drive file ID.
        """
        model_path = Path(model_path)
        folder = self.config.models_folder
        if job_id:
            remote_path = f"{folder}/{job_id}/{model_path.name}"
        else:
            remote_path = f"{folder}/{model_path.name}"
        
        return self.upload_file(model_path, remote_path=remote_path)

    def upload_model_metadata(
        self,
        metadata: dict,
        *,
        job_id: str | None = None,
        filename: str = "model_metadata.json",
    ) -> str:
        """Upload model metadata to Google Drive.
        
        Args:
            metadata: Dictionary containing model metadata.
            job_id: Optional job ID for organizing uploads.
            filename: Name of the metadata file.
            
        Returns:
            Google Drive file ID.
        """
        import tempfile
        
        folder = self.config.models_folder
        if job_id:
            remote_path = f"{folder}/{job_id}/{filename}"
        else:
            remote_path = f"{folder}/{filename}"
        
        # Create temporary file with metadata
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(metadata, tmp, indent=2, default=str)
            tmp_path = Path(tmp.name)
        
        try:
            return self.upload_file(tmp_path, remote_path=remote_path, content_type="application/json")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def upload_training_data(
        self,
        data_path: Path | str,
        *,
        job_id: str | None = None,
        data_type: str = "dataset",
    ) -> str:
        """Upload training data to Google Drive.
        
        Args:
            data_path: Path to training data file or directory.
            job_id: Optional job ID for organizing uploads.
            data_type: Type of training data (e.g., "features", "dataset", "labels").
            
        Returns:
            Google Drive file or folder ID.
        """
        data_path = Path(data_path)
        folder = self.config.training_data_folder
        
        if job_id:
            base_path = f"{folder}/{job_id}/{data_type}"
        else:
            base_path = f"{folder}/{data_type}"
        
        if data_path.is_file():
            remote_path = f"{base_path}/{data_path.name}"
            return self.upload_file(data_path, remote_path=remote_path)
        else:
            remote_path = f"{base_path}/{data_path.name}"
            return self.upload_directory(data_path, remote_path=remote_path)

    # ------------------------------------------------------------------ #
    # Download helpers
    # ------------------------------------------------------------------ #
    def download_file(
        self,
        file_id: str,
        local_path: Path | str,
    ) -> Path:
        """Download a file from Google Drive.
        
        Args:
            file_id: Google Drive file ID.
            local_path: Path to save the file.
            
        Returns:
            Path to the downloaded file.
        """
        service = self._ensure_service()
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        request = service.files().get_media(fileId=file_id)
        
        with open(local_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        
        LOGGER.debug("Downloaded Google Drive file %s -> %s", file_id, local_path)
        return local_path

    def list_files(
        self,
        folder_id: str | None = None,
        file_type: str | None = None,
    ) -> List[Dict[str, str]]:
        """List files in a Google Drive folder.
        
        Args:
            folder_id: Folder ID to list (None for root).
            file_type: Optional MIME type filter.
            
        Returns:
            List of file dictionaries with 'id', 'name', 'mimeType'.
        """
        service = self._ensure_service()
        
        query_parts = ["trashed=false"]
        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")
        if file_type:
            query_parts.append(f"mimeType='{file_type}'")
        
        query = " and ".join(query_parts)
        
        results = service.files().list(
            q=query,
            spaces="drive",
            fields="files(id, name, mimeType, size, createdTime)",
        ).execute()
        
        return results.get("files", [])


# ------------------------------------------------------------------ #
# Module-level helpers
# ------------------------------------------------------------------ #
_CACHED_MANAGER: GoogleDriveManager | None = None


def get_drive_manager(force_refresh: bool = False) -> GoogleDriveManager | None:
    """Return a cached :class:`GoogleDriveManager` if Google Drive is enabled.
    
    Args:
        force_refresh: Whether to force re-initialization.
        
    Returns:
        GoogleDriveManager instance if enabled, None otherwise.
    """
    global _CACHED_MANAGER
    
    if not force_refresh and _CACHED_MANAGER is not None:
        return _CACHED_MANAGER
    
    config = build_drive_config_from_env()
    
    if not config.enabled:
        LOGGER.debug("Google Drive integration is disabled")
        return None
    
    _CACHED_MANAGER = GoogleDriveManager(config)
    return _CACHED_MANAGER


def is_drive_enabled() -> bool:
    """Check if Google Drive integration is enabled."""
    config = build_drive_config_from_env()
    return config.enabled


__all__ = [
    "DriveConfig",
    "GoogleDriveManager",
    "build_drive_config_from_env",
    "get_drive_manager",
    "is_drive_enabled",
]
