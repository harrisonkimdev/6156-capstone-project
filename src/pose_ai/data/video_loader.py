"""Locate and validate video assets for processing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi")


def iter_video_files(root: Path | str, *, recursive: bool = False) -> Iterator[Path]:
    """Yield video files from ``root`` filtering by known extensions."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Video directory not found: {root_path}")

    glob_pattern = "**/*" if recursive else "*"
    for candidate in root_path.glob(glob_pattern):
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
            yield candidate


def ensure_directory(path: Path | str) -> Path:
    """Create a directory if it does not exist."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


__all__ = ["iter_video_files", "ensure_directory", "VIDEO_EXTENSIONS"]
