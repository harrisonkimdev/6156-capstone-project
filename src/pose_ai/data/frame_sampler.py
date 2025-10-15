"""Frame extraction utilities mirrored from the legacy BetaMove notebook."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2

from .video_loader import ensure_directory

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FrameExtractionResult:
    """Metadata describing an extraction run."""

    video_path: Path
    frame_directory: Path
    manifest_path: Optional[Path]
    fps: float
    interval_seconds: float
    total_frames: int
    saved_frames: int
    frame_paths: List[Path]


def extract_frames_every_n_seconds(
    video_path: Path | str,
    *,
    interval_seconds: float = 1.0,
    output_root: Path | str = Path("data") / "frames",
    write_manifest: bool = True,
    overwrite: bool = False,
) -> FrameExtractionResult:
    """Extract frames from ``video_path`` and store them under ``output_root``."""
    source_path = Path(video_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Video file not found: {source_path}")

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive.")

    frames_root = ensure_directory(output_root)
    frame_dir = ensure_directory(frames_root / source_path.stem)

    if overwrite:
        for existing in frame_dir.glob("*.jpg"):
            existing.unlink()

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"CV2 failed to open video file: {source_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_interval = max(int(round(fps * interval_seconds)), 1) if fps > 0 else 1

    frame_idx = 0
    saved_idx = 0
    saved_paths: list[Path] = []
    manifest_records: list[dict[str, object]] = []

    LOGGER.info(
        "Starting frame extraction: video=%s, fps=%.2f, interval=%.2fs",
        source_path,
        fps,
        interval_seconds,
    )

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_idx % frame_interval == 0:
            frame_filename = f"{source_path.stem}_frame_{saved_idx:04d}.jpg"
            frame_path = frame_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            saved_paths.append(frame_path)

            timestamp_seconds = frame_idx / fps if fps > 0 else saved_idx * interval_seconds
            manifest_records.append(
                {
                    "frame_index": int(frame_idx),
                    "saved_index": int(saved_idx),
                    "timestamp_seconds": float(timestamp_seconds),
                    "relative_path": frame_filename,
                }
            )
            saved_idx += 1

        frame_idx += 1

    capture.release()

    manifest_path: Optional[Path] = None
    if write_manifest:
        manifest_path = frame_dir / "manifest.json"
        manifest_payload = {
            "video": str(source_path),
            "fps": fps,
            "interval_seconds": interval_seconds,
            "frame_interval": frame_interval,
            "total_frames": frame_idx,
            "saved_frames": saved_idx,
            "frames": manifest_records,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    LOGGER.info(
        "Extraction complete: saved=%d frames (directory=%s)",
        saved_idx,
        frame_dir,
    )

    return FrameExtractionResult(
        video_path=source_path,
        frame_directory=frame_dir,
        manifest_path=manifest_path,
        fps=fps,
        interval_seconds=interval_seconds,
        total_frames=frame_idx,
        saved_frames=saved_idx,
        frame_paths=saved_paths,
    )


__all__ = ["extract_frames_every_n_seconds", "FrameExtractionResult"]
