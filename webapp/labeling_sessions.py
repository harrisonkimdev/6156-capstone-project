"""Labeling session management for SAM-based hold annotation.

This module manages the state of labeling sessions, including frame tracking,
segment labels, and dataset export functionality.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

LOGGER = logging.getLogger(__name__)


class SessionStatus(str, Enum):
    """Status of a labeling session."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPORTED = "exported"
    FAILED = "failed"


@dataclass
class FrameLabels:
    """Labels for segments in a single frame."""
    
    frame_index: int
    frame_path: str
    segments: Dict[str, Dict]  # segment_id -> {hold_type, is_hold, bbox, etc}
    labeled_count: int = 0
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "frame_index": self.frame_index,
            "frame_path": str(self.frame_path) if isinstance(self.frame_path, Path) else self.frame_path,
            "segments": self.segments,
            "labeled_count": self.labeled_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> FrameLabels:
        """Create from dict."""
        return cls(
            frame_index=data["frame_index"],
            frame_path=data["frame_path"],
            segments=data["segments"],
            labeled_count=data.get("labeled_count", 0),
        )


@dataclass
class LabelingSession:
    """A labeling session for annotating holds in video frames.
    
    Attributes:
        id: Unique session identifier
        name: Human-readable session name
        frame_dir: Directory containing extracted frames
        frames: List of frame paths
        labels: Frame-wise label data
        status: Current session status
        created_at: Creation timestamp
        completed_at: Completion timestamp
        sam_checkpoint: Path to SAM model checkpoint used
        total_segments: Total number of segments across all frames
        labeled_segments: Number of segments labeled as holds
    """
    
    id: str
    name: str
    frame_dir: Path
    frames: List[Path] = field(default_factory=list)
    labels: Dict[int, FrameLabels] = field(default_factory=dict)
    status: SessionStatus = SessionStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    sam_checkpoint: Optional[str] = None
    total_segments: int = 0
    labeled_segments: int = 0
    metadata: Dict = field(default_factory=dict)
    
    def get_session_dir(self, base_dir: Path) -> Path:
        """Get the session's data directory."""
        return base_dir / self.id
    
    def add_frame(self, frame_path: Path, segments: List[Dict]) -> None:
        """Add a frame with its SAM segments to the session.
        
        Args:
            frame_path: Path to frame image
            segments: List of segment dicts from SAM
        """
        frame_index = len(self.frames)
        self.frames.append(frame_path)
        
        # Initialize empty labels for this frame
        segment_dict = {}
        for seg in segments:
            segment_dict[seg["segment_id"]] = {
                "bbox": seg["bbox"],
                "area": seg["area"],
                "predicted_iou": seg["predicted_iou"],
                "stability_score": seg["stability_score"],
                "hold_type": None,
                "is_hold": False,
                "user_confirmed": False,
            }
        
        self.labels[frame_index] = FrameLabels(
            frame_index=frame_index,
            frame_path=str(frame_path),
            segments=segment_dict,
            labeled_count=0,
        )
        
        self.total_segments += len(segments)
        
        if self.status == SessionStatus.PENDING:
            self.status = SessionStatus.IN_PROGRESS
    
    def update_frame_segments(self, frame_index: int, segments: List[Dict]) -> None:
        """Update segments for an existing frame.
        
        Args:
            frame_index: Index of the frame
            segments: List of segment dicts from SAM
        """
        if frame_index not in self.labels:
            return
        
        # Build segment dict
        segment_dict = {}
        for seg in segments:
            segment_dict[seg["segment_id"]] = {
                "bbox": seg["bbox"],
                "area": seg["area"],
                "predicted_iou": seg["predicted_iou"],
                "stability_score": seg["stability_score"],
                "hold_type": None,
                "is_hold": False,
                "user_confirmed": False,
            }
        
        # Update existing frame labels
        self.labels[frame_index].segments = segment_dict
        self.total_segments += len(segments)
        
        if self.status == SessionStatus.PENDING:
            self.status = SessionStatus.IN_PROGRESS
    
    def update_segment_label(
        self,
        frame_index: int,
        segment_id: str,
        hold_type: Optional[str],
        is_hold: bool,
    ) -> None:
        """Update the label for a specific segment.
        
        Args:
            frame_index: Index of the frame
            segment_id: ID of the segment
            hold_type: Hold type classification (or None if not a hold)
            is_hold: Whether this segment is a hold
        """
        if frame_index not in self.labels:
            raise ValueError(f"Frame index {frame_index} not found in session")
        
        frame_labels = self.labels[frame_index]
        if segment_id not in frame_labels.segments:
            raise ValueError(f"Segment {segment_id} not found in frame {frame_index}")
        
        segment = frame_labels.segments[segment_id]
        was_labeled = segment["user_confirmed"]
        
        segment["hold_type"] = hold_type
        segment["is_hold"] = is_hold
        segment["user_confirmed"] = True
        
        # Update counters
        if not was_labeled and is_hold:
            frame_labels.labeled_count += 1
            self.labeled_segments += 1
        elif was_labeled and not is_hold and segment.get("is_hold"):
            frame_labels.labeled_count = max(0, frame_labels.labeled_count - 1)
            self.labeled_segments = max(0, self.labeled_segments - 1)
    
    def get_frame_labels(self, frame_index: int) -> Optional[FrameLabels]:
        """Get labels for a specific frame."""
        return self.labels.get(frame_index)
    
    def get_labeled_segments(self) -> List[Dict]:
        """Get all segments that have been labeled as holds.
        
        Returns:
            List of dicts with frame_index, segment_id, hold_type, bbox, etc.
        """
        labeled = []
        for frame_idx, frame_labels in self.labels.items():
            for seg_id, seg_data in frame_labels.segments.items():
                if seg_data["is_hold"] and seg_data["user_confirmed"]:
                    labeled.append({
                        "frame_index": frame_idx,
                        "frame_path": frame_labels.frame_path,
                        "segment_id": seg_id,
                        "hold_type": seg_data["hold_type"],
                        "bbox": seg_data["bbox"],
                    })
        return labeled
    
    def get_progress(self) -> Dict[str, int]:
        """Get labeling progress statistics."""
        total_frames = len(self.frames)
        labeled_frames = sum(1 for fl in self.labels.values() if fl.labeled_count > 0)
        
        return {
            "total_frames": total_frames,
            "labeled_frames": labeled_frames,
            "total_segments": self.total_segments,
            "labeled_segments": self.labeled_segments,
            "completion_percent": int((self.labeled_segments / max(1, self.total_segments)) * 100),
        }
    
    def mark_completed(self) -> None:
        """Mark session as completed."""
        self.status = SessionStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()
    
    def mark_exported(self) -> None:
        """Mark session as exported to YOLO dataset."""
        self.status = SessionStatus.EXPORTED
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "id": self.id,
            "name": self.name,
            "frame_dir": str(self.frame_dir),
            "frames": [str(f) for f in self.frames],
            "labels": {
                idx: fl.to_dict() for idx, fl in self.labels.items()
            },
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "sam_checkpoint": self.sam_checkpoint,
            "total_segments": self.total_segments,
            "labeled_segments": self.labeled_segments,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> LabelingSession:
        """Create from dict."""
        session = cls(
            id=data["id"],
            name=data["name"],
            frame_dir=Path(data["frame_dir"]),
            frames=[Path(f) for f in data["frames"]],
            labels={
                int(idx): FrameLabels.from_dict(fl)
                for idx, fl in data["labels"].items()
            },
            status=SessionStatus(data["status"]),
            created_at=data["created_at"],
            completed_at=data.get("completed_at"),
            sam_checkpoint=data.get("sam_checkpoint"),
            total_segments=data.get("total_segments", 0),
            labeled_segments=data.get("labeled_segments", 0),
            metadata=data.get("metadata", {}),
        )
        return session


class LabelingManager:
    """Manager for labeling sessions with persistence."""
    
    def __init__(self, sessions_dir: Path):
        """Initialize manager.
        
        Args:
            sessions_dir: Directory to store session data
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        self._sessions: Dict[str, LabelingSession] = {}
        self._load_sessions()
        
        LOGGER.info("LabelingManager initialized: %s", self.sessions_dir)
    
    def _load_sessions(self) -> None:
        """Load existing sessions from disk."""
        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            session_file = session_dir / "session.json"
            if not session_file.exists():
                continue
            
            try:
                data = json.loads(session_file.read_text())
                session = LabelingSession.from_dict(data)
                self._sessions[session.id] = session
                LOGGER.debug("Loaded session: %s", session.id)
            except Exception as exc:
                LOGGER.error("Failed to load session %s: %s", session_dir.name, exc)
    
    def create_session(
        self,
        name: str,
        frame_dir: Path,
        sam_checkpoint: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> LabelingSession:
        """Create a new labeling session.
        
        Args:
            name: Human-readable session name
            frame_dir: Directory containing frames to label
            sam_checkpoint: Path to SAM checkpoint used
            metadata: Optional metadata
            
        Returns:
            New LabelingSession
        """
        session_id = uuid4().hex[:12]
        
        # Scan frame directory for images
        frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
        if not frame_paths:
            LOGGER.warning("No .jpg frames found in %s", frame_dir)
        
        session = LabelingSession(
            id=session_id,
            name=name,
            frame_dir=Path(frame_dir),
            frames=frame_paths,  # Add frames list
            sam_checkpoint=sam_checkpoint,
            metadata=metadata or {},
        )
        
        # Initialize empty labels for each frame (will be populated by SAM later if enabled)
        for frame_idx, frame_path in enumerate(frame_paths):
            session.labels[frame_idx] = FrameLabels(
                frame_index=frame_idx,
                frame_path=frame_path,
                segments={},  # Empty segments - will be filled by SAM
            )
        
        self._sessions[session_id] = session
        self._save_session(session)
        
        # Create session directory
        session_dir = session.get_session_dir(self.sessions_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        
        LOGGER.info("Created labeling session: %s (%s) with %d frames", session_id, name, len(frame_paths))
        return session
    
    def get_session(self, session_id: str) -> Optional[LabelingSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    def list_sessions(self) -> List[LabelingSession]:
        """List all sessions."""
        return list(self._sessions.values())
    
    def update_session(self, session: LabelingSession) -> None:
        """Update and persist a session."""
        self._sessions[session.id] = session
        self._save_session(session)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its data.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        # Remove from memory
        del self._sessions[session_id]
        
        # Remove from disk
        session_dir = session.get_session_dir(self.sessions_dir)
        if session_dir.exists():
            shutil.rmtree(session_dir)
        
        LOGGER.info("Deleted session: %s", session_id)
        return True
    
    def _save_session(self, session: LabelingSession) -> None:
        """Persist session to disk."""
        session_dir = session.get_session_dir(self.sessions_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        
        session_file = session_dir / "session.json"
        session_file.write_text(json.dumps(session.to_dict(), indent=2))
        LOGGER.debug("Saved session: %s", session.id)
    
    def get_stats(self) -> Dict:
        """Get overall statistics across all sessions."""
        total_sessions = len(self._sessions)
        total_frames = sum(len(s.frames) for s in self._sessions.values())
        total_labeled = sum(s.labeled_segments for s in self._sessions.values())
        
        by_status = {}
        for session in self._sessions.values():
            status = session.status.value
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            "total_sessions": total_sessions,
            "total_frames": total_frames,
            "total_labeled_holds": total_labeled,
            "sessions_by_status": by_status,
        }


__all__ = [
    "LabelingSession",
    "LabelingManager",
    "SessionStatus",
    "FrameLabels",
]
