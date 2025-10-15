"""Service helpers for exporting pose-derived features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pose_ai.features import HoldDefinition, summarize_features
from pose_ai.pose.estimator import PoseFrame, PoseLandmark
from pose_ai.service.pose_service import estimate_poses_from_manifest


def _load_pose_results(path: Path) -> List[PoseFrame]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    frames: List[PoseFrame] = []
    for frame_payload in data.get("frames", []):
        landmarks = [
            PoseLandmark(
                name=lm["name"],
                x=float(lm["x"]),
                y=float(lm["y"]),
                z=float(lm["z"]),
                visibility=float(lm.get("visibility", 0.0)),
            )
            for lm in frame_payload.get("landmarks", [])
        ]
        frames.append(
            PoseFrame(
                image_path=Path(frame_payload["image_path"]),
                timestamp_seconds=float(frame_payload["timestamp_seconds"]),
                landmarks=landmarks,
                detection_score=float(frame_payload.get("detection_score", 0.0)),
            )
        )
    return frames


def _load_holds(holds_path: Optional[Path]) -> Optional[Dict[str, HoldDefinition]]:
    if holds_path is None:
        return None
    payload = json.loads(Path(holds_path).read_text(encoding="utf-8"))
    holds: Dict[str, HoldDefinition] = {}
    for name, cfg in payload.items():
        holds[name] = HoldDefinition(
            name=name,
            coords=cfg["coords"],
            normalized=cfg.get("normalized", False),
            hold_type=cfg.get("hold_type", "auto"),
            notes=cfg.get("notes"),
        )
    return holds


def export_features_for_manifest(
    manifest_path: Path | str,
    *,
    holds_path: Optional[Path] = None,
    output_root: Path | None = None,
    estimator_kwargs: Optional[dict] = None,
) -> Path:
    manifest_path = Path(manifest_path)
    frame_dir = manifest_path.parent
    pose_results_path = frame_dir / "pose_results.json"

    if not pose_results_path.exists():
        estimate_poses_from_manifest(
            manifest_path,
            save_json=True,
            estimator_kwargs=estimator_kwargs,
        )

    frames = _load_pose_results(pose_results_path)
    holds = _load_holds(holds_path)

    feature_rows = summarize_features(frames, holds=holds)
    output_dir = output_root or frame_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pose_features.json"
    Path(output_path).write_text(json.dumps(feature_rows, indent=2), encoding="utf-8")
    return output_path
