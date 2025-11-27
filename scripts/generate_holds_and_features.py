"""Generate holds.json and enriched pose_features.json for an existing frame directory.

Usage:
    python scripts/generate_holds_and_features.py /path/to/frames/manifest.json --model yolov8m.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pose_ai.service.feature_service import export_features_for_manifest  # type: ignore
from pose_ai.service.hold_extraction import (
    extract_and_cluster_holds,
    export_holds_json,
)  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hold extraction + enriched feature export")
    p.add_argument("manifest", type=Path, help="Path to manifest.json")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO model weights")
    p.add_argument("--device", default=None, help="Optional torch device")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    frame_dir = manifest_path.parent
    image_paths = sorted(frame_dir.glob("*.jpg"))
    if not image_paths:
        print("No frame images found; aborting hold extraction.")
        holds_path = None
    else:
        print(f"Detecting holds in {len(image_paths)} frames using {args.model}")
        clustered = extract_and_cluster_holds(image_paths, model_name=args.model, device=args.device)
        
        if clustered:
            holds_path = export_holds_json(clustered, output_path=frame_dir / "holds.json")
            print(f"Exported {len(clustered)} clustered holds to {holds_path}")
        else:
            holds_path = None
            print("No holds detected.")
    print("Exporting enriched pose features (with wall angle & holds)...")
    export_features_for_manifest(manifest_path, holds_path=holds_path, auto_wall_angle=True)
    print("Done.")


if __name__ == "__main__":
    main()
