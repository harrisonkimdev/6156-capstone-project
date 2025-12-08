#!/usr/bin/env python3
"""
Performs MediaPipe pose estimation and load analysis on new workflow frames.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src directory to Python path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pose_ai.pose import PoseEstimator
from pose_ai.service import estimate_poses_from_manifest, export_features_for_manifest
from scripts.visualize_pose import visualize_pose_results


def create_manifest_for_frames(frames_dir: Path) -> Path:
    """Create manifest.json for image frames"""
    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        raise ValueError(f"No .jpg files found in {frames_dir}")
    
    # Calculate timestamp assuming 30 FPS
    fps = 30.0
    manifest_data = {
        "video_name": frames_dir.name,
        "fps": fps,
        "frames": []
    }
    
    for idx, frame_path in enumerate(frame_files):
        timestamp = idx / fps
        manifest_data["frames"].append({
            "image_path": str(frame_path),
            "timestamp_seconds": timestamp
        })
    
    manifest_path = frames_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created manifest with {len(frame_files)} frames at {manifest_path}")
    return manifest_path


def main():
    # Configure source and destination directories
    source_dir = ROOT_DIR / "data" / "workflow_frames" / "test_video_frames" / "IMG_3571_converted"
    
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)
    
    print(f"Processing frames in: {source_dir}")
    
    # Step 1: Create manifest.json if it doesn't exist
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = create_manifest_for_frames(source_dir)
    else:
        print(f"Using existing manifest: {manifest_path}")
    
    # Step 2: Run pose estimation
    print("Running pose estimation...")
    try:
        pose_frames = estimate_poses_from_manifest(manifest_path, save_json=True)
        pose_results_path = source_dir / "pose_results.json"
        print(f"Pose estimation complete. Results saved to: {pose_results_path}")
        print(f"Processed {len(pose_frames)} frames")
    except Exception as e:
        print(f"Error during pose estimation: {e}")
        sys.exit(1)
    
    # Step 3: Feature extraction for load analysis
    print("Extracting features for load analysis...")
    try:
        export_features_for_manifest(manifest_path)
        features_path = source_dir / "pose_features.json"
        print(f"Feature extraction complete. Results saved to: {features_path}")
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        features_path = None
    
    # Step 4: Generate load visualization
    print("Generating load visualization...")
    try:
        # Save to visualized_load directory
        output_dir = source_dir / "visualized_load"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_count = visualize_pose_results(
            pose_results_path=pose_results_path,
            output_dir=output_dir,
            include_missing=False,
            min_score=0.2,
            min_visibility=0.2,
            draw_com=True,
            com_box_scale=0.3,
            mode="load",
            features_path=features_path
        )
        
        print(f"Load visualization complete. Generated {processed_count} visualized images in: {output_dir}")
        
        # Print a few sample image paths
        viz_images = list(output_dir.glob("*_viz.jpg"))
        print(f"\nSample output files:")
        for img_path in viz_images[:5]:  # Print first 5 only
            print(f"  - {img_path}")
        if len(viz_images) > 5:
            print(f"  ... and {len(viz_images) - 5} more files")
            
    except Exception as e:
        print(f"Error during visualization: {e}")
        sys.exit(1)
    
    print("\nAnalysis complete!")
    print(f"Results saved in: {source_dir}")
    print(f"Load visualizations: {source_dir / 'visualized_load'}")


if __name__ == "__main__":
    main()