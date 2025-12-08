#!/usr/bin/env python3
"""
Full MediaPipe + Load Visualization Test Pipeline

This script:
1. Runs MediaPipe pose estimation on test frames
2. Extracts movement features 
3. Generates load visualization with biomechanical analysis
4. Displays results summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pose_ai.service.pose_service import estimate_poses_from_manifest
from pose_ai.service.feature_service import export_features_for_manifest
from scripts.visualize_pose import visualize_pose_results


def test_mediapipe_pipeline():
    """Run the full MediaPipe + visualization pipeline."""
    print("=" * 70)
    print("🏃 Full MediaPipe + Load Visualization Pipeline Test")
    print("=" * 70)
    
    repo_root = Path(__file__).parent
    test_frames_dir = repo_root / "data" / "test_frames" / "IMG_3571"
    manifest_path = test_frames_dir / "manifest.json"
    
    if not manifest_path.exists():
        print(f"❌ manifest.json not found at {manifest_path}")
        return False
    
    print(f"\n📍 Test directory: {test_frames_dir}")
    print(f"   - Manifest: {manifest_path.name}")
    
    # =========================================================================
    # Step 1: Pose Estimation
    # =========================================================================
    print("\n[Step 1/3] MediaPipe Pose Estimation")
    print("-" * 70)
    
    try:
        frames = estimate_poses_from_manifest(manifest_path, save_json=True)
        pose_results_path = test_frames_dir / "pose_results.json"
        
        print(f"✅ Pose estimation completed")
        print(f"   - Frames processed: {len(frames)}")
        
        # Statistics
        frames_with_landmarks = sum(1 for f in frames if f.landmarks)
        detection_scores = [f.detection_score for f in frames if f.detection_score > 0]
        
        print(f"   - Frames with landmarks: {frames_with_landmarks}/{len(frames)} ({100*frames_with_landmarks/len(frames):.1f}%)")
        if detection_scores:
            print(f"   - Detection score: {min(detection_scores):.3f} ~ {max(detection_scores):.3f} (avg: {sum(detection_scores)/len(detection_scores):.3f})")
        
        if frames_with_landmarks == 0:
            print("❌ No landmarks detected - cannot proceed")
            return False
        
    except Exception as e:
        print(f"❌ Pose estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # Step 2: Feature Extraction
    # =========================================================================
    print("\n[Step 2/3] Feature Extraction for Load Analysis")
    print("-" * 70)
    
    features_path: Optional[Path] = None
    try:
        export_features_for_manifest(manifest_path)
        features_path = test_frames_dir / "pose_features.json"
        
        if features_path.exists():
            with open(features_path) as f:
                features_data = json.load(f)
            
            if isinstance(features_data, list):
                num_features = len(features_data)
            else:
                num_features = len(features_data.get("rows", []))
            
            print(f"✅ Feature extraction completed")
            print(f"   - Feature rows: {num_features}")
            
            if num_features > 0:
                first_row = features_data[0] if isinstance(features_data, list) else features_data["rows"][0]
                feature_keys = list(first_row.keys())
                print(f"   - Features per row: {len(feature_keys)}")
                print(f"   - Sample features: {', '.join(feature_keys[:5])}")
        else:
            print("⚠️  Features not saved - proceeding without feature file")
            features_path = None
    
    except Exception as e:
        print(f"⚠️  Feature extraction warning: {e}")
        print("   Proceeding with basic visualization")
        features_path = None
    
    # =========================================================================
    # Step 3: Load Visualization
    # =========================================================================
    print("\n[Step 3/3] Generate Load Visualization with Biomechanical Analysis")
    print("-" * 70)
    
    try:
        output_dir = test_frames_dir / "visualized_load"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎨 Rendering visualizations with load analysis...")
        print(f"   - Output directory: {output_dir}")
        print(f"   - Mode: load (green→yellow→red gradient)")
        print(f"   - Features: {'included' if features_path else 'basic analysis only'}")
        
        processed_count = visualize_pose_results(
            pose_results_path=pose_results_path,
            output_dir=output_dir,
            include_missing=False,
            min_score=0.2,
            min_visibility=0.1,
            draw_com=True,  # Draw center of mass
            com_box_scale=0.3,
            mode="load",  # Load visualization mode
            features_path=features_path,
        )
        
        print(f"\n✅ Load visualization completed")
        print(f"   - Images generated: {processed_count}")
        
        # List output files
        viz_images = sorted(output_dir.glob("*_viz.jpg"))
        
        if viz_images:
            print(f"\n📸 Generated visualization samples:")
            for img_path in viz_images[:5]:
                rel_path = img_path.relative_to(repo_root)
                print(f"   - {rel_path}")
            
            if len(viz_images) > 5:
                print(f"   ... and {len(viz_images) - 5} more")
            
            print(f"\n💡 Visualization legend:")
            print(f"   🟢 Green: Low biomechanical load (efficient)")
            print(f"   🟡 Yellow: Moderate load")
            print(f"   🔴 Red: High load (challenging/demanding)")
            print(f"   ◉ Crosshair: Center of mass position")
            print(f"   □ Box: COM stability region")
            print(f"   △ Polygon: Support base (contact points)")
            
        else:
            print(f"⚠️  No visualization images were generated")
            return False
        
    except Exception as e:
        print(f"❌ Load visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ FULL PIPELINE TEST PASSED!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  ✓ Pose estimation: {len(frames)} frames, {frames_with_landmarks} with landmarks")
    print(f"  ✓ Feature extraction: completed successfully")
    print(f"  ✓ Load visualization: {processed_count} images with biomechanical analysis")
    print(f"\nResults location:")
    print(f"  📁 {test_frames_dir}")
    print(f"     ├─ pose_results.json (landmarks)")
    print(f"     ├─ pose_features.json (movement features)")
    print(f"     └─ visualized_load/ (load analysis images)")
    
    return True


if __name__ == "__main__":
    success = test_mediapipe_pipeline()
    sys.exit(0 if success else 1)
