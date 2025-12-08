"""
Test script to verify MediaPipe climber movement analysis works correctly.

This script:
1. Loads test frames from /data/test_frames/IMG_3571
2. Runs pose estimation using MediaPipe
3. Extracts movement features from the poses
4. Validates the analysis pipeline
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pose_ai.service.pose_service import estimate_poses_from_manifest
from pose_ai.features import summarize_features, HoldDefinition
from pose_ai.pose.estimator import PoseFrame, PoseLandmark


def load_holds_from_json(holds_path: Path) -> Optional[Dict[str, HoldDefinition]]:
    """Load hold definitions from holds.json."""
    if not holds_path.exists():
        return None
    
    with open(holds_path) as f:
        holds_data = json.load(f)
    
    holds_dict = {}
    for hold_id, hold_dict in holds_data.items():
        coords = hold_dict.get('coords', [0, 0])
        hold = HoldDefinition(
            id=hold_id,
            center_x=coords[0] if len(coords) > 0 else 0,
            center_y=coords[1] if len(coords) > 1 else 0,
            label=hold_dict.get('label', 'hold'),
        )
        holds_dict[hold_id] = hold
    
    return holds_dict if holds_dict else None


def print_pose_frame_info(frames: List[PoseFrame]) -> None:
    """Print detailed information about pose frames."""
    print(f"\n📊 Pose Frame Statistics:")
    print(f"   Total frames: {len(frames)}")
    
    if not frames:
        return
    
    # Frames with landmarks
    frames_with_landmarks = sum(1 for f in frames if f.landmarks)
    print(f"   Frames with landmarks: {frames_with_landmarks}/{len(frames)} ({100*frames_with_landmarks/len(frames):.1f}%)")
    
    # Average landmarks per frame
    if frames_with_landmarks > 0:
        avg_landmarks = sum(len(f.landmarks) for f in frames) / frames_with_landmarks
        print(f"   Average landmarks per frame: {avg_landmarks:.1f}")
    
    # Detection scores
    scores = [f.detection_score for f in frames if f.detection_score > 0]
    if scores:
        print(f"   Detection score range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"   Average detection score: {sum(scores)/len(scores):.3f}")
    
    # Sample landmarks from first frame with detections
    for frame in frames:
        if frame.landmarks:
            print(f"\n   Sample landmarks from {frame.image_path.name}:")
            for landmark in frame.landmarks[:3]:
                print(f"      - {landmark.name:20s}: ({landmark.x:.3f}, {landmark.y:.3f}, {landmark.z:.3f}) visibility={landmark.visibility:.2f}")
            if len(frame.landmarks) > 3:
                print(f"      ... and {len(frame.landmarks)-3} more")
            break


def print_feature_row_info(feature_rows: List[Dict]) -> None:
    """Print detailed information about extracted features."""
    print(f"\n🎯 Feature Extraction Statistics:")
    print(f"   Total feature rows: {len(feature_rows)}")
    
    if not feature_rows:
        return
    
    first_row = feature_rows[0]
    print(f"   Features per row: {len(first_row)}")
    
    # Categorize features
    angle_features = [k for k in first_row.keys() if 'angle' in k.lower()]
    contact_features = [k for k in first_row.keys() if 'contact' in k.lower()]
    distance_features = [k for k in first_row.keys() if 'distance' in k.lower() or 'reach' in k.lower()]
    motion_features = [k for k in first_row.keys() if 'velocity' in k.lower() or 'acceleration' in k.lower()]
    
    print(f"\n   Feature categories:")
    print(f"      - Angle features: {len(angle_features)}")
    print(f"      - Contact features: {len(contact_features)}")
    print(f"      - Distance/reach features: {len(distance_features)}")
    print(f"      - Motion features: {len(motion_features)}")
    
    if angle_features:
        print(f"\n   Sample angle features:")
        for feat in angle_features[:3]:
            val = first_row.get(feat)
            print(f"      - {feat}: {val}")
    
    if contact_features:
        print(f"\n   Sample contact features:")
        for feat in contact_features[:3]:
            val = first_row.get(feat)
            print(f"      - {feat}: {val}")
    
    # Check for null values
    null_count = 0
    for row in feature_rows:
        null_count += sum(1 for v in row.values() if v is None)
    
    print(f"\n   Data quality:")
    total_values = len(feature_rows) * len(first_row)
    null_pct = 100 * null_count / total_values if total_values > 0 else 0
    print(f"      - Null values: {null_count}/{total_values} ({null_pct:.1f}%)")


def main():
    """Run MediaPipe climber movement analysis."""
    print("🏃 Testing MediaPipe Climber Movement Analysis")
    print("=" * 60)
    
    # Setup paths
    repo_root = Path(__file__).parent
    test_frames_dir = repo_root / "data" / "test_frames" / "IMG_3571"
    manifest_path = test_frames_dir / "manifest.json"
    holds_path = test_frames_dir / "holds.json"
    
    if not manifest_path.exists():
        print(f"❌ manifest.json not found at {manifest_path}")
        return False
    
    print(f"\n✅ Found test data at {test_frames_dir}")
    
    # Step 1: Estimate poses
    print("\n[1/3] Running MediaPipe Pose Estimation...")
    print("-" * 60)
    
    try:
        frames = estimate_poses_from_manifest(
            manifest_path,
            save_json=False,  # Don't overwrite existing results
        )
        print(f"✅ Pose estimation completed")
        print_pose_frame_info(frames)
        
        if len(frames) == 0:
            print("❌ No poses were detected")
            return False
        
        if not any(f.landmarks for f in frames):
            print("⚠️  WARNING: No landmarks detected in any frame")
            print("   This may indicate MediaPipe is not working correctly")
            return False
        
    except ImportError as e:
        print(f"❌ Import error (missing dependency?): {e}")
        return False
    except Exception as e:
        print(f"❌ Pose estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Load holds data
    print("\n[2/3] Loading Hold Detection Data...")
    print("-" * 60)
    
    holds = None
    if holds_path.exists():
        try:
            holds = load_holds_from_json(holds_path)
            if holds:
                print(f"✅ Loaded {len(holds)} holds")
                for hold_id, hold in list(holds.items())[:2]:
                    print(f"   - {hold_id}: ({hold.center_x:.3f}, {hold.center_y:.3f}) label={hold.label}")
                if len(holds) > 2:
                    print(f"   ... and {len(holds)-2} more")
            else:
                print("⚠️  No holds found in holds.json")
        except Exception as e:
            print(f"⚠️  Failed to load holds: {e}")
    else:
        print(f"⚠️  holds.json not found - proceeding without hold data")
    
    # Step 3: Extract features
    print("\n[3/3] Extracting Movement Features...")
    print("-" * 60)
    
    try:
        feature_rows = summarize_features(
            frames,
            holds=holds,
            auto_estimate_wall=False,  # Skip wall angle estimation for speed
        )
        print(f"✅ Feature extraction completed")
        print_feature_row_info(feature_rows)
        
        # Save results for inspection
        output_path = test_frames_dir / "pose_analysis_test.json"
        with open(output_path, 'w') as f:
            # Convert for JSON serialization
            serializable = []
            for row in feature_rows:
                row_copy = {}
                for k, v in row.items():
                    if v is None:
                        row_copy[k] = None
                    elif isinstance(v, (int, float, str, bool)):
                        row_copy[k] = v
                    else:
                        row_copy[k] = str(v)
                serializable.append(row_copy)
            
            json.dump({
                "frame_count": len(feature_rows),
                "feature_count": len(feature_rows[0]) if feature_rows else 0,
                "features": serializable
            }, f, indent=2)
        
        print(f"\n💾 Results saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ MediaPipe climber movement analysis test PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
