#!/usr/bin/env python3
"""
Generate visualizations for climbing motion analysis in various modes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src directory to Python path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.visualize_pose import visualize_pose_results


def main():
    parser = argparse.ArgumentParser(description="Generate various analysis visualizations for climbing motion.")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("data/workflow_frames/test_video_frames/IMG_3571_converted"),
        help="Path to directory containing frames to analyze"
    )
    parser.add_argument(
        "--mode",
        choices=["load", "balance", "dynamics", "strategy"],
        default="load",
        help="Visualization mode (default: load)"
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Output directory suffix (default: use mode name)"
    )
    
    args = parser.parse_args()
    
    frames_dir = ROOT_DIR / args.frames_dir
    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}")
        sys.exit(1)
    
    pose_results_path = frames_dir / "pose_results.json"
    features_path = frames_dir / "pose_features.json"
    
    if not pose_results_path.exists():
        print(f"Error: pose_results.json not found in {frames_dir}")
        print("Please run analyze_workflow_frames.py first.")
        sys.exit(1)
    
    # Configure output directory
    output_suffix = args.output_suffix or args.mode
    output_dir = frames_dir / f"visualized_{output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Mode: {args.mode}")
    print(f"Input: {frames_dir}")
    print(f"Output: {output_dir}")
    print(f"Features: {'Available' if features_path.exists() else 'Not available'}")
    
    # Mode-specific configurations
    mode_configs = {
        "load": {
            "draw_com": True,
            "description": "Display load on each limb with color coding (green=low, red=high)"
        },
        "balance": {
            "draw_com": True,
            "description": "Analyze balance between center of mass (COM) and support points to evaluate stability"
        },
        "dynamics": {
            "draw_com": True,
            "description": "Visualize movement vectors and dynamic changes to analyze motion patterns"
        },
        "strategy": {
            "draw_com": False,
            "description": "Analyze climbing strategy and movement paths to evaluate efficiency"
        }
    }
    
    config = mode_configs.get(args.mode, mode_configs["load"])
    
    print(f"{config['description']}")
    print("\nProcessing...")
    
    try:
        processed_count = visualize_pose_results(
            pose_results_path=pose_results_path,
            output_dir=output_dir,
            include_missing=False,
            min_score=0.2,
            min_visibility=0.2,
            draw_com=config["draw_com"],
            com_box_scale=0.3,
            mode=args.mode,
            features_path=features_path if features_path.exists() else None
        )
        
        print(f"\nComplete! Generated {processed_count} visualization images.")
        
        # Check generated image files
        viz_images = list(output_dir.glob("*_viz.jpg"))
        print(f"Total {len(viz_images)} images saved:")
        print(f"   {output_dir}")
        
        # Display sample images
        for i, img_path in enumerate(sorted(viz_images)[:3]):
            print(f"   {img_path.name}")
        if len(viz_images) > 3:
            print(f"   ... and {len(viz_images) - 3} more files")
            
        # Suggest other modes
        other_modes = [m for m in mode_configs.keys() if m != args.mode]
        if other_modes:
            print(f"\nTry other analysis modes:")
            for mode in other_modes:
                print(f"   python generate_visualizations.py --mode {mode}")
                print(f"      -> {mode_configs[mode]['description']}")
                
    except Exception as e:
        print(f"Error during visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()