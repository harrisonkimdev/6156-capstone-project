#!/usr/bin/env python3
"""
Analysis Report: MediaPipe Load Visualization Results

Generate a comprehensive report of the biomechanical analysis results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def analyze_features(features_path: Path) -> Dict:
    """Analyze feature statistics across all frames."""
    with open(features_path) as f:
        features_data = json.load(f)
    
    features_list = features_data if isinstance(features_data, list) else features_data.get("rows", [])
    
    if not features_list:
        return {}
    
    # Extract numeric columns
    numeric_columns = {}
    
    for feature_dict in features_list:
        for key, value in feature_dict.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key not in numeric_columns:
                    numeric_columns[key] = []
                numeric_columns[key].append(float(value))
    
    # Compute statistics
    stats = {}
    angle_features = []
    contact_features = []
    velocity_features = []
    
    for col_name, values in numeric_columns.items():
        if not values or len(values) == 0:
            continue
        
        valid_values = [v for v in values if v is not None and not (isinstance(v, float) and str(v) == 'nan')]
        if not valid_values:
            continue
        
        col_stats = {
            "min": min(valid_values),
            "max": max(valid_values),
            "mean": statistics.mean(valid_values),
            "median": statistics.median(valid_values),
            "count": len(valid_values),
        }
        
        if len(valid_values) > 1:
            col_stats["stdev"] = statistics.stdev(valid_values)
        
        stats[col_name] = col_stats
        
        # Categorize
        if "angle" in col_name.lower():
            angle_features.append(col_name)
        elif "contact" in col_name.lower():
            contact_features.append(col_name)
        elif "velocity" in col_name.lower() or "_v" in col_name.lower():
            velocity_features.append(col_name)
    
    return {
        "total_features": len(numeric_columns),
        "total_frames": len(features_list),
        "angle_features": angle_features,
        "contact_features": contact_features,
        "velocity_features": velocity_features,
        "statistics": stats,
    }


def generate_report(test_dir: Path) -> str:
    """Generate comprehensive analysis report."""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("📊 MEDIAPIPE LOAD VISUALIZATION ANALYSIS REPORT")
    report_lines.append("=" * 80)
    
    # File summary
    report_lines.append("\n📁 Files Generated:")
    report_lines.append("-" * 80)
    
    pose_results = test_dir / "pose_results.json"
    features_file = test_dir / "pose_features.json"
    viz_dir = test_dir / "visualized_load"
    
    if pose_results.exists():
        with open(pose_results) as f:
            pose_data = json.load(f)
        num_frames = len(pose_data.get("frames", []))
        report_lines.append(f"✓ pose_results.json ({num_frames} frames)")
    
    if features_file.exists():
        report_lines.append(f"✓ pose_features.json")
    
    if viz_dir.exists():
        viz_images = list(viz_dir.glob("*_viz.jpg"))
        report_lines.append(f"✓ visualized_load/ ({len(viz_images)} visualization images)")
    
    # Feature analysis
    if features_file.exists():
        report_lines.append("\n🎯 Feature Analysis:")
        report_lines.append("-" * 80)
        
        analysis = analyze_features(features_file)
        
        if analysis:
            report_lines.append(f"Total Features: {analysis.get('total_features', 'N/A')}")
            report_lines.append(f"Total Frames: {analysis.get('total_frames', 'N/A')}")
            
            angle_features = analysis.get("angle_features", [])
            contact_features = analysis.get("contact_features", [])
            velocity_features = analysis.get("velocity_features", [])
            
            if angle_features:
                report_lines.append(f"\nJoint Angle Features ({len(angle_features)}):")
                for feat in angle_features[:5]:
                    stats = analysis["statistics"].get(feat, {})
                    if stats:
                        report_lines.append(f"  • {feat:30s}: {stats.get('mean', 0):.2f}° (range: {stats.get('min', 0):.2f}° - {stats.get('max', 0):.2f}°)")
                if len(angle_features) > 5:
                    report_lines.append(f"  ... and {len(angle_features) - 5} more")
            
            if velocity_features:
                report_lines.append(f"\nMovement Velocity Features ({len(velocity_features)}):")
                for feat in velocity_features[:3]:
                    stats = analysis["statistics"].get(feat, {})
                    if stats:
                        report_lines.append(f"  • {feat:30s}: {stats.get('mean', 0):.4f} (max: {stats.get('max', 0):.4f})")
                if len(velocity_features) > 3:
                    report_lines.append(f"  ... and {len(velocity_features) - 3} more")
    
    # Visualization modes
    report_lines.append("\n🎨 Visualization Features:")
    report_lines.append("-" * 80)
    report_lines.append("✓ Limb Load Visualization (Green→Yellow→Red gradient)")
    report_lines.append("  - Green (0.0): Low biomechanical load (efficient movement)")
    report_lines.append("  - Yellow (0.5): Moderate load")
    report_lines.append("  - Red (1.0): High load (demanding/challenging movement)")
    report_lines.append("\n✓ Center of Mass Analysis")
    report_lines.append("  - Crosshair: COM position in frame")
    report_lines.append("  - Box: COM stability region")
    report_lines.append("  - Trail: COM trajectory (balance mode)")
    report_lines.append("\n✓ Support Base Analysis")
    report_lines.append("  - Polygon: Convex hull of contact points")
    report_lines.append("  - Vertices: Hand/foot contact points")
    
    # Interpretation guide
    report_lines.append("\n💡 Interpretation Guide:")
    report_lines.append("-" * 80)
    report_lines.append("The load visualization uses biomechanical analysis to estimate")
    report_lines.append("the forces applied through each limb during climbing:")
    report_lines.append("")
    report_lines.append("1. LOAD DISTRIBUTION")
    report_lines.append("   • Estimated from contact status + COM projection distance")
    report_lines.append("   • Adjusted by movement velocity and joint visibility")
    report_lines.append("   • Shows muscular effort required at each contact point")
    report_lines.append("")
    report_lines.append("2. CENTER OF MASS (COM)")
    report_lines.append("   • Computed from weighted joint positions")
    report_lines.append("   • Distance from support base indicates balance difficulty")
    report_lines.append("   • Trail shows COM trajectory through movement sequence")
    report_lines.append("")
    report_lines.append("3. SUPPORT POLYGON")
    report_lines.append("   • Convex hull of hand/foot contact points")
    report_lines.append("   • COM should remain inside polygon for stability")
    report_lines.append("   • Larger polygon = more stable base")
    report_lines.append("   • Color intensity shows COM deviation from center")
    report_lines.append("")
    report_lines.append("4. EFFICIENCY INSIGHTS")
    report_lines.append("   • Consistent green indicates energy-efficient technique")
    report_lines.append("   • Red spikes suggest challenging holds/positions")
    report_lines.append("   • Yellow shows controlled movement transitions")
    
    # Next steps
    report_lines.append("\n🚀 Next Steps:")
    report_lines.append("-" * 80)
    report_lines.append("1. Review visualization images in visualized_load/")
    report_lines.append("2. Identify high-load frames (red) for technique improvement")
    report_lines.append("3. Analyze hold transition patterns from feature data")
    report_lines.append("4. Use insights to refine hold_filters.py logic")
    report_lines.append("5. Integrate load scoring into efficiency model")
    
    report_lines.append("\n" + "=" * 80)
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    repo_root = Path(__file__).parent
    test_dir = repo_root / "data" / "test_frames" / "IMG_3571"
    
    report = generate_report(test_dir)
    print(report)
    
    # Save report
    report_path = test_dir / "analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"\n💾 Report saved to: {report_path}")
