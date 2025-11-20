#!/usr/bin/env python3
"""Train XGBoost model for route difficulty estimation (V0-V10).

Usage:
    python scripts/train_route_grader.py \
        --data data/routes_annotated \
        --model-out models/route_grader.json \
        --epochs 100
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except ImportError:
    print("Error: Required packages not found. Install with: pip install xgboost scikit-learn numpy")
    sys.exit(1)

from pose_ai.ml.route_grading import extract_route_features
from pose_ai.recommendation.efficiency import StepEfficiencyResult
from pose_ai.segmentation.steps import StepSegment


def load_route_data(route_dir: Path) -> tuple[list[dict], list[float]]:
    """Load route data with ground-truth grades.
    
    Expected directory structure:
    route_dir/
        route1/
            pose_features.json
            step_efficiency.json
            holds.json
            grade.txt  # Ground truth grade (V0-V10)
        route2/
            ...
    
    Args:
        route_dir: Directory containing route subdirectories
    
    Returns:
        (features_list, grades_list)
    """
    features_list = []
    grades_list = []
    
    for route_path in sorted(route_dir.iterdir()):
        if not route_path.is_dir():
            continue
        
        # Load ground truth grade
        grade_file = route_path / "grade.txt"
        if not grade_file.exists():
            print(f"Warning: No grade.txt in {route_path}, skipping")
            continue
        
        try:
            grade = float(grade_file.read_text().strip())
            if not (0.0 <= grade <= 10.0):
                print(f"Warning: Invalid grade {grade} in {route_path}, skipping")
                continue
        except (ValueError, TypeError):
            print(f"Warning: Could not parse grade in {route_path}, skipping")
            continue
        
        # Load pipeline outputs
        features_file = route_path / "pose_features.json"
        efficiency_file = route_path / "step_efficiency.json"
        holds_file = route_path / "holds.json"
        
        if not features_file.exists():
            print(f"Warning: No pose_features.json in {route_path}, skipping")
            continue
        
        try:
            # Load feature rows
            with open(features_file, "r") as f:
                feature_rows = json.load(f)
            
            # Load step segments (from feature rows metadata or separate file)
            step_segments = []
            step_efficiency = []
            
            if efficiency_file.exists():
                with open(efficiency_file, "r") as f:
                    efficiency_data = json.load(f)
                    for i, eff_dict in enumerate(efficiency_data):
                        step_segments.append(StepSegment(
                            step_id=eff_dict.get("step_id", i),
                            start_index=0,  # Will be inferred if needed
                            end_index=0,
                            start_time=eff_dict.get("start_time", 0.0),
                            end_time=eff_dict.get("end_time", 0.0),
                            label=eff_dict.get("label", "unknown"),
                        ))
                        step_efficiency.append(StepEfficiencyResult(
                            step_id=eff_dict.get("step_id", i),
                            score=eff_dict.get("score", 0.0),
                            components=eff_dict.get("components", {}),
                            start_time=eff_dict.get("start_time", 0.0),
                            end_time=eff_dict.get("end_time", 0.0),
                        ))
            
            # Load holds
            holds = []
            if holds_file.exists():
                with open(holds_file, "r") as f:
                    holds_data = json.load(f)
                    if isinstance(holds_data, dict):
                        # Old format: dict of hold_id -> hold_data
                        holds = list(holds_data.values())
                    elif isinstance(holds_data, list):
                        # New format: list of holds
                        holds = holds_data
            
            # Extract wall angle from feature rows
            wall_angle = None
            for row in feature_rows:
                angle = row.get("wall_angle")
                if angle is not None:
                    wall_angle = float(angle)
                    break
            
            # Extract route features
            route_features = extract_route_features(
                feature_rows=feature_rows,
                step_segments=step_segments,
                step_efficiency=step_efficiency,
                holds=holds,
                wall_angle=wall_angle,
            )
            
            features_list.append(route_features)
            grades_list.append(grade)
            
        except Exception as e:
            print(f"Error processing {route_path}: {e}")
            continue
    
    return features_list, grades_list


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model for route difficulty estimation")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory containing route subdirectories with annotated grades"
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/route_grader.json"),
        help="Output path for trained model (default: models/route_grader.json)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size ratio (default: 0.2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of boosting rounds (default: 100)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth (default: 6)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=10,
        help="Early stopping rounds (default: 10)"
    )
    parser.add_argument(
        "--importance-out",
        type=Path,
        help="Optional path to save feature importance CSV"
    )
    
    args = parser.parse_args()
    
    if not args.data.exists():
        print(f"Error: Data directory not found: {args.data}")
        sys.exit(1)
    
    print("=" * 60)
    print("Route Difficulty Model Training")
    print("=" * 60)
    
    # Load data
    print("\nLoading route data...")
    features_list, grades_list = load_route_data(args.data)
    
    if not features_list:
        print("Error: No valid route data found")
        sys.exit(1)
    
    print(f"Loaded {len(features_list)} routes")
    
    # Convert to arrays
    feature_names = sorted(features_list[0].keys())
    X = np.array([[f.get(name, 0.0) for name in feature_names] for f in features_list])
    y = np.array(grades_list)
    
    print(f"Features: {len(feature_names)}")
    print(f"Feature names: {', '.join(feature_names[:10])}..." if len(feature_names) > 10 else f"Feature names: {', '.join(feature_names)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    print(f"\nTrain set: {len(X_train)} routes")
    print(f"Test set: {len(X_test)} routes")
    
    # Train model
    print("\nTraining XGBoost model...")
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
    )
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=True,
    )
    
    # Evaluate
    print("\nEvaluating model...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nTraining Metrics:")
    print(f"  RMSE: {train_rmse:.3f}")
    print(f"  MAE:  {train_mae:.3f}")
    print(f"  R²:   {train_r2:.3f}")
    
    print("\nTest Metrics:")
    print(f"  RMSE: {test_rmse:.3f}")
    print(f"  MAE:  {test_mae:.3f}")
    print(f"  R²:   {test_r2:.3f}")
    
    # Save model
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.model_out))
    print(f"\nModel saved to: {args.model_out}")
    
    # Save feature importance
    if args.importance_out:
        importances = model.feature_importances_
        importance_data = [
            {"feature": name, "importance": float(imp)}
            for name, imp in zip(feature_names, importances)
        ]
        importance_data.sort(key=lambda x: x["importance"], reverse=True)
        
        args.importance_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.importance_out, "w") as f:
            json.dump(importance_data, f, indent=2)
        print(f"Feature importance saved to: {args.importance_out}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

