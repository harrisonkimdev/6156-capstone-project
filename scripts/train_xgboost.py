"""Train an XGBoost model on pose feature data (CLI)."""

from __future__ import annotations

import argparse
from pathlib import Path

from pose_ai.ml.xgb_trainer import TrainParams, params_from_dict, train_from_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost on pose feature rows.")
    parser.add_argument("features", type=Path, help="Path to pose_features.json")

    # Data/label
    parser.add_argument("--label-column", default="detection_score")
    parser.add_argument("--label-threshold", type=float, default=None)
    parser.add_argument("--drop-columns", nargs="*", default=["image_path"])
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    # XGBoost hyperparameters
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--scale-pos-weight", type=float, default=None)
    parser.add_argument("--n-jobs", type=int, default=0)
    parser.add_argument("--tree-method", default=None, help="e.g., hist or gpu_hist")

    # Training behaviour
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--eval-metric-cls", default="logloss")
    parser.add_argument("--eval-metric-reg", default="rmse")

    # Outputs
    parser.add_argument("--model-out", type=Path, default=Path("models/xgb_pose.json"))
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--feature-out", type=Path, default=None)
    parser.add_argument("--importance-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    params = params_from_dict(
        {
            "task": args.task,
            "label_column": args.label_column,
            "label_threshold": args.label_threshold,
            "drop_columns": args.drop_columns,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "scale_pos_weight": args.scale_pos_weight,
            "n_jobs": args.n_jobs,
            "tree_method": args.tree_method,
            "early_stopping_rounds": args.early_stopping_rounds,
            "eval_metric_cls": args.eval_metric_cls,
            "eval_metric_reg": args.eval_metric_reg,
            "model_out": args.model_out,
            "metrics_out": args.metrics_out,
            "feature_out": args.feature_out,
            "importance_out": args.importance_out,
        }
    )

    metrics = train_from_file(args.features, params)
    print(f"Model saved to {params.model_out}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
