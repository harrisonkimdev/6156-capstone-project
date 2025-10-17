"""Train an XGBoost model on pose feature data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("xgboost is required. Install with `pip install xgboost`.") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost on pose feature rows.")
    parser.add_argument("features", type=Path, help="Path to pose_features.json")
    parser.add_argument(
        "--label-column",
        default="detection_score",
        help="Column name to use as the target (default: detection_score).",
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=None,
        help="If set (classification), binarise target via label >= threshold.",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=["image_path"],
        help="Columns to drop before training (default: image_path).",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="Training objective.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/xgb_pose.json"),
        help="Path to save trained model (default: models/xgb_pose.json).",
    )
    parser.add_argument(
        "--feature-out",
        type=Path,
        default=None,
        help="Optional CSV path to store predictions appended to features.",
    )
    return parser.parse_args()


def _load_features(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("feature file contains no rows")
    return df


def _prepare_features(df: pd.DataFrame, drop_columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in drop_columns:
        if col in df.columns:
            df = df.drop(columns=col)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)
    return df


def _train_classifier(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    print(classification_report(y_val, preds))
    report = classification_report(y_val, preds, output_dict=True)
    metrics = {
        "accuracy": report["accuracy"],
        "precision": report.get("1", {}).get("precision", float("nan")),
        "recall": report.get("1", {}).get("recall", float("nan")),
        "f1": report.get("1", {}).get("f1-score", float("nan")),
    }
    return model, metrics


def _train_regressor(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    metrics = {
        "mse": float(mean_squared_error(y_val, preds)),
        "mae": float(mean_absolute_error(y_val, preds)),
    }
    print(f"Validation MSE: {metrics['mse']:.4f}")
    print(f"Validation MAE: {metrics['mae']:.4f}")
    return model, metrics


def main() -> None:
    args = _parse_args()

    df = _load_features(args.features)
    if args.label_column not in df.columns:
        raise ValueError(f"label column {args.label_column} not present in features")

    y_raw = df[args.label_column]
    feature_df = _prepare_features(df.drop(columns=[args.label_column]), args.drop_columns)
    feature_df = feature_df.fillna(0.0)

    if args.task == "classification":
        if args.label_threshold is None:
            raise ValueError("--label-threshold is required for classification tasks")
        y = (y_raw >= args.label_threshold).astype(int)
    else:
        y = y_raw.astype(float)

    stratify = y if args.task == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        feature_df,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    if args.task == "classification":
        model, metrics = _train_classifier(X_train, y_train, X_val, y_val)
    else:
        model, metrics = _train_regressor(X_train, y_train, X_val, y_val)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.model_out)
    print(f"Model saved to {args.model_out}")
    print("Metrics:", metrics)

    if args.feature_out:
        preds = model.predict(feature_df)
        export = feature_df.copy()
        export["prediction"] = preds
        args.feature_out.parent.mkdir(parents=True, exist_ok=True)
        export.to_csv(args.feature_out, index=False)
        print(f"Predictions saved to {args.feature_out}")


if __name__ == "__main__":
    main()
