from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Literal, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("xgboost is required. Install with `pip install xgboost`.") from exc


Task = Literal["classification", "regression"]


@dataclass
class TrainParams:
    task: Task = "classification"
    label_column: str = "detection_score"
    label_threshold: float | None = 0.6
    drop_columns: tuple[str, ...] = ("image_path",)
    test_size: float = 0.2
    random_state: int = 42

    # XGBoost hyperparameters (shared sensible defaults)
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: float | None = None  # classification only (optional)
    n_jobs: int = 0
    tree_method: str | None = None  # e.g., "hist" or "gpu_hist"

    # Training behaviour
    early_stopping_rounds: int | None = 30
    eval_metric_cls: str = "logloss"
    eval_metric_reg: str = "rmse"

    # Outputs
    model_out: Path = field(default_factory=lambda: Path("models/xgb_pose.json"))
    metrics_out: Path | None = None
    feature_out: Path | None = None
    importance_out: Path | None = None  # CSV of feature importances


def load_features(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("feature file contains no rows")
    return df


def prepare_features(df: pd.DataFrame, drop_columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in drop_columns:
        if col in df.columns:
            df = df.drop(columns=col)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)
    return df.fillna(0.0)


def _build_model(params: TrainParams, task: Task):
    common = dict(
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        max_depth=params.max_depth,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        random_state=params.random_state,
        n_jobs=params.n_jobs,
    )
    if params.tree_method:
        common["tree_method"] = params.tree_method

    if task == "classification":
        if params.label_threshold is None:
            raise ValueError("label_threshold is required for classification tasks")
        if params.scale_pos_weight is not None:
            common["scale_pos_weight"] = params.scale_pos_weight
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric=params.eval_metric_cls, **common)
    else:
        return xgb.XGBRegressor(objective="reg:squarederror", eval_metric=params.eval_metric_reg, **common)


def _evaluate(task: Task, y_true, y_pred) -> Dict[str, float]:
    from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error  # type: ignore

    if task == "classification":
        report = classification_report(y_true, y_pred, output_dict=True)
        return {
            "task": 0.0 if task == "classification" else 1.0,  # typed float for JSON stability
            "accuracy": float(report.get("accuracy", float("nan"))),
            "precision": float(report.get("1", {}).get("precision", float("nan"))),
            "recall": float(report.get("1", {}).get("recall", float("nan"))),
            "f1": float(report.get("1", {}).get("f1-score", float("nan"))),
        }
    else:
        return {
            "task": 1.0,
            "mse": float(mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }


def train_from_dataframe(df: pd.DataFrame, params: TrainParams) -> Tuple[object, Dict[str, float]]:
    if params.label_column not in df.columns:
        raise ValueError(f"label column {params.label_column} not present in features")

    y_raw = df[params.label_column]
    X = prepare_features(df.drop(columns=[params.label_column]), params.drop_columns)

    from sklearn.model_selection import train_test_split  # type: ignore

    task: Task = params.task
    if task == "classification":
        y = (y_raw >= float(params.label_threshold)).astype(int)
        stratify = y
    else:
        y = y_raw.astype(float)
        stratify = None

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=params.test_size,
        random_state=params.random_state,
        stratify=stratify,
    )

    model = _build_model(params, task)
    eval_set = [(X_val, y_val)]
    fit_kwargs: Dict[str, object] = {"eval_set": eval_set, "verbose": False}
    if params.early_stopping_rounds is not None and params.early_stopping_rounds > 0:
        fit_kwargs["early_stopping_rounds"] = int(params.early_stopping_rounds)

    model.fit(X_train, y_train, **fit_kwargs)
    preds = model.predict(X_val)
    metrics = _evaluate(task, y_val, preds)

    # Optional outputs
    if params.importance_out:
        try:
            importances = getattr(model, "feature_importances_", None)
            if importances is not None:
                imp_df = pd.DataFrame({"feature": list(X.columns), "importance": importances})
                params.importance_out.parent.mkdir(parents=True, exist_ok=True)
                imp_df.to_csv(params.importance_out, index=False)
        except Exception:  # pragma: no cover
            pass

    if params.feature_out:
        export = X.copy()
        export["prediction"] = preds
        params.feature_out.parent.mkdir(parents=True, exist_ok=True)
        export.to_csv(params.feature_out, index=False)

    return model, metrics


def train_from_file(features_path: Path, params: TrainParams) -> Dict[str, float]:
    df = load_features(features_path)
    model, metrics = train_from_dataframe(df, params)

    # Save model and metrics
    if params.model_out:
        params.model_out.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(params.model_out)

    if params.metrics_out:
        params.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        params.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def params_from_dict(d: Dict[str, object]) -> TrainParams:
    # Convert incoming values (possibly from JSON) to TrainParams safely
    return TrainParams(
        task=str(d.get("task", "classification")),
        label_column=str(d.get("label_column", "detection_score")),
        label_threshold=(None if d.get("label_threshold") is None else float(d.get("label_threshold", 0.6))),
        drop_columns=tuple(d.get("drop_columns", ("image_path",))),
        test_size=float(d.get("test_size", 0.2)),
        random_state=int(d.get("random_state", 42)),
        n_estimators=int(d.get("n_estimators", 300)),
        learning_rate=float(d.get("learning_rate", 0.05)),
        max_depth=int(d.get("max_depth", 4)),
        subsample=float(d.get("subsample", 0.8)),
        colsample_bytree=float(d.get("colsample_bytree", 0.8)),
        scale_pos_weight=(None if d.get("scale_pos_weight") is None else float(d.get("scale_pos_weight"))),
        n_jobs=int(d.get("n_jobs", 0)),
        tree_method=(None if d.get("tree_method") in (None, "") else str(d.get("tree_method"))),
        early_stopping_rounds=(None if d.get("early_stopping_rounds") in (None, "") else int(d.get("early_stopping_rounds", 30))),
        eval_metric_cls=str(d.get("eval_metric_cls", "logloss")),
        eval_metric_reg=str(d.get("eval_metric_reg", "rmse")),
        model_out=Path(str(d.get("model_out", "models/xgb_pose.json"))),
        metrics_out=(None if d.get("metrics_out") is None else Path(str(d.get("metrics_out")))),
        feature_out=(None if d.get("feature_out") is None else Path(str(d.get("feature_out")))),
        importance_out=(None if d.get("importance_out") is None else Path(str(d.get("importance_out")))),
    )


__all__ = [
    "TrainParams",
    "train_from_dataframe",
    "train_from_file",
    "load_features",
    "prepare_features",
    "params_from_dict",
]
