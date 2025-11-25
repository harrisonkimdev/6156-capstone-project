from __future__ import annotations

from pathlib import Path
from typing import Dict

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pose_ai.cloud.gcs import get_gcs_manager
from pose_ai.ml.xgb_trainer import params_from_dict, train_from_file  # type: ignore
from webapp.training_jobs import TrainJob

GCS_MANAGER = get_gcs_manager()


def execute_training(job: TrainJob) -> None:
    job.start()
    try:
        path = Path(job.features_path)
        # map incoming job params to our trainer params
        params = params_from_dict(job.params)
        metrics = train_from_file(path, params)
        
        # GCS upload is optional - only if upload_to_gcs is True
        model_uri = None
        upload_to_gcs = job.params.get("upload_to_gcs", False)
        if upload_to_gcs:
            model_uri = GCS_MANAGER.upload_model(params.model_out, job_id=job.id)
            job.log(f"Uploaded model to {model_uri}")
        
        job.complete(metrics=metrics, model_path=str(params.model_out), model_uri=model_uri)
        job.log("Training completed")
    except Exception as exc:  # pragma: no cover
        job.fail(exc)
        job.log(f"Training failed: {exc}")
