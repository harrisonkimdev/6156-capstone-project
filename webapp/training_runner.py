from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pose_ai.ml.xgb_trainer import params_from_dict, train_from_file  # type: ignore
from webapp.training_jobs import TrainJob


def execute_training(job: TrainJob) -> None:
    """Execute XGBoost training job.
    
    Always uploads metadata to storage. Model weights and training data are
    uploaded based on job parameters:
    - upload_to_gcs: Whether to upload model weights.
    - upload_training_data: Whether to upload training data.
    
    Args:
        job: TrainJob instance.
    """
    job.start()
    try:
        path = Path(job.features_path)
        # Map incoming job params to our trainer params
        params = params_from_dict(job.params)
        metrics = train_from_file(path, params)
        
        # Prepare metadata
        metadata = {
            "job_id": job.id,
            "model_type": "xgboost",
            "hyperparameters": {
                "task": params.task,
                "n_estimators": params.n_estimators,
                "learning_rate": params.learning_rate,
                "max_depth": params.max_depth,
                "subsample": params.subsample,
                "colsample_bytree": params.colsample_bytree,
                "test_size": params.test_size,
                "random_state": params.random_state,
            },
            "metrics": metrics,
            "model_path": str(params.model_out),
            "features_path": job.features_path,
            "created_at": datetime.now().isoformat(),
        }
        
        # Upload to storage using unified storage manager
        metadata_uri = None
        model_uri = None
        training_data_uri = None
        drive_model_id = None
        drive_metadata_id = None
        
        try:
            from pose_ai.cloud.storage import get_storage_manager
            storage = get_storage_manager()
            
            # Always upload metadata
            job.log("Uploading model metadata to storage...")
            metadata_result = storage.upload_model_metadata(metadata, job_id=job.id)
            metadata_uri = metadata_result.gcs_uri
            drive_metadata_id = metadata_result.drive_id
            if metadata_uri:
                job.log(f"Metadata uploaded to GCS: {metadata_uri}")
            if drive_metadata_id:
                job.log(f"Metadata uploaded to Google Drive: {drive_metadata_id}")
            
            # Optionally upload model weights
            upload_to_gcs = job.params.get("upload_to_gcs", False)
            if upload_to_gcs:
                job.log("Uploading model weights to storage...")
                model_result = storage.upload_model(params.model_out, job_id=job.id)
                model_uri = model_result.gcs_uri
                drive_model_id = model_result.drive_id
                if model_uri:
                    job.log(f"Model uploaded to GCS: {model_uri}")
                if drive_model_id:
                    job.log(f"Model uploaded to Google Drive: {drive_model_id}")
            
            # Optionally upload training data
            upload_training_data = job.params.get("upload_training_data", False)
            if upload_training_data and path.exists():
                job.log("Uploading training data to storage...")
                data_result = storage.upload_training_data(path, job_id=job.id, data_type="features")
                training_data_uri = data_result.gcs_uri
                if training_data_uri:
                    job.log(f"Training data uploaded to GCS: {training_data_uri}")
                if data_result.drive_id:
                    job.log(f"Training data uploaded to Google Drive: {data_result.drive_id}")
            
        except Exception as exc:
            job.log(f"Storage upload failed: {exc}")
        
        job.complete(
            metrics=metrics,
            model_path=str(params.model_out),
            model_uri=model_uri,
            metadata_uri=metadata_uri,
            training_data_uri=training_data_uri,
            drive_model_id=drive_model_id,
            drive_metadata_id=drive_metadata_id,
        )
        job.log("Training completed")
    except Exception as exc:  # pragma: no cover
        job.fail(exc)
        job.log(f"Training failed: {exc}")
