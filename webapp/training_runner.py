"""Training runner for XGBoost and BiLSTM models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from webapp.training_jobs import TrainJob


def execute_training(job: TrainJob) -> None:
    """Execute training job based on model_type.
    
    Dispatches to appropriate training function based on job.params['model_type'].
    
    Args:
        job: TrainJob instance.
    """
    model_type = job.params.get("model_type", "xgboost")
    
    if model_type == "bilstm":
        execute_bilstm_training(job)
    else:
        execute_xgboost_training(job)


def execute_xgboost_training(job: TrainJob) -> None:
    """Execute XGBoost training job.
    
    Always uploads metadata to storage. Model weights and training data are
    uploaded based on job parameters:
    - upload_to_gcs: Whether to upload model weights.
    - upload_training_data: Whether to upload training data.
    
    Args:
        job: TrainJob instance.
    """
    from pose_ai.ml.xgb_trainer import params_from_dict, train_from_file  # type: ignore

    job.start()
    job.log("Starting XGBoost training...")
    
    try:
        path = Path(job.features_path)
        # Map incoming job params to our trainer params
        params = params_from_dict(job.params)
        job.log(f"Training parameters: n_estimators={params.n_estimators}, "
                f"learning_rate={params.learning_rate}, max_depth={params.max_depth}")
        
        metrics = train_from_file(path, params)
        job.log(f"Training completed. Metrics: {metrics}")
        
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
        job.log("XGBoost training completed successfully")
    except Exception as exc:  # pragma: no cover
        job.fail(exc)
        job.log(f"XGBoost training failed: {exc}")


def execute_bilstm_training(job: TrainJob) -> None:
    """Execute BiLSTM training job.
    
    Trains a BiLSTM model for beta prediction using pose and holds data.
    
    Args:
        job: TrainJob instance.
    """
    job.start()
    job.log("Starting BiLSTM training...")
    
    try:
        # Import required modules
        import json
        
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError:
            raise RuntimeError("PyTorch is required for BiLSTM training. Install with: pip install torch")
        
        from pose_ai.ml.models import BiLSTMMultitaskModel, ModelConfig  # type: ignore
        from pose_ai.ml.train import Trainer, TrainingConfig  # type: ignore
        from pose_ai.ml.dataset import ClimbingWindowDataset, load_features_from_json  # type: ignore
        
        # Load parameters from job
        params = job.params
        features_path = Path(job.features_path)
        
        # Model architecture params
        hidden_dim = params.get("hidden_dim", 128)
        num_layers = params.get("num_layers", 2)
        dropout = params.get("dropout", 0.3)
        use_attention = params.get("use_attention", True)
        
        # Training params
        epochs = params.get("epochs", 100)
        batch_size = params.get("batch_size", 32)
        learning_rate = params.get("learning_rate", 0.001)
        weight_decay = params.get("weight_decay", 0.0001)
        patience = params.get("patience", 10)
        lr_patience = params.get("lr_patience", 5)
        lr_factor = params.get("lr_factor", 0.5)
        device = params.get("device", "cuda")
        
        # Data processing params
        window_size = params.get("window_size", 32)
        stride = params.get("stride", 1)
        train_split = params.get("train_split", 0.7)
        val_split = params.get("val_split", 0.2)
        
        # Output params
        checkpoint_dir = Path(params.get("checkpoint_dir", "models/checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        job.log(f"Model config: hidden_dim={hidden_dim}, num_layers={num_layers}, "
                f"dropout={dropout}, use_attention={use_attention}")
        job.log(f"Training config: epochs={epochs}, batch_size={batch_size}, "
                f"lr={learning_rate}, device={device}")
        
        # Load feature data
        job.log(f"Loading features from {features_path}...")
        if features_path.is_file():
            feature_rows = load_features_from_json(features_path)
        elif features_path.is_dir():
            # Load all JSON files from directory
            feature_rows = []
            for json_path in sorted(features_path.glob("*.json")):
                try:
                    rows = load_features_from_json(json_path)
                    feature_rows.extend(rows)
                    job.log(f"  Loaded {len(rows)} rows from {json_path.name}")
                except Exception as e:
                    job.log(f"  Warning: Failed to load {json_path.name}: {e}")
        else:
            raise FileNotFoundError(f"Features path not found: {features_path}")
        
        if not feature_rows:
            raise ValueError("No feature data loaded")
        
        job.log(f"Total feature rows: {len(feature_rows)}")
        
        # Split data into train/val/test
        total = len(feature_rows)
        train_end = int(total * train_split)
        val_end = int(total * (train_split + val_split))
        
        train_rows = feature_rows[:train_end]
        val_rows = feature_rows[train_end:val_end]
        test_rows = feature_rows[val_end:]
        
        job.log(f"Data split: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")
        
        # Create datasets
        job.log("Creating datasets...")
        train_dataset = ClimbingWindowDataset(
            train_rows,
            window_size=window_size,
            stride=stride,
            video_id="train",
            normalize=True,
        )
        
        val_dataset = ClimbingWindowDataset(
            val_rows,
            window_size=window_size,
            stride=stride,
            video_id="val",
            normalize=False,
        )
        
        # Apply training normalization to validation set
        if train_dataset.feature_mean is not None:
            val_dataset.feature_matrix = (
                val_dataset.feature_matrix - train_dataset.feature_mean
            ) / train_dataset.feature_std
            val_dataset.feature_mean = train_dataset.feature_mean
            val_dataset.feature_std = train_dataset.feature_std
        
        job.log(f"Dataset sizes: train={len(train_dataset)} windows, val={len(val_dataset)} windows")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Create model
        job.log("Creating BiLSTM model...")
        model_config = ModelConfig(
            input_dim=60,  # Feature dimension from dataset
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_action_classes=5,  # 0: no_change, 1-4: limb contacts
            bidirectional=True,
            use_attention=use_attention,
        )
        model = BiLSTMMultitaskModel(model_config)
        
        num_params = sum(p.numel() for p in model.parameters())
        job.log(f"Model created with {num_params:,} parameters")
        
        # Create trainer config
        training_config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            device=device,
            checkpoint_dir=checkpoint_dir,
            log_interval=10,
        )
        
        # Create trainer
        job.log("Setting up trainer...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
        )
        
        # Custom training loop with logging to job
        job.log(f"Starting training for {epochs} epochs on {trainer.device}...")
        
        for epoch in range(epochs):
            # Train epoch
            train_losses = trainer.train_epoch()
            
            # Validate
            val_losses, val_metrics = trainer.validate()
            trainer.val_losses.append(val_losses)
            
            # Log progress
            job.log(f"Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={train_losses['total']:.4f}, "
                    f"val_loss={val_losses['total']:.4f}, "
                    f"eff_mae={val_metrics['eff_mae']:.4f}, "
                    f"action_acc={val_metrics['action_acc']:.4f}")
            
            # Learning rate scheduling
            trainer.scheduler.step(val_losses['total'])
            
            # Save best model
            if val_losses['total'] < trainer.best_val_loss:
                trainer.best_val_loss = val_losses['total']
                trainer.best_epoch = epoch + 1
                trainer.save_checkpoint("best_model.pt", val_metrics)
                job.log(f"  -> New best model saved (loss={trainer.best_val_loss:.4f})")
            
            # Early stopping check
            if trainer.early_stopping(val_losses['total']):
                job.log(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Final metrics
        final_val_losses, final_val_metrics = trainer.validate()
        
        metrics = {
            "best_epoch": trainer.best_epoch,
            "best_val_loss": trainer.best_val_loss,
            "final_eff_mae": final_val_metrics["eff_mae"],
            "final_eff_rmse": final_val_metrics["eff_rmse"],
            "final_eff_corr": final_val_metrics["eff_corr"],
            "final_action_acc": final_val_metrics["action_acc"],
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        }
        
        job.log(f"Training complete! Best epoch: {trainer.best_epoch}, "
                f"Best loss: {trainer.best_val_loss:.4f}")
        
        # Save final model
        final_model_path = checkpoint_dir / "bilstm_multitask.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": model_config.input_dim,
                "hidden_dim": model_config.hidden_dim,
                "num_layers": model_config.num_layers,
                "dropout": model_config.dropout,
                "num_action_classes": model_config.num_action_classes,
                "bidirectional": model_config.bidirectional,
                "use_attention": model_config.use_attention,
            },
            "normalization": {
                "mean": train_dataset.feature_mean.tolist() if train_dataset.feature_mean is not None else None,
                "std": train_dataset.feature_std.tolist() if train_dataset.feature_std is not None else None,
            },
            "metrics": metrics,
        }, final_model_path)
        job.log(f"Final model saved to {final_model_path}")
        
        # Prepare metadata
        metadata = {
            "job_id": job.id,
            "model_type": "bilstm",
            "hyperparameters": {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "use_attention": use_attention,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "patience": patience,
                "window_size": window_size,
                "stride": stride,
            },
            "metrics": metrics,
            "model_path": str(final_model_path),
            "features_path": job.features_path,
            "created_at": datetime.now().isoformat(),
        }
        
        # Upload to storage
        metadata_uri = None
        model_uri = None
        drive_model_id = None
        drive_metadata_id = None
        
        try:
            from pose_ai.cloud.storage import get_storage_manager
            storage = get_storage_manager()
            
            # Upload metadata
            job.log("Uploading model metadata to storage...")
            metadata_result = storage.upload_model_metadata(metadata, job_id=job.id)
            metadata_uri = metadata_result.gcs_uri
            drive_metadata_id = metadata_result.drive_id
            if metadata_uri:
                job.log(f"Metadata uploaded to GCS: {metadata_uri}")
            
            # Optionally upload model weights
            upload_to_gcs = params.get("upload_to_gcs", False)
            if upload_to_gcs:
                job.log("Uploading model weights to storage...")
                model_result = storage.upload_model(final_model_path, job_id=job.id)
                model_uri = model_result.gcs_uri
                drive_model_id = model_result.drive_id
                if model_uri:
                    job.log(f"Model uploaded to GCS: {model_uri}")
                    
        except Exception as exc:
            job.log(f"Storage upload failed: {exc}")
        
        job.complete(
            metrics=metrics,
            model_path=str(final_model_path),
            model_uri=model_uri,
            metadata_uri=metadata_uri,
            drive_model_id=drive_model_id,
            drive_metadata_id=drive_metadata_id,
        )
        job.log("BiLSTM training completed successfully")
        
    except Exception as exc:  # pragma: no cover
        import traceback
        job.log(f"BiLSTM training failed: {exc}")
        job.log(traceback.format_exc())
        job.fail(exc)
