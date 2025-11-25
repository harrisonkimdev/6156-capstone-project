"""Automatic weekly training scheduler for models using GCS pose features.

This script is designed to run as a scheduled job (e.g., via cron or cloud scheduler)
to automatically train models on accumulated pose feature data from GCS.

NOTE: This is a future feature and is currently commented out.
Uncomment and configure when ready to enable automatic weekly training.
"""

# from __future__ import annotations
#
# import json
# import logging
# import sys
# from datetime import datetime, timedelta
# from pathlib import Path
# from tempfile import TemporaryDirectory
# from typing import List
#
# ROOT_DIR = Path(__file__).resolve().parents[1]
# SRC_DIR = ROOT_DIR / "src"
# if str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))
#
# from pose_ai.cloud.gcs import get_gcs_manager
# from pose_ai.ml.xgb_trainer import TrainParams, train_from_file
#
# LOGGER = logging.getLogger(__name__)
#
#
# def download_features_from_gcs(
#     gcs_manager,
#     bucket_name: str,
#     prefix: str,
#     days_back: int = 7,
#     output_dir: Path,
# ) -> List[Path]:
#     """Download pose_features.json files from GCS that were created in the last N days.
#
#     Args:
#         gcs_manager: CloudStorageManager instance
#         bucket_name: GCS bucket name containing frame directories
#         prefix: Prefix path in bucket (e.g., "videos/frames")
#         days_back: Number of days to look back for features
#         output_dir: Local directory to download files to
#
#     Returns:
#         List of paths to downloaded pose_features.json files
#     """
#     client = gcs_manager._ensure_client()
#     cutoff_date = datetime.now() - timedelta(days=days_back)
#     downloaded_files = []
#
#     # List all blobs in the frame bucket with pose_features.json pattern
#     blobs = client.list_blobs(bucket_name, prefix=prefix)
#     for blob in blobs:
#         if not blob.name.endswith("pose_features.json"):
#             continue
#         # Check if blob was created/modified within the time window
#         if blob.time_created and blob.time_created >= cutoff_date:
#             # Download to local temp directory
#             local_path = output_dir / blob.name.replace(prefix, "").replace("/", "_")
#             blob.download_to_filename(str(local_path))
#             downloaded_files.append(local_path)
#             LOGGER.info(f"Downloaded {blob.name} -> {local_path}")
#
#     return downloaded_files
#
#
# def merge_feature_files(feature_files: List[Path], output_path: Path) -> Path:
#     """Merge multiple pose_features.json files into a single file.
#
#     Args:
#         feature_files: List of paths to pose_features.json files
#         output_path: Path to write merged features to
#
#     Returns:
#         Path to merged features file
#     """
#     all_features = []
#     for file_path in feature_files:
#         try:
#             data = json.loads(file_path.read_text(encoding="utf-8"))
#             if isinstance(data, list):
#                 all_features.extend(data)
#             else:
#                 LOGGER.warning(f"Unexpected format in {file_path}, skipping")
#         except Exception as exc:
#             LOGGER.error(f"Failed to load {file_path}: {exc}")
#             continue
#
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     output_path.write_text(json.dumps(all_features, indent=2), encoding="utf-8")
#     LOGGER.info(f"Merged {len(all_features)} feature rows into {output_path}")
#     return output_path
#
#
# def run_weekly_training(
#     *,
#     frame_bucket: str,
#     frame_prefix: str = "videos/frames",
#     days_back: int = 7,
#     model_output: str = "models/xgb_auto_weekly.json",
#     upload_to_gcs: bool = True,
# ) -> None:
#     """Run automatic weekly training on GCS pose features.
#
#     This function:
#     1. Downloads pose_features.json files from GCS (last N days)
#     2. Merges them into a single training dataset
#     3. Trains an XGBoost model
#     4. Optionally uploads the model to GCS
#
#     Args:
#         frame_bucket: GCS bucket name containing frame directories
#         frame_prefix: Prefix path in bucket
#         days_back: Number of days to look back for features
#         model_output: Local path to save trained model
#         upload_to_gcs: Whether to upload trained model to GCS
#     """
#     gcs_manager = get_gcs_manager()
#     config = gcs_manager.config
#
#     if not config.frame_bucket:
#         raise ValueError("GCS_FRAME_BUCKET must be configured for automatic training")
#
#     with TemporaryDirectory() as tmpdir:
#         tmp_path = Path(tmpdir)
#
#         # Step 1: Download features from GCS
#         LOGGER.info(f"Downloading features from gs://{frame_bucket}/{frame_prefix} (last {days_back} days)")
#         feature_files = download_features_from_gcs(
#             gcs_manager,
#             bucket_name=frame_bucket,
#             prefix=frame_prefix,
#             days_back=days_back,
#             output_dir=tmp_path,
#         )
#
#         if not feature_files:
#             LOGGER.warning("No feature files found in the specified time window. Skipping training.")
#             return
#
#         # Step 2: Merge features
#         merged_path = tmp_path / "merged_features.json"
#         merge_feature_files(feature_files, merged_path)
#
#         # Step 3: Train model
#         LOGGER.info(f"Training model on {len(feature_files)} feature files")
#         params = TrainParams(
#             task="regression",
#             label_column="detection_score",
#             label_threshold=None,
#             test_size=0.2,
#             random_state=42,
#             model_out=Path(model_output),
#         )
#         metrics = train_from_file(merged_path, params)
#         LOGGER.info(f"Training completed. Metrics: {metrics}")
#
#         # Step 4: Upload to GCS if requested
#         if upload_to_gcs:
#             job_id = f"auto_weekly_{datetime.now().strftime('%Y%m%d')}"
#             model_uri = gcs_manager.upload_model(params.model_out, job_id=job_id)
#             LOGGER.info(f"Uploaded model to {model_uri}")
#
#
# if __name__ == "__main__":
#     # Example usage (commented out):
#     # run_weekly_training(
#     #     frame_bucket="your-frame-bucket",
#     #     frame_prefix="videos/frames",
#     #     days_back=7,
#     #     model_output="models/xgb_auto_weekly.json",
#     #     upload_to_gcs=True,
#     # )
#     pass

