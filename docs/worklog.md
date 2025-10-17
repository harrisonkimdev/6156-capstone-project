# Worklog (English Summary)

## 1) What This Project Does
- Goal: Analyze short-form bouldering videos to extract pose data, segment the climb into meaningful phases, compute metrics, and prepare datasets for training recommendation/efficiency models.
- Outputs: Modular Python library (`pose_ai`), CLI scripts, end-to-end pipeline, visualization, and a baseline XGBoost trainer.

## 2) Architecture Overview
- Data (`pose_ai.data`): video→frame extraction (manifest), route/hold utilities, manifest→FrameMetrics adapters.
- Pose (`pose_ai.pose`): MediaPipe wrapper and landmark smoothing.
- Features (`pose_ai.features`): frame-level features (joint angles, COM, hold proximity) and segment-level aggregation.
- Segmentation (`pose_ai.segmentation`): rule-based rest/movement with dynamic thresholds; helpers to derive FrameMetrics from frame features.
- Service (`pose_ai.service`): orchestration for pose→features→segment reports.
- Scripts: one-click pipeline runner, stage CLIs, pose visualization, and XGBoost trainer.

## 3) Data Artifacts
- `manifest.json`: extracted frames with timestamps and file names.
- `pose_results.json`: per-frame MediaPipe landmarks + detection score.
- `pose_features.json`: per-frame derived features (angles/COM/hold).
- `segment_metrics.json`: per-segment metrics (COM displacement/path, joint ranges/velocity, contact changes).
- `visualized/*.jpg`: frames overlaid with landmarks and connections.

## 4) Milestones Completed
- Frame extraction + manifest generation.
- Pose scaffold + smoothing; integration points for MediaPipe.
- Frame-level feature extraction and configuration.
- Rule-based segmentation upgraded with dynamic thresholds and feature-derived motion.
- Segment metrics + segment report generation.
- One-click pipeline; pose visualization with confidence thresholds.
- XGBoost baseline trainer for quick tabular experiments.

## 5) How To Run (conda env: `6156-capstone`)
- End-to-end pipeline:
  `conda run -n 6156-capstone env PYTHONPATH=src python scripts/run_pipeline.py BetaMove/videos --out data/frames --interval 1.5`
- Visualization (filter low confidence):
  `conda run -n 6156-capstone env PYTHONPATH=src python scripts/visualize_pose.py data/frames/<video>/pose_results.json --min-score 0.3 --min-visibility 0.2`
- Train XGBoost baseline:
  `conda run -n 6156-capstone env PYTHONPATH=src python scripts/train_xgboost.py data/frames/<video>/pose_features.json --label-column detection_score --label-threshold 0.6 --model-out models/xgb_pose.json`

## 6) Next Steps
- Replace provisional labels with human ratings or success/failure labels; expand to segment-level learning.
- Tune segmentation thresholds per wall/video; consider hold definition files for accurate contact analysis.
- Add sequence models (LSTM/Temporal CNN/Transformers) when moving beyond tabular baselines.
