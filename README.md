# 6156-capstone-project

Climbing video analysis system with pose estimation, hold detection, efficiency scoring, and next-action recommendations.

**📖 Complete Documentation**: See [PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md) for comprehensive system overview, API reference, and implementation details.

**🎯 Project Planning**:

- [Project Backlog](https://docs.google.com/spreadsheets/d/113DbJu6Vg53PxX8Kgu5pkwsuLrqYH-i9JJcEuDvWjus/edit?gid=1139408620#gid=1139408620)
- [Sprint Backlog](https://github.com/users/harrisonkimdev/projects/9/views/1?sortedBy%5Bdirection%5D=asc&sortedBy%5BcolumnId%5D=221588758&sortedBy%5Bdirection%5D=asc&sortedBy%5BcolumnId%5D=Assignees)
- [Implementation Backlog](docs/IMPLEMENTATION_BACKLOG.md) — Detailed feature roadmap

---

## Features

- **Pose Estimation**: MediaPipe 33-landmark detection
- **Hold Detection**: YOLOv8n/m with DBSCAN clustering and temporal tracking
- **Hold Type Classification**: YOLOv8 fine-tuning for hold types (crimp, sloper, jug, pinch, foot_only, volume)
- **Wall Angle**: Automatic estimation (Hough + PCA)
- **Efficiency Scoring**: 7-component physics-based metric with technique bonuses
- **Next-Action Recommendations**: Rule-based planner with hold type awareness
- **ML Models**: BiLSTM and Transformer multitask models (efficiency + next-action)
- **Web UI**: FastAPI with real-time job status and results
- **Cloud Storage**: Optional GCS integration for videos/frames/models
- **Training UI**: XGBoost model training with parameter tuning

---

## Setup

### Conda Environment (Recommended)

Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate 6156-capstone
```

Update an existing environment:

```bash
conda env update -f environment.yml --prune
```

**Alternative** (if `environment.yml` is not available):

```bash
conda create -n 6156-capstone python=3.10 -y
conda activate 6156-capstone
pip install -r requirements.txt
```

### Jupyter Kernel (Optional)

Register a Jupyter kernel for notebooks:

```bash
conda activate 6156-capstone
python -m ipykernel install --user --name 6156-capstone --display-name "6156 (py3.10)"
```

**Or use Make** (one-shot setup with venv):

```bash
make init VENV=.venv PYTHON=python3.10 KERNEL_NAME=6156-capstone DISPLAY_NAME="6156 (py3.10)"
```

---

## Quick Start

All commands assume the project root and a Conda env named `6156-capstone`:

```bash
conda run -n 6156-capstone env PYTHONPATH=src <command>
```

### Run Complete Pipeline

Process videos through frame extraction, pose estimation, hold detection, wall angle estimation, feature engineering, segmentation, and efficiency scoring:

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/run_pipeline.py BetaMove/videos --out data/frames --interval 1.5
```

### Run Individual Stages

#### 1. Extract Frames

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/extract_frames.py BetaMove/videos --output data/frames --interval 1.5
```

#### 2. Pose Estimation

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/run_pose_estimation.py --frames-root data/frames
```

#### 3. Export Features

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/run_feature_export.py data/frames/<video>/manifest.json
```

#### 4. Generate Segment Metrics

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/run_segment_report.py data/frames/<video>/manifest.json
```

#### 5. Visualize Pose

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/visualize_pose.py data/frames/<video>/pose_results.json --min-score 0.3 --min-visibility 0.2
```

#### 6. Run Tests

```bash
conda run -n 6156-capstone env PYTHONPATH=src pytest tests/unit
```

---

## ML Model Training

### Hold Type Annotation & Classification

#### Annotate Holds

Interactive tool for labeling hold types:

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/annotate_holds.py \
  data/raw_images \
  --output data/holds_training
```

Controls: `c`=crimp, `s`=sloper, `j`=jug, `p`=pinch, `f`=foot_only, `v`=volume, `u`=undo, `n`=next, `q`=quit

#### Train YOLOv8 for Hold Types

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/train_yolo_holds.py \
  --data data/holds_training/dataset.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device cuda
```

Output: `runs/detect/train/weights/best.pt`

### Sequence Models (BiLSTM & Transformer)

Train temporal models for efficiency regression and next-action classification:

#### Train BiLSTM

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/train_model.py \
  --data data/features \
  --model-type bilstm \
  --epochs 100 \
  --batch-size 32 \
  --hidden-dim 128 \
  --num-layers 2 \
  --device cuda
```

Key parameters: `--hidden-dim`, `--num-layers`, `--no-attention`, `--lr`, `--patience`

#### Train Transformer

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/train_model.py \
  --data data/features \
  --model-type transformer \
  --epochs 100 \
  --batch-size 32 \
  --d-model 128 \
  --num-layers 4 \
  --num-heads 8 \
  --device cuda
```

Key parameters: `--d-model`, `--num-heads`, `--num-layers`, `--dim-feedforward`, `--pooling`, `--positional-encoding`

#### Evaluate Model

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/evaluate_model.py \
  --model models/checkpoints/best_model.pt \
  --data data/features \
  --split test \
  --device cuda
```

Outputs: Efficiency metrics (MAE, RMSE, Correlation, R²) and action metrics (accuracy, confusion matrix)

### XGBoost Baseline

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/train_xgboost.py \
  data/frames/<video>/pose_features.json \
  --label-column detection_score \
  --label-threshold 0.6 \
  --model-out models/xgb_pose.json
```

Options: `--feature-out`, `--early-stopping-rounds`, `--tree-method`, `--importance-out`

---

## Web UI

Run the pipeline from a browser via FastAPI.

### Setup

1. Install dependencies:

   ```bash
   conda run -n 6156-capstone pip install fastapi 'uvicorn[standard]' jinja2
   ```

2. Launch server:

   ```bash
   conda run -n 6156-capstone env PYTHONPATH=src uvicorn webapp.main:app --reload
   ```

3. Open http://127.0.0.1:8000

### Features

- Upload videos or point to existing directories
- Real-time job status and logs
- Inline pose visualizations
- Training UI at `/training` for XGBoost model training

### API Endpoints

See [PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md#web-api--ui) for complete API documentation:

- `POST /api/upload` — Upload video files
- `POST /api/jobs` — Create pipeline job
- `GET /api/jobs/{job_id}` — Get job status and artifacts
- `GET /api/jobs/{job_id}/analysis` — Get efficiency scores and recommendations
- `GET /api/jobs/{job_id}/ml_predictions` — Get ML predictions
- `DELETE /api/jobs/{job_id}` — Clear job
- `POST /api/training/jobs` — Create training job
- `GET /api/training/jobs/{job_id}` — Get training status

---

## Cloud Storage (Optional)

Mirror videos, frames, and models to Google Cloud Storage.

### Setup

1. Set environment variables:

   ```bash
   export GCS_PROJECT=<gcp-project-id>
   export GCS_VIDEO_BUCKET=<raw-video-bucket>
   export GCS_FRAME_BUCKET=<frame-bucket>
   export GCS_MODEL_BUCKET=<model-bucket>
   # Optional prefixes
   export GCS_VIDEO_PREFIX=videos/raw
   export GCS_FRAME_PREFIX=videos/frames
   export GCS_MODEL_PREFIX=models
   ```

2. Install SDK:
   ```bash
   conda run -n 6156-capstone pip install google-cloud-storage
   ```

### Retention Script

Prune stale artifacts:

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/prune_gcs_artifacts.py \
  --kind videos --days 30
```

Options: `--kind frames|models`, `--bucket`, `--prefix`

---

## Command Reference

| Task                | Command                  | Key Options                               |
| ------------------- | ------------------------ | ----------------------------------------- |
| **Pipeline**        |                          |                                           |
| Extract frames      | `extract_frames.py`      | `--interval`, `--output`                  |
| Pose estimation     | `run_pose_estimation.py` | `--frames-root`                           |
| Feature export      | `run_feature_export.py`  | `<manifest.json>`                         |
| Segment metrics     | `run_segment_report.py`  | `<manifest.json>`                         |
| Full pipeline       | `run_pipeline.py`        | `--interval`, `--out`                     |
| **Hold Annotation** |                          |                                           |
| Annotate holds      | `annotate_holds.py`      | `--output`                                |
| Train YOLO holds    | `train_yolo_holds.py`    | `--data`, `--model`, `--epochs`           |
| **ML Training**     |                          |                                           |
| Train BiLSTM        | `train_model.py`         | `--model-type bilstm`, `--hidden-dim`     |
| Train Transformer   | `train_model.py`         | `--model-type transformer`, `--num-heads` |
| Evaluate model      | `evaluate_model.py`      | `--model`, `--split`, `--data`            |
| **XGBoost**         |                          |                                           |
| Train XGBoost       | `train_xgboost.py`       | `--label-column`, `--model-out`           |
| **Visualization**   |                          |                                           |
| Visualize pose      | `visualize_pose.py`      | `--min-score`, `--min-visibility`         |
