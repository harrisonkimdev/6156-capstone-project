# 6156-capstone-project

Climbing video analysis system with pose estimation, hold detection, efficiency scoring, and next-action recommendations.

**📖 Complete Documentation**: See [PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md) for comprehensive system overview, API reference, and implementation details.

**🎯 Project Planning**:
- [Project Backlog](https://docs.google.com/spreadsheets/d/113DbJu6Vg53PxX8Kgu5pkwsuLrqYH-i9JJcEuDvWjus/edit?gid=1139408620#gid=1139408620)
- [Sprint Backlog](https://github.com/users/harrisonkimdev/projects/9/views/1?sortedBy%5Bdirection%5D=asc&sortedBy%5BcolumnId%5D=221588758&sortedBy%5Bdirection%5D=asc&sortedBy%5BcolumnId%5D=Assignees)
- [Implementation Backlog](docs/IMPLEMENTATION_BACKLOG.md) — Detailed feature roadmap

---

## Quick Start

The pipeline processes climbing videos through frame extraction, pose estimation, hold detection, wall angle estimation, feature engineering, segmentation, and efficiency scoring. No video duration limits.

### Key Features

- **Pose Estimation**: MediaPipe 33-landmark detection
- **Hold Detection**: YOLOv8n/m with DBSCAN clustering
- **Wall Angle**: Automatic estimation (Hough + PCA)
- **Efficiency Scoring**: 5-component rule-based metric
- **Next-Action Recommendations**: Distance-based hold suggestions
- **Web UI**: FastAPI with real-time job status and results
- **Cloud Storage**: Optional GCS integration for videos/frames/models
- **Training UI**: XGBoost model training with parameter tuning

---

## Jupyter Kernel (One‑Shot Setup)

If you want a Python 3.10 venv and a Jupyter kernel in one go, run from the repo root:

```bash
make init VENV=.venv PYTHON=python3.10 KERNEL_NAME=6156-capstone DISPLAY_NAME="6156 (py3.10)"
```

Then, in VS Code or Jupyter, select the kernel named "6156 (py3.10)".

### Conda (recommended)

Create the environment from `environment.yml` and register the kernel:

```bash
conda env create -f environment.yml
conda activate 6156-capstone
python -m ipykernel install --user --name 6156-capstone --display-name "6156 (py3.10)"
```

Updating an existing env to match `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

Alternative (Conda):

```bash
conda create -n 6156-capstone python=3.10 -y
conda activate 6156-capstone
pip install -r requirements.txt ipykernel jupyterlab
python -m ipykernel install --user --name 6156-capstone --display-name "6156 (py3.10)"
```

## Running the Pipeline Locally

All commands assume the project root and a Conda env named `6156-capstone` with dependencies (`mediapipe`, `opencv-python`, `numpy`, `pytest`, etc.) installed.

```bash
conda run -n 6156-capstone env PYTHONPATH=src <command>
```

### One-Command Pipeline
```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/run_pipeline.py BetaMove/videos --out data/frames --interval 1.5
```

This runs the complete pipeline including:
1. Frame extraction (customizable interval, no duration limits)
2. Hold detection (YOLOv8)
3. Wall angle estimation (automatic)
4. Pose estimation (MediaPipe)
5. Feature extraction
6. Segmentation
7. Efficiency scoring

### Individual Pipeline Stages

You can run each stage separately:

#### 1. Extract Frames
```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/extract_frames.py BetaMove/videos --output data/frames --interval 1.5
```

#### 2. Run Pose Estimation
```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/run_pose_estimation.py --frames-root data/frames
```

#### 3. Export Frame-Level Features
```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/run_feature_export.py data/frames/<video>/manifest.json
```

#### 4. Generate Segment Metrics
```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/run_segment_report.py data/frames/<video>/manifest.json
```

#### 5. Run Unit Tests
```bash
conda run -n 6156-capstone env PYTHONPATH=src pytest tests/unit
```

#### 6. Train XGBoost Baseline
```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/train_xgboost.py data/frames/<video>/pose_features.json --label-column detection_score --label-threshold 0.6 --model-out models/xgb_pose.json
```
Adjust `--label-column`/`--label-threshold` to match your ground-truth labels. Optional `--feature-out` writes predictions to CSV. You can tune training via `--early-stopping-rounds 30`, `--tree-method hist|gpu_hist`, and export feature importances with `--importance-out models/xgb_pose_importance.csv`.

#### 7. Pose Visualization
Overlay landmarks and connections on frames using:
```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/visualize_pose.py data/frames/<video>/pose_results.json --min-score 0.3 --min-visibility 0.2
```
Adjust `--min-score` / `--min-visibility` to filter low-confidence detections. Outputs are saved alongside frames under `visualized/` by default.

---

## Web UI

Run the entire pipeline from a browser via the FastAPI app in `webapp/`.

1. Install the extra dependencies (once):
   ```bash
   conda run -n 6156-capstone pip install fastapi 'uvicorn[standard]' jinja2
   ```
2. Launch the server from the project root:
   ```bash
   conda run -n 6156-capstone env PYTHONPATH=src uvicorn webapp.main:app --reload
   ```
3. Open http://127.0.0.1:8000 to access the UI. Either point the form at an existing directory of videos or upload files directly in the browser. Logs stream live on the page, inline previews show generated pose visualizations, and sample pose rows are rendered alongside the job status. The API is available under `/api/*` for automation.

### API Endpoints

See [PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md#web-api--ui) for complete API documentation including:

- `POST /api/upload` — Upload video files
- `POST /api/jobs` — Create pipeline job with YOLO configuration
- `GET /api/jobs/{job_id}` — Get job status and artifacts
- `GET /api/jobs/{job_id}/analysis` — Get efficiency scores and recommendations
- `DELETE /api/jobs/{job_id}` — Clear job from active list
- `POST /api/training/jobs` — Create XGBoost training job
- `GET /api/training/jobs/{job_id}` — Get training status and metrics

### Training UI
- Navigate to `http://127.0.0.1:8000/training` to upload a features JSON file and start an XGBoost training job with optional parameters (defaults provided). The page shows job status, saved model path, and metrics when complete.

---

## Cloud Storage Integration

Videos, extracted frames, and trained models can be mirrored to Google Cloud Storage. Set the following environment variables before launching the web app or scripts (only buckets you configure will be used):

```bash
export GCS_PROJECT=<gcp-project-id>
export GCS_VIDEO_BUCKET=<raw-video-bucket>
export GCS_FRAME_BUCKET=<frame-bucket>
export GCS_MODEL_BUCKET=<model-bucket>
# Optional custom prefixes
export GCS_VIDEO_PREFIX=videos/raw
export GCS_FRAME_PREFIX=videos/frames
export GCS_MODEL_PREFIX=models
```

Install the SDK once in your Conda env:

```bash
conda run -n 6156-capstone pip install google-cloud-storage
```

When configured, the upload API stores the user’s video under the raw bucket, pipeline runs sync each frame directory after pose/feature extraction, and training jobs push the resulting model artifacts. GCS URIs are exposed per job under the `artifacts` (pipeline) or `model_uri` (training) fields.

### Retention Script

Prune stale artifacts with the helper script (ideal for a cron/Scheduler job):

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/prune_gcs_artifacts.py \
  --kind videos --days 30
```

Use `--kind frames` or `--kind models` to target other buckets, or override `--bucket` / `--prefix` for custom locations.
