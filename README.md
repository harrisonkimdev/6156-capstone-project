# 6156-capstone-project

Climbing video analysis system with pose estimation, hold detection, efficiency scoring, and next-action recommendations.

**Project Planning**:

- [Project Backlog](https://docs.google.com/spreadsheets/d/113DbJu6Vg53PxX8Kgu5pkwsuLrqYH-i9JJcEuDvWjus/edit?gid=1139408620#gid=1139408620)
- [Sprint Backlog](https://github.com/users/harrisonkimdev/projects/9/views/1?sortedBy%5Bdirection%5D=asc&sortedBy%5BcolumnId%5D=221588758&sortedBy%5Bdirection%5D=asc&sortedBy%5BcolumnId%5D=Assignees)
- [Implementation Backlog](archive/IMPLEMENTATION_BACKLOG.md) — Detailed feature roadmap

---

## Features

- **Video Processing**: Extract frames using interval-based, motion-based, or motion+pose similarity methods
- **Segmentation**: Pixel-level segmentation to separate wall, holds, and climber regions (YOLO or HSV color-based methods)
- **Color-Based Route Grouping**: Automatically group holds by color to identify climbing routes/problems
- **Pose Estimation**: MediaPipe 33-landmark detection with confidence filtering
- **Hold Detection**: YOLOv8n/m object detection with DBSCAN spatial clustering and temporal tracking
- **Hold Type Classification**: YOLOv8 fine-tuning for hold types (crimp, sloper, jug, pinch, foot_only, volume)
- **Wall Angle**: IMU sensor integration (±1° accuracy) with vision-based fallback (±5°)
- **Efficiency Scoring**: 7-component physics-based metric with technique bonuses
- **Climber Personalization**: Height, wingspan, and flexibility parameters for personalized recommendations
- **Next-Action Recommendations**: Rule-based planner with hold type awareness and personalized reach constraints
- **ML Models**: BiLSTM and Transformer multitask models for efficiency and next-action prediction
- **Route Difficulty Estimation**: XGBoost model for V0-V10 grade prediction
- **Web UI**: FastAPI with background job management, real-time status updates, and grading UI
- **Cloud Storage**: Optional GCS integration for videos, frames, and models

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

---

## Quick Start

All commands assume the project root and a Conda env named `6156-capstone`:

```bash
conda run -n 6156-capstone env PYTHONPATH=src <command>
```

### Run Complete Pipeline

Process videos through frame extraction, pose estimation, hold detection, wall angle estimation, feature engineering, segmentation, and efficiency scoring:

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/run_pipeline.py data/videos --out data/frames --interval 1.5
```

### Run Individual Stages

#### 1. Extract Frames

**Interval-based (default)**:

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/extract_frames.py data/videos --output data/frames --interval 1.5
```

**Motion-based**:

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/extract_frames.py data/videos --output data/frames \
  --method motion \
  --motion-threshold 5.0 \
  --min-frame-interval 5 \
  --initial-sampling-rate 0.1
```

**Motion + Pose Similarity**:

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/extract_frames.py data/videos --output data/frames \
  --method motion_pose \
  --motion-threshold 5.0 \
  --similarity-threshold 0.8 \
  --min-frame-interval 5 \
  --initial-sampling-rate 0.1
```

**With YOLO Segmentation**:

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/extract_frames.py data/videos --output data/frames \
  --segmentation --seg-method yolo --seg-model yolov8n-seg.pt
```

**With HSV segmentation**:

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/extract_frames.py data/videos --output data/frames \
  --segmentation --seg-method hsv \
  --hsv-hue-tolerance 5 \
  --hsv-sat-tolerance 50 \
  --hsv-val-tolerance 40
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

#### Upload Video

```http
POST /api/upload
Content-Type: multipart/form-data

file: <video_file>
```

#### Create Pipeline Job

```http
POST /api/jobs
Content-Type: application/json

{
  "video_dir": "data/videos",
  "output_dir": "data/frames",
  "interval": 1.5,
  "metadata": {
    "route_name": "V5 Problem",
    "imu_quaternion": [0.7071, 0.0, 0.7071, 0.0],
    "imu_euler_angles": [85.5, 2.0, 0.0],
    "climber_height": 175.0,
    "climber_wingspan": 180.0,
    "climber_flexibility": 0.7
  },
  "yolo": {
    "enabled": true,
    "model_name": "yolov8n.pt",
    "min_confidence": 0.35
  },
  "frame_extraction": {
    "method": "motion_pose",
    "motion_threshold": 5.0,
    "similarity_threshold": 0.8,
    "min_frame_interval": 5,
    "use_optical_flow": true,
    "use_pose_similarity": true,
    "initial_sampling_rate": 0.1
  },
  "segmentation": {
    "enabled": true,
    "method": "yolo",
    "model_name": "yolov8n-seg.pt",
    "export_masks": true,
    "group_by_color": true,
    "hue_tolerance": 10,
    "sat_tolerance": 50,
    "val_tolerance": 50
  }
}
```

**Frame Extraction Options**:

- `method`: `"interval"` (time-based), `"motion"` (motion-based), or `"motion_pose"` (motion + pose similarity)
- `motion_threshold`: Minimum motion score (default: 5.0)
- `similarity_threshold`: Maximum pose similarity for `motion_pose` method (default: 0.8)
- `min_frame_interval`: Minimum frames between selections (default: 5)
- `use_optical_flow`: Enable optical flow for motion detection (default: true)
- `use_pose_similarity`: Enable pose similarity filtering (default: true, only for `motion_pose`)
- `initial_sampling_rate`: Initial frame sampling rate in seconds (default: 0.1)

**Segmentation Options**:

- `enabled`: Enable YOLO segmentation (default: false)
- `method`: Segmentation method: `"yolo"`, `"hsv"`, or `"none"` (default: "yolo")
- `model_name`: YOLO segmentation model name (default: "yolov8n-seg.pt")
- `export_masks`: Export segmentation masks as images (default: true)
- `group_by_color`: Group holds by color to identify routes (default: true)
- `hue_tolerance`: Hue tolerance for color clustering (default: 10)
- `sat_tolerance`: Saturation tolerance for color clustering (default: 50)
- `val_tolerance`: Value tolerance for color clustering (default: 50)
- `hsv_hue_tolerance`: HSV method: Hue tolerance for hold detection (default: 5)
- `hsv_sat_tolerance`: HSV method: Saturation tolerance for hold detection (default: 50)
- `hsv_val_tolerance`: HSV method: Value tolerance for hold detection (default: 40)

#### Other Endpoints

- `GET /api/jobs/{job_id}` — Get job status and artifacts
- `GET /api/jobs/{job_id}/analysis` — Get efficiency scores and recommendations
- `GET /api/jobs/{job_id}/ml_predictions` — Get ML predictions
- `GET /api/jobs/{job_id}/route_grade` — Get route difficulty grade (V0-V10)
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

| Task                | Command                  | Key Options                                                                                            |
| ------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------ |
| **Pipeline**        |                          |                                                                                                        |
| Extract frames      | `extract_frames.py`      | `--interval`, `--output`, `--method`, `--motion-threshold`, `--similarity-threshold`, `--segmentation` |
| Pose estimation     | `run_pose_estimation.py` | `--frames-root`                                                                                        |
| Feature export      | `run_feature_export.py`  | `<manifest.json>`                                                                                      |
| Segment metrics     | `run_segment_report.py`  | `<manifest.json>`                                                                                      |
| Full pipeline       | `run_pipeline.py`        | `--interval`, `--out`                                                                                  |
| **Hold Annotation** |                          |                                                                                                        |
| Annotate holds      | `annotate_holds.py`      | `--output`                                                                                             |
| Train YOLO holds    | `train_yolo_holds.py`    | `--data`, `--model`, `--epochs`                                                                        |
| **ML Training**     |                          |                                                                                                        |
| Train BiLSTM        | `train_model.py`         | `--model-type bilstm`, `--hidden-dim`                                                                  |
| Train Transformer   | `train_model.py`         | `--model-type transformer`, `--num-heads`                                                              |
| Evaluate model      | `evaluate_model.py`      | `--model`, `--split`, `--data`                                                                         |
| **XGBoost**         |                          |                                                                                                        |
| Train XGBoost       | `train_xgboost.py`       | `--label-column`, `--model-out`                                                                        |
| **Visualization**   |                          |                                                                                                        |
| Visualize pose      | `visualize_pose.py`      | `--min-score`, `--min-visibility`                                                                      |

---

## System Architecture

```
┌─────────────┐
│  Video      │
│  Upload     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  Pipeline Runner (webapp/pipeline_runner.py)            │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 1. Frame Extraction (scripts/extract_frames.py) │   │
│  │    - Interval-based (default)                    │   │
│  │    - Motion-based (optical flow)                 │   │
│  │    - Motion + Pose similarity                    │   │
│  └────────────────┬────────────────────────────────┘   │
│                   ▼                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 2. Segmentation (segmentation/                  │  │
│  │    yolo_segmentation.py or hsv_segmentation.py) │  │
│  │    - OPTIONAL                                    │  │
│  │    - YOLO: Wall, holds, climber pixel-level     │  │
│  │      masks with color-based route grouping       │  │
│  │    - HSV: Color-based hold detection             │  │
│  └────────────────┬─────────────────────────────────┘  │
│                   ▼                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 3. Hold Detection (service/hold_extraction.py)   │  │
│  │    - YOLOv8 inference on frames                  │  │
│  │    - DBSCAN spatial clustering                   │  │
│  │    - Temporal tracking (IoU + Kalman)            │  │
│  └────────────────┬─────────────────────────────────┘  │
│                   ▼                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 4. Wall Angle Estimation (wall/angle.py)         │  │
│  │    - IMU sensor data (priority)                  │  │
│  │    - Hough line detection + RANSAC               │  │
│  │    - PCA fallback for edge-rich frames           │  │
│  └────────────────┬─────────────────────────────────┘  │
│                   ▼                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 5. Pose Estimation (scripts/run_pose_estimation) │  │
│  │    - MediaPipe 33 landmarks per frame            │  │
│  └────────────────┬─────────────────────────────────┘  │
│                   ▼                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 6. Feature Extraction (features/aggregation.py)  │  │
│  │    - Joint angles, COM, velocities               │  │
│  │    - Hold proximity & contact inference          │  │
│  │    - Wall alignment metrics                      │  │
│  └────────────────┬─────────────────────────────────┘  │
│                   ▼                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 7. Segmentation (segmentation/rule_based.py)     │  │
│  │    - Movement vs rest classification             │  │
│  └────────────────┬─────────────────────────────────┘  │
│                   ▼                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 8. Efficiency Scoring (recommendation/           │  │
│  │    efficiency.py)                                │  │
│  │    - 7-component weighted score                  │  │
│  │    - Next-hold recommendations                   │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│  Results    │
│  - JSON     │
│  - Web UI   │
│  - GCS      │
└─────────────┘
```

---

## Data Pipeline

### Frame Extraction Methods

#### Interval-Based (Default)

Extracts frames at specified time intervals. Simple and fast.

#### Motion-Based

Uses optical flow to detect motion and select frames with significant movement. Better for capturing dynamic actions.

#### Motion + Pose Similarity

Combines motion detection with pose similarity to select diverse frames. Best for capturing key poses while avoiding redundancy.

**Algorithm**:

1. Extract frames at high rate (initial_sampling_rate)
2. Compute motion scores using optical flow
3. Filter frames by motion threshold
4. Run pose estimation on high-motion frames
5. Compare pose keypoints between consecutive frames
6. Select frames where pose similarity < threshold (significant pose change)
7. Apply minimum interval constraint

### Segmentation

Two methods are available for pixel-level segmentation:

#### YOLO Segmentation

Uses YOLO segmentation model to separate wall, holds, and climber regions at pixel level.

**Features**:

- Pixel-level masks for each class (wall, holds, climber)
- Color-based route grouping (same color = same route/problem)
- Automatic hold color extraction and clustering
- Requires pre-trained YOLO model

**Output**:

- `masks/`: Binary mask images for each class
- `segmentation_results.json`: Metadata with mask paths
- `routes.json`: Color-based route groupings

#### HSV Segmentation

Uses HSV color masking to detect holds based on color similarity. This method is useful when you have a reference color or want to detect holds of a specific color.

**Features**:

- HSV color-based hold detection
- Background removal using HSV thresholds
- No model required (works with color information only)
- Configurable hue, saturation, and value tolerances

**Parameters**:

- `hue_tolerance`: Hue tolerance for color matching (0-179, default: 5)
- `sat_tolerance`: Saturation tolerance (0-255, default: 50)
- `val_tolerance`: Value tolerance (0-255, default: 40)

**Output**:

- `masks/`: Binary mask images for detected holds
- `segmentation_results.json`: Metadata with mask paths

### Hold Detection

YOLOv8-based detection with:

- DBSCAN spatial clustering for stable hold positions
- Temporal tracking (IoU + Kalman filter) for consistency
- Hold type classification (crimp, sloper, jug, pinch, foot_only, volume)

### Wall Angle Estimation

**Priority Order**:

1. **IMU Sensor Data** (if provided): Most accurate (±1°)
2. **Vision-based Estimation** (fallback): Hough + PCA (±5°)

### Feature Extraction

Computes derived features from pose landmarks:

- Joint angles (elbow, knee, hip, shoulder)
- Center of Mass (COM) position, velocity, acceleration
- Hold relationships (distance, contact inference)
- Wall alignment metrics
- Kinematic derivatives (velocity, acceleration, jerk)

### Efficiency Scoring

7-component physics-based metric:

1. **Support Polygon Stability** (0.35): COM distance to convex hull
2. **Path Efficiency** (0.25): Net displacement vs actual path length
3. **Support Penalty** (0.20): Penalty if support count < 2
4. **Wall Penalty** (0.10): COM distance from wall
5. **Jerk Penalty** (0.07): Smoothness based on jerk
6. **Reach Penalty** (0.03): Extreme limb extensions
7. **Technique Bonuses**: Bicycle (0.05) + Back-flag (0.05) + Drop-knee (0.03)

**Score Interpretation**:

- 0.8-1.0: Excellent efficiency
- 0.6-0.8: Good efficiency
- 0.4-0.6: Moderate efficiency
- 0.2-0.4: Poor efficiency
- 0.0-0.2: Very poor efficiency

---

## Technology Stack

- **Core**: Python 3.10+
- **Computer Vision**: MediaPipe 0.10.9, OpenCV, Ultralytics YOLOv8
- **ML**: PyTorch (BiLSTM, Transformer), XGBoost, scikit-learn
- **Web**: FastAPI, Uvicorn, Jinja2
- **Storage**: Local filesystem + optional Google Cloud Storage
- **Testing**: pytest

---

## Limitations & Known Issues

### Current Limitations

- **MediaPipe z-coordinate**: Relative depth only, not absolute
- **Viewpoint Sensitivity**: Strong angle changes reduce accuracy
- **Frame Blur**: No automatic filtering for motion blur
- **Hold Tracking**: Frame-by-frame detection (temporal tracking available but optional)

### Performance Targets

- Hold detection mAP@0.5: yolov8n ≥ 0.60, yolov8m ≥ 0.68
- Wall angle MAE: vertical ≤ 5°, overhang ≤ 8° (vision-based), ≤ 1° (IMU)
- Efficiency vs expert correlation: ≥ 0.5
- Next-hold top-3 hit rate: ≥ 0.6

---

## References

- **Implementation Backlog**: [`archive/IMPLEMENTATION_BACKLOG.md`](archive/IMPLEMENTATION_BACKLOG.md) — Detailed feature roadmap
- **Source Code**: [`src/pose_ai/`](src/pose_ai/) — Core library modules
- **Scripts**: [`scripts/`](scripts/) — CLI tools for pipeline stages
- **Web Application**: [`webapp/`](webapp/) — FastAPI server and UI
- **Tests**: [`tests/unit/`](tests/unit/) — Unit test suite
