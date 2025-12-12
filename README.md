# 6156-capstone-project

Climbing video analysis system with pose estimation, hold detection, efficiency scoring, and next-action recommendations.

**Project Planning**:

- [Project Backlog](https://docs.google.com/spreadsheets/d/113DbJu6Vg53PxX8Kgu5pkwsuLrqYH-i9JJcEuDvWjus/edit?gid=1139408620#gid=1139408620)
- [Sprint Backlog](https://github.com/users/harrisonkimdev/projects/9/views/1?sortedBy%5Bdirection%5D=asc&sortedBy%5BcolumnId%5D=221588758&sortedBy%5Bdirection%5D=asc&sortedBy%5BcolumnId%5D=Assignees)
- [Implementation Backlog](archive/IMPLEMENTATION_BACKLOG.md) — Detailed feature roadmap

---

## Features

- **Video Processing**: Extract frames using interval-based, motion-based, or motion+pose similarity methods
- **YOLO Segmentation**: Pixel-level segmentation to separate wall, holds, and climber regions
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
- **Cloud Storage**: Required GCS integration for videos, frames, and models

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

**Tip:** Run `make init` to create or update the Conda environment from `environment.yml` and register the `6156 (py3.10)` Jupyter kernel. The repository never stores a `.venv`; always recreate the environment locally when you clone or move the project.

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
  --segmentation --seg-model yolov8n-seg.pt
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

**Production API** - Simplified for external users. Only `video_dir` and optional `metadata` are required. All other options (output directory, segmentation, YOLO model, frame extraction) are automatically handled using production defaults.

```http
POST /api/jobs
Content-Type: application/json

{
  "video_dir": "data/uploads/IMG_3708_251211_105719AM",
  "metadata": {
    "route_name": "V5 Problem",
    "gym_location": "Climbing Gym",
    "climber_height": 175.0,
    "climber_wingspan": 180.0,
    "climber_flexibility": 0.7,
    "imu_quaternion": [0.7071, 0.0, 0.7071, 0.0],
    "imu_euler_angles": [85.5, 2.0, 0.0],
    "camera_orientation": "portrait",
    "notes": "First attempt",
    "tags": ["overhang", "crimps"]
  }
}
```

**Request Fields**:

- `video_dir` (required): Directory containing source videos (typically from `/api/upload`)
- `metadata` (optional): Capture metadata including:
  - `route_name`: Route or boulder name
  - `gym_location`: Venue or wall identifier
  - `climber_height`: Height in cm (0-250)
  - `climber_wingspan`: Wingspan in cm (0-300)
  - `climber_flexibility`: Flexibility score 0-1
  - `imu_quaternion`: Device orientation as quaternion [w, x, y, z]
  - `imu_euler_angles`: Device orientation as Euler angles [pitch, roll, yaw] in degrees
  - `imu_timestamp`: IMU reading timestamp (unix milliseconds)
  - `camera_orientation`: "portrait", "landscape", or "other"
  - `notes`: Freeform notes
  - `tags`: List of search tags

**Production Defaults** (automatically applied):

- **Output Directory**: Same as `video_dir` (all artifacts stored in the same directory, workflow style)
- **Segmentation**: Always enabled (YOLO segmentation with default settings)
- **YOLO Model**: Production model from `PRODUCTION_YOLO_MODEL` environment variable
- **Frame Extraction**: All frames extracted, then BiLSTM model selects key frames (from `PRODUCTION_FRAME_SELECTOR_MODEL`)
- **GCS Upload**: Handled automatically via `.env` configuration

#### Other Endpoints

- `GET /api/jobs/{job_id}` — Get job status and artifacts
- `GET /api/jobs/{job_id}/analysis` — Get efficiency scores and recommendations
- `GET /api/jobs/{job_id}/ml_predictions` — Get ML predictions
- `GET /api/jobs/{job_id}/route_grade` — Get route difficulty grade (V0-V10)
- `DELETE /api/jobs/{job_id}` — Clear job
- `POST /api/training/jobs` — Create training job
- `GET /api/training/jobs/{job_id}` — Get training status

---

## Cloud Storage (Required)

All pipeline artifacts (videos, frames, and models) are automatically uploaded to Google Cloud Storage. GCS configuration is **required** for the application to run.

### Setup

1. **Create a `.env` file** (recommended for local development):

   Create a `.env` file in the project root with the following variables:

   ```bash
   # Required: GCP Project ID
   GCS_PROJECT=your-gcp-project-id
   # Alternative: GOOGLE_CLOUD_PROJECT=your-gcp-project-id

   # Required: GCS Bucket Names
   GCS_VIDEO_BUCKET=your-video-bucket-name
   GCS_FRAME_BUCKET=your-frame-bucket-name
   GCS_MODEL_BUCKET=your-model-bucket-name

   # Optional: Custom Prefixes (defaults shown)
   # GCS_VIDEO_PREFIX=videos/raw
   # GCS_FRAME_PREFIX=videos/frames
   # GCS_MODEL_PREFIX=models

   # Required: Authentication
   # Option 1: Service Account JSON file path
   # Recommended: Store the JSON file in the project root directory
   GOOGLE_APPLICATION_CREDENTIALS=./gcs-key.json
   # Or use absolute path:
   # GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json

   # Option 2: Use gcloud auth application-default login instead
   # (then you don't need GOOGLE_APPLICATION_CREDENTIALS)
   ```

   **Note:**

   - The `.env` file is already in `.gitignore` and will not be committed to version control.
   - Service account JSON key files (e.g., `*-key.json`, `*service-account*.json`, `gcs-key.json`) are also in `.gitignore` for security.
   - You can safely store the JSON key file in the project root directory.

   **Alternative:** You can also set environment variables directly in your shell:

   ```bash
   export GCS_PROJECT=<gcp-project-id>
   export GCS_VIDEO_BUCKET=<raw-video-bucket>
   export GCS_FRAME_BUCKET=<frame-bucket>
   export GCS_MODEL_BUCKET=<model-bucket>
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   ```

2. **Create Service Account and download JSON key:**

   - Go to [Google Cloud Console](https://console.cloud.google.com) → IAM & Admin → Service Accounts
   - Create a new service account with **Storage Object Admin** role (or **Storage Admin** for full access)
   - Create a JSON key and download it
   - Save the JSON file in the project root directory (e.g., `gcs-key.json` or `betamove-gcs-key.json`)
   - Update `.env` file with the path: `GOOGLE_APPLICATION_CREDENTIALS=./gcs-key.json`

   **Security Note:** The JSON key file is automatically ignored by Git (see `.gitignore`). Never commit it to version control.

3. **Configure authentication** (choose one):

   - **Service account JSON file:** Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env` (recommended: use relative path like `./gcs-key.json`)
   - **Google Cloud SDK default credentials:**
     ```bash
     gcloud auth application-default login
     ```

4. **Install dependencies:**
   ```bash
   conda run -n 6156-capstone pip install google-cloud-storage python-dotenv
   ```

### Error Handling

If any required environment variables are missing, the application will raise a `ValueError` at startup with details about which variables need to be set. GCS upload failures will cause the entire operation to fail.

### Retention Script

Prune stale artifacts:

```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/prune_gcs_artifacts.py \
  --kind videos --days 30
```

Options: `--kind frames|models`, `--bucket`, `--prefix`

---

## Google Drive Integration (Optional)

Google Drive can be used as an alternative or additional storage backend, especially useful for training models on Google Colab.

### Setup

1. **Enable Google Drive API:**

   - Go to [Google Cloud Console](https://console.cloud.google.com) → APIs & Services → Library
   - Search for "Google Drive API" and enable it

2. **Create Service Account (for local automation):**

   - Go to IAM & Admin → Service Accounts
   - Create a new service account
   - Grant "Editor" role
   - Create JSON key and download it
   - Save as `drive-service-account.json` in the project root

3. **Add to `.env` file:**

   ```bash
   # Google Drive (optional)
   GOOGLE_DRIVE_ENABLED=true
   GOOGLE_DRIVE_ROOT_FOLDER_ID=your-folder-id
   GOOGLE_DRIVE_SERVICE_ACCOUNT_PATH=./drive-service-account.json

   # Storage backend selection (gcs, drive, both)
   STORAGE_BACKEND=both
   ```

4. **Find your Google Drive Folder ID:**

   - Open the folder in Google Drive
   - The folder ID is in the URL after `folders/`
   - Example: `https://drive.google.com/drive/folders/1a2b3c4d5e` → ID: `1a2b3c4d5e`

### Storage Backend Options

| Backend | Description                                              |
| ------- | -------------------------------------------------------- |
| `gcs`   | Use only Google Cloud Storage (default)                  |
| `drive` | Use only Google Drive                                    |
| `both`  | Use both GCS and Google Drive (redundant storage)        |

---

## Training on Google Colab

For training models using Google Colab's GPU resources:

1. **Open the Colab notebook:**

   - Navigate to `notebooks/train_on_colab.ipynb`
   - Open in Google Colab

2. **Prepare training data:**

   Upload your training data to Google Drive with this structure:
   ```
   MyDrive/BetaMove/
   ├── training_data/
   │   ├── hold_detection/      # YOLO dataset
   │   │   ├── dataset.yaml
   │   │   ├── images/
   │   │   └── labels/
   │   └── pose_features/       # XGBoost features
   │       └── features.json
   └── models/                  # Trained models (output)
   ```

3. **Run training:**

   - Enable GPU runtime: Runtime → Change runtime type → GPU
   - Execute the notebook cells sequentially
   - Models are saved to Google Drive automatically

4. **Use trained models:**

   - Download models from `MyDrive/BetaMove/models/`
   - Place in the project's `models/` directory
   - Or configure the application to load from Drive

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
│  │ 2. YOLO Segmentation (segmentation/             │  │
│  │    yolo_segmentation.py) - OPTIONAL              │  │
│  │    - Wall, holds, climber pixel-level masks     │  │
│  │    - Color-based route grouping                 │  │
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

### YOLO Segmentation

Uses YOLO segmentation model to separate wall, holds, and climber regions at pixel level.

**Features**:

- Pixel-level masks for each class (wall, holds, climber)
- Color-based route grouping (same color = same route/problem)
- Automatic hold color extraction and clustering

**Output**:

- `masks/`: Binary mask images for each class
- `segmentation_results.json`: Metadata with mask paths
- `routes.json`: Color-based route groupings

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
- **Storage**: Local filesystem (temporary) + required Google Cloud Storage
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
