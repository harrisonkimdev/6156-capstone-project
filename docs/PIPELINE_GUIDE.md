# Climbing Video Analysis Pipeline — Complete Guide

**Version**: 1.0  
**Last Updated**: November 18, 2025  
**Source of Truth**: This document consolidates all project documentation and reflects the current implementation state.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Data Pipeline](#data-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Efficiency Scoring](#efficiency-scoring)
6. [Next-Action Recommendations](#next-action-recommendations)
7. [Web API & UI](#web-api--ui)
8. [Machine Learning](#machine-learning)
9. [Notebook Usage](#notebook-usage)
10. [Limitations & Known Issues](#limitations--known-issues)
11. [Future Work](#future-work)

---

## System Overview

### Purpose

This system analyzes bouldering climbing videos to:

1. Extract pose keypoints using MediaPipe
2. Detect climbing holds using YOLOv8
3. Estimate wall angle automatically
4. Compute move efficiency scores
5. Recommend next actions for climbers

### Key Features

- **Video Processing**: Extract frames at configurable intervals (no duration limits)
- **Pose Estimation**: MediaPipe 33-landmark detection with confidence filtering
- **Hold Detection**: YOLOv8n/m object detection with DBSCAN spatial clustering
- **Wall Angle**: Automatic estimation using Hough line detection + PCA fallback
- **Efficiency Scoring**: 5-component rule-based metric (detection quality, joint angles, COM stability, contact count, wall alignment)
- **Recommendations**: Distance-based next-hold suggestions with recency weighting
- **Cloud Storage**: Optional GCS integration for videos, frames, and models
- **Web Interface**: FastAPI with background job management and real-time status updates

### Technology Stack

- **Core**: Python 3.10+
- **Computer Vision**: MediaPipe 0.10.9, OpenCV, Ultralytics YOLOv8
- **ML**: XGBoost, scikit-learn
- **Web**: FastAPI, Uvicorn, Jinja2
- **Storage**: Local filesystem + optional Google Cloud Storage
- **Testing**: pytest

---

## Architecture

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
│  └────────────────┬────────────────────────────────┘   │
│                   ▼                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 2. Hold Detection (service/hold_extraction.py)   │  │
│  │    - YOLOv8 inference on frames                  │  │
│  │    - DBSCAN spatial clustering                   │  │
│  └────────────────┬─────────────────────────────────┘  │
│                   ▼                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 3. Wall Angle Estimation (wall/angle.py)         │  │
│  │    - Hough line detection + RANSAC               │  │
│  │    - PCA fallback for edge-rich frames           │  │
│  └────────────────┬─────────────────────────────────┘  │
│                   ▼                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 4. Pose Estimation (scripts/run_pose_estimation) │  │
│  │    - MediaPipe 33 landmarks per frame            │  │
│  └────────────────┬─────────────────────────────────┘  │
│                   ▼                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 5. Feature Extraction (features/aggregation.py)  │  │
│  │    - Joint angles, COM, velocities               │  │
│  │    - Hold proximity & contact inference          │  │
│  │    - Wall alignment metrics                      │  │
│  └────────────────┬─────────────────────────────────┘  │
│                   ▼                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 6. Segmentation (segmentation/rule_based.py)     │  │
│  │    - Movement vs rest classification             │  │
│  └────────────────┬─────────────────────────────────┘  │
│                   ▼                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 7. Efficiency Scoring (recommendation/           │  │
│  │    efficiency.py)                                │  │
│  │    - 5-component weighted score                  │  │
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

### Core Modules

- **`src/pose_ai/pose/estimator.py`**: MediaPipe wrapper with smoothing and dropout handling
- **`src/pose_ai/service/hold_extraction.py`**: YOLO-based hold detection and clustering
- **`src/pose_ai/wall/angle.py`**: Automatic wall angle estimation
- **`src/pose_ai/features/aggregation.py`**: Feature engineering from pose frames
- **`src/pose_ai/segmentation/rule_based.py`**: Movement/rest classification
- **`src/pose_ai/recommendation/efficiency.py`**: Efficiency scoring and recommendations
- **`webapp/pipeline_runner.py`**: Background job orchestrator
- **`webapp/main.py`**: FastAPI application with REST endpoints

---

## Data Pipeline

### 1. Frame Extraction

**Script**: [`scripts/extract_frames.py`](../scripts/extract_frames.py)

Extracts frames from videos at specified intervals with no duration limits.

```bash
python scripts/extract_frames.py BetaMove/videos --output data/frames --interval 1.5
```

**Output**: `data/frames/<video_id>/frame_*.jpg` + `manifest.json`

**Manifest Schema**:
```json
{
  "video_name": "video01.mp4",
  "fps": 30.0,
  "total_frames": 2700,
  "duration_sec": 90.0,
  "frame_interval": 1.5,
  "output_dir": "data/frames/video01",
  "frames": [
    {"frame_number": 0, "timestamp_sec": 0.0, "file_path": "frame_0000.jpg"},
    {"frame_number": 45, "timestamp_sec": 1.5, "file_path": "frame_0045.jpg"}
  ]
}
```

### 2. Hold Detection

**Module**: [`src/pose_ai/service/hold_extraction.py`](../src/pose_ai/service/hold_extraction.py)

Uses YOLOv8 to detect climbing holds across all frames, then applies DBSCAN clustering to generate stable hold positions.

```python
from pose_ai.service.hold_extraction import extract_holds_from_frames, cluster_holds

# Extract per-frame detections
detections = extract_holds_from_frames(
    frame_paths=frame_files,
    model_name="yolov8n",  # or "yolov8m" for higher accuracy
    conf_threshold=0.25
)

# Cluster into stable holds
holds = cluster_holds(detections, eps=30, min_samples=3)
```

**Output**: `holds.json`
```json
{
  "holds": [
    {
      "cluster_id": 0,
      "centroid": [245.3, 180.7],
      "confidence": 0.82,
      "detection_count": 47,
      "bbox_mean": [230, 170, 260, 195]
    }
  ]
}
```

**Key Parameters**:
- `model_name`: `"yolov8n"` (fast, mAP ≥ 0.60) or `"yolov8m"` (accurate, mAP ≥ 0.68)
- `conf_threshold`: Detection confidence threshold (default: 0.25)
- `eps`: DBSCAN clustering radius in pixels (default: 30)
- `min_samples`: Minimum detections to form a cluster (default: 3)

### 3. Wall Angle Estimation

**Module**: [`src/pose_ai/wall/angle.py`](../src/pose_ai/wall/angle.py)

Automatically estimates wall angle from video frames using edge detection and geometric analysis.

```python
from pose_ai.wall.angle import estimate_wall_angle

angle, confidence = estimate_wall_angle(
    frame_paths=representative_frames,
    method="hough"  # or "pca"
)
```

**Algorithm**:
1. **Hough Line Detection**: Detect vertical/near-vertical lines in edge-filtered frames
2. **RANSAC Fitting**: Robust line fitting to handle outliers
3. **PCA Fallback**: Use principal component analysis if insufficient lines detected
4. **Confidence Scoring**: Based on line consensus and edge density

**Output**: Angle in degrees (0° = vertical, 90° = horizontal overhang), confidence score 0-1

**Typical Performance**:
- Vertical walls: MAE ≤ 5°
- Overhangs: MAE ≤ 8°

### 4. Pose Estimation

**Module**: [`src/pose_ai/pose/estimator.py`](../src/pose_ai/pose/estimator.py)  
**Script**: [`scripts/run_pose_estimation.py`](../scripts/run_pose_estimation.py)

Extracts 33 MediaPipe landmarks per frame with confidence filtering.

```bash
python scripts/run_pose_estimation.py --frames-root data/frames
```

**Output**: `pose_results.json`
```json
{
  "frames": [
    {
      "frame_path": "frame_0000.jpg",
      "timestamp": 0.0,
      "landmarks": [
        {"id": 0, "name": "nose", "x": 0.51, "y": 0.23, "z": -0.12, "visibility": 0.95},
        {"id": 11, "name": "left_shoulder", "x": 0.45, "y": 0.35, "z": -0.08, "visibility": 0.92}
      ]
    }
  ]
}
```

**Key Landmarks** (for climbing analysis):
- Shoulders (11, 12)
- Elbows (13, 14)
- Wrists (15, 16)
- Hips (23, 24)
- Knees (25, 26)
- Ankles (27, 28)

### 5. Feature Extraction

**Module**: [`src/pose_ai/features/aggregation.py`](../src/pose_ai/features/aggregation.py)  
**Script**: [`scripts/run_feature_export.py`](../scripts/run_feature_export.py)

Computes derived features from pose landmarks and hold positions.

```bash
python scripts/run_feature_export.py data/frames/video01/manifest.json
```

**Feature Categories**:

1. **Joint Angles**:
   - Elbow angle (shoulder-elbow-wrist)
   - Knee angle (hip-knee-ankle)
   - Hip angle (shoulder-hip-knee)
   - Shoulder angle (elbow-shoulder-hip)

2. **Center of Mass (COM)**:
   - Weighted average of major landmarks
   - COM velocity and acceleration
   - COM position relative to support polygon

3. **Hold Relationships**:
   - Distance from each limb to nearest hold
   - Contact inference (distance + velocity thresholds)
   - Contact duration and stability

4. **Wall Alignment** (new in current version):
   - `wall_angle`: Estimated wall inclination
   - `hip_alignment_error`: Hip deviation from wall normal
   - `com_along_wall`: COM projection along wall surface
   - `com_perp_wall`: COM distance perpendicular to wall

5. **Kinematics**:
   - Joint velocities (frame-to-frame deltas)
   - Joint accelerations (velocity deltas)
   - Jerk (acceleration deltas, optional)

**Output**: `pose_features.json`

### 6. Segmentation

**Module**: [`src/pose_ai/segmentation/rule_based.py`](../src/pose_ai/segmentation/rule_based.py)  
**Script**: [`scripts/run_segmentation.py`](../scripts/run_segmentation.py)

Classifies frames into movement vs rest segments using velocity and position stability heuristics.

**Classification Logic**:
- **Movement**: High joint velocities (above threshold), changing COM position
- **Rest**: Low velocities, stable position for minimum duration

**Output**: `segment_metrics.json`
```json
{
  "segments": [
    {
      "start_frame": 0,
      "end_frame": 45,
      "duration_sec": 1.5,
      "segment_type": "movement",
      "avg_velocity": 0.23,
      "max_velocity": 0.45
    }
  ]
}
```

---

## Feature Engineering

### Coordinate Normalization

All spatial features are normalized for scale and view invariance:

1. **Root-Relative**: Subtract hip center `(hip_left + hip_right) / 2`
2. **Scale Normalization**: Divide by body scale (shoulder width)
3. **View Normalization** (optional): Apply homography if wall calibration available

### Kinematic Derivatives

Frame-to-frame changes provide motion context:

- **Velocity**: `v_t = (p_t - p_{t-1}) / dt`
- **Acceleration**: `a_t = (v_t - v_{t-1}) / dt`
- **Jerk**: `j_t = (a_t - a_{t-1}) / dt` (optional, for smoothness metrics)

Focus joints: wrists, ankles, hips, shoulders, COM

### Hold Contact Inference

**Current Implementation** (basic distance thresholding):
```python
def infer_contact(joint_pos, holds, threshold=0.25):
    """Check if joint is in contact with any hold"""
    distances = [np.linalg.norm(joint_pos - hold.position) for hold in holds]
    min_dist = min(distances) if distances else float('inf')
    return min_dist <= threshold * body_scale
```

**Planned Enhancement** (per efficiency_calculation.md):
- Distance threshold with hysteresis (`r_on` vs `r_off`)
- Velocity condition (`|v| <= v_hold`)
- Minimum duration filter (`min_on_frames >= 3`)
- Smear detection (foot near wall, no hold within radius)

---

## Efficiency Scoring

**Module**: [`src/pose_ai/recommendation/efficiency.py`](../src/pose_ai/recommendation/efficiency.py)

### Current Implementation (Rule-Based MVP)

5-component weighted score normalized to 0-1 range:

```python
def compute_efficiency_score(pose_frames, holds, wall_angle):
    """
    Components:
    1. detection_quality (0.20): Mean landmark visibility
    2. joint_smoothness (0.25): Inverse of joint angle variance
    3. com_stability (0.25): Inverse of COM movement variance
    4. contact_count (0.15): Number of holds in contact
    5. hip_wall_alignment (0.15): Hip alignment with wall normal
    """
    score = (
        0.20 * detection_quality +
        0.25 * joint_smoothness +
        0.25 * com_stability +
        0.15 * normalize_contact_count(contact_count) +
        0.15 * hip_wall_alignment
    )
    return clamp(score, 0.0, 1.0)
```

**Interpretation**:
- **0.8-1.0**: Excellent efficiency, smooth and controlled
- **0.6-0.8**: Good efficiency, minor inefficiencies
- **0.4-0.6**: Moderate efficiency, significant room for improvement
- **0.2-0.4**: Poor efficiency, unstable or jerky movement
- **0.0-0.2**: Very poor efficiency, loss of control or missing data

### Planned Enhancement (per efficiency_calculation.md)

7-component physics-based scoring:

1. **Support Polygon Stability** (w1=0.35):
   - Build convex hull from contact points
   - Compute COM distance to polygon
   - `stab = exp(-α * dist_normalized)`

2. **Support Count Penalty** (w2=0.20):
   - Strong penalty if fewer than 2 points of contact
   - Penalty for frequent contact switching

3. **Wall-Body Distance Penalty** (w3=0.10):
   - Forearm load increases with COM distance from wall
   - `pen_wall = w3 * ReLU(z_COM - z_ref)`

4. **Path Efficiency** (w4=0.25):
   - Net displacement vs actual path length
   - `eff_path = net_disp / (path_len + ε)`

5. **Smoothness Penalty** (w5=0.07):
   - Normalized jerk for COM and key limbs
   - Direction change penalties

6. **Reach-Limit Penalty** (w6=0.03):
   - Penalize extreme limb extensions
   - `pen_reach = w6 * max(0, reach_norm - τ_reach)`

7. **Technique Bonuses**:
   - Bicycle, back-flag, drop-knee detection
   - Add confidence-weighted bonuses

**Formula**:
```
score_eff = w1*stab + w4*eff_path 
            - (pen_support + pen_wall + pen_jerk + pen_reach)
            + technique_bonus
```

---

## Next-Action Recommendations

**Module**: [`src/pose_ai/recommendation/efficiency.py`](../src/pose_ai/recommendation/efficiency.py)

### Current Implementation (Distance-Based Heuristic)

```python
def suggest_next_holds(current_pose, holds, recently_used_holds, top_k=3):
    """
    Recommend next holds based on:
    1. Distance from current COM
    2. Recency penalty (avoid just-released holds)
    3. Directional bias (upward preferred)
    """
    candidates = []
    for hold in holds:
        dist = np.linalg.norm(hold.position - current_com)
        recency_penalty = 0.5 if hold.id in recently_used_holds else 0.0
        vertical_bonus = 0.2 if hold.position[1] < current_com[1] else 0.0
        
        score = 1.0 / (dist + 1e-6) + vertical_bonus - recency_penalty
        candidates.append((hold, score))
    
    return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
```

**Output Example**:
```json
{
  "recommendations": [
    {
      "hold_id": 12,
      "position": [245, 180],
      "score": 0.87,
      "limb": "right_hand",
      "reasoning": "Closest unused hold, upward progression"
    }
  ]
}
```

### Planned Enhancement (per efficiency_calculation.md)

**Rule-Based v1**:
1. Sample K candidate holds based on current support and target direction
2. For each candidate, simulate new support set → recompute support polygon
3. Estimate new efficiency score `score_eff'`
4. Prefer moves that:
   - Maintain/increase support count
   - Keep COM inside polygon
   - Respect reach and crossing constraints
5. Return best `score_eff'` candidate with limb/direction/hold

**Model-Based v2** (BiLSTM/Transformer):
- Input: Sliding window (T=32 frames @ 25fps ≈ 1.28s)
- Features: Normalized keypoints, v/a, COM, contact embeddings, efficiency metrics
- Architecture: BiLSTM (128-256) + attention pooling
- Head 1: Efficiency regression (Huber loss)
- Head 2: Next-action classification (limb × direction or hold cluster, CrossEntropy)
- Training: Weak labels from heuristic + fine-tune on human annotations

---

## Web API & UI

**Application**: [`webapp/main.py`](../webapp/main.py)  
**Job Runner**: [`webapp/pipeline_runner.py`](../webapp/pipeline_runner.py)

### Running the Server

```bash
# Install dependencies
pip install fastapi 'uvicorn[standard]' jinja2

# Launch server
PYTHONPATH=src uvicorn webapp.main:app --reload
```

Open http://127.0.0.1:8000

### REST API Endpoints

#### 1. Upload Video

```http
POST /api/upload
Content-Type: multipart/form-data

file: <video_file>
```

**Response**:
```json
{
  "filename": "video01.mp4",
  "upload_id": "c091d1fd722b4b328dc08dedceddbc85",
  "path": "data/uploads/c091d1fd722b4b328dc08dedceddbc85/video01.mp4"
}
```

#### 2. Create Pipeline Job

```http
POST /api/jobs
Content-Type: application/json

{
  "video_path": "data/uploads/.../video01.mp4",
  "frame_interval": 1.5,
  "yolo_model": "yolov8n",
  "yolo_conf": 0.25
}
```

**Response**:
```json
{
  "job_id": "job_20251118_143022_abc123",
  "status": "running",
  "created_at": "2025-11-18T14:30:22Z"
}
```

#### 3. Get Job Status

```http
GET /api/jobs/{job_id}
```

**Response**:
```json
{
  "job_id": "job_20251118_143022_abc123",
  "status": "completed",
  "progress": 100,
  "stages": {
    "extract_frames": "completed",
    "detect_holds": "completed",
    "estimate_wall_angle": "completed",
    "pose_estimation": "completed",
    "feature_extraction": "completed",
    "segmentation": "completed",
    "efficiency_scoring": "completed"
  },
  "artifacts": {
    "frames": "data/frames/video01",
    "holds": "data/frames/video01/holds.json",
    "pose_results": "data/frames/video01/pose_results.json",
    "features": "data/frames/video01/pose_features.json",
    "segments": "data/frames/video01/segment_metrics.json"
  },
  "gcs_uris": {
    "video": "gs://bucket/videos/raw/video01.mp4",
    "frames": "gs://bucket/videos/frames/video01/"
  }
}
```

#### 4. Get Analysis Results (New)

```http
GET /api/jobs/{job_id}/analysis
```

**Response**:
```json
{
  "job_id": "job_20251118_143022_abc123",
  "efficiency_score": 0.73,
  "components": {
    "detection_quality": 0.89,
    "joint_smoothness": 0.68,
    "com_stability": 0.71,
    "contact_count": 0.80,
    "hip_wall_alignment": 0.76
  },
  "recommendations": [
    {
      "hold_id": 12,
      "position": [245, 180],
      "score": 0.87,
      "limb": "right_hand"
    }
  ],
  "wall_angle": 15.3,
  "wall_confidence": 0.82
}
```

#### 5. Clear Job (New)

```http
DELETE /api/jobs/{job_id}
```

Removes job from active list (does not delete artifacts).

#### 6. Training Jobs

```http
POST /api/training/jobs
Content-Type: application/json

{
  "features_path": "data/frames/video01/pose_features.json",
  "label_column": "detection_score",
  "label_threshold": 0.6,
  "model_name": "xgb_v1"
}
```

**Response**:
```json
{
  "job_id": "train_20251118_150000_xyz789",
  "status": "running"
}
```

**Check Training Status**:
```http
GET /api/training/jobs/{job_id}
```

**Response**:
```json
{
  "job_id": "train_20251118_150000_xyz789",
  "status": "completed",
  "model_path": "models/xgb_v1.json",
  "metrics": {
    "accuracy": 0.87,
    "precision": 0.84,
    "recall": 0.89,
    "f1": 0.86
  },
  "gcs_uri": "gs://bucket/models/xgb_v1.json"
}
```

### Web UI Features

**Main Page** (`/`):
- Video upload form with YOLO configuration
- Job status table with real-time updates
- Inline pose visualization previews
- Download links for all artifacts

**Training Page** (`/training`):
- Features file upload
- Training parameter configuration
- Job status and metrics display
- Model download links

**Efficiency & Recommendations Card** (new):
- Overall efficiency score with color coding
- Component breakdown (5 metrics)
- Next-hold recommendations with visual indicators
- Wall angle estimate

---

## Machine Learning

### Current State

**XGBoost Baseline** ([`scripts/train_xgboost.py`](../scripts/train_xgboost.py)):
- Binary classification for pose quality (`detection_score`)
- Input: Frame-level features from `pose_features.json`
- Output: Trained model saved to `models/xgb_pose.json`

```bash
python scripts/train_xgboost.py \
  data/frames/video01/pose_features.json \
  --label-column detection_score \
  --label-threshold 0.6 \
  --model-out models/xgb_pose.json \
  --importance-out models/xgb_importance.csv
```

**Supported Parameters**:
- `--early-stopping-rounds`: Early stopping patience (default: 10)
- `--tree-method`: `hist` (CPU) or `gpu_hist` (GPU acceleration)
- `--test-size`: Train/test split ratio (default: 0.2)

### Planned Models (per efficiency_calculation.md)

#### 1. BiLSTM Multitask (v1)

**Architecture**:
```
Input: [T=32 frames, F features]
  ↓
BiLSTM(128-256, 1-2 layers)
  ↓
Attention Pooling
  ↓
├─ Head1: Efficiency Regression (Huber Loss)
└─ Head2: Next-Action Classification (CrossEntropy)
```

**Features**:
- Normalized keypoints (selected joints)
- Velocities and accelerations
- COM trajectory
- Contact embeddings (one-hot or learned)
- Per-frame efficiency metrics
- Hold type embeddings

**Training**:
1. Generate weak labels from heuristic efficiency scorer
2. Pre-train on weak labels
3. Fine-tune on small human-annotated dataset
4. Evaluate: MAE/R² (efficiency), top-1/top-3 accuracy (next action)

#### 2. Temporal Transformer / TCN (v2)

- Replace BiLSTM with Transformer encoder (2-4 layers, 4-8 heads)
- Alternative: Temporal Convolutional Network (TCN) for comparison
- Same multitask heads as v1
- Refine Head2 to predict next-hold ID or cluster directly

**Dataset Builder**:
- Sliding windows: length T=32, stride=1
- Inputs: [keypoints, v/a, COM, contacts, efficiency_features]
- Targets: [efficiency_score, next_action_label]

---

## Notebook Usage

**Jupyter Notebook**: [`notebooks/scripts_collection.ipynb`](../notebooks/scripts_collection.ipynb)

All pipeline components are demonstrated in functional notebook cells (English, no emojis).

### Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate 6156-capstone

# Register kernel
python -m ipykernel install --user --name 6156-capstone --display-name "6156 (py3.10)"
```

### Notebook Sections

1. **Environment Setup**: Imports and path configuration
2. **Frame Extraction Demo**: Extract frames from sample video
3. **Hold Detection Demo** (Cells 24-25): 
   - YOLOv8 inference
   - DBSCAN clustering
   - Visualization
4. **Wall Angle Estimation** (Cells 22-23):
   - Hough + PCA algorithm
   - Confidence scoring
5. **Pose Estimation Demo**: MediaPipe landmark detection
6. **Feature Engineering**: Joint angles, COM, velocities
7. **Efficiency Scoring** (Cells 26-27):
   - 5-component breakdown
   - Score visualization
8. **Complete Pipeline** (Cells 28-29):
   - End-to-end demonstration
   - Results analysis

### Running Cells

All cells are ready to execute in order. Markdown cells provide explanations; code cells include clear function-by-function breakdowns.

---

## Limitations & Known Issues

### Current Limitations

1. **Contact Inference**: Basic distance thresholding only
   - No velocity filtering
   - No hysteresis or minimum duration
   - No smear detection
   - **Planned**: Full algorithm per efficiency_calculation.md

2. **Efficiency Scoring**: Rule-based heuristic
   - 5 components vs planned 7
   - Fixed weights (not learned)
   - No support polygon analysis
   - No technique pattern detection
   - **Planned**: Physics-based scoring + technique bonuses

3. **Step Segmentation**: Simple movement/rest classification
   - No step boundary detection based on contact changes
   - No duration constraints (0.2-4s)
   - **Planned**: Confirmed contact-based segmentation

4. **Next-Action Recommendations**: Distance-only heuristic
   - No efficiency simulation
   - No support polygon constraints
   - No reach/crossing checks
   - **Planned**: Rule-based planner v1 → BiLSTM v2

5. **Hold Detection**: Generic YOLOv8
   - No transfer learning for climbing-specific holds
   - No hold type classification (crimp, sloper, jug, etc.)
   - **Planned**: Fine-tune on climbing hold dataset

6. **Wall Calibration**: Single-angle estimation
   - No multi-view calibration
   - No RANSAC plane fitting for complex walls
   - **Planned**: Advanced geometric calibration

### Known Issues

1. **MediaPipe z-coordinate**: Relative depth only, not absolute
   - Wall distance requires calibration or learned proxy
   - Current `com_perp_wall` is approximate

2. **Viewpoint Sensitivity**: Strong angle changes reduce accuracy
   - Recommend fixed camera position
   - Standardize capture guidelines

3. **Frame Blur**: No automatic filtering
   - Motion blur affects pose detection quality
   - **Planned**: Auto-detect and skip blurry frames

4. **Hold Tracking**: Frame-by-frame detection, no temporal consistency
   - **Planned**: IoU tracking + Kalman filter + embeddings

### Performance Targets (Initial)

- Hold detection mAP@0.5: yolov8n ≥ 0.60, yolov8m ≥ 0.68
- Wall angle MAE: vertical ≤ 5°, overhang ≤ 8°
- Efficiency vs expert correlation: ≥ 0.5
- Next-hold top-3 hit rate: ≥ 0.6

---

## Future Work

See [`IMPLEMENTATION_BACKLOG.md`](IMPLEMENTATION_BACKLOG.md) for detailed roadmap.

### Short-Term (Next 2-3 Weeks)

1. **Advanced Contact Inference**:
   - Implement hysteresis (r_on/r_off)
   - Add velocity condition
   - Minimum duration filter
   - Smear detection

2. **Step Segmentation**:
   - Contact-based step boundaries
   - Duration constraints (0.2-4s)
   - Segment labeling (Reach, Stabilize, FootAdjust, DynamicMove, Rest, Finish)

3. **Efficiency Enhancement**:
   - Support polygon stability
   - Path efficiency metric
   - Jerk/smoothness penalties
   - Full 7-component formula

4. **Hold Detection Improvements**:
   - Transfer learning: yolov8n vs yolov8m comparison
   - Hold type classification
   - Tracking across frames

### Medium-Term (1-2 Months)

1. **BiLSTM Multitask Model**:
   - Dataset builder (sliding windows)
   - Weak label generation
   - Training pipeline
   - Evaluation metrics

2. **Rule-Based Planner v1**:
   - Candidate hold sampling
   - Efficiency simulation
   - Support polygon constraints
   - Reach/crossing checks

3. **Technique Pattern Detection**:
   - Bicycle detection
   - Back-flag detection
   - Drop-knee detection
   - Confidence scoring

4. **Wall Calibration**:
   - RANSAC plane fitting
   - Multi-view geometry
   - Manual calibration UI

### Long-Term (3+ Months)

1. **Transformer/TCN Models**:
   - Architecture comparison
   - Hold-cluster prediction
   - Temporal attention analysis

2. **Advanced Features**:
   - Multi-sensor fusion (IMU, force plates)
   - Climber profiling and personalization
   - Route difficulty estimation
   - Collaborative filtering from community data

3. **Production Readiness**:
   - Model versioning and registry
   - A/B testing framework
   - Real-time inference optimization
   - Monitoring and retraining pipeline

4. **Data Infrastructure**:
   - DVC for dataset versioning
   - Parquet artifact format
   - Experiment manifests
   - CI/CD pipeline (lint, test, coverage, Docker)

---

## References

- **Authoritative Spec**: [`efficiency_calculation.md`](efficiency_calculation.md) — Complete algorithm specifications
- **Implementation Backlog**: [`IMPLEMENTATION_BACKLOG.md`](IMPLEMENTATION_BACKLOG.md) — Prioritized feature roadmap
- **Original Roadmap**: [`beta_model_plan.md`](beta_model_plan.md) — High-level project phases
- **Source Code**: [`src/pose_ai/`](../src/pose_ai/) — Core library modules
- **Scripts**: [`scripts/`](../scripts/) — CLI tools for pipeline stages
- **Web Application**: [`webapp/`](../webapp/) — FastAPI server and UI
- **Tests**: [`tests/unit/`](../tests/unit/) — Unit test suite

---

**For questions or contributions, see project documentation in `/docs` or contact the development team.**
