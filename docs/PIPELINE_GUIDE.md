# Climbing Video Analysis Pipeline — Complete Guide

**Version**: 2.0  
**Last Updated**: November 20, 2025  
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
12. [References](#references)
13. [Appendix A: Efficiency Calculation Specification](#appendix-a-efficiency-calculation-specification)
14. [Appendix B: Project Roadmap](#appendix-b-project-roadmap)
15. [Appendix C: Testing Guide](#appendix-c-testing-guide)

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
- **Hold Detection**: YOLOv8n/m object detection with DBSCAN spatial clustering and temporal tracking
- **Hold Type Classification**: YOLOv8 fine-tuning for hold types (crimp, sloper, jug, pinch, foot_only, volume)
- **Wall Angle**: IMU sensor integration (±1° accuracy) with vision-based fallback (±5°)
- **Efficiency Scoring**: 7-component physics-based metric with technique bonuses
- **Climber Personalization**: Height, wingspan, and flexibility parameters for personalized recommendations
- **Next-Action Recommendations**: Rule-based planner with hold type awareness and personalized reach constraints
- **ML Models**: BiLSTM and Transformer multitask models for efficiency and next-action prediction
- **Route Difficulty Estimation**: XGBoost model for V0-V10 grade prediction
- **Cloud Storage**: Optional GCS integration for videos, frames, and models
- **Web Interface**: FastAPI with background job management, real-time status updates, and grading UI

### Technology Stack

- **Core**: Python 3.10+
- **Computer Vision**: MediaPipe 0.10.9, OpenCV, Ultralytics YOLOv8
- **ML**: PyTorch (BiLSTM, Transformer), XGBoost, scikit-learn
- **Web**: FastAPI, Uvicorn, Jinja2
- **Storage**: Local filesystem + optional Google Cloud Storage
- **Testing**: pytest
- **Documentation**: See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive testing specifications

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

Automatically estimates wall angle from video frames using edge detection and geometric analysis. **Now supports IMU sensor data for improved accuracy.**

**Priority Order**:
1. **IMU Sensor Data** (if provided): Most accurate (±1°)
2. **Vision-based Estimation** (fallback): Hough + PCA (±5°)

**IMU Integration**:
```python
from pose_ai.wall.angle import compute_wall_angle_from_imu

# From quaternion
result = compute_wall_angle_from_imu(
    quaternion=[0.7071, 0.0, 0.7071, 0.0]  # [w, x, y, z]
)

# Or from Euler angles
result = compute_wall_angle_from_imu(
    euler_angles=[85.5, 2.0, 0.0]  # [pitch, roll, yaw] in degrees
)
```

**Vision-based Algorithm** (fallback):
1. **Hough Line Detection**: Detect vertical/near-vertical lines in edge-filtered frames
2. **RANSAC Fitting**: Robust line fitting to handle outliers
3. **PCA Fallback**: Use principal component analysis if insufficient lines detected
4. **Confidence Scoring**: Based on line consensus and edge density

**Output**: Angle in degrees (0° = horizontal, 90° = vertical), confidence score 0-1

**Typical Performance**:
- **IMU sensor**: MAE ≤ 1° (highly accurate)
- **Vision-based**: Vertical walls MAE ≤ 5°, Overhangs MAE ≤ 8°

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

4. **Wall Alignment**:
   - `wall_angle`: Estimated wall inclination (from IMU or vision)
   - `hip_alignment_error`: Hip deviation from wall normal
   - `com_along_wall`: COM projection along wall surface
   - `com_perp_wall`: COM distance perpendicular to wall

5. **Climber Personalization** (NEW):
   - `climber_height`: Height in cm (for body scale normalization)
   - `climber_wingspan`: Wingspan in cm (for personalized reach constraints)
   - `climber_flexibility`: Flexibility score 0-1 (for threshold adjustments)
   - `body_scale_normalized`: Body scale adjusted by height (expected shoulder width = height × 0.16)

**Personalization Effects**:
- **Body Scale Normalization**: Adjusts for different body proportions using climber height
- **Reach Constraints**: Personalized reach limits based on wingspan/height ratio and flexibility
- **Efficiency Scoring**: Flexibility-adjusted reach penalty thresholds (5-10% bonus for flexible climbers)

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

### Current Implementation (Physics-Based with Technique Bonuses)

7-component weighted score with technique bonuses:

```python
def compute_efficiency_score(pose_frames, holds, wall_angle):
    """
    Components:
    1. stability (0.35): Support polygon stability (COM distance to convex hull)
    2. path_efficiency (0.25): Net displacement vs actual path length
    3. support_penalty (0.20): Penalty if support count < 2 + contact switching
    4. wall_penalty (0.10): COM distance from wall (forearm load proxy)
    5. jerk_penalty (0.07): Smoothness based on jerk (third derivative)
    6. reach_penalty (0.03): Penalty for extreme limb extensions
    7. technique_bonus: Bicycle (0.05) + Back-flag (0.05) + Drop-knee (0.03)
    """
    score = (
        0.35 * stability +
        0.25 * path_efficiency
        - (0.20 * support_penalty +
           0.10 * wall_penalty +
           0.07 * jerk_penalty +
           0.03 * reach_penalty)
        + technique_bonus
    )
    return score  # Range approximately 0.0-1.0
```

**Interpretation**:
- **0.8-1.0**: Excellent efficiency, smooth and controlled
- **0.6-0.8**: Good efficiency, minor inefficiencies
- **0.4-0.6**: Moderate efficiency, significant room for improvement
- **0.2-0.4**: Poor efficiency, unstable or jerky movement
- **0.0-0.2**: Very poor efficiency, loss of control or missing data

### Implementation Details

**Support Polygon Stability**:
- Uses `scipy.spatial.ConvexHull` when available (3+ contact points)
- Falls back to simple polygon if scipy not available or < 3 points
- Stability score: `exp(-α * distance_to_polygon / body_scale)`
- Default α = 4.0

**Technique Detection** (per efficiency_calculation.md):
- **Bicycle**: Both feet on same/near hold, opposing toe vectors
- **Back-flag**: Free leg extended behind body, hip rotation counter
- **Drop-knee**: Knee internal rotation with pelvic twist
- Confidence scores (0-1) computed per frame, averaged across step

**Path Efficiency**:
- Accumulates COM path length frame-by-frame
- Compares to net displacement: `net_disp / (path_len + ε)`
- Rewards direct, economical movement

**Jerk Penalty**:
- Third derivative of position (computed via `kinematics.py`)
- Penalizes jerky, uncontrolled movements
- Normalized by body scale

**Contact Inference Integration**:
- Uses advanced contact filter with hysteresis (r_on/r_off)
- Velocity condition enforced (low speed required for contact)
- Minimum duration filter (≥3 frames)
- Smear detection for feet near wall

---

## Next-Action Recommendations

**Module**: [`src/pose_ai/recommendation/efficiency.py`](../src/pose_ai/recommendation/efficiency.py)

### Current Implementation

**Basic Version** (`suggest_next_actions`):
- Distance-based heuristic with COM proximity
- Recency penalty (avoid just-released holds)
- Directional bias (upward preferred)
- Fast, simple, no simulation

**Advanced Version** (`suggest_next_actions_advanced`):
- Rule-based planner with efficiency simulation
- Candidate hold sampling (K=10, upward bias)
- Support simulation (computes new contact set)
- Efficiency simulation (recomputes stability with new support polygon)
- Constraint checking:
  - Support count ≥ 2
  - COM inside/near polygon (tolerance)
  - **Personalized reach limit check** (based on wingspan, height, flexibility)
  - No limb crossing
- Ranking by simulated efficiency gain (Δeff)
- **Hold type awareness**: Prefers jugs when support is low, crimp/sloper for reach moves

```python
from pose_ai.recommendation.planner import NextMovePlanner

planner = NextMovePlanner()
candidates = planner.plan_next_move(current_row, holds, top_k=3)
# Returns MoveCandidate objects with efficiency estimates
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

### Future Enhancement (Model-Based v2)

**BiLSTM/Transformer**:
- Input: Sliding window (T=32 frames @ 25fps ≈ 1.28s)
- Features: Normalized keypoints, v/a, COM, contact embeddings, efficiency metrics
- Architecture: BiLSTM (128-256) + attention pooling
- Head 1: Efficiency regression (Huber loss)
- Head 2: Next-action classification (limb × direction or hold cluster, CrossEntropy)
- Training: Weak labels from heuristic + fine-tune on human annotations

See [IMPLEMENTATION_BACKLOG.md](IMPLEMENTATION_BACKLOG.md) for detailed roadmap.

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
  }
}
```

**Metadata Fields**:
- `imu_quaternion`: Device orientation as quaternion [w, x, y, z] (for accurate wall angle)
- `imu_euler_angles`: Device orientation as Euler angles [pitch, roll, yaw] in degrees
- `climber_height`: Height in cm (for body scale normalization)
- `climber_wingspan`: Wingspan in cm (for personalized reach constraints)
- `climber_flexibility`: Flexibility score 0-1 (for threshold adjustments)

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

#### 5. Get ML Predictions

```http
GET /api/jobs/{job_id}/ml_predictions
```

**Response**:
```json
{
  "job_id": "job_20251118_143022_abc123",
  "model_type": "bilstm",
  "num_predictions": 25,
  "predictions": [
    {
      "frame_index": 16,
      "efficiency_score": 0.72,
      "next_action": "left_hand",
      "next_action_class": 1,
      "next_action_probs": [0.1, 0.6, 0.2, 0.05, 0.05]
    },
    ...
  ]
}
```

**Note**: Requires trained model at `models/checkpoints/bilstm_multitask.pt` or `transformer_multitask.pt`. Auto-detects model type from checkpoint.

#### 6. Get Route Grade

```http
GET /api/jobs/{job_id}/route_grade
```

**Response**:
```json
{
  "job_id": "job_20251118_143022_abc123",
  "grade": 5.2,
  "confidence": 0.85,
  "calibrated_grade": "Advanced (V5)",
  "features": {
    "hold_density": 12.5,
    "hold_count": 25,
    "hold_spacing_mean": 0.15,
    "wall_angle": 90.0,
    "step_count": 8,
    "avg_efficiency": 0.65,
    "contact_switches_per_second": 2.3,
    "hold_type_ratio_jug": 0.4,
    "route_length": 0.8,
    "duration_seconds": 12.5
  }
}
```

**Note**: Requires completed pipeline job. Returns `grade: null` if model not trained.

#### 7. Clear Job

```http
DELETE /api/jobs/{job_id}
```

Removes job from active list (does not delete artifacts).

#### 8. Training Jobs

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

**Route Grading Page** (`/grading`) — NEW:
- Route selection dropdown (completed jobs only)
- Predicted difficulty grade display (V0-V10)
- Color-coded difficulty indicators (green=easy, red=hard)
- Confidence indicator with progress bar
- Feature breakdown grid (hold density, complexity, etc.)
- Gym calibration display

**Efficiency & Recommendations Card**:
- Overall efficiency score with color coding
- Component breakdown (7 metrics)
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

### Implemented Models

#### 1. BiLSTM Multitask Model (v1) — ✅ IMPLEMENTED

**Status**: Fully implemented with training, evaluation, and inference pipelines

**Architecture**:
```
Input: [T=32 frames, F=60 features]
  ↓
BiLSTM(hidden_dim=128, num_layers=2, bidirectional=True)
  ↓
Attention Pooling (optional)
  ↓
├─ Head1: Efficiency Regression (Huber Loss)
└─ Head2: Next-Action Classification (5 classes, CrossEntropy)
```

**Features** (60 dimensions):
- Joint positions (x, y) for 8 selected joints: 16 features
- Joint velocities (vx, vy): 16 features
- Joint accelerations (ax, ay): 16 features
- COM position and velocity: 4 features
- Contact states (4 limbs): 4 features
- Support count: 1 feature
- Body scale: 1 feature
- Wall distance: 1 feature
- Efficiency: 1 feature

**Training**:
```bash
# Train BiLSTM model
python scripts/train_model.py \
  --data data/features \
  --model-type bilstm \
  --epochs 100 \
  --hidden-dim 128 \
  --num-layers 2 \
  --device cuda
```

**Evaluation**:
```bash
# Evaluate model (auto-detects model type)
python scripts/evaluate_model.py \
  --model models/checkpoints/bilstm_multitask.pt \
  --data data/features \
  --split test
```

**Output Metrics**:
- Efficiency: MAE, RMSE, R², correlation coefficient
- Action: Top-1 accuracy, per-class accuracy, confusion matrix

#### 2. Transformer Multitask Model (v2) — ✅ IMPLEMENTED

**Status**: Fully implemented alongside BiLSTM, supports model selection via CLI

**Architecture**:
```
Input: [T=32 frames, F=60 features]
  ↓
Input Projection (if input_dim != d_model)
  ↓
Positional Encoding (sinusoidal or learnable)
  ↓
Transformer Encoder (2-4 layers, 4-8 heads)
  ↓
Pooling (mean/max/cls)
  ↓
├─ Head1: Efficiency Regression (Huber Loss)
└─ Head2: Next-Action Classification (5 classes, CrossEntropy)
```

**Key Features**:
- Multi-head self-attention (4-8 heads)
- Positional encoding (sinusoidal or learnable)
- Configurable pooling strategies (mean, max, cls token)
- Same feature set as BiLSTM

**Training**:
```bash
# Train Transformer model
python scripts/train_model.py \
  --data data/features \
  --model-type transformer \
  --epochs 100 \
  --d-model 128 \
  --num-layers 4 \
  --num-heads 8 \
  --pooling mean \
  --device cuda
```

**Model Selection**:
- Unified training script: `scripts/train_model.py` with `--model-type` flag
- Auto-detection in evaluation and inference
- Backward compatible: `train_bilstm.py` still works (wrapper)

#### 3. Route Difficulty Estimation — ✅ IMPLEMENTED

**Status**: XGBoost model for route difficulty prediction (V0-V10)

**Purpose**: Predict route difficulty grade from video analysis

**Features** (20+ route-level features):
- Hold density: Holds per square meter
- Hold spacing: Mean/median/std/min/max distances
- Wall angle: From IMU or vision
- Move complexity: Step count, efficiency stats, reach penalties, contact switches
- Hold type distribution: Ratios for jug/crimp/sloper/pinch/foot_only/volume
- Route length: Vertical distance climbed
- Duration: Total climbing time

**Training**:
```bash
# Train route grader model
python scripts/train_route_grader.py \
  --data data/routes_annotated \
  --model-out models/route_grader.json \
  --n-estimators 100 \
  --max-depth 6 \
  --learning-rate 0.1
```

**Expected Data Structure**:
```
data/routes_annotated/
  route1/
    pose_features.json
    step_efficiency.json
    holds.json
    grade.txt  # Ground truth: 5.0 (V5)
  route2/
    ...
```

**Model Usage**:
```python
from pose_ai.ml.route_grading import RouteDifficultyModel, extract_route_features

# Extract features
features = extract_route_features(
    feature_rows, step_segments, step_efficiency, holds, wall_angle
)

# Predict grade
model = RouteDifficultyModel(model_path=Path("models/route_grader.json"))
grade, confidence = model.predict_with_confidence(features)
# Returns: (5.2, 0.85)  # V5.2 with 85% confidence
```

**Gym Calibration**:
```python
from pose_ai.ml.route_grading import GymGradeCalibration

calibration = GymGradeCalibration()
calibrated = calibration.calibrate(5.2)
# Returns: "Advanced (V5)"
```

**Default Grade Mapping**:
- V0-V2: Beginner
- V3-V4: Intermediate
- V5-V6: Advanced
- V7-V8: Expert
- V9-V10: Elite

### Dataset Builder

**Module**: [`src/pose_ai/ml/dataset.py`](../src/pose_ai/ml/dataset.py)

**Features**:
- Sliding windows: length T=32, stride=1 (configurable)
- Z-score normalization per feature dimension
- Next-action label extraction with lookahead window (5 frames)
- Train/val/test split support (default 70/20/10)

**Usage**:
```python
from pose_ai.ml.dataset import create_datasets_from_directory

train_dataset, val_dataset, test_dataset = create_datasets_from_directory(
    data_dir="data/features",
    window_size=32,
    stride=1,
    train_split=0.7,
    val_split=0.2,
    normalize=True,
)
```

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

1. **✅ Contact Inference**: FULLY IMPLEMENTED
   - Hysteresis (r_on/r_off) ✅
   - Velocity filtering ✅
   - Minimum duration (≥3 frames) ✅
   - Smear detection ✅

2. **✅ Efficiency Scoring**: FULLY IMPLEMENTED
   - 7-component physics-based formula ✅
   - Support polygon with ConvexHull ✅
   - Technique detection (bicycle, back-flag, drop-knee) ✅
   - Technique bonuses integrated ✅

3. **✅ Step Segmentation**: FULLY IMPLEMENTED
   - Contact-based segmentation ✅
   - Duration constraints (0.2-4s) ✅
   - Step labels (Reach, Stabilize, FootAdjust, DynamicMove, Rest, Finish) ✅

4. **✅ Next-Action Recommendations**: RULE-BASED PLANNER IMPLEMENTED
   - Basic distance heuristic available
   - Advanced planner with efficiency simulation ✅
   - Support polygon constraints ✅
   - Reach/crossing checks ✅
   - **Future**: BiLSTM/Transformer model (v2)

5. **Hold Detection**: Functional but not optimized
   - Generic YOLOv8n/m (mAP ≥ 0.60/0.68)
   - No hold type classification yet
   - **Planned**: Transfer learning, type classification (crimp, sloper, jug, etc.)

6. **Wall Calibration**: Basic single-angle estimation
   - Hough + PCA (MAE ≤ 5-8°)
   - **Planned**: RANSAC plane fitting, multi-view calibration

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

- **Implementation Backlog**: [`IMPLEMENTATION_BACKLOG.md`](IMPLEMENTATION_BACKLOG.md) — Detailed feature roadmap and implementation status
- **Source Code**: [`src/pose_ai/`](../src/pose_ai/) — Core library modules
- **Scripts**: [`scripts/`](../scripts/) — CLI tools for pipeline stages
- **Web Application**: [`webapp/`](../webapp/) — FastAPI server and UI
- **Tests**: [`tests/unit/`](../tests/unit/) — Unit test suite

**Note**: The following documents have been consolidated into this guide:
- `efficiency_calculation.md` → See [Appendix A: Efficiency Calculation Specification](#appendix-a-efficiency-calculation-specification)
- `beta_model_plan.md` → See [Appendix B: Project Roadmap](#appendix-b-project-roadmap)
- `TESTING_GUIDE.md` → See [Appendix C: Testing Guide](#appendix-c-testing-guide)

---

## Appendix A: Efficiency Calculation Specification

**Source**: Consolidated from `efficiency_calculation.md` (authoritative specification)

### Purpose

Compute **move/step-level efficiency** and use it to power **next-action recommendations**.

**Input**: Uploaded bouldering video → frame sequence with MediaPipe keypoints, hold locations, optional wall calibration  
**Output**: Step-level efficiency score (0–1) and diagnostic metrics

### Contact Inference Algorithm

Contact is determined by **distance**, **velocity**, and **temporal hysteresis**.

1. **Distance Threshold**: `d(J,H) ≤ r_th` where `r_th = k_r × body_scale` (recommend 0.25–0.35 × shoulder width)
2. **Velocity Condition**: `|v_J| ≤ v_hold` where `v_hold = k_v × body_scale / fps` (recommend 0.02–0.05)
3. **Hysteresis**: Dual thresholds `r_on < r_off` to reduce flicker
   - **On**: `d ≤ r_on` AND `|v| ≤ v_hold`
   - **Stay on**: `d ≤ r_off`
4. **Minimum Duration**: Enforce `min_on_frames ≥ 3` for confirmed contact
5. **Smear Detection**: Foot near wall (`|z_foot − z_wall| ≤ z_eps`), no hold within `r_smear`, low speed
6. **Technique Patterns**: Bicycle, back-flag, drop-knee detection with confidence scores (0–1)

**Constants**:
- `r_on = 0.22 × body_scale`
- `r_off = 0.28 × body_scale`
- `v_hold = 0.03 × body_scale/fps`
- `min_on_frames = 3`
- `z_eps = 0.03` (smear detection)

### Efficiency Formula (7 Components)

**Frame-level computation, aggregated to step-level**:

```
score_eff = w1*stab + w4*eff_path 
            - (pen_support + pen_wall + pen_jerk + pen_reach)
            + technique_bonus
```

**Component Details**:

1. **Support Polygon Stability** (w1=0.35):
   - Build convex hull from contact points
   - Compute `dist(COM, polygon)` normalized by body_scale
   - `stab = exp(-α × dist_normalized)` where α=4.0

2. **Support Count/Switch Penalties** (w2=0.20):
   - Strong penalty if `n_support < 2`
   - Penalty for frequent contact switching

3. **Wall-Body Distance Penalty** (w3=0.10):
   - `pen_wall = w3 × ReLU(z_COM - z_ref)`
   - Proxy for forearm load

4. **Path Efficiency** (w4=0.25):
   - `net_disp = ||COM_end - COM_start||`
   - `path_len = Σ||COM_t - COM_{t-1}||`
   - `eff_path = clamp(net_disp / (path_len + ε), 0, 1)`

5. **Smoothness Penalty** (w5=0.07):
   - Mean normalized jerk for COM and key limbs
   - Direction change penalties

6. **Reach-Limit Penalty** (w6=0.03):
   - Penalize extreme limb extensions
   - `pen_reach = w6 × max(0, reach_norm - τ_reach)`
   - Personalized based on climber flexibility

7. **Technique Bonuses**:
   - Bicycle: 0.05 × confidence
   - Back-flag: 0.05 × confidence
   - Drop-knee: 0.03 × confidence

**Step Aggregation**: Weighted mean of frame scores (or top-quantile)

### Step Segmentation

- Split on confirmed contact changes (limb changes hold)
- Priority: Single-limb change while others maintain support
- Duration constraints: 0.2s ≤ step_duration ≤ 4s
- Segment labels: Reach, Stabilize, FootAdjust, DynamicMove, Rest, Finish

### Data Representation

**Frame-Level Features**:
- Keypoints: Shoulders, elbows, wrists, hips, knees, ankles (normalized, root-relative)
- Kinematic derivatives: Velocity, acceleration, jerk
- COM: Center of mass position, velocity, acceleration
- Contact states: Per-limb contact (on/off, hold_id, type)
- Technique scores: Bicycle, back-flag, drop-knee confidence

**Normalization**:
- Root-relative: Subtract hip center from all joints
- Scale normalization: Divide by body reference (shoulder width)
- View normalization: Optional homography to wall coordinate system

---

## Appendix B: Project Roadmap

**Source**: Consolidated from `beta_model_plan.md`

### Implementation Status (November 2025)

**✅ Completed Features**:
- Hold Detection: YOLOv8n/m with DBSCAN clustering and temporal tracking
- Wall Angle: IMU sensor integration (±1° accuracy) with vision fallback
- Efficiency Scoring: 7-component physics-based metric with technique bonuses
- Step Segmentation: Contact-based with duration constraints and labels
- Next-Action Recommendations: Rule-based planner v1 with hold type awareness
- Hold Type Classification: YOLOv8 fine-tuning infrastructure
- BiLSTM Multitask Model: Full training/evaluation/inference pipeline
- Transformer Multitask Model: Alternative architecture with model selection
- IMU Sensor Integration: Device orientation for accurate wall angle
- Climber Personalization: Height, wingspan, flexibility parameters
- Route Difficulty Estimation: XGBoost model for V0-V10 prediction

**🔄 In Progress**:
- Model training on real climbing data
- Performance comparison: BiLSTM vs Transformer

**📋 Planned**:
- Production infrastructure (CI/CD, monitoring)
- Advanced wall calibration (RANSAC, multi-view)
- Climber profiling database (external app integration)

### Development Phases

**Phase 1**: Core pipeline (✅ Completed)
- Frame extraction, pose estimation, hold detection
- Basic efficiency scoring and recommendations

**Phase 2**: Advanced features (✅ Completed)
- Advanced contact inference, step segmentation
- Full efficiency formula, rule-based planner
- Hold type classification, ML models

**Phase 3**: Personalization & grading (✅ Completed)
- IMU sensor integration
- Climber personalization
- Route difficulty estimation

**Phase 4**: Production readiness (🔄 Planned)
- CI/CD pipeline
- Monitoring and logging
- Model registry and versioning

See [IMPLEMENTATION_BACKLOG.md](IMPLEMENTATION_BACKLOG.md) for detailed task breakdown.

---

## Appendix C: Testing Guide

**Source**: Consolidated from `TESTING_GUIDE.md`

### Overview

Comprehensive testing documentation for all pipeline steps. Each step includes input/output specifications, validation criteria, and test cases.

### Pipeline Steps

1. **Frame Extraction**: Extract frames at intervals, validate timestamps and image quality
2. **Pose Estimation**: MediaPipe landmarks, validate detection scores and coordinate ranges
3. **Hold Detection**: YOLO detection + clustering, validate hold count and positions
4. **Feature Extraction**: Frame-level features, validate feature completeness and temporal consistency
5. **Step Segmentation**: Contact-based segmentation, validate duration constraints and labels
6. **Efficiency Scoring**: Step-level scores, validate component values and score ranges
7. **ML Model Inference**: BiLSTM/Transformer predictions, validate efficiency and action outputs
8. **Route Grading**: V0-V10 prediction, validate feature extraction and grade ranges

### Test Data Requirements

- Sample videos: Short (5-10s), normal (30-60s), long (2+ min)
- Known difficulty routes: V3, V5, V7 for validation
- Edge cases: No holds, single frame, extreme angles, occlusion

### Validation Criteria

Each step has specific validation criteria:
- **Frame Extraction**: Frame count, sequential timestamps, readable images
- **Pose Estimation**: 33 landmarks per frame, valid coordinates, detection scores
- **Hold Detection**: Hold count > 0, valid coordinates, confidence scores
- **Feature Extraction**: Feature completeness, value ranges, temporal consistency
- **Step Segmentation**: Duration constraints, valid labels, no overlaps
- **Efficiency Scoring**: Score ranges, component completeness, temporal smoothness
- **ML Inference**: Prediction ranges, probability distributions, frame indices
- **Route Grading**: Grade range [0-10], feature completeness, confidence scores

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
python scripts/run_pipeline.py test_data/sample_climb.mp4 --out test_output

# Manual testing via web UI
# Navigate to http://localhost:8000
```

**See [TESTING_GUIDE.md](TESTING_GUIDE.md) for complete test case specifications and detailed input/output formats.**

---

**For questions or contributions, see project documentation in `/docs` or contact the development team.**
