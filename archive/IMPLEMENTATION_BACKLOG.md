# Implementation Backlog

**Last Updated**: November 20, 2025  
**Purpose**: Detailed feature roadmap documenting gaps between current implementation and complete specification per [efficiency_calculation.md](efficiency_calculation.md)

---

## Priority Legend

- **P0**: Critical for core functionality
- **P1**: High impact, needed for production readiness
- **P2**: Important enhancements
- **P3**: Nice-to-have, future optimization

---

## Short-Term (Next 2-3 Weeks)

### ✅ P0: Advanced Contact Inference — COMPLETED

**Status**: Fully implemented in `src/pose_ai/features/contacts.py`

**Completed Implementation** (per efficiency_calculation.md):

- Hysteresis thresholds: `r_on` (contact start) vs `r_off` (contact end)
- Velocity condition: `|v_joint| <= v_hold` (low speed required for contact)
- Minimum duration filter: `min_on_frames >= 3` (ignore brief touches)
- Smear detection: Foot near wall, no hold within radius
- Optional: HMM/Viterbi temporal smoothing for noisy sequences

**Completed Tasks**:

1. [x] Add `r_on`/`r_off` parameters to contact inference (r_on=0.22, r_off=0.28 × body_scale)
2. [x] Compute joint velocities in feature aggregation (`kinematics.py`)
3. [x] Apply velocity threshold (`v_hold = 0.03 × body_scale/fps`)
4. [x] Implement contact state machine with hysteresis
5. [x] Add minimum duration filter with pending_on counter
6. [x] Detect smears: `|z_foot - z_wall| <= z_eps` AND `no hold within r_smear`
7. [x] Technique detection integrated (bicycle, back-flag, drop-knee)

**Files Modified**:

- `src/pose_ai/features/contacts.py` (full implementation)
- `src/pose_ai/features/kinematics.py` (velocity/acceleration/jerk)
- `src/pose_ai/service/feature_service.py` (pipeline integration)

---

### ✅ P0: Step Segmentation with Contact Boundaries — COMPLETED

**Status**: Fully implemented in `src/pose_ai/segmentation/steps.py`

**Completed Implementation** (per efficiency_calculation.md):

- Split on confirmed contact changes (limb changes hold)
- Priority: Single-limb change while others maintain support
- Duration constraints: 0.2s ≤ step_duration ≤ 4s
- Segment labels: Reach, Stabilize, FootAdjust, DynamicMove, Rest, Finish

**Completed Tasks**:

1. [x] Detect contact change events from contact filter output
2. [x] Implement step boundary detection (prefer single-limb transitions)
3. [x] Enforce duration constraints (min 0.2s, max 4s)
4. [x] Rule-based segment labeling with `_classify_step_label` function:
   - **Reach**: Contact count increases or high velocity with hand movement
   - **Stabilize**: Contacts maintained, low velocity
   - **FootAdjust**: Only foot changed, hands stable
   - **DynamicMove**: Multiple limbs changed or very high velocity
   - **Rest**: Very low velocity, stable contacts (≥3 support)
   - **Finish**: At top position, sustained stability
5. [x] Added segment metadata (step_id, start/end indices, timestamps, label)
6. [x] Integration with pipeline via `segment_steps_by_contacts`

**Files Modified**:

- `src/pose_ai/segmentation/steps.py` (complete implementation)
- `src/pose_ai/service/feature_service.py` (pipeline integration)

---

### ✅ P1: Full 7-Component Efficiency Formula — COMPLETED

**Status**: Fully implemented in `src/pose_ai/recommendation/efficiency.py`

**Completed Implementation** (per efficiency_calculation.md):

#### Component Breakdown

1. **Support Polygon Stability** (w1=0.35):

   - Build convex hull from contact points
   - Compute `dist(COM, polygon)` (0 if inside)
   - Normalize by body_scale
   - `stab = exp(-α * dist_normalized)`

2. **Support Count/Switch Penalties** (w2=0.20):

   - Strong penalty if `n_support < 2`
   - Penalty for frequent contact switching
   - `pen_support = w2 * (1 if n_support < 2 else 0) + pen_switch`

3. **Wall-Body Distance Penalty** (w3=0.10):

   - Forearm load proxy: `pen_wall = w3 * ReLU(z_COM - z_ref)`
   - Requires wall plane calibration

4. **Path Efficiency** (w4=0.25):

   - Net displacement vs actual path length
   - `net_disp = ||COM_end - COM_start||`
   - `path_len = Σ||COM_t - COM_{t-1}||`
   - `eff_path = clamp(net_disp / (path_len + ε), 0, 1)`

5. **Smoothness Penalty** (w5=0.07):

   - Mean normalized jerk for COM and key limbs
   - Direction change penalties
   - `pen_jerk = w5 * mean_jerk_norm`

6. **Reach-Limit Penalty** (w6=0.03):

   - Penalize extreme limb extensions
   - `reach_norm = limb_length / (p95_reach × body_scale)`
   - `pen_reach = w6 * max(0, reach_norm - τ_reach)`

7. **Technique Bonuses**:
   - Bicycle, back-flag, drop-knee detection
   - Confidence-weighted bonuses

**Formula**:

```
score_eff = w1*stab + w4*eff_path
            - (pen_support + pen_wall + pen_jerk + pen_reach)
            + technique_bonus
```

**Completed Tasks**:

1. [x] Implement convex hull from contact points (`_compute_convex_hull` with scipy.spatial.ConvexHull)
2. [x] Compute COM-to-polygon distance (`_distance_to_polygon`, `_point_in_polygon`)
3. [x] Track contact switching frequency per step
4. [x] Compute wall-body distance (uses `com_perp_wall` from features)
5. [x] Path efficiency: accumulate COM path length vs net displacement
6. [x] Jerk computation: third derivative via `kinematics.py`
7. [x] Reach normalization: limb length relative to body_scale
8. [x] Technique pattern detection (bicycle, back-flag, drop-knee via `contacts.py`)
9. [x] Aggregate frame-level scores to step-level (mean of frame scores)
10. [x] Updated efficiency scoring with `StepEfficiencyComputer` class

**Formula Implemented**:

```python
score = (
    w1 * stability +
    w4 * path_efficiency
    - (w2 * support_penalty +
       w3 * wall_penalty +
       w5 * jerk_penalty +
       w6 * reach_penalty)
    + technique_bonus
)
```

**Files Modified**:

- `src/pose_ai/recommendation/efficiency.py` (complete implementation)
- `src/pose_ai/features/kinematics.py` (jerk computation)
- `src/pose_ai/features/contacts.py` (technique detection)

---

### ✅ P1: Hold Type Classification — COMPLETED

**Status**: Fully implemented with annotation tools, training script, and integration

**Completed Implementation**:

**Phase 1.1: Annotation Infrastructure**

- [x] Created `HoldAnnotation` dataclass with YOLO format support
- [x] Interactive annotation tool (`scripts/annotate_holds.py`) with OpenCV
- [x] Dataset structure initialization (`data/holds_training/`)
- [x] 6 hold type classes: crimp, sloper, jug, pinch, foot_only, volume
- [x] YOLO dataset.yaml configuration generator

**Phase 1.2: YOLOv8 Fine-Tuning**

- [x] Training script (`scripts/train_yolo_holds.py`) with full hyperparameter control
- [x] Support for augmentation, early stopping, checkpointing
- [x] ONNX export capability
- [x] Validation metrics (mAP@0.5, mAP@0.5:0.95, per-class metrics)

**Phase 1.3: Hold Extraction Integration**

- [x] Added `hold_type` and `type_confidence` fields to `HoldDetection` and `ClusteredHold`
- [x] Dominant type aggregation in `cluster_holds` function
- [x] Type information exported in `holds.json` schema
- [x] Automatic type detection from YOLO labels (if model predicts specific types)

**Phase 1.4: Recommendation Integration**

- [x] Enhanced `PlannerConfig` with hold type preferences:
  - `prefer_jug_when_low_support`: Prefer jugs when support count < 3
  - `jug_bonus`: Efficiency bonus for jug holds (default 0.05)
  - `reach_hold_bonus`: Bonus for crimp/sloper on reach moves (default 0.03)
- [x] Type-aware candidate scoring in `NextMovePlanner.plan_next_move`
- [x] Hold type included in `MoveCandidate` reasoning
- [x] Type bonuses applied to efficiency delta calculations

**Completed Tasks**:

1. [x] Annotation infrastructure and dataset structure
2. [x] YOLOv8 fine-tuning script with full configuration
3. [x] Hold extraction updated to include type predictions
4. [x] Type stored in `holds.json` schema
5. [x] Type-aware recommendations (prefer jugs when low support, crimp/sloper for reach)

**Files Created/Modified**:

- `src/pose_ai/service/hold_annotation.py` (new: annotation utilities)
- `scripts/annotate_holds.py` (new: interactive annotation tool)
- `scripts/train_yolo_holds.py` (new: YOLOv8 training script)
- `data/holds_training/` (new: dataset structure with README)
- `src/pose_ai/service/hold_extraction.py` (modified: added type fields)
- `src/pose_ai/recommendation/planner.py` (modified: type-aware scoring)

**Next Steps** (for actual training):

- Annotate training dataset with hold types (≥500 examples per class recommended)
- Train YOLOv8 model: `python scripts/train_yolo_holds.py --data data/holds_training/dataset.yaml`
- Use trained model in hold extraction pipeline

---

## Medium-Term (1-2 Months)

### ✅ Rule-Based Planner v1 — COMPLETED

**Status**: Fully implemented in `src/pose_ai/recommendation/planner.py`

**Completed Implementation**:

- [x] Candidate hold sampling (K=10, upward bias=0.3)
- [x] Support simulation (new contact set generation)
- [x] Efficiency simulation (stability recomputation with new support polygon)
- [x] Constraint checking:
  - Support count ≥ 2
  - COM inside/near polygon (tolerance=0.15)
  - Reach limit (max_reach_ratio=1.2)
  - Limb crossing detection
- [x] Ranking by efficiency gain (Δeff)
- [x] Integration with `suggest_next_actions_advanced` in efficiency.py

**Classes & Methods**:

- `NextMovePlanner` class with configurable `PlannerConfig`
- `MoveCandidate` dataclass for structured results
- `plan_next_move` method returns top-k candidates with reasoning

**Files Created**:

- `src/pose_ai/recommendation/planner.py` (complete implementation)

**Usage**:

```python
from pose_ai.recommendation.efficiency import suggest_next_actions_advanced
recommendations = suggest_next_actions_advanced(current_row, holds, top_k=3)
```

---

### ✅ P1: BiLSTM Multitask Model (v1) — COMPLETED

**Status**: Fully implemented with dataset builder, model architecture, training pipeline, evaluation, and API integration

**Completed Implementation**:

**Architecture** (per efficiency_calculation.md):

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

**Phase 2.1: Dataset Builder** (`src/pose_ai/ml/dataset.py`)

- [x] `ClimbingWindowDataset` with sliding window extraction (T=32, stride=1)
- [x] 60-feature vectors extracted from frame data:
  - Joint positions (x, y) for 8 selected joints: 16 features
  - Joint velocities (vx, vy): 16 features
  - Joint accelerations (ax, ay): 16 features
  - COM position and velocity: 4 features
  - Contact states (4 limbs): 4 features
  - Support count: 1 feature
  - Body scale: 1 feature
  - Wall distance: 1 feature
  - Efficiency: 1 feature
- [x] Z-score normalization per feature dimension
- [x] Next-action label extraction with lookahead window (5 frames)
- [x] Train/val/test split support (default 70/20/10)
- [x] `load_features_from_json` and `create_datasets_from_directory` utilities

**Phase 2.2: Model Architecture** (`src/pose_ai/ml/models.py`)

- [x] `BiLSTMMultitaskModel` with configurable `ModelConfig`
- [x] Bidirectional LSTM encoder (default: 128 hidden, 2 layers)
- [x] `AttentionPooling` layer for sequence aggregation (optional)
- [x] Dual task heads:
  - Efficiency regression: Linear(64) → ReLU → Dropout → Linear(1)
  - Action classification: Linear(64) → ReLU → Dropout → Linear(5)
- [x] `MultitaskLoss` with configurable task weights (default: eff=1.0, action=0.5)
- [x] Model save/load utilities with checkpoint format

**Phase 2.3: Training Pipeline** (`src/pose_ai/ml/train.py`, `scripts/train_bilstm.py`)

- [x] `Trainer` class with full training loop
- [x] Early stopping with patience-based monitoring
- [x] `ReduceLROnPlateau` scheduler
- [x] Model checkpointing (best model + final model)
- [x] Comprehensive CLI with hyperparameter control:
  - Epochs, batch size, learning rate, weight decay
  - Hidden dimension, layers, dropout
  - Early stopping patience, LR scheduling
  - Device selection (CUDA/CPU)
- [x] Training metrics logging (loss per task, validation metrics)

**Phase 2.4: Evaluation** (`scripts/evaluate_bilstm.py`)

- [x] Comprehensive evaluation metrics:
  - Efficiency: MAE, RMSE, R², correlation coefficient
  - Action: Top-1 accuracy, per-class accuracy, confusion matrix
- [x] Support for train/val/test split evaluation
- [x] JSON export of results
- [x] Detailed per-class breakdown and confusion matrix visualization

**Phase 2.5: Pipeline Integration** (`src/pose_ai/ml/inference.py`, `webapp/main.py`)

- [x] `BiLSTMInference` class for production inference
- [x] Sliding window inference with normalization
- [x] Batch inference utilities (`batch_inference`)
- [x] FastAPI endpoint: `GET /api/jobs/{job_id}/ml_predictions`
- [x] Returns frame-by-frame efficiency scores + next-action predictions
- [x] Error handling for missing model/normalization files

**Completed Tasks**:

1. [x] Dataset builder with sliding windows and normalization
2. [x] Weak labels: Uses heuristic efficiency scorer from features, next-action from contact sequences
3. [x] Model architecture: PyTorch BiLSTM with attention and dual heads
4. [x] Training pipeline: Full loop with early stopping, LR scheduling, checkpointing
5. [x] Evaluation metrics: MAE, RMSE, R², correlation, accuracy, confusion matrix
6. [x] Integration: Inference engine + API endpoint

**Files Created**:

- `src/pose_ai/ml/dataset.py` (sliding window dataset)
- `src/pose_ai/ml/models.py` (BiLSTM architecture)
- `src/pose_ai/ml/train.py` (training loop)
- `src/pose_ai/ml/inference.py` (inference engine)
- `scripts/train_bilstm.py` (training CLI)
- `scripts/evaluate_bilstm.py` (evaluation CLI)

**Files Modified**:

- `webapp/main.py` (added `/api/jobs/{job_id}/ml_predictions` endpoint)

**Usage**:

```bash
# Train model
python scripts/train_bilstm.py --data data/features --epochs 100 --device cuda

# Evaluate model
python scripts/evaluate_bilstm.py --model models/checkpoints/bilstm_multitask.pt --data data/features

# API endpoint
GET /api/jobs/{job_id}/ml_predictions
```

**Next Steps** (for actual training):

- Collect training data: Run feature export on multiple climbing videos
- Train model: `python scripts/train_bilstm.py --data data/features --epochs 100`
- Evaluate on test set and compare with rule-based baseline
- Fine-tune hyperparameters based on validation performance

---

### ✅ P2: Technique Pattern Detection — COMPLETED

**Status**: Fully implemented in `src/pose_ai/features/contacts.py`

**Completed Patterns** (per efficiency_calculation.md):

1. **Bicycle**: Opposing toes on same/near holds

   - Both feet contact same hold
   - Opposing velocity vectors (foot directions)
   - Confidence: 0-1 based on velocity opposition

2. **Back-Flag**: Extended leg behind body

   - Both feet on same side of hip center
   - Hip rotation counter-balance
   - Confidence: 0-1 based on hip alignment

3. **Drop-Knee**: Knee rotated inward
   - Knee-ankle separation vs hip-ankle distance
   - Pelvic rotation indicator
   - Confidence: 0-1 based on angle ratios

**Completed Tasks**:

1. [x] Lightweight heuristic implementations in `detect_techniques` function
2. [x] Per-frame confidence scores (0-1)
3. [x] Integration into feature pipeline via `annotate_techniques`
4. [x] Technique bonuses integrated into efficiency scoring:
   - `technique_bonus = 0.05 * bicycle + 0.05 * back_flag + 0.03 * drop_knee`
5. [x] Technique features added to frame data: `technique_bicycle`, `technique_back_flag`, `technique_drop_knee`

**Files Modified**:

- `src/pose_ai/features/contacts.py` (technique detection functions)
- `src/pose_ai/recommendation/efficiency.py` (technique bonus integration)
- `src/pose_ai/service/feature_service.py` (pipeline integration)

---

### ✅ P2: Hold Tracking Across Frames — COMPLETED

**Status**: Fully implemented with IoU matching and Kalman filtering

**Completed Implementation**:

**Phase 1: Core Tracking Module** (`src/pose_ai/service/hold_tracking.py`)

- [x] `HoldTrack` dataclass: Track state with Kalman filter, history, age/hits/misses
- [x] `KalmanFilter2D` class: 2D position/velocity tracking
  - State: [x, y, vx, vy] with constant velocity motion model
  - Predict: Linear motion extrapolation
  - Update: Measurement fusion with detection
  - TODO comments for adaptive noise, acceleration model, outlier rejection
- [x] `IoUTracker` class: Detection-to-track matching
  - `compute_iou()`: Bbox intersection over union calculation
  - `match_detections_to_tracks()`: Greedy matching (TODO: Hungarian algorithm)
  - `create_new_track()`: Initialize new tracks from unmatched detections
  - `update_tracks()`: Frame-by-frame tracking pipeline
  - `prune_tracks()`: Remove tracks with excessive misses
  - Configurable parameters: IoU threshold, max_age, min_hits
  - TODO comments for visual features, MHT, track splitting/merging

**Phase 2: Integration** (`src/pose_ai/service/hold_extraction.py`)

- [x] `track_holds()`: Main tracking pipeline (detections → confirmed tracks)
  - Groups detections by frame, processes sequentially
  - Uses `IoUTracker` for temporal association
  - Returns confirmed tracks with min_hits threshold
- [x] `cluster_tracks()`: Final clustering of tracked holds
  - Uses Kalman-filtered positions (more stable than raw detections)
  - Aggregates track properties (label, type, confidence, hits)
  - DBSCAN clustering on track centroids
- [x] Modified `extract_and_cluster_holds()`:
  - Added `use_tracking: bool = True` parameter
  - If True: detect → track → cluster_tracks (new method)
  - If False: detect → cluster (old DBSCAN-only, backward compatible)
  - Configurable tracking parameters exposed in API
- [x] Updated `__all__` exports

**Completed Tasks**:

1. [x] IoU tracker: match detections across consecutive frames (IoU > 0.5)
2. [x] Kalman filter: predict hold position, correct with detections
3. [~] Re-identification: Prepared with TODO comments (visual features not yet implemented)
4. [x] Track management: spawn new tracks, terminate lost tracks
5. [x] Cluster stable tracks for final hold positions

**Key Features**:

- Temporal consistency: Tracks holds across frames vs frame-by-frame
- Kalman filtering: Smooth position estimates, handles occlusion
- IoU matching: Associate detections to existing tracks
- Track management: Create, confirm, delete tracks based on age/hits
- Backward compatible: Can disable tracking for comparison (`use_tracking=False`)
- Extensive TODO comments for future enhancements

**Files Created/Modified**:

- `src/pose_ai/service/hold_tracking.py` (new: 500+ lines)
- `src/pose_ai/service/hold_extraction.py` (modified: added tracking functions)

**Usage**:

```python
# With tracking (default)
clustered = extract_and_cluster_holds(
    image_paths,
    use_tracking=True,  # Enable temporal tracking
    iou_threshold=0.5,
    max_age=5,
    min_hits=3,
)

# Without tracking (old method)
clustered = extract_and_cluster_holds(
    image_paths,
    use_tracking=False,  # Use DBSCAN-only clustering
)
```

**Future Enhancements** (TODO comments in code):

- Visual feature extraction (ResNet18/34) for re-identification after occlusion
- Hungarian algorithm for optimal detection-track assignment
- Multi-hypothesis tracking (MHT) for handling ambiguous associations
- Track splitting/merging logic for holds that separate or combine
- Adaptive IoU threshold based on track confidence
- Constant acceleration model for dynamic holds

**Expected Benefits**:

- Track fragmentation rate < 20% (target)
- Position variance reduction > 30% vs DBSCAN-only (target)
- No duplicate IDs for same physical hold
- Better handling of detection noise and occlusions

---

## Long-Term (3+ Months)

### ✅ P1: Transformer/TCN Models (v2) — COMPLETED (Transformer only)

**Status**: Transformer architecture implemented and integrated. TCN skipped. BiLSTM kept as baseline.

**Completed Implementation**:

1. [x] Transformer encoder implementation (PyTorch)

   - `TransformerMultitaskModel` with 2-4 layers, 4-8 heads
   - Positional encoding (sinusoidal and learnable options)
   - Configurable pooling strategies (mean/max/cls)
   - Same dual-head architecture as BiLSTM (efficiency + next-action)

2. [x] Positional encoding for frame sequences

   - `PositionalEncoding` class with sinusoidal/learnable modes
   - Automatic sequence length handling

3. [-] TCN implementation (skipped)

   - Decision: Transformer provides sufficient comparison baseline
   - Can be added later if needed

4. [x] Unified training pipeline

   - `scripts/train_model.py` with `--model-type bilstm/transformer`
   - Full hyperparameter control for both architectures
   - Backward compatibility with `train_bilstm.py` (wrapper)

5. [-] Architecture comparison (pending actual training)

   - Infrastructure ready for comparison
   - Requires training on real dataset
   - Can generate performance metrics once trained

6. [-] Attention visualization (future enhancement)
   - Not critical for initial deployment
   - Can be added for model interpretability later

**Completed Tasks**:

- [x] Add `TransformerMultitaskModel`, `TransformerConfig`, `PositionalEncoding` to `src/pose_ai/ml/models.py`
- [x] Create unified `scripts/train_model.py` (replaces `train_transformer.py`)
- [x] Create unified `scripts/evaluate_model.py` with auto-detection
- [x] Update `src/pose_ai/ml/inference.py` with `ModelInference` (auto-detects BiLSTM/Transformer)
- [x] Add `model_type` metadata to checkpoints for auto-detection
- [x] Update `webapp/main.py` to use `ModelInference`
- [x] Comprehensive README.md documentation with command examples

**Files Modified**:

- `src/pose_ai/ml/models.py` (added Transformer architecture)
- `scripts/train_model.py` (new: unified training)
- `scripts/evaluate_model.py` (new: unified evaluation with auto-detection)
- `scripts/train_bilstm.py` (converted to wrapper for backward compatibility)
- `scripts/evaluate_bilstm.py` (converted to wrapper for backward compatibility)
- `src/pose_ai/ml/inference.py` (ModelInference with auto-detection)
- `webapp/main.py` (uses ModelInference)
- `README.md` (complete ML training documentation)

**Usage**:

```bash
# Train Transformer model
python scripts/train_model.py \
  --data data/features \
  --model-type transformer \
  --num-layers 4 \
  --num-heads 8 \
  --d-model 128 \
  --epochs 100

# Evaluate (auto-detects model type)
python scripts/evaluate_model.py \
  --model models/checkpoints/transformer_multitask.pt \
  --data data/features \
  --split test
```

**Next Steps**:

- Train both models on real climbing data
- Compare performance metrics (MAE, accuracy, inference time)
- Add attention visualization if needed for interpretability
- Consider TCN if Transformer doesn't meet performance targets

---

### ✅ P2: IMU Sensor & Climber Personalization — COMPLETED

**Status**: Fully implemented with IMU-based wall angle and personalized efficiency/recommendations

**Completed Implementation**:

**Phase 1: IMU Sensor Integration**

- [x] API model extensions (`MediaMetadata` in `webapp/main.py`)

  - `imu_quaternion`: Device orientation as quaternion [w, x, y, z]
  - `imu_euler_angles`: Device orientation as Euler angles [pitch, roll, yaw]
  - `imu_timestamp`: IMU reading timestamp

- [x] IMU wall angle computation (`src/pose_ai/wall/angle.py`)

  - `quaternion_to_euler()`: Convert quaternion to Euler angles
  - `compute_wall_angle_from_imu()`: Derive wall angle from IMU data
  - Returns `WallAngleResult` with method="imu_sensor"
  - Confidence scoring based on device orientation

- [x] Priority-based angle selection (`src/pose_ai/features/aggregation.py`)

  - Priority 1: Pre-computed angle (if provided)
  - Priority 2: IMU sensor data (if available) ← NEW
  - Priority 3: Vision-based estimation (Hough+PCA, fallback)

- [x] Pipeline integration (`webapp/pipeline_runner.py`, `src/pose_ai/service/feature_service.py`)
  - Extract IMU data from job metadata
  - Pass to `export_features_for_manifest()` → `summarize_features()`
  - IMU angle computed before vision estimation

**Phase 2: Climber Physical Parameters**

- [x] API model extensions

  - `climber_height`: Height in cm (for body scale normalization)
  - `climber_wingspan`: Wingspan in cm (for reach constraints)
  - `climber_flexibility`: Flexibility score 0-1 (for threshold adjustments)

- [x] Body scale normalization (`src/pose_ai/features/aggregation.py`)

  - Compute `body_scale_normalized` using climber height
  - Expected shoulder width = height × 0.16
  - Adjust for different body proportions

- [x] Personalized reach limits (`src/pose_ai/recommendation/planner.py`)

  - `PlannerConfig.get_adjusted_reach_ratio()`: Compute personalized reach ratio
  - Wingspan adjustment: ±10% per 0.1 deviation from average wingspan/height ratio
  - Flexibility bonus: 5-10% reach increase for flexible climbers
  - Applied in `_simulate_efficiency()` constraint checking

- [x] Personalized efficiency scoring (`src/pose_ai/recommendation/efficiency.py`)
  - Flexibility-adjusted reach penalty threshold
  - 5-10% higher threshold for flexible climbers
  - Frame-by-frame personalization

**Completed Tasks**:

1. [x] IMU raw data fields in API (`MediaMetadata`)
2. [x] IMU → wall angle conversion functions
3. [x] IMU priority in feature aggregation
4. [x] Climber params stored in each feature row
5. [x] Body scale normalization with height
6. [x] Personalized reach ratio in planner
7. [x] Personalized reach penalty in efficiency scoring
8. [x] Pipeline integration (metadata → features → planner/efficiency)

**Files Modified**:

- `webapp/main.py` (MediaMetadata with IMU and climber fields)
- `src/pose_ai/wall/angle.py` (IMU functions)
- `src/pose_ai/features/aggregation.py` (IMU priority, climber params, body scale normalization)
- `src/pose_ai/service/feature_service.py` (pass IMU/climber params)
- `webapp/pipeline_runner.py` (extract metadata and pass to pipeline)
- `src/pose_ai/recommendation/planner.py` (personalized reach limits)
- `src/pose_ai/recommendation/efficiency.py` (personalized reach penalty)

**Usage Example**:

```json
POST /api/jobs
{
  "video_dir": "data/videos",
  "metadata": {
    "imu_quaternion": [0.7071, 0.0, 0.7071, 0.0],
    "climber_height": 175.0,
    "climber_wingspan": 180.0,
    "climber_flexibility": 0.7
  }
}
```

Mobile app (React Native / Flutter):

```javascript
// Read IMU sensor
const { rotation } = await DeviceMotion.getDeviceMotionAsync();
const quaternion = [rotation.w, rotation.x, rotation.y, rotation.z];

// Send to API
fetch("/api/jobs", {
  method: "POST",
  body: JSON.stringify({
    metadata: {
      imu_quaternion: quaternion,
      climber_height: 175,
      climber_wingspan: 180,
      climber_flexibility: 0.7,
    },
  }),
});
```

**Expected Benefits**:

- Wall angle accuracy: ±1° with IMU vs ±5° vision-only
- Personalized reach constraints prevent unrealistic recommendations
- Flexibility-adjusted penalties reduce false negatives for flexible climbers
- Better recommendation quality across diverse climber body types

---

### P2: Advanced Wall Calibration — DEFERRED

**Status**: Deferred in favor of IMU sensor approach

**Reasoning**:

- IMU sensors provide superior accuracy (±1° vs ±5° vision-based)
- Faster and more reliable than vision-based methods
- No need for complex RANSAC/Homography for standard vertical walls
- Can be revisited for complex multi-angle walls or volumes if needed

**Potential Future Work** (if complex walls require it):

1. [ ] RANSAC plane fitting for volumes and complex multi-angle walls
2. [ ] Homography estimation for distance calibration
3. [ ] Multi-view calibration for 3D reconstruction
4. [ ] Manual calibration UI for ground-truth correction

**Current State**: Single-angle estimation from Hough + PCA (fallback when IMU unavailable)

---

### P2: Climber Profiling and Personalization

**Target State**:

- Track climber history (routes attempted, success rate)
- Learn climber-specific metrics (height, wingspan, flexibility)
- Personalized recommendations based on climber profile

**Implementation Tasks**:

1. [ ] Climber database schema (ID, physical attributes, history)
2. [ ] Profile inference from videos (estimate height, reach)
3. [ ] Personalized efficiency scoring (adjust weights by climber type)
4. [ ] Recommendation filtering (avoid moves incompatible with climber)

**Affected Files**:

- `src/pose_ai/data/climber_profile.py` (new module)
- `webapp/main.py` (add climber management endpoints)

---

### P3: Production Infrastructure

**Data Infrastructure**:

1. [ ] DVC for dataset versioning (track video/annotation changes)
2. [ ] Parquet format for artifacts (faster I/O than JSON)
3. [ ] Experiment manifests (track parameters + dataset hash)
4. [ ] Model registry (versioned models with metadata)

**CI/CD**:

1. [ ] GitHub Actions: lint, test, coverage on PR
2. [ ] Docker multi-stage build (dev, prod)
3. [ ] Automated model retraining on new data
4. [ ] Deployment pipeline (staging → prod with rollback)

**Monitoring**:

1. [ ] Prometheus metrics (API latency, job throughput)
2. [ ] Model performance tracking (efficiency correlation, action accuracy)
3. [ ] Data drift detection (feature distribution monitoring)
4. [ ] A/B testing framework (compare model versions)

**Affected Files**:

- `.github/workflows/ci.yml` (new CI config)
- `Dockerfile` (new container definition)
- `docker-compose.yml` (new: app + monitoring stack)
- `monitoring/prometheus.yml` (new metrics config)

---

### ✅ P3: Route Difficulty Estimation — COMPLETED

**Status**: Fully implemented with feature extraction, XGBoost model, API endpoint, and UI page

**Completed Implementation**:

1. [x] Feature engineering (`src/pose_ai/ml/route_grading.py`)

   - Hold density: Holds per square meter
   - Hold spacing: Mean/median/std/min/max distances
   - Wall angle: From IMU or vision
   - Move complexity: Step count, efficiency stats, reach penalties, contact switches
   - Hold type distribution: Ratios for jug/crimp/sloper/pinch/foot_only/volume
   - Route length: Vertical distance climbed
   - Duration: Total climbing time

2. [x] XGBoost regression model

   - `RouteDifficultyModel` class for prediction (V0-V10)
   - `predict_with_confidence()` method
   - Training script: `scripts/train_route_grader.py`
   - Model parameters: max_depth=6, learning_rate=0.1, early_stopping=10

3. [x] Gym-specific calibration

   - `GymGradeCalibration` class
   - Default mapping: V0-V2 (Beginner), V3-V4 (Intermediate), V5-V6 (Advanced), V7-V8 (Expert), V9-V10 (Elite)
   - Extensible for custom gym scales

4. [x] API integration

   - Endpoint: `GET /api/jobs/{job_id}/route_grade`
   - Returns: grade, confidence, calibrated_grade, features
   - Works with completed pipeline jobs

5. [x] New UI page
   - Route: `/grading`
   - Clean interface for route selection and grade display
   - Feature breakdown visualization
   - Color-coded difficulty indicators

**Completed Tasks**:

- [x] `extract_route_features()` function with 20+ route-level features
- [x] `RouteDifficultyModel` XGBoost wrapper
- [x] `GymGradeCalibration` for grade mapping
- [x] Training script with data loading, train/test split, evaluation metrics
- [x] API endpoint integration
- [x] Grading UI page (`webapp/templates/grading.html`)
- [x] Route handler in `webapp/main.py`

**Files Created**:

- `src/pose_ai/ml/route_grading.py` (feature extraction + model + calibration)
- `scripts/train_route_grader.py` (training CLI)
- `webapp/templates/grading.html` (new UI page)
- `docs/TESTING_GUIDE.md` (comprehensive testing documentation)

**Files Modified**:

- `webapp/main.py` (added `/grading` route and `/api/jobs/{job_id}/route_grade` endpoint)

**Usage**:

```bash
# Train route grader model
python scripts/train_route_grader.py \
  --data data/routes_annotated \
  --model-out models/route_grader.json \
  --n-estimators 100

# API endpoint
GET /api/jobs/{job_id}/route_grade

# UI
Navigate to http://localhost:8000/grading
```

**Next Steps** (for actual training):

- Annotate training videos with ground-truth grades (V0-V10)
- Collect 50+ routes for training dataset
- Train model: `python scripts/train_route_grader.py --data data/routes_annotated`
- Evaluate on test set and fine-tune hyperparameters
- Extend gym calibration mappings for specific gyms

---

## References

- **Authoritative Spec**: [efficiency_calculation.md](efficiency_calculation.md)
- **Current Roadmap**: [beta_model_plan.md](beta_model_plan.md)
- **System Guide**: [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)
- **Testing Guide**: [TESTING_GUIDE.md](TESTING_GUIDE.md)
- **Source Code**: [`src/pose_ai/`](../src/pose_ai/)

---

**Note**: Priorities may shift based on user feedback and evaluation results. Revisit this backlog monthly to adjust priorities and add new features.
