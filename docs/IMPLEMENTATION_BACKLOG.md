# Implementation Backlog

**Last Updated**: November 18, 2025  
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

### P1: Transformer/TCN Models (v2)

**Target State**:

- Replace BiLSTM with Transformer encoder (2-4 layers, 4-8 heads)
- Alternative: Temporal Convolutional Network (TCN) for comparison
- Same multitask heads as BiLSTM v1
- Refine Head2 to predict next-hold ID or cluster directly

**Implementation Tasks**:

1. [ ] Transformer encoder implementation (PyTorch)
2. [ ] Positional encoding for frame sequences
3. [ ] TCN implementation for comparison
4. [ ] Training pipeline (same as BiLSTM)
5. [ ] Architecture comparison: BiLSTM vs Transformer vs TCN
6. [ ] Attention visualization (which frames matter for predictions?)

**Validation Criteria**:

- Efficiency MAE < BiLSTM v1 by ≥5%
- Next-action accuracy improvement ≥10%
- Inference time comparable to BiLSTM

**Affected Files**:

- `src/pose_ai/ml/models.py` (add Transformer/TCN)
- `scripts/train_transformer.py` (new CLI)
- `docs/model_comparison.md` (new analysis doc)

---

### P2: Advanced Wall Calibration

**Current State**: Single-angle estimation from Hough + PCA

**Target State**:

- RANSAC plane fitting for complex walls (multi-angle, volumes)
- Multi-view geometry for accurate 3D wall reconstruction
- Manual calibration UI for ground-truth correction

**Implementation Tasks**:

1. [ ] RANSAC plane fitting from edge points
2. [ ] Multi-view calibration (2+ camera angles)
3. [ ] Homography estimation for wall coordinates
4. [ ] Manual calibration UI (click wall corners → compute transform)
5. [ ] 3D wall mesh reconstruction (optional)

**Validation Criteria**:

- Plane fitting RMSE < 5cm on test walls
- Multi-view calibration error < 2% of wall dimensions

**Affected Files**:

- `src/pose_ai/wall/calibration.py` (new module)
- `webapp/templates/calibration.html` (new UI page)

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

### P3: Route Difficulty Estimation

**Target State**:

- Predict route grade (V0-V10) from video
- Features: hold density, wall angle, move complexity
- Integration with climbing gym databases

**Implementation Tasks**:

1. [ ] Annotate training videos with route grades
2. [ ] Feature engineering: hold spacing, angle distribution, move count
3. [ ] Train regression model (XGBoost → Neural Network)
4. [ ] Calibration with gym-specific grade scales

**Affected Files**:

- `src/pose_ai/ml/route_grading.py` (new module)
- `scripts/train_route_grader.py` (new CLI)

---

## References

- **Authoritative Spec**: [efficiency_calculation.md](efficiency_calculation.md)
- **Current Roadmap**: [beta_model_plan.md](beta_model_plan.md)
- **System Guide**: [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)
- **Source Code**: [`src/pose_ai/`](../src/pose_ai/)

---

**Note**: Priorities may shift based on user feedback and evaluation results. Revisit this backlog monthly to adjust priorities and add new features.
