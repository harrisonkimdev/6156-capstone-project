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

### P1: Hold Type Classification

**Current State**: Generic object detection, no type labels

**Target State**:

- Classify holds: crimp, sloper, jug, pinch, foot-only, volume
- Confidence score per type
- Integration into hold metadata

**Implementation Tasks**:

1. [ ] Annotate training dataset with hold types (≥500 examples per class)
2. [ ] Fine-tune YOLOv8 with hold type head
3. [ ] Update hold extraction to include type predictions
4. [ ] Store type in `holds.json` schema
5. [ ] Use type in next-action recommendations (prefer jugs when low support)

**Validation Criteria**:

- Hold type accuracy > 75% on test set
- No significant mAP degradation for position detection

**Affected Files**:

- `src/pose_ai/service/hold_extraction.py` (add type output)
- `data/holds_training/` (new dataset folder)
- `scripts/train_yolo_holds.py` (new training script)

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

### P1: BiLSTM Multitask Model (v1)

**Architecture** (per efficiency_calculation.md):

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

**Features** (per window):

- Normalized keypoints (selected joints: wrists, ankles, hips, shoulders)
- Velocities and accelerations
- COM trajectory
- Contact embeddings (one-hot or learned)
- Per-frame efficiency metrics
- Hold type embeddings (if available)

**Implementation Tasks**:

1. [ ] Dataset builder:
   - Sliding window extraction (T=32, stride=1)
   - Feature normalization (z-score per joint)
   - Contact encoding (one-hot: LH/RH/LF/RF × on/off)
   - Hold embeddings (learned or one-hot by cluster ID)
2. [ ] Generate weak labels:
   - Run heuristic efficiency scorer on all training videos
   - Label next-action from ground-truth contact sequences
3. [ ] Model architecture:
   - PyTorch BiLSTM with attention
   - Two output heads (regression + classification)
   - Multi-task loss: `L = λ1*Huber(eff) + λ2*CE(action)`
4. [ ] Training pipeline:
   - Train/val/test split (70/15/15)
   - Early stopping on validation loss
   - Learning rate schedule (ReduceLROnPlateau)
   - Model checkpointing
5. [ ] Evaluation metrics:
   - Efficiency: MAE, R², correlation with expert ratings
   - Next-action: top-1/top-3 accuracy, confusion matrix
   - Ablations: features on/off, single-task vs multitask
6. [ ] Integration:
   - Add model inference to pipeline runner
   - Serve predictions via API endpoint
   - Compare with rule-based baseline

**Validation Criteria**:

- Efficiency MAE < 0.10 on test set
- Next-action top-3 accuracy > 60%
- Inference time < 100ms per window on CPU

**Affected Files**:

- `src/pose_ai/ml/dataset.py` (new: sliding window dataset)
- `src/pose_ai/ml/models.py` (new: BiLSTM architecture)
- `src/pose_ai/ml/train.py` (new: training loop)
- `scripts/train_bilstm.py` (new CLI)
- `webapp/pipeline_runner.py` (integrate model inference)

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

### P2: Hold Tracking Across Frames

**Current State**: Frame-by-frame detection, no temporal consistency

**Target State**:

- Track hold IDs across frames using IoU + Kalman filter
- Resolve detection ambiguities with learned embeddings
- Stable hold positions despite detection noise

**Implementation Tasks**:

1. [ ] IoU tracker: match detections across consecutive frames (IoU > 0.5)
2. [ ] Kalman filter: predict hold position, correct with detections
3. [ ] Re-identification: extract visual embeddings from hold patches
4. [ ] Track maintenance: spawn new tracks, terminate lost tracks
5. [ ] Cluster stable tracks for final hold positions

**Validation Criteria**:

- Track fragmentation rate < 20%
- Position variance reduction > 30% vs frame-by-frame
- No duplicate IDs for same physical hold

**Affected Files**:

- `src/pose_ai/service/hold_tracking.py` (new module)
- `src/pose_ai/service/hold_extraction.py` (integrate tracker)

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
