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

### P0: Advanced Contact Inference

**Current State**: Basic distance thresholding (`distance <= threshold * body_scale`)

**Target State** (per efficiency_calculation.md):
- Hysteresis thresholds: `r_on` (contact start) vs `r_off` (contact end)
- Velocity condition: `|v_joint| <= v_hold` (low speed required for contact)
- Minimum duration filter: `min_on_frames >= 3` (ignore brief touches)
- Smear detection: Foot near wall, no hold within radius
- Optional: HMM/Viterbi temporal smoothing for noisy sequences

**Implementation Tasks**:
1. [ ] Add `r_on`/`r_off` parameters to contact inference (r_on=0.22, r_off=0.28 × body_scale)
2. [ ] Compute joint velocities in feature aggregation
3. [ ] Apply velocity threshold (`v_hold = 0.03 × body_scale/fps`)
4. [ ] Implement contact state machine with hysteresis
5. [ ] Add minimum duration filter (morphological close→open)
6. [ ] Detect smears: `|z_foot - z_wall| <= z_eps` AND `no hold within r_smear`
7. [ ] Unit tests for contact filter edge cases

**Validation Criteria**:
- Contact start/end aligned with video observation (±2 frames)
- False positive rate < 10% on test videos
- Smear detection accuracy > 80%

**Affected Files**:
- `src/pose_ai/features/aggregation.py` (add velocity computation)
- `src/pose_ai/recommendation/efficiency.py` (update contact logic)
- `tests/unit/test_contact_inference.py` (new test suite)

---

### P0: Step Segmentation with Contact Boundaries

**Current State**: Simple movement/rest classification based on velocity

**Target State** (per efficiency_calculation.md):
- Split on confirmed contact changes (limb changes hold)
- Priority: Single-limb change while others maintain support
- Duration constraints: 0.2s ≤ step_duration ≤ 4s
- Segment labels: Reach, Stabilize, FootAdjust, DynamicMove, Rest, Finish

**Implementation Tasks**:
1. [ ] Detect contact change events from contact filter output
2. [ ] Implement step boundary detection (prefer single-limb transitions)
3. [ ] Enforce duration constraints (merge short steps, split long ones)
4. [ ] Rule-based segment labeling:
   - **Reach**: Contact count increases, high velocity
   - **Stabilize**: Contact maintained, decreasing velocity
   - **FootAdjust**: Foot contact change, hands stable
   - **DynamicMove**: Multiple simultaneous contact changes
   - **Rest**: All contacts stable, near-zero velocity
   - **Finish**: Top hold reached, sustained stability
5. [ ] Add segment metadata (start/end frame, limb involved, hold IDs)
6. [ ] Validate against manual annotations

**Validation Criteria**:
- Step boundaries within ±3 frames of manual labels
- Segment label accuracy > 70% on test set
- No steps < 0.2s or > 4s after constraints

**Affected Files**:
- `src/pose_ai/segmentation/rule_based.py` (replace current logic)
- `src/pose_ai/segmentation/step_detector.py` (new module)
- `scripts/run_segmentation.py` (update to use new segmenter)
- `tests/unit/test_step_segmentation.py` (new tests)

---

### P1: Full 7-Component Efficiency Formula

**Current State**: 5-component rule-based heuristic (detection, joint, COM, contact, hip-wall)

**Target State** (per efficiency_calculation.md):

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

**Implementation Tasks**:
1. [ ] Implement convex hull from contact points (scipy.spatial.ConvexHull)
2. [ ] Compute COM-to-polygon distance (point-in-polygon test)
3. [ ] Track contact switching frequency per step
4. [ ] Compute wall-body distance (requires wall plane calibration)
5. [ ] Path efficiency: accumulate COM path length vs net displacement
6. [ ] Jerk computation: third derivative of position
7. [ ] Reach normalization: limb length vs percentile distribution
8. [ ] Technique pattern detection (see P2 section below)
9. [ ] Aggregate frame-level scores to step-level (mean or top-quantile)
10. [ ] Update efficiency scoring module with new formula

**Validation Criteria**:
- Correlation with expert ratings ≥ 0.6 (vs current ≥ 0.5)
- Component contributions interpretable (SHAP analysis)
- Score distribution covers full 0-1 range on test videos

**Affected Files**:
- `src/pose_ai/recommendation/efficiency.py` (complete rewrite)
- `src/pose_ai/features/support_polygon.py` (new module)
- `src/pose_ai/features/kinematics.py` (new module for jerk)
- `tests/unit/test_efficiency_scoring.py` (comprehensive tests)

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

### P1: Rule-Based Planner v1

**Current State**: Distance-based heuristic with recency penalty

**Target State** (per efficiency_calculation.md):
1. Sample K candidate holds based on current support and target direction
2. For each candidate, simulate new support set
3. Recompute support polygon and estimate new efficiency `score_eff'`
4. Apply constraints:
   - Maintain/increase support count
   - Keep COM inside polygon
   - Respect reach limits (avoid overextension)
   - Avoid crossing limbs
5. Return best candidate with reasoning

**Implementation Tasks**:
1. [ ] Candidate sampling:
   - Filter holds by direction (upward bias)
   - Filter by reach distance (realistic range)
   - Sample K=10 candidates
2. [ ] Support simulation:
   - Copy current contacts
   - Replace one limb's contact with candidate hold
   - Recompute support polygon
   - Compute new COM position (assume linear interpolation)
3. [ ] Efficiency simulation:
   - Apply full 7-component formula with simulated state
   - Compute `Δeff = score_eff' - score_eff`
4. [ ] Constraint checking:
   - Support count must be ≥ 2 during transition
   - COM must remain inside polygon (or within tolerance)
   - Limb reach < threshold (e.g., 95th percentile)
   - No limb crossings (check for intersections)
5. [ ] Ranking and selection:
   - Rank by `Δeff` (prefer efficiency gains)
   - Filter out constraint violations
   - Return top-3 with reasoning strings

**Validation Criteria**:
- Top-3 hit rate > 70% on manual next-action labels
- No constraint violations in top-5 recommendations
- Reasoning strings align with video observation

**Affected Files**:
- `src/pose_ai/recommendation/planner.py` (new module)
- `src/pose_ai/recommendation/efficiency.py` (expose simulation interface)
- `webapp/main.py` (update analysis endpoint)

---

### P2: Technique Pattern Detection

**Target Patterns** (per efficiency_calculation.md):

1. **Bicycle**: Opposing toes on same/near holds
   - Angle between legs ≈ 60-90°
   - Both feet in contact
   - Confidence based on angle consistency

2. **Back-Flag**: Extended leg behind body
   - Hip-wall alignment > threshold
   - One leg extended, other supporting
   - Knee angle of flagging leg ≈ 50°

3. **Drop-Knee**: Knee rotated inward
   - Knee rotation angle > threshold
   - Torso twist toward inside hip
   - Foot contact stable

**Implementation Tasks**:
1. [ ] Compute leg angles (hip-knee-ankle)
2. [ ] Compute torso orientation (shoulder-hip vector)
3. [ ] Implement bicycle detector:
   - Check both feet in contact
   - Measure angle between thighs
   - Return confidence 0-1
4. [ ] Implement back-flag detector:
   - Check one foot contact, other extended
   - Measure hip-wall alignment
   - Knee angle of extended leg
5. [ ] Implement drop-knee detector:
   - Knee rotation from neutral position
   - Torso twist correlation
6. [ ] Integrate into efficiency formula as bonuses:
   - `technique_bonus = 0.05 * bicycle_conf + 0.05 * backflag_conf + 0.03 * dropknee_conf`
7. [ ] Add technique labels to step metadata

**Validation Criteria**:
- Bicycle detection precision > 80%
- Back-flag detection recall > 70%
- Drop-knee detection balanced accuracy > 75%

**Affected Files**:
- `src/pose_ai/features/techniques.py` (new module)
- `src/pose_ai/recommendation/efficiency.py` (integrate bonuses)
- `tests/unit/test_technique_detection.py` (new tests)

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
