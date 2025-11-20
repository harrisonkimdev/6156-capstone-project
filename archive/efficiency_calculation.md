# Purpose / Scope

* **Goal**: Compute **move/step-level efficiency** and use it to power **next-action recommendations**.
* **Input**: Uploaded bouldering video → frame sequence. Each frame includes MediaPose **keypoints (x, y, z, visibility)**. Hold 2D locations (optionally type/size/color), and optional wall plane calibration.
* **Output**:

  1. Step-level efficiency score (0–1) and diagnostic metrics
  2. Next-action recommendation (rule-based v1 → model-based v2)

---

## Terminology

* **Frame**: One timepoint of the video.
* **Move/Step**: A contiguous frame segment that constitutes a single action (e.g., place left foot → settle).
* **Time series**: Frames sampled at fixed ∆t.
* We unify terms: use **time series**; “sequence” is a synonym here.

---

## Data Representation (Frame-Level Features)

### 1) Keypoint selection (must-have)

* Shoulders L/R, Elbows L/R, Wrists L/R
* Hips L/R (or hip center), Knees L/R, Ankles L/R (or foot index)
* One head/neck proxy (e.g., nose or neck) for body orientation

### 2) Coordinate normalization

* **Root-relative**: subtract hip center (=(hip_L+hip_R)/2) or torso center from all joints.
* **Scale normalization**: divide all coords/velocities/distances by a body reference (e.g., shoulder width).
* **View normalization (optional)**: if camera is fixed, apply left/right flip or homography to a wall coordinate system.

### 3) Kinematic derivatives

* Velocity: `v_t = p_t − p_{t−1}`
* Acceleration: `a_t = v_t − v_{t−1}`
* Jerk (optional): `j_t = a_t − a_{t−1}`
* Focus joints: wrists/ankles/hips/shoulders + COM (below)

### 4) Center of Mass (COM) approximation

* Use a simple mean or weighted mean of major joints to approximate 2D COM.
* If wall calibration exists, project to the wall coordinate frame.

### 5) Hold metadata (recommended)

* For each hold: `id`, `(x, y)`, `type` (crimp, pinch, sloper, jug, foot-only, etc.), optional size/color, and difficulty tag.

---

## Hold Contact Inference (with continuity & velocity filters)

Contact is determined by **distance**, **velocity**, and **temporal hysteresis**.

### 1) Candidates by distance

* For each limb joint J ∈ {LH, RH, LF, RF} and each hold H, compute `d(J,H)`.
* Candidate if `d(J,H) ≤ r_th`. Choose the nearest hold.

  * Recommend `r_th = k_r × body_scale` (e.g., 0.25–0.35 × shoulder width)

### 2) Velocity condition

* True contact usually implies near-zero joint speed. Require `|v_J| ≤ v_hold`.

  * Recommend `v_hold = k_v × body_scale / fps` (e.g., 0.02–0.05)
* **Swing/glance suppression**: if `|v_J|` is large, defer contact.

### 3) Continuity / Hysteresis

* Consider previous frame contact state `S_{t−1}(J)`.
* **On**: `d ≤ r_on` **and** `|v| ≤ v_hold` (with `r_on < r_off`).
* **Stay on**: if `d ≤ r_off`.
* Use dual thresholds (`r_on`, `r_off`) to reduce flicker.
* Enforce minimum on-duration: `min_on_frames` (e.g., ≥3 frames) → “confirmed contact”.

### 4) Temporal smoothing (optional)

* Use HMM/Viterbi with transition costs for `on↔off`.
* Lightweight alternative: morphological closing→opening on the binary time series.

### 5) Smearing detection

* Prereq: wall plane known or usable depth proxy.
* Smear if **all** hold distances `> r_th`, foot `z` is close to the wall plane (`|z_foot−z_wall| ≤ z_eps`), and foot speed is low (`|v_foot| ≤ v_hold`).
* Mark as contact with `type="smear"`, `hold_id=null`.

### 6) Technique patterns (heuristic v1)

* **Bicycle**: both feet contact the same or near-identical hold/cluster, toe vectors oppose each other (foot–knee vector dot < −cos θ).
* **Back-flag**: free leg crosses behind/outside the supporting leg to counter hip rotation (thigh vs body axis angle criteria).
* **Drop-knee**: knee internal rotation + pelvic rotation while foot maintains contact (thigh vector angle below a threshold).
* Record each with a confidence score (0–1) to be used as features in v2.

---

## Efficiency Metrics (frame-level → step aggregation)

Compute frame metrics, then aggregate over the step (mean/min/variance or top-quantile).

### 1) Support polygon stability

* Build convex polygon P from current support contacts (holds/smears).
* Compute `dist(COM, P)` as distance to polygon (0 if inside). Normalize by body_scale.
* `stab = exp(−α · dist)` → in [0,1]. Higher is better.

### 2) Support count / switch penalties

* `n_support ∈ {0..4(+smear)}`.
* `pen_support = w2 · 1{n_support < 2}` (strong penalty if <2).
* Add `pen_switch` for frequent contact switching.

### 3) Wall–body distance penalty (grip-load proxy)

* Assume forearm load rises as z_COM increases away from the wall.
* `pen_wall = w3 · ReLU(z_COM − z_ref)` (requires calibration or learned proxy).

### 4) Path efficiency (path vs displacement)

* `net_disp = ||COM_T − COM_0||`, `path_len = Σ||COM_t − COM_{t−1}||`.
* `eff_path = clamp(net_disp / (path_len + ε), 0, 1)`.

### 5) Smoothness (jerk / direction changes)

* Use mean normalized jerk and velocity direction flips for COM and key limbs.
* `pen_jerk = w5 · mean_jerk_norm`.

### 6) Reach-limit penalty

* If limb reach (root-relative, scale-normalized) exceeds a high percentile (e.g., p95), penalize:
* `pen_reach = w6 · max(0, reach_norm − τ_reach)`.

### 7) Aggregate score (example)

```
score_eff =
  w1*stab  + w4*eff_path
  − (pen_support + pen_wall + pen_jerk + pen_reach)
```

* Initial weights (tune later):
  `w1=0.35, w4=0.25, w2=0.20, w3=0.10, w5=0.07, w6=0.03`.
* Step score = weighted mean of frame scores (or top-q quantile).

---

## Step Segmentation

* Start a new step when a **confirmed contact** event occurs (limb changes hold).
* Priority: split on single-limb change while others maintain support.
* Constrain step duration (e.g., 0.2s–4s) to avoid over/under-segmentation.

---

## Modeling (v1 → v2)

### v1: Rules + BiLSTM (multitask)

* **Input**: sliding window length T (e.g., 32 frames @ 25fps ≈ 1.28s), stride=1.
* **Features**:

  * normalized keypoints (selected joints), v/a, COM, contact (one-hot/embedding),
  * per-frame efficiency features (stab, path, etc.),
  * hold type embeddings (if available).
* **Architecture**: BiLSTM (128–256) → attention pooling →

  * Head1: efficiency regression (0–1, Huber)
  * Head2: next-action classification (e.g., {LH/RH/LF/RF} × 8-dir or hold-cluster, CE)
* **Training**: use heuristic efficiency as **weak labels**; fine-tune with a small set of human labels.

### v2: Temporal Transformer / TCN + finer recommendation

* Same inputs. Replace encoder with Transformer (2–4 layers, 4–8 heads) or TCN and compare.
* Refine Head2 to predict **next-hold id** (or cluster). Keep efficiency loss as an auxiliary task.

---

## Next-Action Recommendation (rule-based v1)

1. Sample K candidate target holds based on current support and target direction (upward/left/right density).
2. For each candidate, simulate the new support set → recompute support polygon → estimate `score_eff'`.
3. Prefer moves that maintain/increase support count, keep COM inside the polygon, and respect reach/crossing constraints. Return the best `score_eff'` candidate (limb/direction/hold).

---

## Calibration / Preprocessing

* **FPS normalization**: resample to 25fps (25–30 recommended).
* **Wall plane**: one-time calibration (click wall corners → homography) to estimate wall coordinates.
* **Hold detection**: start with manual/semi-auto labels; introduce YOLO/segmentation later.

---

## File / Module Layout (example)

```
project/
 ├─ data/
 │   ├─ raw_videos/
 │   ├─ holds/{video_id}.json          # holds: id, x, y, type
 │   └─ calib/{video_id}.yaml          # wall/homography/scale
 ├─ features/
 │   ├─ pose_extract.py                # MediaPose wrapper
 │   ├─ feature_build.py               # normalization, v/a/jerk, COM, contacts
 │   └─ schemas.py                     # JSON schemas
 ├─ modeling/
 │   ├─ datasets.py                    # sliding windows, collate
 │   ├─ models.py                      # BiLSTM/TCN/Transformer
 │   ├─ losses.py                      # Huber/BCE/CE
 │   └─ train.py                       # train loop/logging
 ├─ rules/
 │   ├─ contact_filter.py              # r_on/off, v_hold, HMM
 │   ├─ techniques.py                  # bicycle/back-flag/drop-knee
 │   └─ plan_next_move.py              # rule-based planner
 └─ eval/
     ├─ metrics.py                     # R^2/MAE/F1/top-k
     └─ visualize.py                   # overlay renderer
```

---

## JSON Schemas (summary)

**FrameFeature**:

```json
{
  "t": 123,
  "keypoints": {
    "wrist_l": {"x":0.12,"y":0.34,"z":-0.01,"vis":0.9},
    "wrist_r": {"x":..., "y":..., "z":..., "vis":...},
    "hip_c":   {"x":..., "y":..., "z":..., "vis":...},
    "ankle_l": {...}, "ankle_r": {...}, "shoulder_l": {...}
  },
  "derived": {
    "v": {"wrist_l": [vx,vy,vz], ...},
    "a": {"wrist_l": [ax,ay,az], ...},
    "com": {"x":..., "y":..., "z":...}
  },
  "contacts": {
    "LH": {"type":"hold","id":12,"conf":0.86},
    "RF": {"type":"smear","id":null,"conf":0.73}
  },
  "techniques": {"bicycle":0.2, "back_flag":0.6, "drop_knee":0.1},
  "meta": {"fps":25, "body_scale":0.42}
}
```

**StepLabel** (for training):

```json
{
  "video_id": "...",
  "step_id": 7,
  "t_start": 210, "t_end": 260,
  "efficiency_score": 0.78,
  "next_action": {
    "limb": "RH",
    "target": {"type":"cluster","cid":3}
  },
  "notes": "mild back-flag, 3-point support"
}
```

---

## Constants / Thresholds (initial guides)

* `fps = 25`
* `body_scale`: shoulder width normalized to 1.0
* Distance: `r_on = 0.22`, `r_off = 0.28` (× body_scale)
* Velocity: `v_hold = 0.03` (× body_scale/frame)
* Min contact duration: `min_on_frames = 3`
* Step duration: `0.2s ≤ len ≤ 4s`
* Smear: `|z_foot − z_wall| ≤ z_eps (=0.03)` and `no hold within r_smear (=0.25)`
* Technique angles: `θ_bicycle ≈ 60°`, `θ_backflag ≈ 50°` (tune empirically)

---

## Efficiency Computation (pseudocode)

```python
for each step in steps:
    scores = []
    for frame in step.frames:
        P = support_polygon(frame.contacts)
        stab = stability(COM(frame), P)               # exp(-α·dist)
        n = num_support(frame.contacts)
        pen_support = w2 * int(n < 2) + pen_switch(frame)
        pen_wall = w3 * relu(COM(frame).z - z_ref)
        eff_path_frame = net_over_path(frame_window(COM_history))
        pen_jerk = w5 * mean_norm_jerk(frame)
        pen_reach = w6 * max(0, reach_norm(frame) - τ_reach)
        score = w1*stab + w4*eff_path_frame - (pen_support+pen_wall+pen_jerk+pen_reach)
        scores.append(score)
    step_score = aggregate(scores)   # mean or top-quantile
    output(step_id, step_score, diagnostics)
```

---

## Contact Filter (pseudocode; with continuity & velocity)

```python
def decide_contact(prev_state, joint_pos, joint_vel, holds, params):
    h_star, d_star = nearest_hold(joint_pos, holds)
    on = prev_state.on

    if not on:
        if d_star <= params.r_on and norm(joint_vel) <= params.v_hold:
            on = True; hold_id = h_star
    else:
        if d_star > params.r_off:
            on = False; hold_id = None
        else:
            hold_id = prev_state.hold_id if prev_state.hold_id is not None else h_star

    on = enforce_min_duration(on, prev_state.buffer, params.min_on_frames)
    return ContactState(on, hold_id)
```

---

## Training Pipeline (v1)

1. Preprocess: unify fps → extract pose → normalize + derivatives → contact/smear inference → step segmentation.
2. Create **weak labels** for efficiency from heuristics (this section) for initial supervision.
3. Dataset: sliding windows (T=32, stride=1) + targets (efficiency score, next-action).
4. Model: BiLSTM (2 layers, 128) + attention; Head1=Huber, Head2=CE.
5. Evaluation: MAE/R^2 (efficiency), top-1/top-3 (next action), ablations (metrics/filters on/off).
6. Visualization: overlay contacts/COM/polygon/recommendation arrows.

---

## Risks / Assumptions

* MediaPose `z` is relative, not absolute depth; wall distance needs calibration or learned proxy.
* Strong viewpoint changes reduce generalization → recommend capture guidelines / preprocessing standardization.
* Hold detection/type may be manual at first → plan staged automation.

---

## Roadmap

* **Sprint A (Weeks 1–2)**: Calibration/normalization/contact filter, stable step segmentation.
* **Sprint B (Weeks 3–4)**: Implement efficiency metrics + weak labels, rule-based planner v1.
* **Sprint C (Weeks 5–6)**: BiLSTM multitask training/eval, visualization dashboard.
* **Sprint D (Week 7+)**: Transformer/TCN comparison, hold-type utilization, technique confidence integration.

---

## AI Implementation Instructions (copy/paste for the dev agent)

1. **Standardize**: resample video to 25fps. Run MediaPose; convert to root-relative, scale-normalized coords; compute v/a/jerk.
2. **Contact filter**: per limb, use nearest-hold distance and velocity threshold with on/off hysteresis (r_on/r_off). Apply `min_on_frames`. Detect smears by (far from holds + near wall plane + low speed).
3. **Step segmentation**: split on confirmed contact changes; enforce 0.2–4s duration constraints.
4. **Efficiency metrics**: per frame, compute support-polygon stability, support count/switch penalties, wall–body penalty, path efficiency, jerk penalty, reach penalty; aggregate to step-level 0–1 score with weighted sum.
5. **Dataset**: slice into windows (T=32, stride=1). Inputs = [normalized keypoints + v/a + COM + contact one-hot/embeddings + metrics], Targets = [efficiency (regression), next action (classification: limb×8-dir or hold-cluster)].
6. **Model (v1)**: BiLSTM (128–256, 1–2 layers) + attention. Head1=HuberLoss, Head2=CrossEntropy. Train with weak labels, then fine-tune on small human-labeled set.
7. **Planner (v1)**: rule-based evaluation of K candidate holds; return the one maximizing predicted efficiency gain under constraints (support count, COM-in-polygon, reach/crossing limits).
8. **Eval/Vis**: MAE/R^2, top-k; ablations. Render overlays (contacts, COM, polygon, arrows).

---

## Notes

* The term “backflip” in prior discussion is interpreted here as **back-flag**; adjust naming if needed.
