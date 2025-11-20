# Testing Guide

**Purpose**: Comprehensive testing documentation for all pipeline steps with input/output specifications and test cases.

**Last Updated**: November 20, 2025

---

## Overview

This guide documents testing requirements for each step of the climbing video analysis pipeline. Each section includes:

- **Input**: Required inputs (files, parameters, data formats)
- **Expected Output**: Output format and validation criteria
- **Test Cases**: Specific test scenarios with expected results
- **Edge Cases**: Error handling and boundary conditions

---

## Pipeline Steps

### 1. Frame Extraction

**Script**: `scripts/extract_frames.py`

**Purpose**: Extract frames from climbing videos at specified intervals.

#### Input

- **Video file**: MP4, MOV, AVI (any format supported by OpenCV)
- **Interval**: Frame extraction interval in seconds (e.g., 1.0, 1.5, 2.0)
- **Output directory**: Target directory for extracted frames

**Command**:
```bash
python scripts/extract_frames.py <video_dir> --output <output_dir> --interval <seconds>
```

#### Expected Output

**Directory structure**:
```
<output_dir>/
  <video_name>/
    frame_000000.jpg
    frame_000001.jpg
    ...
    manifest.json
```

**Manifest.json schema**:
```json
{
  "video_name": "sample_climb.mp4",
  "fps": 30.0,
  "total_frames": 30,
  "frames": [
    {
      "frame_number": 0,
      "timestamp_seconds": 0.0,
      "image_path": "frame_000000.jpg"
    },
    ...
  ]
}
```

#### Validation Criteria

- [ ] All frames extracted (count matches expected: `duration / interval`)
- [ ] Timestamps are sequential and match interval
- [ ] Images are readable (OpenCV can load)
- [ ] No duplicate timestamps
- [ ] Image dimensions match video resolution
- [ ] Manifest.json is valid JSON
- [ ] All frame paths in manifest exist

#### Test Cases

**Test Case 1.1: Normal Video Extraction**

**Input**:
- Video: `test_data/sample_climb.mp4` (30 seconds, 1920x1080, 30 fps)
- Interval: 1.0 seconds
- Output: `data/frames/sample_climb/`

**Expected Output**:
- Frame count: 30 frames
- Timestamps: 0.0, 1.0, 2.0, ..., 29.0 seconds
- Image format: JPEG, 1920x1080
- Manifest: Valid JSON with 30 frame entries

**Test Case 1.2: Short Video (Edge Case)**

**Input**:
- Video: `test_data/short_climb.mp4` (2 seconds)
- Interval: 1.0 seconds

**Expected Output**:
- Frame count: 2 frames
- Timestamps: 0.0, 1.0

**Test Case 1.3: High Interval (Edge Case)**

**Input**:
- Video: `test_data/sample_climb.mp4` (30 seconds)
- Interval: 5.0 seconds

**Expected Output**:
- Frame count: 6 frames (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)

**Test Case 1.4: Invalid Video File (Error Handling)**

**Input**:
- Video: `test_data/corrupted.mp4` (corrupted file)

**Expected Behavior**:
- Error message logged
- No frames extracted
- No manifest created

---

### 2. Pose Estimation

**Script**: `scripts/run_pose_estimation.py`

**Purpose**: Extract MediaPipe pose landmarks from frame images.

#### Input

- **Frames directory**: Directory containing extracted frames
- **Manifest.json**: Frame manifest from step 1
- **MediaPipe model**: Default or custom MediaPipe pose model

**Command**:
```bash
python scripts/run_pose_estimation.py --frames-root <frames_dir>
```

#### Expected Output

**pose_results.json schema**:
```json
{
  "video_name": "sample_climb",
  "fps": 30.0,
  "frames": [
    {
      "image_path": "frame_000000.jpg",
      "timestamp_seconds": 0.0,
      "detection_score": 0.95,
      "landmarks": [
        {
          "name": "left_wrist",
          "x": 0.45,
          "y": 0.32,
          "z": -0.01,
          "visibility": 0.92
        },
        ...
      ]
    },
    ...
  ]
}
```

#### Validation Criteria

- [ ] All frames processed (count matches manifest)
- [ ] Each frame has 33 landmarks (MediaPipe standard)
- [ ] Detection scores in range [0.0, 1.0]
- [ ] Coordinates normalized (x, y typically in [0, 1], z in [-1, 1])
- [ ] Visibility scores in range [0.0, 1.0]
- [ ] Landmark names match MediaPipe standard names
- [ ] Timestamps match manifest timestamps

#### Test Cases

**Test Case 2.1: Normal Pose Estimation**

**Input**:
- Frames: 30 frames from step 1
- Manifest: Valid manifest.json

**Expected Output**:
- pose_results.json with 30 frame entries
- Each frame: 33 landmarks
- Detection scores > 0.3 (reasonable threshold)
- All landmarks have valid coordinates

**Test Case 2.2: Low Detection Score (Edge Case)**

**Input**:
- Frame with partial occlusion or poor lighting

**Expected Output**:
- Detection score < 0.5
- Some landmarks may have low visibility (< 0.5)
- Coordinates still present (may be estimated)

**Test Case 2.3: Missing Frame (Error Handling)**

**Input**:
- Manifest references frame_000999.jpg (doesn't exist)

**Expected Behavior**:
- Error logged for missing frame
- Other frames still processed
- Missing frame entry skipped or marked as error

---

### 3. Hold Detection

**Module**: `src/pose_ai/service/hold_extraction.py`

**Purpose**: Detect and cluster holds using YOLOv8.

#### Input

- **Frame images**: JPEG images from step 1
- **YOLO model**: YOLOv8 model weights (yolov8n.pt, yolov8m.pt, etc.)
- **Hold labels**: Target labels (hold, foot_hold, volume, etc.)

**Command** (via pipeline):
```python
from pose_ai.service.hold_extraction import extract_and_cluster_holds

holds = extract_and_cluster_holds(
    image_paths,
    model_name="yolov8n.pt",
    use_tracking=True,
)
```

#### Expected Output

**holds.json schema**:
```json
{
  "hold_0": {
    "coords": [0.45, 0.32],
    "normalized": true,
    "hold_type": "jug",
    "confidence": 0.85,
    "cluster_size": 5
  },
  ...
}
```

#### Validation Criteria

- [ ] Hold count > 0 (at least some holds detected)
- [ ] Coordinates in normalized range [0, 1]
- [ ] Hold types valid (jug, crimp, sloper, pinch, foot_only, volume, or null)
- [ ] Confidence scores in range [0.0, 1.0]
- [ ] Cluster sizes >= 1
- [ ] No duplicate hold IDs

#### Test Cases

**Test Case 3.1: Normal Hold Detection**

**Input**:
- 30 frame images
- YOLOv8n model
- Tracking enabled

**Expected Output**:
- 10-50 holds detected (typical range)
- Hold types distributed (not all same type)
- Confidence scores > 0.35 (YOLO threshold)
- Stable tracks (min_hits=3)

**Test Case 3.2: No Holds Detected (Edge Case)**

**Input**:
- Frames with no visible holds (wall only, climber blocking)

**Expected Output**:
- Empty holds.json or minimal holds
- Warning logged
- Pipeline continues (holds optional)

**Test Case 3.3: Tracking vs No Tracking**

**Input**:
- Same frames, `use_tracking=True` vs `use_tracking=False`

**Expected Output**:
- Tracking: More stable hold positions, fewer duplicates
- No tracking: More noisy positions, potential duplicates
- Both produce valid holds.json

---

### 4. Feature Extraction

**Module**: `src/pose_ai/service/feature_service.py`

**Purpose**: Extract frame-level features from pose and holds.

#### Input

- **pose_results.json**: From step 2
- **holds.json**: From step 3 (optional)
- **IMU data**: Optional (quaternion, Euler angles)
- **Climber params**: Optional (height, wingspan, flexibility)

**Command**:
```python
from pose_ai.service.feature_service import export_features_for_manifest

export_features_for_manifest(
    manifest_path,
    holds_path=holds_path,
    imu_quaternion=[...],
    climber_height=175.0,
)
```

#### Expected Output

**pose_features.json schema**:
```json
[
  {
    "image_path": "frame_000000.jpg",
    "timestamp": 0.0,
    "detection_score": 0.95,
    "com_x": 0.5,
    "com_y": 0.6,
    "com_z": -0.02,
    "body_scale": 0.42,
    "wall_angle": 90.0,
    "left_hand_x": 0.45,
    "left_hand_y": 0.32,
    "left_hand_contact_on": true,
    "left_hand_contact_hold": "hold_0",
    "technique_bicycle": 0.2,
    "climber_height": 175.0,
    ...
  },
  ...
]
```

#### Validation Criteria

- [ ] Feature count matches frame count
- [ ] All required features present (com_x, com_y, body_scale, etc.)
- [ ] Temporal derivatives computed (com_velocity, com_acceleration, com_jerk)
- [ ] Contact states valid (true/false, hold IDs match holds.json)
- [ ] Technique scores in range [0.0, 1.0]
- [ ] Wall angle present (from IMU or vision)
- [ ] Climber params propagated (if provided)

#### Test Cases

**Test Case 4.1: Normal Feature Extraction**

**Input**:
- pose_results.json (30 frames)
- holds.json (20 holds)
- No IMU/climber params

**Expected Output**:
- 30 feature rows
- All features computed
- Contact inference applied
- Techniques detected
- Wall angle from vision (if auto_estimate_wall=True)

**Test Case 4.2: With IMU Data**

**Input**:
- pose_results.json
- IMU quaternion: [0.7071, 0.0, 0.7071, 0.0]

**Expected Output**:
- wall_angle computed from IMU (priority over vision)
- Higher confidence angle value

**Test Case 4.3: With Climber Params**

**Input**:
- pose_results.json
- climber_height: 175.0
- climber_wingspan: 180.0
- climber_flexibility: 0.7

**Expected Output**:
- body_scale_normalized computed
- climber_height, climber_wingspan, climber_flexibility in each row
- Personalized features available

---

### 5. Step Segmentation

**Module**: `src/pose_ai/segmentation/steps.py`

**Purpose**: Segment frames into steps based on contact changes.

#### Input

- **pose_features.json**: From step 4

**Function**:
```python
from pose_ai.segmentation.steps import segment_steps_by_contacts

segments = segment_steps_by_contacts(feature_rows)
```

#### Expected Output

**Step segments** (used internally, not saved separately):
```python
[
  StepSegment(
    step_id=0,
    start_index=0,
    end_index=15,
    start_time=0.0,
    end_time=0.6,
    label="Reach"
  ),
  ...
]
```

**Integrated in step_efficiency.json**:
```json
[
  {
    "step_id": 0,
    "score": 0.75,
    "start_time": 0.0,
    "end_time": 0.6,
    "label": "Reach",
    "components": {...}
  },
  ...
]
```

#### Validation Criteria

- [ ] Segment count > 0
- [ ] All segments have valid indices (within feature_rows range)
- [ ] Duration constraints: 0.2s ≤ duration ≤ 4.0s
- [ ] Labels valid: Reach, Stabilize, FootAdjust, DynamicMove, Rest, Finish
- [ ] No overlapping segments
- [ ] Segments cover all frames (or most frames)

#### Test Cases

**Test Case 5.1: Normal Segmentation**

**Input**:
- pose_features.json (30 frames, 1.2 seconds total)

**Expected Output**:
- 3-8 segments (typical)
- Each segment: 0.2-4.0 seconds duration
- Labels distributed (not all same label)
- Segments sequential (no gaps)

**Test Case 5.2: Very Short Video (Edge Case)**

**Input**:
- pose_features.json (5 frames, 0.2 seconds)

**Expected Output**:
- 1 segment (minimum duration)
- Label: "Reach" or "Stabilize"

**Test Case 5.3: No Contact Changes (Edge Case)**

**Input**:
- pose_features.json with no contact changes (all frames same contacts)

**Expected Output**:
- 1 segment (entire video)
- Label: "Rest" or "Stabilize"

---

### 6. Efficiency Scoring

**Module**: `src/pose_ai/recommendation/efficiency.py`

**Purpose**: Compute step-level efficiency scores.

#### Input

- **pose_features.json**: From step 4
- **Step segments**: From step 5

**Function**:
```python
from pose_ai.recommendation.efficiency import score_steps

efficiency_results = score_steps(feature_rows, step_segments)
```

#### Expected Output

**step_efficiency.json schema**:
```json
[
  {
    "step_id": 0,
    "score": 0.75,
    "start_time": 0.0,
    "end_time": 0.6,
    "stability": 0.85,
    "path_efficiency": 0.70,
    "support_penalty": 0.0,
    "wall_penalty": 0.05,
    "jerk_penalty": 0.10,
    "reach_penalty": 0.05,
    "technique_bonus": 0.08
  },
  ...
]
```

#### Validation Criteria

- [ ] Efficiency count matches segment count
- [ ] Scores in reasonable range (typically 0.0-1.0, can be negative)
- [ ] All components present (stability, path_efficiency, penalties, bonus)
- [ ] Component values in valid ranges
- [ ] Scores correlate with movement quality (higher = better)

#### Test Cases

**Test Case 6.1: Normal Efficiency Scoring**

**Input**:
- pose_features.json (30 frames)
- 5 step segments

**Expected Output**:
- 5 efficiency results
- Scores: 0.3-0.9 (typical range)
- Components sum to score (weighted)

**Test Case 6.2: Low Efficiency (Edge Case)**

**Input**:
- Features with many contact switches, high jerk, low support

**Expected Output**:
- Low efficiency score (< 0.3)
- High penalties (support_penalty, jerk_penalty)
- Low stability

**Test Case 6.3: High Efficiency (Edge Case)**

**Input**:
- Features with stable contacts, smooth movement, good technique

**Expected Output**:
- High efficiency score (> 0.7)
- Low penalties
- High stability, path_efficiency
- Technique bonus applied

---

### 7. ML Model Inference

**Module**: `src/pose_ai/ml/inference.py`

**Purpose**: Predict efficiency and next-action using trained BiLSTM/Transformer.

#### Input

- **pose_features.json**: From step 4
- **Model checkpoint**: `models/checkpoints/bilstm_multitask.pt` or `transformer_multitask.pt`
- **Normalization params**: `models/checkpoints/normalization.npz` (optional)

**API Endpoint**:
```bash
GET /api/jobs/{job_id}/ml_predictions
```

#### Expected Output

**API Response**:
```json
{
  "job_id": "abc123",
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

#### Validation Criteria

- [ ] Predictions count > 0 (sliding window inference)
- [ ] Efficiency scores in reasonable range (model-dependent)
- [ ] Next-action classes valid (0-4: no_change, left_hand, right_hand, left_foot, right_foot)
- [ ] Action probabilities sum to ~1.0
- [ ] Frame indices within feature_rows range

#### Test Cases

**Test Case 7.1: Normal ML Inference**

**Input**:
- pose_features.json (30 frames)
- Trained BiLSTM model
- Normalization params

**Expected Output**:
- ~25 predictions (30 frames - 32 window + 1, stride=5)
- Efficiency scores: 0.0-1.0 range
- Action probabilities valid

**Test Case 7.2: Transformer Model**

**Input**:
- Same features
- Trained Transformer model

**Expected Output**:
- Similar prediction count
- Auto-detected model_type: "transformer"
- Comparable score ranges

**Test Case 7.3: Missing Model (Error Handling)**

**Input**:
- pose_features.json
- No model checkpoint

**Expected Behavior**:
- HTTP 503 error
- Message: "ML model not available"

---

### 8. Route Grading (NEW)

**Module**: `src/pose_ai/ml/route_grading.py`

**Purpose**: Predict route difficulty (V0-V10) from route-level features.

#### Input

- **pose_features.json**: From step 4
- **step_efficiency.json**: From step 6
- **holds.json**: From step 3
- **Trained model**: `models/route_grader.json` (optional)

**API Endpoint**:
```bash
GET /api/jobs/{job_id}/route_grade
```

#### Expected Output

**API Response**:
```json
{
  "job_id": "abc123",
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

#### Validation Criteria

- [ ] Grade in range [0.0, 10.0] (V0-V10)
- [ ] Confidence in range [0.0, 1.0]
- [ ] All route features present
- [ ] Feature values in reasonable ranges
- [ ] Calibrated grade string valid

#### Test Cases

**Test Case 8.1: Normal Route Grading**

**Input**:
- Complete pipeline outputs (features, efficiency, holds)
- Trained route grader model

**Expected Output**:
- Grade: 3.0-7.0 (typical range)
- Confidence: > 0.5
- All features extracted correctly

**Test Case 8.2: No Model (Edge Case)**

**Input**:
- Complete pipeline outputs
- No trained model

**Expected Output**:
- grade: null
- confidence: 0.0
- Features still extracted (for future training)

**Test Case 8.3: Minimal Holds (Edge Case)**

**Input**:
- Features with only 2-3 holds detected

**Expected Output**:
- Low hold_density
- Grade may be unreliable (low confidence)
- Features still computed

---

## End-to-End Pipeline Test

### Complete Pipeline Flow

**Input**:
- Video file: `test_data/complete_climb.mp4` (60 seconds)
- IMU data: Quaternion [0.7071, 0.0, 0.7071, 0.0]
- Climber params: height=175, wingspan=180, flexibility=0.7

**Expected Outputs**:

1. **Frames**: 60 frames (interval=1.0s)
2. **Pose**: 60 frames × 33 landmarks
3. **Holds**: 15-30 holds detected
4. **Features**: 60 feature rows with all fields
5. **Segments**: 8-15 step segments
6. **Efficiency**: 8-15 efficiency scores
7. **ML Predictions**: ~50 predictions (if model available)
8. **Route Grade**: V3-V6 (typical)

**Validation**:
- [ ] All steps complete without errors
- [ ] Output files exist and are valid JSON
- [ ] Data flows correctly between steps
- [ ] No data loss or corruption
- [ ] Performance acceptable (< 5 minutes for 60s video)

---

## Test Data Requirements

### Sample Videos

1. **Short video**: 5-10 seconds (edge case testing)
2. **Normal video**: 30-60 seconds (typical use case)
3. **Long video**: 2+ minutes (stress testing)
4. **Known difficulty routes**: V3, V5, V7 (for route grading validation)

### Ground Truth Data

- **Route grades**: Manually annotated V-scale grades
- **Step labels**: Manually segmented steps (optional, for validation)
- **Hold annotations**: Manually labeled hold types (optional)

### Edge Cases

- **No holds visible**: Wall-only frames
- **Single frame**: Minimal video
- **Extreme angles**: Overhangs (>90°), slabs (<90°)
- **Occlusion**: Climber blocking holds
- **Poor lighting**: Low detection scores

---

## Running Tests

### Unit Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run specific test file
pytest tests/unit/test_feature_extraction.py
```

### Integration Tests

```bash
# Run pipeline on test video
python scripts/run_pipeline.py test_data/sample_climb.mp4 --out test_output

# Validate outputs
python scripts/validate_pipeline_outputs.py test_output/
```

### Manual Testing

1. Upload video via web UI
2. Monitor job status
3. Check API endpoints
4. Verify UI displays

---

## Test Coverage Goals

- **Unit tests**: >80% code coverage
- **Integration tests**: All pipeline steps
- **Edge cases**: All documented edge cases
- **Error handling**: All error paths tested

---

## References

- **Pipeline Guide**: [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)
- **Efficiency Calculation**: [efficiency_calculation.md](efficiency_calculation.md)
- **Implementation Backlog**: [IMPLEMENTATION_BACKLOG.md](IMPLEMENTATION_BACKLOG.md)

