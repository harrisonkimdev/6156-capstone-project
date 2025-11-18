# Beta Recommendation Model Roadmap

이 문서는 볼더링 베타 추천 AI를 구축하기 위한 고레벨 로드맵과 작업 순서를 정리합니다. 각 페이즈는 앞선 작업의 산출물을 활용해 다음 단계로 확장하도록 설계했습니다.

**Last Updated**: November 18, 2025  
**Status**: Phase 1 core features implemented; transitioning to advanced capabilities

---

## Phase 0 — 현재 파이프라인 이해 (Existing Assets)
- [x] 현 `pose_features.json` 및 세그먼트 메트릭 구조 파악
- [x] 수집 가능한 로그/메타데이터 정리 (루트, 클라이머, 세그먼트)
- [x] 현행 스크립트/웹 UI 기능 간 의존성 다이어그램 업데이트

## Phase 1 — 문제 정의와 성공 지표 (Problem Definition) ✅ COMPLETED
- [x] 사용자 요구와 베타 추천 형태 정의 (동작 시퀀스 vs. 설명 텍스트 등)
- [x] 예측 타겟/레이블 스키마 설계 (예: 추천 동작 집합, 성공 확률, 난이도 조정)
- [x] 성공 지표(KPI) 및 평가 프로토콜 합의 (예: Top-k 정확도, 사용자 피드백 점수)

### Completed Deliverables (Phase 1)
- **Hold Detection**: YOLOv8n/m with DBSCAN clustering ([`src/pose_ai/service/hold_extraction.py`](../src/pose_ai/service/hold_extraction.py))
- **Wall Angle Estimation**: Hough + PCA automatic inference ([`src/pose_ai/wall/angle.py`](../src/pose_ai/wall/angle.py))
- **Efficiency Scoring MVP**: 5-component rule-based metric ([`src/pose_ai/recommendation/efficiency.py`](../src/pose_ai/recommendation/efficiency.py))
- **Next-Hold Recommendations**: Distance-based heuristic with recency weighting
- **Web UI Integration**: Real-time efficiency display and recommendation cards
- **API Endpoints**: `/api/jobs/{job_id}/analysis` for efficiency and recommendations

## Phase 2 — 데이터 인벤토리 및 갭 분석 (Data Inventory & Gap Analysis)
- [x] 현재 파이프라인에서 추출 가능한 피처 목록화 (포즈, 세그먼트, 시간 통계)
- [x] 필요한 추가 신호 정의 (홀드 위치/종류, 벽 각도, 클라이머 프로필 등)
- [ ] 데이터 확보 전략 수립 (수동 라벨링, 컴퓨터 비전, 외부 API 등)
- [ ] 데이터 거버넌스/버전 관리 방안 초안 작성

## Phase 3 — 데이터 파이프라인 확장 (Instrumentation & Collection)
- [x] 홀드/벽 메타데이터 캡처 파이프라인 설계 및 PoC
- [ ] 클라이머 프로필 수집 및 익명화 전략 구현
- [ ] 포즈+홀드+벽 정보를 결합한 세션 단위 데이터셋 스키마 정의
- [ ] 지속 수집을 위한 스토리지/ETL/검증 자동화 설정

## Phase 4 — 모델링 프로토타입 (Model Prototyping)
- [x] 베이스라인 라벨 태스크 구현 (예: 성공 여부/세그먼트 분류) — Gradient Boosting/XGBoost
- [ ] 모델 설명 및 중요도 분석 체계 마련 (SHAP, feature importance 등)
- [x] 학습 자동화 스크립트/노트북 정비, 반복 실험 로깅

## Phase 5 — 평가 및 제품 통합 (Evaluation & Integration)
- [ ] 오프라인 평가 결과 공유 및 피드백 회수
- [x] 베타 추천 UX 프로토타입 설계 (CLI/Web UI/리포트)
- [x] 모델 서빙 전략 결정 (배치추천 vs. 실시간)
- [ ] 모니터링 및 재학습 계획 수립

## Phase 6 — 장기 발전 (Future Enhancements)
- [ ] 추가 센서/모션 캡처 연동 검토
- [ ] 협업 필터링/커뮤니티 데이터와의 결합
- [ ] 시퀀스/멀티모달 모델 실험 (LSTM/Transformer) — See [IMPLEMENTATION_BACKLOG.md](IMPLEMENTATION_BACKLOG.md)

---

## Current Implementation Status (November 2025)

### ✅ Completed Features

**Hold Detection & Clustering**:
- YOLOv8n (fast, mAP ≥ 0.60) and YOLOv8m (accurate, mAP ≥ 0.68) support
- DBSCAN spatial clustering for stable hold positions
- Per-frame detection with frame-wise confidence tracking
- Integration: Web UI, CLI script, pipeline runner

**Wall Angle Estimation**:
- Hough line detection + RANSAC fitting
- PCA fallback for edge-rich frames
- Confidence scoring based on line consensus
- Performance: Vertical walls MAE ≤ 5°, Overhangs MAE ≤ 8°

**Efficiency Scoring (Rule-Based MVP)**:
- 5-component metric: detection quality (0.20), joint smoothness (0.25), COM stability (0.25), contact count (0.15), hip-wall alignment (0.15)
- Normalized 0-1 score with clear interpretation ranges
- Frame-level and aggregated scoring

**Next-Hold Recommendations**:
- Distance-based heuristic with recency penalty
- Vertical progression bias (upward preferred)
- Top-k selection with confidence scoring

**Extended Features**:
- Wall alignment metrics: `wall_angle`, `hip_alignment_error`, `com_along_wall`, `com_perp_wall`
- Integration into feature aggregation pipeline

**Web API & UI**:
- FastAPI endpoints for job management
- Real-time status updates with background task execution
- Efficiency display card with component breakdown
- Next-hold recommendation visualization
- Training UI for XGBoost model experimentation
- GCS integration for cloud storage (optional)

**Infrastructure**:
- Background job management with unique job IDs
- Artifact tracking (frames, holds, pose, features, segments)
- Pipeline stage monitoring
- CLI tools for all stages

---

## Short-Term Priorities (Next 2-3 Weeks)

### 1. Advanced Contact Inference

**Current Gap**: Basic distance thresholding, no velocity or temporal filtering  
**Target**: Full algorithm per [efficiency_calculation.md](efficiency_calculation.md)

Tasks:
- [ ] Implement hysteresis (separate r_on/r_off thresholds)
- [ ] Add velocity condition (`|v| <= v_hold`)
- [ ] Minimum duration filter (`min_on_frames >= 3`)
- [ ] Smear detection (foot near wall, no hold within radius)
- [ ] Optional: HMM/Viterbi temporal smoothing

**Impact**: More accurate contact inference → better efficiency scoring and step segmentation

### 2. Step Segmentation Enhancement

**Current Gap**: Simple movement/rest classification, no step boundaries  
**Target**: Contact-based step detection with duration constraints

Tasks:
- [ ] Split on confirmed contact changes (single limb priority)
- [ ] Enforce step duration constraints (0.2s - 4s)
- [ ] Segment labeling: Reach, Stabilize, FootAdjust, DynamicMove, Rest, Finish
- [ ] Validate against manual annotations

**Impact**: Meaningful step-level analysis for efficiency and recommendations

### 3. Full Efficiency Formula

**Current Gap**: 5-component heuristic, no support polygon or path metrics  
**Target**: 7-component physics-based formula per efficiency_calculation.md

Tasks:
- [ ] Support polygon stability (convex hull from contacts, COM distance)
- [ ] Support count/switch penalties (strong penalty if <2 contacts)
- [ ] Wall-body distance penalty (forearm load proxy)
- [ ] Path efficiency (net displacement vs path length)
- [ ] Smoothness penalty (jerk, direction changes)
- [ ] Reach-limit penalty (extreme limb extensions)
- [ ] Technique bonuses (bicycle, back-flag, drop-knee)

**Impact**: More accurate efficiency scores aligned with climbing physics

### 4. Hold Detection Improvements

Tasks:
- [ ] Transfer learning comparison: yolov8n vs yolov8m mAP/latency
- [ ] Hold type classification (crimp, sloper, jug, pinch, foot-only)
- [ ] Frame selection for wall angle (automatic rest segment detection)
- [ ] Hold tracking across frames (IoU + Kalman filter)

---

## Medium-Term Goals (1-2 Months)

### BiLSTM Multitask Model (v1)

**Architecture**:
- Input: Sliding windows (T=32 frames @ 25fps)
- Features: Normalized keypoints, v/a, COM, contact embeddings, efficiency metrics
- BiLSTM (128-256, 1-2 layers) + attention pooling
- Head 1: Efficiency regression (Huber loss)
- Head 2: Next-action classification (CrossEntropy)

Tasks:
- [ ] Dataset builder (sliding windows, stride=1)
- [ ] Weak label generation from heuristic efficiency
- [ ] Training pipeline with early stopping
- [ ] Evaluation: MAE/R² (efficiency), top-1/top-3 (next action)
- [ ] Fine-tuning on human annotations

### Rule-Based Planner v1

Tasks:
- [ ] Candidate hold sampling (K holds based on direction)
- [ ] Efficiency simulation (recompute score with new support set)
- [ ] Support polygon constraints (COM inside polygon)
- [ ] Reach and crossing limits
- [ ] Return best candidate with reasoning

### Technique Pattern Detection

Tasks:
- [ ] Bicycle detection (opposing toes, angle thresholds)
- [ ] Back-flag detection (hip-wall alignment, knee angle)
- [ ] Drop-knee detection (knee rotation, torso twist)
- [ ] Confidence scoring (0-1 per technique)
- [ ] Integration into efficiency formula as bonuses

---

## Long-Term Vision (3+ Months)

See [IMPLEMENTATION_BACKLOG.md](IMPLEMENTATION_BACKLOG.md) for detailed roadmap including:

- Transformer/TCN models (v2)
- Advanced wall calibration (RANSAC plane, multi-view)
- Climber profiling and personalization
- Route difficulty estimation
- Model versioning and registry
- Production deployment (monitoring, retraining, A/B testing)
- Data infrastructure (DVC, Parquet, CI/CD)

---
