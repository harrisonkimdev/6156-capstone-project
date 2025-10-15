# 6156 Capstone – Climbing Pose Analysis Platform

## 1. Vision & Scope
- **Goal**: ingest short-form bouldering videos, segment the climb into meaningful phases, extract pose-derived metrics, and return strategy suggestions through a backend API.
- **Environment**: Python service deployed on servers (container-friendly), interacting with background workers and storage.
- **Deliverables**: modular Python library (`pose_ai`), FastAPI service, asynchronous processing pipeline, documentation and automated tests.

## 2. End-to-End Pipeline
1. **Ingestion** – accept uploaded media or URLs, persist raw assets, enqueue processing jobs.
2. **Segmentation** – split the video into movement phases via rule-based heuristic, scene-change hybrid, or learned temporal models.
3. **Pose Extraction** – run MediaPipe/MoveNet on frames; smooth landmarks and handle dropouts.
4. **Feature Aggregation** – compute joint angles, center-of-mass trajectories, hold contact stats per segment.
5. **Recommendation** – analyse segments, detect climbing patterns, generate improvement suggestions.
6. **Delivery** – expose results via API, dashboards, or batch export.

## 3. Architectural Principles
- **Layered design**: data access, segmentation, pose, features, recommendation, services, and web API separated for clarity.
- **Dependency inversion**: higher layers depend on abstract interfaces; implementations can swap (e.g., different pose models).
- **Asynchronous processing**: long-running tasks run on queue workers (Celery/RQ/Arq) with Redis/SQS brokers.
- **Observability-first**: structured logging, Prometheus metrics, and persistent job metadata for traceability.
- **Testable units**: each module ships with dedicated unit tests; pipeline and API have integration/e2e tests.

## 4. Proposed Package Layout
```
pose_ai/
  __init__.py
  config/settings.py
  utils/logger.py
  utils/timecode.py
  data/video_loader.py
  data/frame_sampler.py
  data/metrics.py
  segmentation/rule_based.py
  segmentation/hybrid.py
  segmentation/model.py
  pose/estimator.py
  pose/filters.py
  features/__init__.py
  features/aggregation.py
  features/config.py
  recommendation/planner.py
  recommendation/reasoner.py
  persistence/repositories.py
  persistence/schemas.py
  service/analyze_video.py
  service/get_insights.py
  service/segmentation_service.py
  service/pose_service.py
  web/api.py
  web/routers/videos.py
  web/schemas.py
  web/deps.py
tests/
  unit/
  integration/
  e2e/
scripts/
  run_pipeline.py
  extract_frames.py
  run_segmentation.py
  run_pose_estimation.py
  run_feature_export.py
```

### Module Responsibilities
- **config** – load environment variables with Pydantic settings; centralise thresholds and model paths.
- **data** – file acquisition, ffmpeg wrappers, frame skipping strategies, plus manifest utilities that convert saved frame metadata into segmentation-ready metrics.
- **segmentation** – successive implementations (rule-based MVP, shot-detect hybrid, trainable temporal models). Current rule-based module (`segment_by_activity`) classifies rest vs movement from per-frame motion/hold-change scores.
- **pose** – wrap estimators (MediaPipe, Alternate models), apply landmark smoothing, manage retries. Current scaffold (`PoseEstimator`) returns placeholder `PoseFrame` objects until MediaPipe integration lands.
- **features** – derive frame/segment metrics (angles, center of mass, hold proximity) and export summarised data.
- **recommendation** – orchestrate expert rules and ML models to produce actionable advice.
- **persistence** – database/file storage abstraction (Postgres + S3/GCS recommended).
- **service** – application use cases (e.g., `AnalyzeVideoService`) and orchestration helpers; segmentation bridge plus pose service that converts manifests into pose results with optional MediaPipe inference.
- **web** – FastAPI routers, request/response schemas, dependency injection.
- **tests** – fixtures, mocks, sample videos, regression snapshots.

## 5. Video Segmentation Strategy
### Rule-Based MVP (`segmentation/rule_based.py`)
- Inputs: frame timestamps, landmark positions, COM velocity, optical flow magnitude, hold contact signals.
- Logic: detect peaks above thresholds → mark boundaries; enforce min segment duration; merge noise.
- Output: `Segment(start, end, label)` list for downstream processing.

### Hybrid Approach (`segmentation/hybrid.py`)
- Stage 1: scene/shot detection (PySceneDetect or histogram diff).
- Stage 2: refine with pose dynamics (movement classification, hold change detection).
- Adaptive thresholds: tune automatically using aggregated job statistics.

### Learned Model (`segmentation/model.py`)
- Dataset: annotated videos with labelled phases (start, crux, top-out, rests).
- Model: Temporal CNN / Transformer over pose features.
- Deployment: export to TorchScript/ONNX, support GPU acceleration.

## 6. Pose Extraction & Smoothing
- `pose/estimator.py` handles model selection, batching, retry on camera failures.
- `PoseFrame` dataclass: image reference, timestamp, landmarks, confidences, raw inference results.
- `pose/filters.py`: OneEuro/Kalman smoothing, interpolation for missing keypoints.
- Error policy: log warnings, skip frames, reinitialise estimator after N failures.

## 7. Feature Engineering
- `features/aggregator.py`: summarise per-segment metrics (mean joint angles, COM path, move duration).
- `features/metrics.py`: derive scoring (efficiency, stability, risk).
- `features/patterns.py`: recognise climbing techniques (drop-knee, heel-hook) via heuristics/ML.
- Outputs stored as Pandas DataFrames, serialised to Parquet/JSON for downstream systems.

## 8. Recommendation Engine
- `recommendation/planner.py`: analyse segments, determine key decisions, highlight improvement opportunities.
- `recommendation/reasoner.py`: compare with historical success cases, surface alternative beta suggestions.
- Supports rule-based and ML models (e.g., retrieval-based reasoning using vector search).

## 9. Backend/API Design
- Framework: FastAPI with Pydantic v2 schemas.
- Endpoints (initial set):
  - `POST /videos/`: submit video or URL, returns job ID.
  - `GET /videos/{id}/status`: check processing status.
  - `GET /videos/{id}/insights`: retrieve processed metrics & recommendations.
  - `POST /videos/{id}/feedback`: capture user feedback for continuous learning.
- Background jobs: Celery/RQ worker invoking `AnalyzeVideoService`.
- Storage: raw media in S3/GCS, metadata in Postgres, cached features in Redis/Parquet.
- Security: API key initially; plan for OAuth/JWT integration later.

## 10. Tooling & Environment
- **Package manager**: Poetry or PDM (`pyproject.toml`).
- **Runtime deps**: mediapipe, opencv-python, numpy, pandas, scikit-learn, torch (optional), fastapi, uvicorn, pydantic-settings, celery/rq, redis.
- **Dev deps**: pytest, pytest-asyncio, httpx, black, ruff, mypy.
- **Configuration**: `.env` + `config/settings.py`; include `.env.example`.
- **Containerisation**: Dockerfile (python:3.11-slim base + mediapipe deps); docker-compose with web, worker, redis, postgres.

## 11. Git Workflow
- Reuse existing repository (`/Users/harrisonkim/code/repos/6156-capstone-project`).
- `.gitignore`: add `__pycache__/`, `.venv/`, `artifacts/`, `*.log`, `*.mp4`, `*.ipynb_checkpoints`, `notebooks/`.
- Branching: `main` + `feature/<task-name>`; use PRs for reviews.
- Conventional commits (`feat`, `fix`, `refactor`, `docs`, `test`) with linked task IDs where applicable.

## 12. Implementation Roadmap
1. **TASK-01 – Requirements Doc**: formalise input/output specs, metrics, SLAs.
2. **TASK-02 – Scaffold**: create package folders, `pyproject.toml`, `.gitignore`, README updates.
3. **TASK-03 – Rule-Based Segmentation MVP**: implement heuristics, unit tests with synthetic data.
4. **TASK-04 – Pose Module**: MediaPipe wrapper, smoothing filters, synthetic regression test.
5. **TASK-05 – Feature Aggregation**: segment KPIs, reproducible dataset exports.
6. **TASK-06 – Recommendation Draft**: rule-based planner, schema definitions.
7. **TASK-07 – FastAPI Service**: endpoints, background queue integration.
8. **TASK-08 – Integration Tests**: sample video pipeline run, snapshot results.
9. **TASK-09 – Deployment Prep**: Docker, docker-compose, CI (lint/test/build).

## 13. Testing & QA
- Unit tests per module (fast, deterministic).
- Initial coverage includes `tests/unit/test_frame_sampler.py` for frame extraction manifests, `tests/unit/test_route_detection.py` for HSV-based hold clustering, `tests/unit/test_manifest_metrics.py` for manifest-to-metric conversion, `tests/unit/test_pose_estimator.py` for the estimation scaffold, `tests/unit/test_pose_filters.py` for smoothing behaviour, `tests/unit/test_pose_service.py` for manifest pose pipelines, `tests/unit/test_features_aggregation.py` for frame-level metrics, `tests/unit/test_feature_service.py` for feature exports, `tests/unit/test_rule_based_segmentation.py` for the activity segmentation heuristics, and `tests/unit/test_segmentation_service.py` for manifest-driven orchestration.
- Integration tests for pipeline stages (use 10–15s sample video).
- E2E tests hitting FastAPI endpoints with async client.
- Regression suite: synthetic pose data, stored snapshots to detect drift.
- Performance benchmarks: measure throughput, GPU/CPU utilisation, memory footprint.

## 14. Operational Considerations
- **Storage strategy**: object storage for media, relational DB for metadata, parquet files for analytics.
- **Scaling**: horizontal scaling of workers; consider GPU pools for pose inference.
- **Monitoring**: metrics on job latency, success/failure counts, per-stage timings.
- **Model lifecycle**: versioned models, canary deployments, rollback procedures.

## 15. Documentation & Conversion
- Store this file as `docs/design.md`.
- For PDF: `pandoc docs/design.md -o docs/design.pdf`.
- Update alongside each milestone to reflect new learnings or design changes.
- **Legacy notebook migration** – the existing `BetaMove/VideoToImage/VideoToImagePush.ipynb` extracts frames with OpenCV into `BetaMove/rawImages/`. When porting to `src/pose_ai/data`, convert it into callable modules (`frame_sampler.py`, `route_detection.py`) that:
  - accepts input paths from `videos/` (or uploaded sources) and persists frames under `data/frames/<video_id>/`.
  - records frame metadata (timestamp, frame index) to a manifest JSON/Parquet for downstream processing.
  - exposes reusable HSV tolerance/hold-detection helpers for the forthcoming route-recognition workflow.
  - remains callable both from CLI scripts and background workers while keeping notebooks as optional demos only.
- **Utility script** – `scripts/extract_frames.py` provides a CLI over `pose_ai.data` helpers to batch process a directory of videos (interval, recursion, manifest toggles), replacing the manual notebook workflow when seeding datasets.
