# 6156-capstone-project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/harrisonkimdev/6156-capstone-project/blob/main/notebooks/analysis.ipynb)

## [Project Backlog](https://docs.google.com/spreadsheets/d/113DbJu6Vg53PxX8Kgu5pkwsuLrqYH-i9JJcEuDvWjus/edit?gid=1139408620#gid=1139408620)

## [Sprint Backlog](https://github.com/users/harrisonkimdev/projects/9/views/1?sortedBy%5Bdirection%5D=asc&sortedBy%5BcolumnId%5D=221588758&sortedBy%5Bdirection%5D=asc&sortedBy%5BcolumnId%5D=Assignees)

## Sprint Planning Meeting

## Scrum Meeting Evaluation

---

## Running the Pipeline Locally

All commands assume the project root and a Conda env named `6156-capstone` with dependencies (`mediapipe`, `opencv-python`, `numpy`, `pytest`, etc.) installed.

```bash
conda run -n 6156-capstone env PYTHONPATH=src <command>
```

### One-Command Pipeline
```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/run_pipeline.py BetaMove/videos --out data/frames --interval 1.5
```

### 1. Extract Frames from Videos
```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/extract_frames.py BetaMove/videos --output data/frames --interval 1.5
```

### 2. Run Pose Estimation on Extracted Frames
```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/run_pose_estimation.py --frames-root data/frames
```

### 3. Export Frame-Level Features
```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/run_feature_export.py data/frames/<video>/manifest.json
```

### 4. Generate Segment Metrics
```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/run_segment_report.py data/frames/<video>/manifest.json
```

### 5. Run Unit Tests
```bash
conda run -n 6156-capstone env PYTHONPATH=src pytest tests/unit
```

### 6. Train XGBoost Baseline
```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/train_xgboost.py data/frames/<video>/pose_features.json --label-column detection_score --label-threshold 0.6 --model-out models/xgb_pose.json
```
Adjust `--label-column`/`--label-threshold` to match your ground-truth labels. Optional `--feature-out` writes predictions to CSV.

### Pose Visualization
Overlay landmarks and connections on frames using:
```bash
conda run -n 6156-capstone env PYTHONPATH=src python scripts/visualize_pose.py data/frames/<video>/pose_results.json --min-score 0.3 --min-visibility 0.2
```
Adjust `--min-score` / `--min-visibility` to filter low-confidence detections. Outputs are saved alongside frames under `visualized/` by default.

---

## Web UI

Run the entire pipeline from a browser via the FastAPI app in `webapp/`.

1. Install the extra dependencies (once):
   ```bash
   conda run -n 6156-capstone pip install fastapi 'uvicorn[standard]' jinja2
   ```
2. Launch the server from the project root:
   ```bash
   conda run -n 6156-capstone env PYTHONPATH=src uvicorn webapp.main:app --reload
   ```
3. Open http://127.0.0.1:8000 to access the UI. Either point the form at an existing directory of videos or upload files directly in the browser. Logs stream live on the page, inline previews show generated pose visualizations, and sample pose rows are rendered alongside the job status. The API is available under `/api/*` for automation.

### Training UI
- Navigate to `http://127.0.0.1:8000/training` to upload a features JSON file and start an XGBoost training job with optional parameters (defaults provided). The page shows job status, saved model path, and metrics when complete.
