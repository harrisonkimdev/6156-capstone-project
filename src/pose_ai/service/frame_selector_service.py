"""End-to-end service for frame selector pipeline.

Handles:
- Hold detection from first frame
- Pose estimation
- Feature extraction  
- Model training
- Inference (key frame prediction)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from pose_ai.features.frame_selector_features import (
    extract_pose_based_features_from_sequence,
    features_to_numpy,
    load_hold_positions,
    normalize_features,
)
from pose_ai.ml.frame_selector_model import FrameSelectorBiLSTM, FrameSelectorConfig
from pose_ai.ml.frame_selector_trainer import (
    FrameSelectorDataset,
    FrameSelectorTrainer,
    TrainerConfig,
    evaluate_model,
    temporal_train_val_test_split,
)
from pose_ai.service.pose_service import estimate_poses_from_manifest

LOGGER = logging.getLogger(__name__)


def detect_holds_from_first_frame(
    workflow_dir: Path,
    model_path: Optional[Path] = None,
) -> Dict[str, List[float]]:
    """Detect holds from the first frame (no climber).
    
    Args:
        workflow_dir: Directory containing all_frames/
        model_path: Optional path to YOLO model
    
    Returns:
        Dictionary of hold positions
    """
    all_frames_dir = workflow_dir / 'all_frames'
    
    if not all_frames_dir.exists():
        raise FileNotFoundError(f"all_frames directory not found: {all_frames_dir}")
    
    # Get first frame
    frame_files = sorted(all_frames_dir.glob('*.jpg'))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {all_frames_dir}")
    
    first_frame = frame_files[0]
    LOGGER.info(f"Detecting holds from first frame: {first_frame.name}")
    
    # TODO: Integrate with YOLO hold detection
    # For now, return dummy data
    holds = {
        'holds': [
            {'id': 0, 'x': 0.3, 'y': 0.4, 'w': 0.05, 'h': 0.06},
            {'id': 1, 'x': 0.5, 'y': 0.5, 'w': 0.04, 'h': 0.05},
            {'id': 2, 'x': 0.7, 'y': 0.6, 'w': 0.06, 'h': 0.07},
        ]
    }
    
    # Save to hold_positions.json
    hold_positions_path = workflow_dir / 'hold_positions.json'
    with open(hold_positions_path, 'w') as f:
        json.dump(holds, f, indent=2)
    
    LOGGER.info(f"Saved {len(holds['holds'])} holds to {hold_positions_path}")
    
    return holds


def extract_features_from_workflow(
    workflow_dir: Path,
    fps: float = 30.0,
) -> Tuple[np.ndarray, List[Dict[str, any]]]:
    """Extract features from workflow directory.
    
    Args:
        workflow_dir: Directory containing manifest.json, hold_positions.json
        fps: Video frame rate
    
    Returns:
        (features_array, pose_sequence)
    """
    manifest_path = workflow_dir / 'manifest.json'
    hold_positions_path = workflow_dir / 'hold_positions.json'
    pose_results_path = workflow_dir / 'pose_results.json'
    
    # Step 1: Detect holds if not already done
    if not hold_positions_path.exists():
        LOGGER.info("Hold positions not found, detecting from first frame...")
        detect_holds_from_first_frame(workflow_dir)
    
    hold_positions = load_hold_positions(hold_positions_path)
    
    # Step 2: Pose estimation (use cached if available)
    if not pose_results_path.exists():
        LOGGER.info("Running pose estimation...")
        pose_frames = estimate_poses_from_manifest(
            manifest_path,
            save_json=True,
            output_path=pose_results_path,
        )
    else:
        LOGGER.info("Loading cached pose results...")
        with open(pose_results_path, 'r') as f:
            pose_data = json.load(f)
        
        # Convert to pose_sequence format
        pose_sequence = []
        for i, frame_data in enumerate(pose_data.get('frames', [])):
            landmarks_dict = {}
            for lm in frame_data.get('landmarks', []):
                landmarks_dict[lm['name']] = {
                    'x': lm['x'],
                    'y': lm['y'],
                    'z': lm['z'],
                    'visibility': lm['visibility'],
                }
            
            pose_sequence.append({
                'frame_idx': i,
                'timestamp': frame_data.get('timestamp_seconds', i / fps),
                'filename': Path(frame_data['image_path']).name,
                'landmarks': landmarks_dict,
            })
    
    # Step 3: Feature extraction
    LOGGER.info(f"Extracting features from {len(pose_sequence)} frames...")
    features_list = extract_pose_based_features_from_sequence(
        pose_sequence,
        hold_positions,
        fps=fps,
    )
    
    features_array = features_to_numpy(features_list)
    LOGGER.info(f"Extracted features shape: {features_array.shape}")
    
    return features_array, pose_sequence


def train_frame_selector_pipeline(
    workflow_dir: Path,
    output_dir: Optional[Path] = None,
    fps: float = 30.0,
    config: Optional[TrainerConfig] = None,
) -> Dict[str, any]:
    """Complete training pipeline for frame selector.
    
    Args:
        workflow_dir: Directory containing:
            - all_frames/
            - human_selected_frames/ (user-selected key frames)
            - manifest.json
        output_dir: Output directory for model and results
        fps: Video frame rate
        config: Training configuration
    
    Returns:
        Results dictionary
    """
    start_time = time.time()
    
    output_dir = output_dir or workflow_dir / 'frame_selector_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract features
    LOGGER.info("Step 1: Extracting features...")
    features, pose_sequence = extract_features_from_workflow(workflow_dir, fps)
    
    # Step 2: Create labels from human_selected_frames
    LOGGER.info("Step 2: Creating labels...")
    human_selected_dir = workflow_dir / 'human_selected_frames'
    
    if not human_selected_dir.exists() or len(list(human_selected_dir.glob('*.jpg'))) == 0:
        raise ValueError("No human-selected frames found")
    
    selected_filenames = {f.name for f in human_selected_dir.glob('*.jpg')}
    labels = np.array([
        1.0 if pose_data['filename'] in selected_filenames else 0.0
        for pose_data in pose_sequence
    ], dtype=np.float32)
    
    n_key_frames = int(labels.sum())
    LOGGER.info(f"Found {n_key_frames} key frames out of {len(labels)} total frames")
    
    if n_key_frames == 0:
        raise ValueError("No key frames found")
    
    # Step 3: Normalize features
    LOGGER.info("Step 3: Normalizing features...")
    features_norm, feat_mean, feat_std = normalize_features(features)
    
    # Save normalization stats
    np.save(output_dir / 'feature_mean.npy', feat_mean)
    np.save(output_dir / 'feature_std.npy', feat_std)
    
    # Step 4: Temporal split
    LOGGER.info("Step 4: Splitting data temporally...")
    train_feats, train_labels, val_feats, val_labels, test_feats, test_labels = \
        temporal_train_val_test_split(features_norm, labels)
    
    # Step 5: Create datasets
    LOGGER.info("Step 5: Creating datasets...")
    trainer_config = config or TrainerConfig(save_dir=output_dir / 'checkpoints')
    
    train_dataset = FrameSelectorDataset(
        train_feats, train_labels,
        sequence_length=trainer_config.sequence_length,
        stride=trainer_config.stride,
    )
    val_dataset = FrameSelectorDataset(
        val_feats, val_labels,
        sequence_length=trainer_config.sequence_length,
        stride=trainer_config.stride,
    )
    test_dataset = FrameSelectorDataset(
        test_feats, test_labels,
        sequence_length=trainer_config.sequence_length,
        stride=trainer_config.stride,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=trainer_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=trainer_config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=trainer_config.batch_size)
    
    LOGGER.info(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    # Step 6: Initialize model
    LOGGER.info("Step 6: Initializing model...")
    model_config = FrameSelectorConfig(input_dim=14)
    model = FrameSelectorBiLSTM(model_config)
    
    # Step 7: Train
    LOGGER.info("Step 7: Training model...")
    trainer = FrameSelectorTrainer(model, train_loader, val_loader, trainer_config)
    history = trainer.train()
    
    # Step 8: Evaluate
    LOGGER.info("Step 8: Evaluating on test set...")
    best_model = trainer.load_best_model()
    test_metrics = evaluate_model(best_model, test_loader, device=str(trainer.device))
    
    # Step 9: Save results
    elapsed_time = time.time() - start_time
    
    results = {
        'status': 'success',
        'model_path': str(trainer_config.save_dir / 'best_model.pt'),
        'normalization': {
            'mean_path': str(output_dir / 'feature_mean.npy'),
            'std_path': str(output_dir / 'feature_std.npy'),
        },
        'metrics': test_metrics,
        'training_history': history,
        'data_stats': {
            'total_frames': len(features),
            'key_frames': n_key_frames,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
        },
        'training_time_seconds': elapsed_time,
        'config': {
            'epochs': trainer_config.epochs,
            'batch_size': trainer_config.batch_size,
            'learning_rate': trainer_config.learning_rate,
            'pos_weight': trainer_config.pos_weight,
            'sequence_length': trainer_config.sequence_length,
        },
    }
    
    # Save results JSON
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    LOGGER.info(f"Training complete! Results saved to {results_path}")
    LOGGER.info(f"Test metrics: {test_metrics}")
    
    return results


def predict_key_frames(
    workflow_dir: Path,
    model_path: Path,
    threshold: float = 0.5,
    fps: float = 30.0,
) -> Dict[str, any]:
    """Predict key frames for a video using trained model.
    
    Args:
        workflow_dir: Directory containing frames and manifest
        model_path: Path to trained model checkpoint
        threshold: Classification threshold
        fps: Video frame rate
    
    Returns:
        Prediction results
    """
    # Step 1: Extract features
    LOGGER.info("Extracting features...")
    features, pose_sequence = extract_features_from_workflow(workflow_dir, fps)
    
    # Step 2: Load normalization stats
    output_dir = model_path.parent.parent
    feat_mean = np.load(output_dir / 'feature_mean.npy')
    feat_std = np.load(output_dir / 'feature_std.npy')
    
    # Normalize
    features_norm, _, _ = normalize_features(features, feat_mean, feat_std)
    
    # Step 3: Load model
    LOGGER.info(f"Loading model from {model_path}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FrameSelectorBiLSTM.load(model_path, device=device)
    model.eval()
    
    # Step 4: Predict
    LOGGER.info("Predicting key frames...")
    features_tensor = torch.from_numpy(features_norm).float().unsqueeze(0).to(device)  # (1, n_frames, 14)
    
    with torch.no_grad():
        probs = model.predict_probs(features_tensor)  # (1, n_frames, 1)
        probs = probs.squeeze().cpu().numpy()  # (n_frames,)
    
    # Step 5: Select frames above threshold
    selected_indices = np.where(probs >= threshold)[0].tolist()
    selected_filenames = [pose_sequence[i]['filename'] for i in selected_indices]
    
    LOGGER.info(f"Selected {len(selected_indices)} key frames (threshold={threshold})")
    
    # Step 6: Save selected frames to selected_frames/
    selected_frames_dir = workflow_dir / 'selected_frames'
    selected_frames_dir.mkdir(exist_ok=True)
    
    all_frames_dir = workflow_dir / 'all_frames'
    for filename in selected_filenames:
        src = all_frames_dir / filename
        dst = selected_frames_dir / filename
        if src.exists() and not dst.exists():
            import shutil
            shutil.copy2(src, dst)
    
    results = {
        'selected_frame_indices': selected_indices,
        'selected_filenames': selected_filenames,
        'frame_probabilities': probs.tolist(),
        'num_frames': len(features),
        'selected_count': len(selected_indices),
        'threshold': threshold,
    }
    
    # Save results
    results_path = workflow_dir / 'prediction_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    LOGGER.info(f"Prediction results saved to {results_path}")
    
    return results
