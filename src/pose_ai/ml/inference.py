"""Inference module for BiLSTM multitask model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False
    torch = None  # type: ignore
    np = None  # type: ignore


@dataclass(slots=True)
class PredictionResult:
    """Result from model inference."""
    frame_index: int
    efficiency_score: float
    next_action_class: int
    next_action_probs: List[float]
    next_action_name: str


class BiLSTMInference:
    """Inference wrapper for BiLSTM multitask model."""
    
    ACTION_NAMES = ["no_change", "left_hand", "right_hand", "left_foot", "right_foot"]
    
    def __init__(
        self,
        model_path: Path,
        normalization_path: Optional[Path] = None,
        device: str = "cpu",
        window_size: int = 32,
    ):
        """Initialize inference engine.
        
        Args:
            model_path: Path to trained model (.pt file)
            normalization_path: Path to normalization parameters (.npz file)
            device: Device to use ("cuda" or "cpu")
            window_size: Window size for sliding windows
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for inference. Install with: pip install torch")
        
        self.window_size = window_size
        self.device = torch.device(device)
        
        # Load model
        from .models import BiLSTMMultitaskModel
        self.model = BiLSTMMultitaskModel.load(model_path, device=device)
        self.model.eval()
        
        # Load normalization parameters
        if normalization_path and normalization_path.exists():
            norm_data = np.load(normalization_path)
            self.feature_mean = norm_data["mean"]
            self.feature_std = norm_data["std"]
        else:
            self.feature_mean = None
            self.feature_std = None
    
    def preprocess_features(self, feature_matrix: "np.ndarray") -> "np.ndarray":
        """Normalize features using training statistics."""
        if self.feature_mean is not None and self.feature_std is not None:
            return (feature_matrix - self.feature_mean) / self.feature_std
        return feature_matrix
    
    def predict_window(
        self,
        features: "np.ndarray",
    ) -> Tuple[float, int, "np.ndarray"]:
        """Predict on a single window.
        
        Args:
            features: Feature array of shape (T, feature_dim)
        
        Returns:
            (efficiency_score, action_class, action_probs)
        """
        # Normalize
        features = self.preprocess_features(features)
        
        # Convert to tensor and add batch dimension
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)  # (1, T, D)
        features_tensor = features_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            eff_pred, action_logits = self.model(features_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
        
        # Extract results
        efficiency_score = float(eff_pred.item())
        action_class = int(action_logits.argmax(dim=-1).item())
        action_probs_np = action_probs.cpu().numpy().flatten()
        
        return efficiency_score, action_class, action_probs_np
    
    def predict_sequence(
        self,
        feature_rows: List[Dict[str, object]],
        stride: int = 1,
    ) -> List[PredictionResult]:
        """Predict on a sequence of feature rows using sliding windows.
        
        Args:
            feature_rows: List of feature dictionaries
            stride: Stride for sliding windows
        
        Returns:
            List of prediction results
        """
        from .dataset import extract_features_from_row
        
        # Extract feature matrix
        feature_matrix = []
        for row in feature_rows:
            features = extract_features_from_row(row)
            feature_matrix.append(features)
        
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        
        # Sliding window inference
        results = []
        for i in range(0, len(feature_matrix) - self.window_size + 1, stride):
            window = feature_matrix[i:i + self.window_size]
            eff_score, action_class, action_probs = self.predict_window(window)
            
            # Center frame of window
            center_idx = i + self.window_size // 2
            
            results.append(PredictionResult(
                frame_index=center_idx,
                efficiency_score=eff_score,
                next_action_class=action_class,
                next_action_probs=action_probs.tolist(),
                next_action_name=self.ACTION_NAMES[action_class],
            ))
        
        return results
    
    def predict_from_json(
        self,
        json_path: Path,
        stride: int = 1,
    ) -> List[PredictionResult]:
        """Predict from a feature JSON file.
        
        Args:
            json_path: Path to feature JSON file
            stride: Stride for sliding windows
        
        Returns:
            List of prediction results
        """
        from .dataset import load_features_from_json
        
        feature_rows = load_features_from_json(json_path)
        return self.predict_sequence(feature_rows, stride=stride)
    
    def predict_efficiency_only(
        self,
        features: "np.ndarray",
    ) -> float:
        """Predict efficiency score only (no action classification)."""
        features = self.preprocess_features(features)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)
        features_tensor = features_tensor.to(self.device)
        
        with torch.no_grad():
            eff_pred = self.model.predict_efficiency(features_tensor)
        
        return float(eff_pred.item())
    
    def predict_action_only(
        self,
        features: "np.ndarray",
    ) -> Tuple[int, "np.ndarray"]:
        """Predict next-action only (no efficiency regression)."""
        features = self.preprocess_features(features)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)
        features_tensor = features_tensor.to(self.device)
        
        with torch.no_grad():
            action_probs = self.model.predict_action(features_tensor)
        
        action_class = int(action_probs.argmax(dim=-1).item())
        action_probs_np = action_probs.cpu().numpy().flatten()
        
        return action_class, action_probs_np


def batch_inference(
    inference_engine: BiLSTMInference,
    feature_paths: Sequence[Path],
    output_dir: Path,
    stride: int = 1,
) -> None:
    """Run batch inference on multiple feature files.
    
    Args:
        inference_engine: Inference engine
        feature_paths: List of feature JSON file paths
        output_dir: Directory to save prediction results
        stride: Stride for sliding windows
    """
    import json
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for feature_path in feature_paths:
        print(f"Processing: {feature_path}")
        
        try:
            results = inference_engine.predict_from_json(feature_path, stride=stride)
            
            # Convert to JSON-serializable format
            output_data = {
                "video": feature_path.stem,
                "predictions": [
                    {
                        "frame_index": r.frame_index,
                        "efficiency_score": r.efficiency_score,
                        "next_action": r.next_action_name,
                        "next_action_class": r.next_action_class,
                        "next_action_probs": r.next_action_probs,
                    }
                    for r in results
                ],
            }
            
            # Save predictions
            output_path = output_dir / f"{feature_path.stem}_predictions.json"
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"  Saved {len(results)} predictions to: {output_path}")
        
        except Exception as e:
            print(f"  Error: {e}")


__all__ = [
    "BiLSTMInference",
    "PredictionResult",
    "batch_inference",
]

