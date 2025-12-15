"""Training pipeline for frame selector model.

Dataset, DataLoader, Trainer, and evaluation utilities.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

LOGGER = logging.getLogger(__name__)


class FrameSelectorDataset(Dataset):
    """Dataset for frame selector training.
    
    Each sample is a sequence of frames with binary labels (key frame or not).
    """
    
    def __init__(
        self,
        features: np.ndarray,               # (n_frames, 14)
        labels: np.ndarray,                 # (n_frames,) binary labels
        sequence_length: int = 30,          # Sliding window size
        stride: int = 1,                    # Sliding window stride
    ):
        """
        Args:
            features: Frame features array
            labels: Binary labels (1=key frame, 0=non-key frame)
            sequence_length: Number of frames in each sequence
            stride: Step size for sliding window
        """
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Calculate number of samples
        self.n_frames = len(features)
        self.n_samples = max(0, (self.n_frames - sequence_length) // stride + 1)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence sample.
        
        Returns:
            (features, labels)
            - features: (seq_len, 14) tensor
            - labels: (seq_len,) tensor
        """
        start = idx * self.stride
        end = start + self.sequence_length
        
        feature_seq = torch.from_numpy(self.features[start:end]).float()
        label_seq = torch.from_numpy(self.labels[start:end]).float()
        
        return feature_seq, label_seq


def temporal_train_val_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data temporally (preserving time order).
    
    Args:
        features: (n_frames, 14) array
        labels: (n_frames,) array
        train_ratio: Fraction for training (0.8 = 80%)
        val_ratio: Fraction for validation (0.1 = 10%)
    
    Returns:
        (train_features, train_labels, val_features, val_labels, test_features, test_labels)
    """
    n_frames = len(features)
    train_end = int(n_frames * train_ratio)
    val_end = int(n_frames * (train_ratio + val_ratio))
    
    train_features = features[:train_end]
    train_labels = labels[:train_end]
    
    val_features = features[train_end:val_end]
    val_labels = labels[train_end:val_end]
    
    test_features = features[val_end:]
    test_labels = labels[val_end:]
    
    LOGGER.info(f"Temporal split: train={len(train_features)}, val={len(val_features)}, test={len(test_features)}")
    
    return train_features, train_labels, val_features, val_labels, test_features, test_labels


@dataclass
class TrainerConfig:
    """Training configuration."""
    
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 10           # Early stopping patience
    pos_weight: float = 10.0     # Loss weight for positive class
    device: str = 'cuda'
    save_dir: Path = Path('models/frame_selector')
    sequence_length: int = 30    # Frames per sequence
    stride: int = 1              # Sliding window stride


class FrameSelectorTrainer:
    """Trainer for frame selector model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainerConfig] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainerConfig()
        
        # Device
        if self.config.device == 'cuda' and not torch.cuda.is_available():
            self.config.device = 'cpu'
            LOGGER.warning("CUDA not available, using CPU")
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Loss function (Binary Cross Entropy with pos_weight for class imbalance)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.config.pos_weight]).to(self.device)
        )
        
        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
        }
        
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)  # (batch, seq_len, 14)
            labels = labels.to(self.device)      # (batch, seq_len)
            
            # Forward
            logits = self.model(features)  # (batch, seq_len, 1)
            logits = logits.squeeze(-1)    # (batch, seq_len)
            
            # Loss
            loss = self.criterion(logits, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> Tuple[float, float]:
        """Validation.
        
        Returns:
            (val_loss, val_f1)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(features)
                logits = logits.squeeze(-1)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # For F1 calculation
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Compute F1 score
        preds = np.concatenate(all_preds).flatten()
        lbls = np.concatenate(all_labels).flatten()
        
        try:
            from sklearn.metrics import f1_score
            f1 = f1_score(lbls, preds, zero_division=0)
        except ImportError:
            LOGGER.warning("sklearn not available, F1 score set to 0")
            f1 = 0.0
        
        return avg_loss, f1
    
    def train(self) -> Dict[str, List[float]]:
        """Full training loop.
        
        Returns:
            Training history
        """
        LOGGER.info(f"Starting training for {self.config.epochs} epochs")
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch()
            val_loss, val_f1 = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            
            LOGGER.info(
                f"Epoch {epoch+1}/{self.config.epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_f1={val_f1:.4f}"
            )
            
            # LR scheduler step
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    LOGGER.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        elapsed = time.time() - start_time
        LOGGER.info(f"Training completed in {elapsed:.2f}s, best epoch: {self.best_epoch+1}")
        
        return self.history
    
    def _save_checkpoint(self, epoch: int) -> None:
        """Save best model."""
        path = self.config.save_dir / 'best_model.pt'
        self.model.save(path)
        LOGGER.info(f"Saved best model at epoch {epoch+1}")
    
    def load_best_model(self) -> nn.Module:
        """Load the best saved model."""
        from .frame_selector_model import FrameSelectorBiLSTM
        path = self.config.save_dir / 'best_model.pt'
        return FrameSelectorBiLSTM.load(path, device=str(self.device))


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu',
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate model on test set.
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs >= threshold).float()
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    probs = np.concatenate(all_probs).flatten()
    preds = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    
    try:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )
        
        metrics = {
            'accuracy': float(accuracy_score(labels, preds)),
            'precision': float(precision_score(labels, preds, zero_division=0)),
            'recall': float(recall_score(labels, preds, zero_division=0)),
            'f1': float(f1_score(labels, preds, zero_division=0)),
            'auc': float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        }
    except ImportError:
        LOGGER.warning("sklearn not available, returning basic metrics")
        metrics = {
            'accuracy': float(np.mean(preds == labels)),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.0,
        }
    
    LOGGER.info(f"Test metrics: {metrics}")
    return metrics
