"""BiLSTM model for frame selection.

Binary classification: key frame (1) vs non-key frame (0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


@dataclass
class FrameSelectorConfig:
    """BiLSTM frame selector model configuration."""
    
    input_dim: int = 14              # Feature dimension
    hidden_dim: int = 128            # LSTM hidden state size
    num_layers: int = 2              # Number of stacked LSTM layers
    dropout: float = 0.3             # Dropout rate
    bidirectional: bool = True       # BiLSTM
    output_dim: int = 1              # Binary classification


class FrameSelectorBiLSTM(nn.Module):
    """BiLSTM for key frame detection.
    
    Architecture:
    - Input: (batch, seq_len, 14) - pose-based features
    - BiLSTM: bidirectional, 2 layers
    - Per-frame classification head: hidden → [0, 1]
    """
    
    def __init__(self, config: Optional[FrameSelectorConfig] = None):
        super().__init__()
        self.config = config or FrameSelectorConfig()
        
        # BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.config.input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
            bidirectional=self.config.bidirectional,
            batch_first=True,
        )
        
        # LSTM output dimension
        lstm_out_dim = self.config.hidden_dim * (2 if self.config.bidirectional else 1)
        
        # Dropout
        self.dropout = nn.Dropout(self.config.dropout)
        
        # Per-frame classification head
        self.frame_classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(64, 1),  # Output: logit (real number)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim=14)
        
        Returns:
            logits: (batch, seq_len, 1) - per-frame logits
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq_len, lstm_out_dim)
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # Per-frame classification
        logits = self.frame_classifier(lstm_out)  # (batch, seq_len, 1)
        
        return logits
    
    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions (0-1).
        
        Args:
            x: Input tensor (batch, seq_len, 14)
        
        Returns:
            probs: (batch, seq_len, 1) - probabilities
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs
    
    def predict_binary(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """Get binary predictions.
        
        Args:
            x: Input tensor (batch, seq_len, 14)
            threshold: Classification threshold
        
        Returns:
            predictions: (batch, seq_len, 1) - 0 or 1
        """
        probs = self.predict_probs(x)
        predictions = (probs >= threshold).float()
        return predictions
    
    def save(self, path: Path) -> None:
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, path)
        LOGGER.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path, device: str = 'cpu') -> FrameSelectorBiLSTM:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            device: Device to load model on
        
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(config=checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        LOGGER.info(f"Model loaded from {path}")
        return model
