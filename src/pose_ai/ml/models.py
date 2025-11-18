"""BiLSTM multitask model for efficiency regression and next-action classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False
    nn = None  # type: ignore
    torch = None  # type: ignore
    F = None  # type: ignore


@dataclass(slots=True)
class ModelConfig:
    """Configuration for BiLSTM multitask model."""
    input_dim: int = 60  # Feature dimension
    hidden_dim: int = 128  # LSTM hidden dimension
    num_layers: int = 2  # Number of LSTM layers
    dropout: float = 0.3  # Dropout rate
    num_action_classes: int = 5  # Number of next-action classes (0-4)
    bidirectional: bool = True  # Use bidirectional LSTM
    use_attention: bool = True  # Use attention pooling


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence features."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        if not HAS_TORCH:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_out: "torch.Tensor") -> "torch.Tensor":
        """Apply attention pooling.
        
        Args:
            lstm_out: (batch, seq_len, hidden_dim)
        
        Returns:
            pooled: (batch, hidden_dim)
        """
        # Compute attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum
        pooled = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_dim)
        return pooled


class BiLSTMMultitaskModel(nn.Module):
    """BiLSTM model with multitask heads for efficiency regression and next-action classification."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if not HAS_TORCH:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        self.config = config or ModelConfig()
        
        # BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.config.input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0,
            bidirectional=self.config.bidirectional,
            batch_first=True,
        )
        
        # Calculate output dimension after LSTM
        lstm_output_dim = self.config.hidden_dim * (2 if self.config.bidirectional else 1)
        
        # Attention pooling (optional)
        if self.config.use_attention:
            self.attention_pool = AttentionPooling(lstm_output_dim)
        else:
            self.attention_pool = None
        
        # Dropout
        self.dropout = nn.Dropout(self.config.dropout)
        
        # Task 1: Efficiency regression head
        self.efficiency_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(64, 1),
        )
        
        # Task 2: Next-action classification head
        self.action_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(64, self.config.num_action_classes),
        )
    
    def forward(
        self,
        x: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        
        Returns:
            (efficiency_pred, action_logits)
            - efficiency_pred: (batch, 1) - efficiency scores
            - action_logits: (batch, num_action_classes) - logits for next-action
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_output_dim)
        
        # Pooling
        if self.attention_pool is not None:
            pooled = self.attention_pool(lstm_out)  # (batch, lstm_output_dim)
        else:
            # Mean pooling
            pooled = lstm_out.mean(dim=1)  # (batch, lstm_output_dim)
        
        pooled = self.dropout(pooled)
        
        # Task 1: Efficiency regression
        efficiency_pred = self.efficiency_head(pooled)  # (batch, 1)
        
        # Task 2: Next-action classification
        action_logits = self.action_head(pooled)  # (batch, num_action_classes)
        
        return efficiency_pred, action_logits
    
    def predict_efficiency(self, x: "torch.Tensor") -> "torch.Tensor":
        """Predict efficiency scores only."""
        efficiency_pred, _ = self.forward(x)
        return efficiency_pred
    
    def predict_action(self, x: "torch.Tensor") -> "torch.Tensor":
        """Predict next-action class probabilities."""
        _, action_logits = self.forward(x)
        return F.softmax(action_logits, dim=-1)
    
    def save(self, path: Path) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "config": {
                "input_dim": self.config.input_dim,
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "dropout": self.config.dropout,
                "num_action_classes": self.config.num_action_classes,
                "bidirectional": self.config.bidirectional,
                "use_attention": self.config.use_attention,
            },
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "BiLSTMMultitaskModel":
        """Load model from checkpoint."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        checkpoint = torch.load(path, map_location=device)
        config_dict = checkpoint["config"]
        config = ModelConfig(**config_dict)
        
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        
        return model


class MultitaskLoss:
    """Combined loss for multitask learning."""
    
    def __init__(
        self,
        efficiency_weight: float = 1.0,
        action_weight: float = 0.5,
        efficiency_loss: str = "huber",
    ):
        """Initialize loss.
        
        Args:
            efficiency_weight: Weight for efficiency regression loss
            action_weight: Weight for action classification loss
            efficiency_loss: Type of regression loss ("huber" or "mse")
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        self.efficiency_weight = efficiency_weight
        self.action_weight = action_weight
        
        if efficiency_loss == "huber":
            self.efficiency_criterion = nn.HuberLoss()
        elif efficiency_loss == "mse":
            self.efficiency_criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown efficiency loss: {efficiency_loss}")
        
        self.action_criterion = nn.CrossEntropyLoss()
    
    def __call__(
        self,
        efficiency_pred: "torch.Tensor",
        efficiency_target: "torch.Tensor",
        action_logits: "torch.Tensor",
        action_target: "torch.Tensor",
    ) -> Tuple["torch.Tensor", Dict[str, float]]:
        """Compute combined loss.
        
        Args:
            efficiency_pred: (batch, 1)
            efficiency_target: (batch,)
            action_logits: (batch, num_classes)
            action_target: (batch,)
        
        Returns:
            (total_loss, loss_dict)
        """
        # Efficiency loss
        efficiency_loss = self.efficiency_criterion(
            efficiency_pred.squeeze(-1),
            efficiency_target
        )
        
        # Action loss
        action_loss = self.action_criterion(action_logits, action_target)
        
        # Combined loss
        total_loss = (
            self.efficiency_weight * efficiency_loss +
            self.action_weight * action_loss
        )
        
        loss_dict = {
            "total": total_loss.item(),
            "efficiency": efficiency_loss.item(),
            "action": action_loss.item(),
        }
        
        return total_loss, loss_dict


def count_parameters(model: "nn.Module") -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "BiLSTMMultitaskModel",
    "ModelConfig",
    "MultitaskLoss",
    "AttentionPooling",
    "count_parameters",
]

