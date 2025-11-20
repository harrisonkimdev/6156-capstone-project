"""Training pipeline for BiLSTM multitask model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import torch
    from torch.utils.data import DataLoader
    import numpy as np
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False
    torch = None  # type: ignore
    DataLoader = None  # type: ignore
    np = None  # type: ignore


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 10  # Early stopping patience
    lr_patience: int = 5  # LR scheduler patience
    lr_factor: float = 0.5  # LR reduction factor
    device: str = "cuda"  # "cuda" or "cpu"
    checkpoint_dir: Path = Path("models/checkpoints")
    log_interval: int = 10  # Log every N batches


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """Check if should stop training.
        
        Args:
            val_loss: Current validation loss
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop


class Trainer:
    """Training manager for BiLSTM multitask model."""
    
    def __init__(
        self,
        model: "torch.nn.Module",
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: BiLSTM multitask model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        
        # Setup device
        if self.config.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.config.device = "cpu"
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Setup loss function
        from .models import MultitaskLoss
        self.criterion = MultitaskLoss(
            efficiency_weight=1.0,
            action_weight=0.5,
            efficiency_loss="huber",
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.lr_factor,
            patience=self.config.lr_patience,
            verbose=True,
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=self.config.patience)
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        
        # Checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        total_loss = 0.0
        total_eff_loss = 0.0
        total_action_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, eff_labels, action_labels) in enumerate(self.train_loader):
            # Move to device
            features = features.to(self.device)
            eff_labels = eff_labels.to(self.device)
            action_labels = action_labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            eff_pred, action_logits = self.model(features)
            
            # Compute loss
            loss, loss_dict = self.criterion(
                eff_pred, eff_labels,
                action_logits, action_labels
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track losses
            total_loss += loss_dict["total"]
            total_eff_loss += loss_dict["efficiency"]
            total_action_loss += loss_dict["action"]
            num_batches += 1
            
            # Log progress
            if (batch_idx + 1) % self.config.log_interval == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(self.train_loader)}: "
                    f"Loss={loss_dict['total']:.4f} "
                    f"(Eff={loss_dict['efficiency']:.4f}, Act={loss_dict['action']:.4f})"
                )
        
        # Average losses
        avg_losses = {
            "total": total_loss / num_batches,
            "efficiency": total_eff_loss / num_batches,
            "action": total_action_loss / num_batches,
        }
        
        return avg_losses
    
    def validate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate model.
        
        Returns:
            (loss_dict, metrics_dict)
        """
        self.model.eval()
        
        total_loss = 0.0
        total_eff_loss = 0.0
        total_action_loss = 0.0
        num_batches = 0
        
        # For metrics
        all_eff_pred = []
        all_eff_true = []
        all_action_pred = []
        all_action_true = []
        
        with torch.no_grad():
            for features, eff_labels, action_labels in self.val_loader:
                # Move to device
                features = features.to(self.device)
                eff_labels = eff_labels.to(self.device)
                action_labels = action_labels.to(self.device)
                
                # Forward pass
                eff_pred, action_logits = self.model(features)
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    eff_pred, eff_labels,
                    action_logits, action_labels
                )
                
                # Track losses
                total_loss += loss_dict["total"]
                total_eff_loss += loss_dict["efficiency"]
                total_action_loss += loss_dict["action"]
                num_batches += 1
                
                # Track predictions for metrics
                all_eff_pred.extend(eff_pred.cpu().numpy().flatten())
                all_eff_true.extend(eff_labels.cpu().numpy())
                
                action_pred_classes = action_logits.argmax(dim=-1)
                all_action_pred.extend(action_pred_classes.cpu().numpy())
                all_action_true.extend(action_labels.cpu().numpy())
        
        # Average losses
        avg_losses = {
            "total": total_loss / num_batches,
            "efficiency": total_eff_loss / num_batches,
            "action": total_action_loss / num_batches,
        }
        
        # Compute metrics
        all_eff_pred = np.array(all_eff_pred)
        all_eff_true = np.array(all_eff_true)
        all_action_pred = np.array(all_action_pred)
        all_action_true = np.array(all_action_true)
        
        # Efficiency metrics
        eff_mae = np.abs(all_eff_pred - all_eff_true).mean()
        eff_rmse = np.sqrt(((all_eff_pred - all_eff_true) ** 2).mean())
        eff_corr = np.corrcoef(all_eff_pred, all_eff_true)[0, 1] if len(all_eff_pred) > 1 else 0.0
        
        # Action metrics
        action_acc = (all_action_pred == all_action_true).mean()
        
        # Top-3 accuracy (if applicable)
        # For simplicity, we'll just use top-1 here
        
        metrics = {
            "eff_mae": float(eff_mae),
            "eff_rmse": float(eff_rmse),
            "eff_corr": float(eff_corr),
            "action_acc": float(action_acc),
        }
        
        return avg_losses, metrics
    
    def train(self) -> None:
        """Run full training loop."""
        print(f"Starting training for {self.config.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            # Train
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            print(f"  Train Loss: {train_losses['total']:.4f} "
                  f"(Eff={train_losses['efficiency']:.4f}, Act={train_losses['action']:.4f})")
            
            # Validate
            val_losses, val_metrics = self.validate()
            self.val_losses.append(val_losses)
            
            print(f"  Val Loss: {val_losses['total']:.4f} "
                  f"(Eff={val_losses['efficiency']:.4f}, Act={val_losses['action']:.4f})")
            print(f"  Val Metrics: MAE={val_metrics['eff_mae']:.4f}, "
                  f"Corr={val_metrics['eff_corr']:.4f}, Acc={val_metrics['action_acc']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_losses['total'])
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.best_epoch = epoch + 1
                self.save_checkpoint("best_model.pt", val_metrics)
                print(f"  ✓ New best model saved (loss={self.best_val_loss:.4f})")
            
            # Early stopping check
            if self.early_stopping(val_losses['total']):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
        
        print(f"\nTraining complete!")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename: str, metrics: Optional[Dict[str, float]] = None) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / filename
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }
        
        if metrics:
            checkpoint["metrics"] = metrics
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_epoch = checkpoint.get("best_epoch", 0)


__all__ = [
    "Trainer",
    "TrainingConfig",
    "EarlyStopping",
]

