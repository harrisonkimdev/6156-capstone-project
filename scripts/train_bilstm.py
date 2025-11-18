#!/usr/bin/env python3
"""Train BiLSTM multitask model.

Usage:
    python scripts/train_bilstm.py \
        --data data/features \
        --checkpoint-dir models/checkpoints \
        --epochs 100 \
        --batch-size 32 \
        --device cuda
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    print("Error: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

from pose_ai.ml.dataset import create_datasets_from_directory
from pose_ai.ml.models import BiLSTMMultitaskModel, ModelConfig, count_parameters
from pose_ai.ml.train import Trainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM multitask model")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory containing feature JSON files"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models/checkpoints"),
        help="Directory for model checkpoints (default: models/checkpoints)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=32,
        help="Window size for sliding windows (default: 32)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for sliding windows (default: 1)"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="LSTM hidden dimension (default: 128)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of LSTM layers (default: 2)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate (default: 0.3)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="Weight decay (default: 0.0001)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda/cpu, default: cuda)"
    )
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Disable attention pooling"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Training split ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)"
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    if not args.data.exists():
        print(f"Error: Data directory not found: {args.data}")
        sys.exit(1)
    
    print("=" * 60)
    print("BiLSTM Multitask Model Training")
    print("=" * 60)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets_from_directory(
        args.data,
        window_size=args.window_size,
        stride=args.stride,
        train_split=args.train_split,
        val_split=args.val_split,
        normalize=True,
    )
    
    if train_dataset is None or val_dataset is None:
        print("Error: Failed to create datasets")
        sys.exit(1)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    if test_dataset:
        print(f"  Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if args.device == "cuda" else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if args.device == "cuda" else False,
    )
    
    # Create model
    print("\nCreating model...")
    model_config = ModelConfig(
        input_dim=60,  # Feature dimension from dataset
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_action_classes=5,
        bidirectional=True,
        use_attention=not args.no_attention,
    )
    
    model = BiLSTMMultitaskModel(model_config)
    num_params = count_parameters(model)
    
    print(f"  Model: BiLSTM (bidirectional={model_config.bidirectional})")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Layers: {args.num_layers}")
    print(f"  Attention: {model_config.use_attention}")
    print(f"  Total parameters: {num_params:,}")
    
    # Create trainer
    print("\nSetting up trainer...")
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=10,
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
    )
    
    # Train
    print("\n" + "=" * 60)
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    final_model_path = args.checkpoint_dir / "bilstm_multitask.pt"
    model.save(final_model_path)
    print(f"Model saved to: {final_model_path}")
    
    # Save normalization parameters
    if train_dataset.feature_mean is not None:
        import numpy as np
        norm_path = args.checkpoint_dir / "normalization.npz"
        np.savez(
            norm_path,
            mean=train_dataset.feature_mean,
            std=train_dataset.feature_std,
        )
        print(f"Normalization parameters saved to: {norm_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

