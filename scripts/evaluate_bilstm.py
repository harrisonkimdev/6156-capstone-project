#!/usr/bin/env python3
"""Evaluate BiLSTM multitask model.

Usage:
    python scripts/evaluate_bilstm.py \
        --model models/checkpoints/bilstm_multitask.pt \
        --data data/features \
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
    import numpy as np
except ImportError:
    print("Error: PyTorch and NumPy are required. Install with: pip install torch numpy")
    sys.exit(1)

from pose_ai.ml.dataset import create_datasets_from_directory
from pose_ai.ml.models import BiLSTMMultitaskModel, MultitaskLoss


def evaluate_model(
    model: "torch.nn.Module",
    data_loader: DataLoader,
    criterion: MultitaskLoss,
    device: str,
) -> dict:
    """Evaluate model on dataset.
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_eff_loss = 0.0
    total_action_loss = 0.0
    num_batches = 0
    
    all_eff_pred = []
    all_eff_true = []
    all_action_pred = []
    all_action_true = []
    
    with torch.no_grad():
        for features, eff_labels, action_labels in data_loader:
            features = features.to(device)
            eff_labels = eff_labels.to(device)
            action_labels = action_labels.to(device)
            
            # Forward pass
            eff_pred, action_logits = model(features)
            
            # Compute loss
            loss, loss_dict = criterion(
                eff_pred, eff_labels,
                action_logits, action_labels
            )
            
            total_loss += loss_dict["total"]
            total_eff_loss += loss_dict["efficiency"]
            total_action_loss += loss_dict["action"]
            num_batches += 1
            
            # Track predictions
            all_eff_pred.extend(eff_pred.cpu().numpy().flatten())
            all_eff_true.extend(eff_labels.cpu().numpy())
            
            action_pred_classes = action_logits.argmax(dim=-1)
            all_action_pred.extend(action_pred_classes.cpu().numpy())
            all_action_true.extend(action_labels.cpu().numpy())
            
            # Top-3 predictions
            action_top3 = action_logits.topk(3, dim=-1)[1]
            all_action_top3 = action_top3.cpu().numpy()
    
    # Convert to arrays
    all_eff_pred = np.array(all_eff_pred)
    all_eff_true = np.array(all_eff_true)
    all_action_pred = np.array(all_action_pred)
    all_action_true = np.array(all_action_true)
    
    # Efficiency metrics
    eff_mae = np.abs(all_eff_pred - all_eff_true).mean()
    eff_rmse = np.sqrt(((all_eff_pred - all_eff_true) ** 2).mean())
    eff_corr = np.corrcoef(all_eff_pred, all_eff_true)[0, 1] if len(all_eff_pred) > 1 else 0.0
    
    # R² score
    ss_res = ((all_eff_true - all_eff_pred) ** 2).sum()
    ss_tot = ((all_eff_true - all_eff_true.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Action metrics
    action_acc = (all_action_pred == all_action_true).mean()
    
    # Per-class accuracy
    per_class_acc = {}
    for class_id in range(5):
        mask = all_action_true == class_id
        if mask.sum() > 0:
            per_class_acc[class_id] = (all_action_pred[mask] == all_action_true[mask]).mean()
    
    # Confusion matrix
    confusion = np.zeros((5, 5), dtype=int)
    for true_label, pred_label in zip(all_action_true, all_action_pred):
        confusion[int(true_label), int(pred_label)] += 1
    
    return {
        "loss": {
            "total": total_loss / num_batches,
            "efficiency": total_eff_loss / num_batches,
            "action": total_action_loss / num_batches,
        },
        "efficiency": {
            "mae": float(eff_mae),
            "rmse": float(eff_rmse),
            "correlation": float(eff_corr),
            "r2": float(r2),
        },
        "action": {
            "accuracy": float(action_acc),
            "per_class": {k: float(v) for k, v in per_class_acc.items()},
            "confusion_matrix": confusion.tolist(),
        },
    }


def print_results(results: dict, split_name: str = "Test") -> None:
    """Print evaluation results."""
    print(f"\n{split_name} Results:")
    print("=" * 60)
    
    print("\nLoss:")
    print(f"  Total: {results['loss']['total']:.4f}")
    print(f"  Efficiency: {results['loss']['efficiency']:.4f}")
    print(f"  Action: {results['loss']['action']:.4f}")
    
    print("\nEfficiency Regression:")
    print(f"  MAE: {results['efficiency']['mae']:.4f}")
    print(f"  RMSE: {results['efficiency']['rmse']:.4f}")
    print(f"  Correlation: {results['efficiency']['correlation']:.4f}")
    print(f"  R²: {results['efficiency']['r2']:.4f}")
    
    print("\nNext-Action Classification:")
    print(f"  Accuracy: {results['action']['accuracy']:.4f}")
    
    print("\n  Per-Class Accuracy:")
    class_names = ["no_change", "left_hand", "right_hand", "left_foot", "right_foot"]
    for class_id, acc in results['action']['per_class'].items():
        print(f"    {class_names[class_id]}: {acc:.4f}")
    
    print("\n  Confusion Matrix:")
    confusion = np.array(results['action']['confusion_matrix'])
    print("    Pred →")
    print("    True ↓ ", end="")
    for i in range(5):
        print(f"{i:>6}", end="")
    print()
    for i in range(5):
        print(f"    {i:>6}", end="")
        for j in range(5):
            print(f"{confusion[i, j]:>6}", end="")
        print(f"  ({class_names[i]})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BiLSTM multitask model")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory containing feature JSON files"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda/cpu, default: cuda)"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which split to evaluate (default: test)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=32,
        help="Window size (default: 32)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save results JSON"
    )
    
    args = parser.parse_args()
    
    # Check model file
    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Check data directory
    if not args.data.exists():
        print(f"Error: Data directory not found: {args.data}")
        sys.exit(1)
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    
    print("=" * 60)
    print("BiLSTM Model Evaluation")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from: {args.model}")
    model = BiLSTMMultitaskModel.load(args.model, device=args.device)
    print(f"Model loaded on: {args.device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets_from_directory(
        args.data,
        window_size=args.window_size,
        stride=1,
        normalize=True,
    )
    
    # Create criterion
    criterion = MultitaskLoss(
        efficiency_weight=1.0,
        action_weight=0.5,
        efficiency_loss="huber",
    )
    
    # Evaluate
    if args.split == "all":
        splits = [
            ("Train", train_dataset),
            ("Val", val_dataset),
            ("Test", test_dataset),
        ]
    elif args.split == "train":
        splits = [("Train", train_dataset)]
    elif args.split == "val":
        splits = [("Val", val_dataset)]
    else:
        splits = [("Test", test_dataset)]
    
    all_results = {}
    
    for split_name, dataset in splits:
        if dataset is None:
            print(f"\n{split_name} dataset not available")
            continue
        
        print(f"\nEvaluating on {split_name} set ({len(dataset)} samples)...")
        
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        results = evaluate_model(model, data_loader, criterion, device)
        all_results[split_name.lower()] = results
        
        print_results(results, split_name)
    
    # Save results if requested
    if args.output:
        import json
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

