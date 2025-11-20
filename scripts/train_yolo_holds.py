#!/usr/bin/env python3
"""Train YOLOv8 model for hold type classification.

Usage:
    python scripts/train_yolo_holds.py --data data/holds_training/dataset.yaml --model yolov8n.pt
    
    # With custom hyperparameters
    python scripts/train_yolo_holds.py \
        --data data/holds_training/dataset.yaml \
        --model yolov8m.pt \
        --epochs 100 \
        --batch 16 \
        --imgsz 640 \
        --device cuda
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics is required. Install with: pip install ultralytics")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for hold type classification")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to dataset.yaml file"
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base model to fine-tune (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)"
    )
    parser.add_argument(
        "--device",
        default="",
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    parser.add_argument(
        "--project",
        default="runs/hold_type",
        help="Project folder (default: runs/hold_type)"
    )
    parser.add_argument(
        "--name",
        default="train",
        help="Experiment name (default: train)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (default: 50)"
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Save checkpoint every n epochs (default: 10)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers (default: 8)"
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate fraction (default: 0.01)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.937,
        help="SGD momentum (default: 0.937)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="Weight decay (default: 0.0005)"
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=3.0,
        help="Warmup epochs (default: 3.0)"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    
    args = parser.parse_args()
    
    # Validate data file
    if not args.data.exists():
        print(f"Error: Dataset file not found: {args.data}")
        sys.exit(1)
    
    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)
    
    print(f"Starting training...")
    print(f"  Dataset: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {args.device or 'auto'}")
    
    # Train model
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save_period=args.save_period,
        workers=args.workers,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        augment=args.augment,
        resume=args.resume,
        verbose=True,
    )
    
    print("\nTraining complete!")
    print(f"Best model saved to: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    
    # Validate model
    print("\nRunning validation...")
    val_results = model.val()
    
    print("\nValidation Results:")
    print(f"  mAP@0.5: {val_results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {val_results.box.map:.4f}")
    
    # Per-class metrics
    if hasattr(val_results.box, 'maps'):
        print("\nPer-Class mAP@0.5:")
        class_names = model.names
        for i, map_val in enumerate(val_results.box.maps):
            class_name = class_names.get(i, f"class_{i}")
            print(f"  {class_name}: {map_val:.4f}")
    
    # Export to ONNX (optional)
    try:
        print("\nExporting to ONNX format...")
        onnx_path = model.export(format="onnx", imgsz=args.imgsz)
        print(f"ONNX model saved to: {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

