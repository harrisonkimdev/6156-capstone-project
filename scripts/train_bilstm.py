#!/usr/bin/env python3
"""Train BiLSTM multitask model.

DEPRECATED: This script is kept for backward compatibility.
Use train_model.py with --model-type bilstm instead.

Usage:
    python scripts/train_bilstm.py \
        --data data/features \
        --checkpoint-dir models/checkpoints \
        --epochs 100 \
        --batch-size 32 \
        --device cuda
"""

import sys
from pathlib import Path

print("=" * 70)
print("WARNING: train_bilstm.py is deprecated!")
print("Please use: python scripts/train_model.py --model-type bilstm")
print("=" * 70)
print()

# Forward to train_model.py with bilstm model type
script_dir = Path(__file__).parent
train_model_script = script_dir / "train_model.py"

# Construct new command
new_args = ["--model-type", "bilstm"] + sys.argv[1:]
new_cmd = [sys.executable, str(train_model_script)] + new_args

import subprocess
sys.exit(subprocess.call(new_cmd))

