#!/usr/bin/env python3
"""Evaluate BiLSTM multitask model.

DEPRECATED: This script is kept for backward compatibility.
Use evaluate_model.py instead (auto-detects model type).

Usage:
    python scripts/evaluate_bilstm.py \
        --model models/checkpoints/bilstm_multitask.pt \
        --data data/features \
        --device cuda
"""

import sys
from pathlib import Path

print("=" * 70)
print("WARNING: evaluate_bilstm.py is deprecated!")
print("Please use: python scripts/evaluate_model.py")
print("=" * 70)
print()

# Forward to evaluate_model.py
script_dir = Path(__file__).parent
evaluate_model_script = script_dir / "evaluate_model.py"

import subprocess
sys.exit(subprocess.call([sys.executable, str(evaluate_model_script)] + sys.argv[1:]))
