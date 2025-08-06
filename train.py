#!/usr/bin/env python3
"""
DEPRECATED: This script uses SB3 which is not GPU-friendly.
Please use train_evorl_only.py instead for pure GPU training.

To use the new EvoRL GPU-only implementation:
  python train_evorl_only.py --symbols BPCL --test-days 30
"""

import sys

print("=" * 80)
print("⚠️  DEPRECATION WARNING")
print("=" * 80)
print("This script (train.py) uses SB3 which is not GPU-friendly.")
print("Please use the new EvoRL GPU-only implementation instead:")
print()
print("  python train_evorl_only.py --symbols BPCL --test-days 30")
print()
print("The new implementation provides:")
print("  • Pure JAX/GPU training (5-10x faster)")
print("  • Test period performance evaluation")
print("  • Deployment capabilities")
print("  • No CPU/GPU transfer overhead")
print()
print("=" * 80)

sys.exit(1)