#!/usr/bin/env python3
"""
Apply GPU optimization to parameters.py
This script updates the parameters for maximum GPU performance
"""

import os
import shutil
from datetime import datetime

def backup_parameters():
    """Backup current parameters.py"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"parameters_backup_{timestamp}.py"
    shutil.copy("parameters.py", backup_file)
    print(f"✅ Backed up parameters.py to {backup_file}")
    return backup_file

def apply_gpu_optimizations(optimization_level="standard"):
    """
    Apply GPU optimizations to parameters.py
    
    Args:
        optimization_level: "standard", "aggressive", or "conservative"
    """
    
    # Read current parameters.py
    with open("parameters.py", "r") as f:
        lines = f.readlines()
    
    # Define optimization parameters based on level
    if optimization_level == "aggressive":
        print("🚀 Applying AGGRESSIVE GPU optimization...")
        replacements = {
            "N_ENVS = 32": "N_ENVS = 256  # AGGRESSIVE: 8x increase for maximum GPU throughput",
            "N_STEPS = 2048": "N_STEPS = 256  # AGGRESSIVE: Shorter rollouts for frequent updates",
            "BATCH_SIZE = 512": "BATCH_SIZE = 4096  # AGGRESSIVE: 8x increase for GPU efficiency",
            "GLOBALLEARNINGRATE = 0.0003": "GLOBALLEARNINGRATE = 0.001  # AGGRESSIVE: Higher LR for large batches",
            "N_CORES = 16": "N_CORES = 8  # AGGRESSIVE: Minimize CPU-GPU sync overhead",
            "SIGNAL_OPT_BATCH_SIZE = 8192": "SIGNAL_OPT_BATCH_SIZE = 32768  # AGGRESSIVE: 4x for GPU",
            "SIGNAL_OPTIMIZATION_WORKERS = 16": "SIGNAL_OPTIMIZATION_WORKERS = 8  # Match N_CORES",
            "GPU_MEMORY_FRACTION = 0.9": "GPU_MEMORY_FRACTION = 0.98  # AGGRESSIVE: Use almost all GPU memory",
        }
    elif optimization_level == "conservative":
        print("🛡️ Applying CONSERVATIVE GPU optimization...")
        replacements = {
            "N_ENVS = 32": "N_ENVS = 64  # CONSERVATIVE: 2x increase for stability",
            "N_STEPS = 2048": "N_STEPS = 1024  # CONSERVATIVE: Balanced rollout size",
            "BATCH_SIZE = 512": "BATCH_SIZE = 1024  # CONSERVATIVE: 2x increase",
            "N_CORES = 16": "N_CORES = 12  # CONSERVATIVE: Balanced CPU usage",
            "SIGNAL_OPT_BATCH_SIZE = 8192": "SIGNAL_OPT_BATCH_SIZE = 16384  # CONSERVATIVE: 2x",
            "SIGNAL_OPTIMIZATION_WORKERS = 16": "SIGNAL_OPTIMIZATION_WORKERS = 12  # Match N_CORES",
        }
    else:  # standard
        print("⚡ Applying STANDARD GPU optimization (Recommended)...")
        replacements = {
            "N_ENVS = 32": "N_ENVS = 128  # OPTIMIZED: 4x increase for GPU throughput",
            "N_STEPS = 2048": "N_STEPS = 512  # OPTIMIZED: Shorter for frequent updates",
            "BATCH_SIZE = 512": "BATCH_SIZE = 2048  # OPTIMIZED: 4x for better GPU utilization",
            "GLOBALLEARNINGRATE = 0.0003": "GLOBALLEARNINGRATE = 0.0005  # OPTIMIZED: Adjusted for larger batches",
            "N_CORES = 16": "N_CORES = 8  # OPTIMIZED: Reduce CPU-GPU sync overhead",
            "SIGNAL_OPT_BATCH_SIZE = 8192": "SIGNAL_OPT_BATCH_SIZE = 16384  # OPTIMIZED: 2x for GPU",
            "SIGNAL_OPTIMIZATION_WORKERS = 16": "SIGNAL_OPTIMIZATION_WORKERS = 8  # Match N_CORES",
            "GPU_MEMORY_FRACTION = 0.9": "GPU_MEMORY_FRACTION = 0.95  # OPTIMIZED: Use more GPU memory",
            "LOGFREQ = min(BASEMODELITERATIONS//2,500)": "LOGFREQ = 1000  # OPTIMIZED: Less frequent logging",
        }
    
    # Apply replacements
    modified = False
    new_lines = []
    for line in lines:
        original_line = line
        for old, new in replacements.items():
            if old in line and not line.strip().startswith("#"):
                line = line.replace(old, new)
                modified = True
                print(f"  Updated: {old} -> {new.split('#')[0].strip()}")
                break
        new_lines.append(line)
    
    # Add optimization timestamp
    if modified:
        # Update optimization timestamp
        for i, line in enumerate(new_lines):
            if "OPTIMIZATION_TIMESTAMP = " in line:
                new_lines[i] = f"OPTIMIZATION_TIMESTAMP = '{datetime.now().strftime('%Y-%m-%d %H:%M')}'\n"
            if "print(f\"GPU Optimization:" in line:
                new_lines[i] = f'    print(f"GPU Optimization ({optimization_level.upper()}): {{GPU_NAME}} with {{N_ENVS}} environments")\n'
    
    # Write updated parameters
    with open("parameters.py", "w") as f:
        f.writelines(new_lines)
    
    print(f"\n✅ GPU optimization applied successfully!")
    
    # Print expected performance
    print("\n📊 Expected Performance Improvements:")
    print("=" * 50)
    if optimization_level == "aggressive":
        print("- Steps/second: ~150-200+ (from 43)")
        print("- GPU utilization: 90-95%")
        print("- Training speed: ~4-5x faster")
        print("⚠️  WARNING: May need learning rate tuning")
    elif optimization_level == "conservative":
        print("- Steps/second: ~70-90 (from 43)")
        print("- GPU utilization: 60-70%")
        print("- Training speed: ~2x faster")
        print("✅ More stable, less tuning needed")
    else:
        print("- Steps/second: ~100-150 (from 43)")
        print("- GPU utilization: 75-85%")
        print("- Training speed: ~3-4x faster")
        print("✅ Good balance of speed and stability")
    
    print("\n🎯 Next Steps:")
    print("1. Run training: python train.py")
    print("2. Monitor GPU usage: watch -n 1 nvidia-smi")
    print("3. Check steps/s in training output")
    print("4. If stable, try 'aggressive' mode for more speed")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Apply GPU optimizations to parameters.py")
    parser.add_argument("--level", choices=["standard", "aggressive", "conservative"], 
                       default="standard", help="Optimization level")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup")
    args = parser.parse_args()
    
    # Backup current parameters
    if not args.no_backup:
        backup_file = backup_parameters()
    
    # Apply optimizations
    apply_gpu_optimizations(args.level)
    
    print(f"\nTo revert changes: cp {backup_file} parameters.py")

if __name__ == "__main__":
    main()