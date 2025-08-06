#!/usr/bin/env python3
"""
GPU Optimization Configuration for RTX 4080
Target: Maximize steps/second while maintaining training stability
"""

import torch
import numpy as np

def get_gpu_optimized_params(gpu_memory_gb=12.0):
    """
    Get optimized parameters based on GPU memory
    RTX 4080 has 12GB VRAM
    """
    
    # Key insight: For PPO, steps/s is primarily affected by:
    # 1. N_ENVS (parallel environments) - MORE IS BETTER for GPU
    # 2. BATCH_SIZE - needs to be divisible by N_ENVS
    # 3. N_STEPS - rollout buffer size
    # 4. Network size (but keep reasonable for trading)
    
    params = {}
    
    # CRITICAL GPU PERFORMANCE PARAMETERS
    # Maximize parallel environments - RTX 4080 can handle many more
    params['N_ENVS'] = 128  # Increased from 32 to 128 (4x)
    
    # Optimize batch size for GPU utilization
    # Rule: BATCH_SIZE should be large but must divide (N_STEPS * N_ENVS)
    params['N_STEPS'] = 512  # Reduced from 2048 to allow more frequent updates
    params['BATCH_SIZE'] = 2048  # Increased from 512 (4x) for better GPU utilization
    
    # Ensure divisibility
    total_samples = params['N_STEPS'] * params['N_ENVS']  # 512 * 128 = 65,536
    assert total_samples % params['BATCH_SIZE'] == 0, f"Batch size must divide total samples: {total_samples} % {params['BATCH_SIZE']} = {total_samples % params['BATCH_SIZE']}"
    
    # Learning parameters - adjusted for larger batch size
    params['GLOBALLEARNINGRATE'] = 0.0005  # Slightly higher for larger batches
    params['N_EPOCHS'] = 10  # Keep same
    params['ENT_COEF'] = 0.01  # Keep same
    params['TARGET_KL'] = 0.02  # Keep same
    params['GAE_LAMBDA'] = 0.95  # Keep same
    
    # GPU-specific optimizations
    params['USE_SDE'] = True
    params['SDE_SAMPLE_FREQ'] = 4
    
    # Network architecture - keep efficient
    params['POLICY_KWARGS'] = {
        'activation_fn': 'ReLU',
        'net_arch': {
            'pi': [256, 256],  # Keep same - good balance
            'vf': [256, 256]
        },
        'ortho_init': True
    }
    
    # Memory optimization
    params['GRADIENT_ACCUMULATION_STEPS'] = 1
    params['MIXED_PRECISION_DTYPE'] = torch.float16
    params['GPU_MEMORY_FRACTION'] = 0.95  # Use more GPU memory
    
    # CPU thread optimization for GPU mode
    params['N_CORES'] = 8  # Reduced from 16 to minimize CPU-GPU sync overhead
    params['OMP_NUM_THREADS'] = 8
    
    # Signal optimization for GPU
    params['SIGNAL_OPT_BATCH_SIZE'] = 16384  # Increased from 8192 (2x)
    params['SIGNAL_OPTIMIZATION_WORKERS'] = 8  # Reduced to match N_CORES
    
    # Logging frequency - adjust for more environments
    params['LOGFREQ'] = 1000  # Less frequent logging to reduce overhead
    
    return params

def get_aggressive_gpu_params():
    """
    Aggressive GPU parameters for maximum throughput
    Use with caution - may require tuning
    """
    params = get_gpu_optimized_params()
    
    # AGGRESSIVE SETTINGS
    params['N_ENVS'] = 256  # Maximum parallel environments (8x original)
    params['N_STEPS'] = 256  # Shorter rollouts for more frequent updates
    params['BATCH_SIZE'] = 4096  # Large batch size (8x original)
    
    # Ensure divisibility
    total_samples = params['N_STEPS'] * params['N_ENVS']  # 256 * 256 = 65,536
    assert total_samples % params['BATCH_SIZE'] == 0
    
    # Adjusted learning rate for very large batches
    params['GLOBALLEARNINGRATE'] = 0.001  # Higher for large batches
    
    # More aggressive GPU memory usage
    params['GPU_MEMORY_FRACTION'] = 0.98
    
    return params

def get_conservative_gpu_params():
    """
    Conservative GPU parameters for stability
    """
    params = get_gpu_optimized_params()
    
    # CONSERVATIVE SETTINGS
    params['N_ENVS'] = 64  # Moderate increase (2x original)
    params['N_STEPS'] = 1024  # Balanced rollout size
    params['BATCH_SIZE'] = 1024  # Moderate batch size (2x original)
    
    # Ensure divisibility
    total_samples = params['N_STEPS'] * params['N_ENVS']  # 1024 * 64 = 65,536
    assert total_samples % params['BATCH_SIZE'] == 0
    
    return params

def print_optimization_summary(params):
    """Print summary of optimization parameters"""
    print("=" * 60)
    print("GPU OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Parallel Environments (N_ENVS): {params['N_ENVS']}")
    print(f"Rollout Steps (N_STEPS): {params['N_STEPS']}")
    print(f"Batch Size: {params['BATCH_SIZE']}")
    print(f"Total samples per update: {params['N_STEPS'] * params['N_ENVS']:,}")
    print(f"Mini-batches per epoch: {(params['N_STEPS'] * params['N_ENVS']) // params['BATCH_SIZE']}")
    print(f"Learning Rate: {params['GLOBALLEARNINGRATE']}")
    print(f"CPU Threads: {params['N_CORES']}")
    print("=" * 60)

if __name__ == "__main__":
    # Test different configurations
    print("\n1. STANDARD GPU OPTIMIZATION (Recommended)")
    standard = get_gpu_optimized_params()
    print_optimization_summary(standard)
    
    print("\n2. AGGRESSIVE GPU OPTIMIZATION (Maximum Speed)")
    aggressive = get_aggressive_gpu_params()
    print_optimization_summary(aggressive)
    
    print("\n3. CONSERVATIVE GPU OPTIMIZATION (Stability)")
    conservative = get_conservative_gpu_params()
    print_optimization_summary(conservative)
    
    print("\nRecommendation: Start with STANDARD, then try AGGRESSIVE if stable")