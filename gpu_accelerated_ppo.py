#!/usr/bin/env python3
"""
GPU Accelerated PPO - Stub module for compatibility
This module is replaced by EvoRL implementation when using train_evorl.py
"""

from stable_baselines3 import PPO


class GPUAcceleratedPPO(PPO):
    """GPU Accelerated PPO - compatibility wrapper"""
    def __init__(self, *args, **kwargs):
        # Remove GPU-specific arguments for base PPO
        kwargs.pop('use_gpu', None)
        kwargs.pop('gpu_batch_size', None)
        kwargs.pop('gpu_memory_fraction', None)
        super().__init__(*args, **kwargs)


def create_optimized_ppo(env, **kwargs):
    """Create optimized PPO model - compatibility function"""
    # For compatibility with existing code
    # When using EvoRL, this is replaced by EvoRL trainer
    # Default policy for PPO
    policy = kwargs.pop('policy', 'MlpPolicy')
    return GPUAcceleratedPPO(policy, env, **kwargs)


# Export for compatibility
__all__ = ['GPUAcceleratedPPO', 'create_optimized_ppo']