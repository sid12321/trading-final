#!/usr/bin/env python3
"""
JAX GPU Initialization for EvoRL Training
"""

import os
import jax
import jax.numpy as jnp


def init_jax_gpu():
    """Initialize JAX for GPU usage"""
    # Set environment variables for GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['JAX_PLATFORMS'] = 'cuda'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    print("✅ JAX GPU environment configured")


def get_jax_status():
    """Get JAX backend and device information"""
    try:
        backend = jax.default_backend()
        devices = jax.devices()
        device_count = len(devices)
        
        status = {
            'backend': backend,
            'devices': [str(d) for d in devices],
            'device_count': device_count
        }
        
        if backend == 'gpu':
            try:
                # Get GPU memory info if available
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_memory = result.stdout.strip().split('\n')[0]
                    status['gpu_memory'] = f"{gpu_memory}MB"
            except:
                status['gpu_memory'] = "Unknown"
        
        return status
        
    except Exception as e:
        return {
            'backend': 'unknown',
            'devices': [],
            'device_count': 0,
            'error': str(e)
        }


if __name__ == "__main__":
    init_jax_gpu()
    status = get_jax_status()
    
    print("JAX Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")