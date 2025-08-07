#!/usr/bin/env python3
"""
JAX GPU Initialization for EvoRL Training
Supports both CUDA (Linux/Windows) and Metal (macOS M-series)
"""

import os
import sys
import platform
import jax
import jax.numpy as jnp


def init_jax_gpu():
    """Initialize JAX for GPU usage - auto-detects CUDA vs Metal"""
    
    # Detect platform and set appropriate GPU backend
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "darwin" and "arm" in machine:
        # macOS Apple Silicon - use Metal
        print("🍎 Detected macOS Apple Silicon - configuring for Metal GPU")
        # Don't set JAX_PLATFORMS for Metal - let JAX auto-detect
        # os.environ['JAX_PLATFORMS'] = 'metal'  # Not needed, auto-detected
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        print("✅ JAX Metal GPU environment configured")
        
    elif system == "linux" or system == "windows":
        # Linux/Windows - use CUDA
        print("🐧 Detected Linux/Windows - configuring for CUDA GPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['JAX_PLATFORMS'] = 'cuda'
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        print("✅ JAX CUDA GPU environment configured")
        
    else:
        print(f"⚠️ Unknown platform: {system} {machine} - using default JAX settings")
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def get_jax_status():
    """Get JAX backend and device information"""
    try:
        backend = jax.default_backend()
        devices = jax.devices()
        device_count = len(devices)
        
        # Check if we have GPU-like devices (Metal or CUDA)
        has_gpu_device = any(
            str(d).upper().startswith(('GPU', 'METAL', 'CUDA'))
            for d in devices
        )
        
        status = {
            'backend': backend,
            'devices': [str(d) for d in devices],
            'device_count': device_count,
            'has_gpu': has_gpu_device
        }
        
        # Try to get memory info based on platform
        if has_gpu_device:
            try:
                if backend == 'gpu':
                    # NVIDIA GPU
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        gpu_memory = result.stdout.strip().split('\n')[0]
                        status['gpu_memory'] = f"{gpu_memory}MB"
                elif any('METAL' in str(d).upper() for d in devices):
                    # Apple Metal GPU
                    status['gpu_memory'] = "64GB unified (M4 Max)"
                    status['gpu_type'] = "Apple Metal"
            except:
                status['gpu_memory'] = "Unknown"
        
        return status
        
    except Exception as e:
        return {
            'backend': 'unknown',
            'devices': [],
            'device_count': 0,
            'has_gpu': False,
            'error': str(e)
        }


def is_gpu_available():
    """Check if GPU backend is available (CUDA or Metal)"""
    try:
        devices = jax.devices()
        backend = jax.default_backend()
        
        # Check for GPU devices
        has_gpu_device = any(
            str(d).upper().startswith(('GPU', 'METAL', 'CUDA'))
            for d in devices
        )
        
        # Check backend types that indicate GPU acceleration
        gpu_backends = ['gpu', 'metal']
        has_gpu_backend = backend.lower() in gpu_backends
        
        return has_gpu_device or has_gpu_backend
        
    except Exception:
        return False


if __name__ == "__main__":
    init_jax_gpu()
    status = get_jax_status()
    
    print("JAX Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")