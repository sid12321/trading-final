#!/usr/bin/env python3
"""
JAX Metal Compatibility Layer for M4 Max
Handles Metal backend limitations with intelligent CPU fallbacks
"""

import jax
import jax.numpy as jnp
import numpy as np
import warnings
from functools import wraps
import os

def with_device(device_type='cpu'):
    """Decorator to force operations on a specific device"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with jax.default_device(jax.devices(device_type)[0]):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def safe_jax_operation(operation_name="unknown"):
    """Decorator to safely execute JAX operations with Metal fallback"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Try on default device first (Metal if available)
                return func(*args, **kwargs)
            except Exception as e:
                if "UNIMPLEMENTED" in str(e) or "default_memory_space" in str(e):
                    # Fall back to CPU for unsupported Metal operations
                    print(f"⚠️  Metal unsupported for {operation_name}, using CPU fallback")
                    try:
                        with jax.default_device(jax.devices('cpu')[0]):
                            result = func(*args, **kwargs)
                            return result
                    except Exception as cpu_e:
                        raise cpu_e
                else:
                    raise e
        return wrapper
    return decorator

class MetalCompatJAX:
    """JAX operations with Metal compatibility"""
    
    @staticmethod
    def random_PRNGKey(seed):
        """Create PRNG key with Metal fallback"""
        try:
            # Try on default device first (Metal if available)
            return jax.random._patched_PRNGKey(seed)
        except Exception as e:
            if "UNIMPLEMENTED" in str(e) or "default_memory_space" in str(e):
                print("⚠️  Metal unsupported for random_key, using CPU fallback")
                with jax.default_device(jax.devices('cpu')[0]):
                    return jax.random._patched_PRNGKey(seed)
            else:
                raise e
    
    @staticmethod
    def random_split(key, num=2):
        """Split random key with Metal fallback"""
        try:
            return jax.random._patched_split(key, num)
        except Exception as e:
            if "UNIMPLEMENTED" in str(e) or "default_memory_space" in str(e):
                print("⚠️  Metal unsupported for random_split, using CPU fallback")
                with jax.default_device(jax.devices('cpu')[0]):
                    # Ensure key is on CPU
                    cpu_key = jax.device_put(key, jax.devices('cpu')[0])
                    return jax.random._patched_split(cpu_key, num)
            else:
                raise e
    
    @staticmethod 
    def random_normal(key, shape, dtype=jnp.float32):
        """Generate random normal with Metal fallback"""
        try:
            return jax.random._patched_normal(key, shape, dtype=dtype)
        except Exception as e:
            if "UNIMPLEMENTED" in str(e) or "default_memory_space" in str(e):
                print("⚠️  Metal unsupported for random_normal, using CPU fallback")
                with jax.default_device(jax.devices('cpu')[0]):
                    # Ensure key is on CPU
                    cpu_key = jax.device_put(key, jax.devices('cpu')[0])
                    return jax.random._patched_normal(cpu_key, shape, dtype=dtype)
            else:
                raise e
        
    @staticmethod
    def random_uniform(key, shape, minval=0.0, maxval=1.0, dtype=jnp.float32):
        """Generate random uniform with Metal fallback"""
        try:
            return jax.random._patched_uniform(key, shape, minval=minval, maxval=maxval, dtype=dtype)
        except Exception as e:
            if "UNIMPLEMENTED" in str(e) or "default_memory_space" in str(e):
                print("⚠️  Metal unsupported for random_uniform, using CPU fallback")
                with jax.default_device(jax.devices('cpu')[0]):
                    return jax.random._patched_uniform(key, shape, minval=minval, maxval=maxval, dtype=dtype)
            else:
                raise e
    
    @staticmethod
    def asarray(array, dtype=None):
        """Convert to JAX array with Metal fallback"""
        try:
            return jnp._patched_asarray(array, dtype=dtype)
        except Exception as e:
            if "UNIMPLEMENTED" in str(e) or "default_memory_space" in str(e):
                print("⚠️  Metal unsupported for asarray, using CPU fallback")
                with jax.default_device(jax.devices('cpu')[0]):
                    return jnp._patched_asarray(array, dtype=dtype)
            else:
                raise e
    
    @staticmethod
    def zeros(shape, dtype=jnp.float32):
        """Create zeros array with Metal fallback"""
        try:
            return jnp._patched_zeros(shape, dtype=dtype)
        except Exception as e:
            if "UNIMPLEMENTED" in str(e) or "default_memory_space" in str(e):
                print("⚠️  Metal unsupported for zeros, using CPU fallback")
                with jax.default_device(jax.devices('cpu')[0]):
                    return jnp._patched_zeros(shape, dtype=dtype)
            else:
                raise e
    
    @staticmethod  
    def ones(shape, dtype=jnp.float32):
        """Create ones array with Metal fallback"""
        try:
            return jnp._patched_ones(shape, dtype=dtype)
        except Exception as e:
            if "UNIMPLEMENTED" in str(e) or "default_memory_space" in str(e):
                print("⚠️  Metal unsupported for ones, using CPU fallback")
                with jax.default_device(jax.devices('cpu')[0]):
                    return jnp._patched_ones(shape, dtype=dtype)
            else:
                raise e
    
    @staticmethod
    def sqrt(x):
        """Square root with Metal fallback"""
        try:
            return jnp._patched_sqrt(x)
        except Exception as e:
            if "UNIMPLEMENTED" in str(e) or "default_memory_space" in str(e):
                print("⚠️  Metal unsupported for sqrt, using CPU fallback")
                with jax.default_device(jax.devices('cpu')[0]):
                    # Convert to CPU if needed
                    cpu_x = jax.device_put(x, jax.devices('cpu')[0])
                    return jnp._patched_sqrt(cpu_x)
            else:
                raise e

def setup_jax_for_metal():
    """Configure JAX for optimal Metal/CPU hybrid execution"""
    
    # Apply Apple's recommended environment variables + M4 Max optimizations
    os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    # M4 Max specific optimizations
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'  # Use 90% of 64GB unified memory
    os.environ['JAX_THREEFRY_PARTITIONABLE'] = '1'      # Better random number performance
    os.environ['JAX_TRANSFER_GUARD'] = 'allow'          # Allow Metal/CPU transfers
    
    # For debugging Metal issues, temporarily disable JIT as recommended by Apple
    # os.environ['JAX_DISABLE_JIT'] = '1'
    
    # Test Metal with compatible JAX versions first
    print("🍎 Testing JAX Metal compatibility with JAX 0.4.26...")
    # Note: Disable CPU forcing to test Metal backend
    # os.environ['JAX_PLATFORMS'] = 'cpu'
    
    # Enable Metal if available, with CPU fallback
    devices = jax.devices()
    has_metal = any('METAL' in str(d).upper() for d in devices)
    
    # Use Metal backend with optimizations
    if has_metal:
        print("🍎 JAX Metal backend detected - enabling hybrid CPU/Metal execution")
        print("   Metal ops: Large matrix operations, supported neural network layers")  
        print("   CPU ops: Random number generation, unsupported operations")
        print("   PJRT compatibility: ENABLED")
    else:
        print("🖥️ No Metal backend - using standard CPU execution")
    
    # Suppress Metal experimental warnings for cleaner output
    warnings.filterwarnings('ignore', message='.*METAL.*experimental.*')
    
    return has_metal

def get_compute_devices():
    """Get available compute devices with capabilities"""
    devices = jax.devices()
    
    device_info = {
        'devices': devices,
        'has_metal': any('METAL' in str(d).upper() for d in devices),
        'has_cpu': any('CPU' in str(d).upper() for d in devices),
        'primary_device': devices[0] if devices else None,
        'cpu_device': None,
        'metal_device': None
    }
    
    # Find specific device types
    for device in devices:
        if 'CPU' in str(device).upper():
            device_info['cpu_device'] = device
        elif 'METAL' in str(device).upper():
            device_info['metal_device'] = device
    
    return device_info

# Monkey patch common JAX functions for Metal compatibility
def patch_jax_for_metal():
    """Apply Metal compatibility patches to common JAX functions"""
    
    # Patch jax.random functions
    jax.random._patched_PRNGKey = jax.random.PRNGKey
    jax.random.PRNGKey = MetalCompatJAX.random_PRNGKey
    
    jax.random._patched_split = jax.random.split  
    jax.random.split = MetalCompatJAX.random_split
    
    jax.random._patched_normal = jax.random.normal
    jax.random.normal = MetalCompatJAX.random_normal
    
    jax.random._patched_uniform = jax.random.uniform
    jax.random.uniform = MetalCompatJAX.random_uniform
    
    # Patch jnp array creation functions
    jnp._patched_asarray = jnp.asarray
    jnp.asarray = MetalCompatJAX.asarray
    
    jnp._patched_zeros = jnp.zeros
    jnp.zeros = MetalCompatJAX.zeros
    
    jnp._patched_ones = jnp.ones  
    jnp.ones = MetalCompatJAX.ones
    
    jnp._patched_sqrt = jnp.sqrt
    jnp.sqrt = MetalCompatJAX.sqrt
    
    print("✅ JAX Metal compatibility patches applied")

def restore_jax_functions():
    """Restore original JAX functions"""
    if hasattr(jax.random, '_patched_PRNGKey'):
        jax.random.PRNGKey = jax.random._patched_PRNGKey
        jax.random.split = jax.random._patched_split
        jax.random.normal = jax.random._patched_normal
        jax.random.uniform = jax.random._patched_uniform
        
        jnp.asarray = jnp._patched_asarray  
        jnp.zeros = jnp._patched_zeros
        jnp.ones = jnp._patched_ones
        jnp.sqrt = jnp._patched_sqrt

if __name__ == "__main__":
    # Test the compatibility layer
    print("Testing JAX Metal Compatibility Layer")
    print("=" * 50)
    
    setup_jax_for_metal()
    patch_jax_for_metal()
    
    try:
        # Test random key creation (the failing operation)
        print("Testing random key creation...")
        key = MetalCompatJAX.random_PRNGKey(42)
        print(f"✅ Random key created: {key}")
        
        # Test array operations  
        print("Testing array operations...")
        arr = MetalCompatJAX.zeros((10, 10))
        device_info = str(arr.device()) if hasattr(arr, 'device') and callable(arr.device) else str(getattr(arr, 'device', 'Unknown'))
        print(f"✅ Zeros array created: {arr.shape} on {device_info}")
        
        # Test random generation
        print("Testing random generation...")
        key1, key2 = MetalCompatJAX.random_split(key)
        rand_arr = MetalCompatJAX.random_normal(key1, (5, 5))
        device_info = str(rand_arr.device()) if hasattr(rand_arr, 'device') and callable(rand_arr.device) else str(getattr(rand_arr, 'device', 'Unknown'))
        print(f"✅ Random array created: {rand_arr.shape} on {device_info}")
        
        print("🎉 JAX Metal compatibility layer working!")
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        
    finally:
        restore_jax_functions()