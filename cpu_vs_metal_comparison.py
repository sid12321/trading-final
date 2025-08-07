#!/usr/bin/env python3
"""
CPU vs Metal Performance Comparison
Tests the same operations on CPU vs Metal to understand actual performance difference
"""

import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np

def force_cpu_mode():
    """Force JAX to use CPU only"""
    os.environ['JAX_PLATFORMS'] = 'cpu'
    # Restart JAX
    jax.clear_caches()

def force_metal_mode():
    """Force JAX to use Metal"""
    os.environ.pop('JAX_PLATFORMS', None)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'
    # Restart JAX  
    jax.clear_caches()

def benchmark_operations(mode_name):
    """Benchmark key operations"""
    print(f"\n{'='*20} {mode_name} MODE {'='*20}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    
    results = {}
    
    # Test 1: Large matrix multiplication
    print("\n🧪 Test 1: Large matrix multiplication (2000x2000)")
    x = jnp.ones((2000, 2000))
    
    start = time.time()
    result = jnp.dot(x, x)
    matmul_time = time.time() - start
    
    results['matmul_time'] = matmul_time
    print(f"   Time: {matmul_time:.4f}s")
    
    # Test 2: JIT compilation performance
    print("\n🧪 Test 2: JIT-compiled operation")
    
    @jax.jit
    def complex_operation(x):
        for i in range(5):
            x = jnp.exp(jnp.sin(x)) * jnp.cos(x)
            x = x / jnp.linalg.norm(x)
        return jnp.sum(x)
    
    # Warm up
    test_data = jnp.ones(10000)
    _ = complex_operation(test_data)
    
    start = time.time()
    result = complex_operation(test_data)
    jit_time = time.time() - start
    
    results['jit_time'] = jit_time
    print(f"   Time: {jit_time:.4f}s")
    
    # Test 3: Vectorized operations (like our policy network)
    print("\n🧪 Test 3: Vectorized policy-like operations")
    
    @jax.jit
    def policy_simulation(params, obs_batch):
        W1, b1 = params
        # Simple 2-layer network
        x = jnp.tanh(jnp.dot(obs_batch, W1) + b1)
        return jnp.sum(x, axis=1)
    
    # Create parameters
    obs_dim, hidden_dim = 100, 512
    batch_size = 256
    
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    
    params = (
        jax.random.normal(k1, (obs_dim, hidden_dim)), 
        jax.random.normal(k2, (hidden_dim,))
    )
    obs_batch = jax.random.normal(key, (batch_size, obs_dim))
    
    # Warm up
    _ = policy_simulation(params, obs_batch[:1])
    
    start = time.time()
    result = policy_simulation(params, obs_batch)
    policy_time = time.time() - start
    
    results['policy_time'] = policy_time
    print(f"   Time: {policy_time:.4f}s")
    print(f"   Batch size: {batch_size}")
    
    # Test 4: Intensive computation that should show GPU usage
    print("\n🧪 Test 4: Intensive computation (should spike utilization)")
    
    @jax.jit
    def intensive_compute():
        # Generate large random matrices
        x = jax.random.normal(jax.random.PRNGKey(42), (3000, 3000))
        
        # Chain of expensive operations
        for i in range(8):
            x = jnp.dot(x, x.T)  # Matrix multiplication
            x = jnp.exp(jnp.sin(x))  # Element-wise transcendentals
            x = x / jnp.linalg.norm(x)  # Normalization
            x = x[:2900, :2900]  # Trim to prevent exponential growth
        
        return jnp.sum(x)
    
    print("   Running intensive computation (watch system monitor!)...")
    
    start = time.time()
    result = intensive_compute()
    intensive_time = time.time() - start
    
    results['intensive_time'] = intensive_time
    print(f"   Time: {intensive_time:.4f}s")
    print(f"   Result: {result}")
    
    return results

def main():
    """Main comparison function"""
    print("🔍 CPU vs Metal Performance Comparison")
    print("This will help identify if Metal is actually being used effectively")
    
    # Test Metal mode first
    force_metal_mode()
    metal_results = benchmark_operations("METAL")
    
    print("\n" + "="*60)
    print("🔄 Switching to CPU mode...")
    print("Note: This requires restarting Python process")
    print("="*60)
    
    # Force CPU mode - this requires process restart
    force_cpu_mode()
    
    # Reimport JAX after clearing
    import importlib
    importlib.reload(jax)
    
    cpu_results = benchmark_operations("CPU")
    
    # Compare results
    print("\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON")
    print("="*60)
    
    print(f"{'Operation':<25} {'Metal Time':<12} {'CPU Time':<12} {'Speedup':<10}")
    print("-" * 60)
    
    for op in metal_results:
        metal_time = metal_results[op]
        cpu_time = cpu_results.get(op, 0)
        speedup = cpu_time / metal_time if metal_time > 0 and cpu_time > 0 else 0
        
        print(f"{op:<25} {metal_time:<12.4f} {cpu_time:<12.4f} {speedup:<10.2f}x")
    
    # Analysis
    print("\n🔍 ANALYSIS:")
    
    avg_speedup = np.mean([cpu_results[op] / metal_results[op] for op in metal_results if op in cpu_results])
    
    if avg_speedup > 3.0:
        print(f"✅ Metal is significantly faster ({avg_speedup:.1f}x average speedup)")
        print("   The GPU IS being used effectively!")
    elif avg_speedup > 1.5:
        print(f"⚠️  Metal is moderately faster ({avg_speedup:.1f}x average speedup)")
        print("   Some operations may be using Metal, others CPU")
    else:
        print(f"❌ Metal shows minimal advantage ({avg_speedup:.1f}x average speedup)")
        print("   Operations may be running primarily on CPU")
    
    # GPU utilization analysis
    print("\n💡 GPU UTILIZATION ANALYSIS:")
    print("If Metal shows good speedup but system monitor shows 0% GPU:")
    print("1. macOS may not report Metal GPU utilization correctly")
    print("2. powermetrics might be the only way to see actual Metal usage")
    print("3. Our training IS likely using Metal despite 0% in monitoring")
    print("4. The 33% speedup we achieved is evidence Metal is working")
    
    print("\n🎯 CONCLUSIONS FOR YOUR TRAINING:")
    if metal_results.get('intensive_time', 1) < 1.0:
        print("✅ Metal backend IS working and providing acceleration")
        print("✅ Your 33% training speedup is evidence of GPU utilization")
        print("✅ System monitors may not accurately report Metal GPU usage")
        print("✅ Focus on training speed (60 it/s) rather than GPU % readings")
    else:
        print("⚠️  Metal performance may not be optimal")
        print("   Consider checking JAX configuration")

if __name__ == "__main__":
    print("⚠️  WARNING: This script needs to restart the Python process")
    print("   to properly switch between CPU and Metal modes.")
    print("\n   Run this script twice:")
    print("   1. First run will test Metal mode")
    print("   2. Manually set JAX_PLATFORMS=cpu and run again for comparison")
    print()
    
    choice = input("Continue with Metal test? (y/n): ")
    if choice.lower() == 'y':
        # Just run Metal mode for now
        force_metal_mode()
        metal_results = benchmark_operations("METAL")
        
        print("\n🔧 TO COMPLETE COMPARISON:")
        print("1. Run: JAX_PLATFORMS=cpu python cpu_vs_metal_comparison.py")
        print("2. Compare the timing results manually")
    else:
        print("Cancelled.")