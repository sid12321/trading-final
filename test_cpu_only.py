#!/usr/bin/env python3
"""
CPU-only performance test for comparison with Metal
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import time
import jax
import jax.numpy as jnp

def benchmark_cpu_operations():
    """Benchmark key operations on CPU only"""
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
    
    # Test 4: Intensive computation
    print("\n🧪 Test 4: Intensive computation")
    
    @jax.jit
    def intensive_compute():
        # Generate large random matrices
        x = jax.random.normal(jax.random.PRNGKey(42), (1500, 1500))  # Smaller for CPU
        
        # Chain of expensive operations
        for i in range(6):  # Fewer iterations for CPU
            x = jnp.dot(x, x.T)  # Matrix multiplication
            x = jnp.exp(jnp.sin(x))  # Element-wise transcendentals
            x = x / jnp.linalg.norm(x)  # Normalization
            x = x[:1400, :1400]  # Trim to prevent exponential growth
        
        return jnp.sum(x)
    
    print("   Running intensive computation on CPU...")
    
    start = time.time()
    result = intensive_compute()
    intensive_time = time.time() - start
    
    results['intensive_time'] = intensive_time
    print(f"   Time: {intensive_time:.4f}s")
    print(f"   Result: {result}")
    
    return results

if __name__ == "__main__":
    print("CPU-Only Performance Test")
    print("========================")
    
    results = benchmark_cpu_operations()
    
    print("\nCPU Results:")
    for op, time_val in results.items():
        print(f"  {op}: {time_val:.4f}s")