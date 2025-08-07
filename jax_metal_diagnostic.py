#!/usr/bin/env python3
"""
JAX Metal Diagnostic Tool for M4 Max
Investigates why Metal GPU shows 0% utilization despite JAX detecting it
"""

import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
import psutil
from datetime import datetime

def setup_metal_environment():
    """Setup Metal environment variables"""
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'
    os.environ['JAX_THREEFRY_PARTITIONABLE'] = '1'

def test_basic_jax_operations():
    """Test basic JAX operations to see if they actually use Metal"""
    print("=" * 60)
    print("🔍 JAX Metal Diagnostic Tool")
    print("=" * 60)
    
    # JAX system info
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Available backends: {jax.lib.xla_bridge.get_backend().platform}")
    
    print("\n" + "=" * 60)
    print("🧪 Testing Basic JAX Operations")
    print("=" * 60)
    
    # Test 1: Basic array creation
    print("Test 1: Array creation")
    start_time = time.time()
    x = jnp.ones((10000, 10000))
    array_time = time.time() - start_time
    print(f"   Created 10000x10000 array in {array_time:.4f}s")
    print(f"   Device: {x.device()}")
    
    # Test 2: Matrix multiplication
    print("\nTest 2: Matrix multiplication")
    start_time = time.time()
    result = jnp.dot(x, x)
    matmul_time = time.time() - start_time
    print(f"   Matrix multiplication in {matmul_time:.4f}s")
    print(f"   Result device: {result.device()}")
    
    # Test 3: JIT compilation
    print("\nTest 3: JIT compilation")
    
    @jax.jit
    def compute_expensive_operation(x):
        """Expensive operation that should benefit from GPU"""
        return jnp.sum(jnp.exp(jnp.sin(x)) * jnp.cos(x))
    
    # Warm up JIT
    _ = compute_expensive_operation(jnp.ones(10))
    
    start_time = time.time()
    result = compute_expensive_operation(jnp.ones(100000))
    jit_time = time.time() - start_time
    print(f"   JIT operation on 100K elements in {jit_time:.4f}s")
    print(f"   Result: {result}")
    
    # Test 4: vmap (vectorization)
    print("\nTest 4: vmap vectorization")
    
    def single_operation(x):
        return jnp.sum(x ** 2)
    
    vectorized_op = jax.vmap(single_operation)
    
    data = jnp.ones((1000, 1000))
    start_time = time.time()
    vmap_result = vectorized_op(data)
    vmap_time = time.time() - start_time
    print(f"   vmap on 1000x1000 in {vmap_time:.4f}s")
    print(f"   Result shape: {vmap_result.shape}")
    
    return {
        'array_time': array_time,
        'matmul_time': matmul_time,
        'jit_time': jit_time,
        'vmap_time': vmap_time
    }

def test_trading_like_operations():
    """Test operations similar to what we do in trading"""
    print("\n" + "=" * 60)
    print("📊 Trading-Like Operations Test")
    print("=" * 60)
    
    # Simulate trading environment operations
    batch_size = 256
    obs_dim = 100
    action_dim = 2
    
    # Test 1: Policy network forward pass simulation
    print("Test 1: Policy network simulation")
    
    @jax.jit
    def policy_forward(params, obs):
        """Simulate policy network forward pass"""
        W1, b1 = params['layer1']
        W2, b2 = params['layer2'] 
        W3, b3 = params['layer3']
        
        x = jnp.tanh(jnp.dot(obs, W1) + b1)
        x = jnp.tanh(jnp.dot(x, W2) + b2)
        mean = jnp.dot(x, W3) + b3
        return mean
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    params = {
        'layer1': (jax.random.normal(k1, (obs_dim, 512)), jax.random.normal(k2, (512,))),
        'layer2': (jax.random.normal(k3, (512, 256)), jax.random.normal(k4, (256,))),
        'layer3': (jax.random.normal(k1, (256, action_dim)), jax.random.normal(k2, (action_dim,)))
    }
    
    # Batch observations
    obs_batch = jax.random.normal(key, (batch_size, obs_dim))
    
    # Warm up
    _ = policy_forward(params, obs_batch[:1])
    
    start_time = time.time()
    actions = policy_forward(params, obs_batch)
    policy_time = time.time() - start_time
    print(f"   Policy forward pass (batch={batch_size}) in {policy_time:.4f}s")
    print(f"   Actions device: {actions.device()}")
    
    # Test 2: Vectorized policy steps
    print("\nTest 2: Vectorized policy steps")
    
    @jax.jit
    def policy_step(params, obs, key):
        """Single policy step with sampling"""
        mean = policy_forward(params, obs[None, :])
        action = mean[0] + 0.1 * jax.random.normal(key, mean[0].shape)
        log_prob = jnp.sum(-0.5 * ((action - mean[0]) / 0.1)**2)
        return action, log_prob
    
    # Vectorize over batch
    vmap_policy_step = jax.jit(jax.vmap(policy_step, in_axes=(None, 0, 0)))
    
    # Generate keys for batch
    step_keys = jax.random.split(key, batch_size)
    
    # Warm up
    _ = vmap_policy_step(params, obs_batch[:2], step_keys[:2])
    
    start_time = time.time()
    batch_actions, batch_log_probs = vmap_policy_step(params, obs_batch, step_keys)
    vmap_policy_time = time.time() - start_time
    print(f"   Vectorized policy steps (batch={batch_size}) in {vmap_policy_time:.4f}s")
    print(f"   Actions shape: {batch_actions.shape}")
    
    # Test 3: GAE computation simulation
    print("\nTest 3: GAE computation simulation")
    
    @jax.jit
    def compute_gae_batch(rewards, values, dones, next_values, gamma=0.99, gae_lambda=0.95):
        """Compute GAE for a batch of episodes"""
        n_steps = rewards.shape[0]
        advantages = jnp.zeros_like(rewards)
        
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_values
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages = advantages.at[t].set(delta + gamma * gae_lambda * next_non_terminal * advantages[t+1] if t < n_steps-1 else delta)
        
        return advantages
    
    # Simulate episode data
    n_steps = 512
    episode_rewards = jax.random.normal(key, (n_steps,))
    episode_values = jax.random.normal(key, (n_steps,))
    episode_dones = jax.random.bernoulli(key, 0.01, (n_steps,))  # 1% done probability
    final_value = jax.random.normal(key, ())
    
    start_time = time.time()
    advantages = compute_gae_batch(episode_rewards, episode_values, episode_dones, final_value)
    gae_time = time.time() - start_time
    print(f"   GAE computation ({n_steps} steps) in {gae_time:.4f}s")
    print(f"   Advantages device: {advantages.device()}")
    
    return {
        'policy_time': policy_time,
        'vmap_policy_time': vmap_policy_time,
        'gae_time': gae_time
    }

def monitor_system_during_computation():
    """Monitor system resources during heavy computation"""
    print("\n" + "=" * 60)
    print("🔍 System Monitoring During Computation")
    print("=" * 60)
    
    # Start monitoring
    print("Starting intensive computation while monitoring system...")
    
    @jax.jit
    def intensive_computation():
        """Heavy computation to trigger GPU usage"""
        # Large matrix operations
        x = jax.random.normal(jax.random.PRNGKey(42), (5000, 5000))
        
        # Multiple operations
        for i in range(10):
            x = jnp.dot(x, x.T)
            x = jnp.exp(jnp.sin(x)) / (1 + jnp.abs(x))
            x = x / jnp.linalg.norm(x)
        
        return jnp.sum(x)
    
    # Pre-compile
    _ = intensive_computation()
    
    # Monitor during execution
    start_time = time.time()
    
    # Get initial system stats
    initial_cpu = psutil.cpu_percent(interval=0.1)
    initial_memory = psutil.virtual_memory().percent
    
    print(f"Initial CPU: {initial_cpu:.1f}%")
    print(f"Initial Memory: {initial_memory:.1f}%")
    
    # Run computation
    print("\nRunning intensive computation...")
    start_compute = time.time()
    result = intensive_computation()
    compute_time = time.time() - start_compute
    
    # Get final system stats
    final_cpu = psutil.cpu_percent(interval=0.1)
    final_memory = psutil.virtual_memory().percent
    
    print(f"\nComputation completed in {compute_time:.4f}s")
    print(f"Result: {result}")
    print(f"Final CPU: {final_cpu:.1f}%")
    print(f"Final Memory: {final_memory:.1f}%")
    print(f"CPU change: {final_cpu - initial_cpu:+.1f}%")
    print(f"Memory change: {final_memory - initial_memory:+.1f}%")
    
    return compute_time

def test_explicit_device_placement():
    """Test explicit device placement to see if operations actually run on Metal"""
    print("\n" + "=" * 60)
    print("🎯 Explicit Device Placement Test")
    print("=" * 60)
    
    # Get available devices
    devices = jax.devices()
    print(f"Available devices: {devices}")
    
    for device in devices:
        print(f"\nTesting on device: {device}")
        
        # Place computation explicitly on this device
        with jax.default_device(device):
            x = jnp.ones((1000, 1000))
            print(f"   Array device: {x.device()}")
            
            start_time = time.time()
            result = jnp.dot(x, x)
            compute_time = time.time() - start_time
            
            print(f"   Matrix multiplication: {compute_time:.4f}s")
            print(f"   Result device: {result.device()}")

def main():
    """Main diagnostic function"""
    # Setup environment
    setup_metal_environment()
    
    # Run all diagnostic tests
    basic_times = test_basic_jax_operations()
    trading_times = test_trading_like_operations()
    intensive_time = monitor_system_during_computation()
    test_explicit_device_placement()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print("Performance Times:")
    print(f"  Array creation:     {basic_times['array_time']:.4f}s")
    print(f"  Matrix multiply:    {basic_times['matmul_time']:.4f}s")
    print(f"  JIT operation:      {basic_times['jit_time']:.4f}s")
    print(f"  vmap operation:     {basic_times['vmap_time']:.4f}s")
    print(f"  Policy forward:     {trading_times['policy_time']:.4f}s")
    print(f"  Vectorized policy:  {trading_times['vmap_policy_time']:.4f}s")
    print(f"  GAE computation:    {trading_times['gae_time']:.4f}s")
    print(f"  Intensive compute:  {intensive_time:.4f}s")
    
    # Analysis
    print("\n🔍 ANALYSIS:")
    if basic_times['matmul_time'] < 0.01:
        print("✅ Matrix operations are VERY fast - likely using Metal GPU")
    elif basic_times['matmul_time'] < 0.1:
        print("⚠️  Matrix operations are moderately fast - uncertain Metal usage")
    else:
        print("❌ Matrix operations are slow - likely CPU only")
    
    if trading_times['policy_time'] < 0.001:
        print("✅ Policy operations are VERY fast - likely using Metal GPU")
    else:
        print("⚠️  Policy operations could be faster")
    
    print("\n💡 RECOMMENDATIONS:")
    print("1. Run this while monitoring with: python simple_monitor.py")
    print("2. Compare times with CPU-only mode")
    print("3. Check if operations show up as Metal processes")
    print("4. Look for GPU utilization spikes during intensive operations")
    
    print("\n🎯 To investigate GPU utilization:")
    print("Terminal 1: python jax_metal_diagnostic.py")
    print("Terminal 2: python simple_monitor.py")
    print("\nWatch for Metal GPU activity correlation!")

if __name__ == "__main__":
    main()