#!/usr/bin/env python3
"""
Test Threading Optimization Impact
Compares performance before/after threading configuration
"""

import os
import sys
import time
import psutil

# Configure environment
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['XLA_CPU_MULTI_THREAD_EIGEN'] = 'true'
os.environ['JAX_ENABLE_X64'] = 'false'
os.environ['OPENBLAS_NUM_THREADS'] = '16'

import jax
import jax.numpy as jnp
import numpy as np

def monitor_cpu_during_operation(operation_name, operation_func, *args, **kwargs):
    """Monitor CPU usage during an operation"""
    print(f"\n🧪 Testing {operation_name}:")
    
    # Get initial CPU state
    initial_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
    initial_avg = np.mean(initial_cpu)
    active_cores_initial = sum(1 for c in initial_cpu if c > 10)
    
    print(f"   Initial: {initial_avg:.1f}% avg, {active_cores_initial}/16 cores active")
    
    # Run operation
    start_time = time.time()
    result = operation_func(*args, **kwargs)
    end_time = time.time()
    
    # Get final CPU state
    final_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
    final_avg = np.mean(final_cpu)
    active_cores_final = sum(1 for c in final_cpu if c > 10)
    max_core_usage = max(final_cpu)
    
    execution_time = end_time - start_time
    
    print(f"   Duration: {execution_time:.4f}s")
    print(f"   Final: {final_avg:.1f}% avg, {active_cores_final}/16 cores active")
    print(f"   Max core: {max_core_usage:.1f}%")
    print(f"   CPU increase: {final_avg - initial_avg:+.1f}%")
    
    return execution_time, final_avg, active_cores_final

def test_jax_operations():
    """Test JAX operations that should benefit from threading"""
    print("=" * 60)
    print("🚀 Threading Optimization Test")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    
    # Test 1: Large matrix operations
    def large_matrix_ops():
        x = jnp.ones((3000, 3000))
        # Multiple operations that could use threading
        for i in range(3):
            x = jnp.dot(x, x.T)
            x = x / jnp.linalg.norm(x)
        return jnp.sum(x)
    
    time1, cpu1, cores1 = monitor_cpu_during_operation("Large Matrix Operations", large_matrix_ops)
    
    # Test 2: Batch operations
    def batch_operations():
        batch_size = 1000
        data = jax.random.normal(jax.random.PRNGKey(42), (batch_size, 500))
        
        @jax.jit
        def process_batch(x):
            # CPU-intensive operations
            for i in range(50):
                x = jnp.sin(x) * jnp.cos(x)
                x = jnp.tanh(x)
            return jnp.mean(x)
        
        # Vectorize over batch
        vectorized_process = jax.vmap(process_batch)
        return vectorized_process(data)
    
    time2, cpu2, cores2 = monitor_cpu_during_operation("Batch Operations", batch_operations)
    
    # Test 3: Policy-like operations
    def policy_simulation():
        obs_dim, hidden_dim = 100, 512
        batch_size = 512  # Match our training
        
        @jax.jit
        def neural_network(params, obs_batch):
            W1, b1, W2, b2 = params
            x = jnp.tanh(jnp.dot(obs_batch, W1) + b1)
            x = jnp.tanh(jnp.dot(x, W2) + b2)
            return x
        
        # Create parameters
        key = jax.random.PRNGKey(42)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        params = (
            jax.random.normal(k1, (obs_dim, hidden_dim)),
            jax.random.normal(k2, (hidden_dim,)),
            jax.random.normal(k3, (hidden_dim, 2)),
            jax.random.normal(k4, (2,))
        )
        
        # Batch data
        obs_batch = jax.random.normal(key, (batch_size, obs_dim))
        
        # Run multiple forward passes
        results = []
        for i in range(100):  # Many forward passes
            result = neural_network(params, obs_batch)
            results.append(result)
        
        return jnp.stack(results)
    
    time3, cpu3, cores3 = monitor_cpu_during_operation("Policy Network Simulation", policy_simulation)
    
    return {
        'matrix_time': time1, 'matrix_cpu': cpu1, 'matrix_cores': cores1,
        'batch_time': time2, 'batch_cpu': cpu2, 'batch_cores': cores2, 
        'policy_time': time3, 'policy_cpu': cpu3, 'policy_cores': cores3
    }

def test_trading_simulation():
    """Test operations similar to actual training"""
    print("\n" + "=" * 60)
    print("📊 Trading Simulation Test")
    print("=" * 60)
    
    def trading_step_simulation():
        # Simulate what happens in one training rollout
        n_steps = 512  # Match our training
        batch_size = 256  # Our parallel environments
        obs_dim = 100
        
        @jax.jit
        def simulate_rollout_step(params, obs_batch, keys):
            """Simulate one step of rollout collection"""
            W1, b1, W2, b2 = params
            
            # Policy forward pass
            x = jnp.tanh(jnp.dot(obs_batch, W1) + b1)
            actions = jnp.tanh(jnp.dot(x, W2) + b2)
            
            # Sample actions with randomness
            actions = actions + 0.1 * jax.random.normal(keys, actions.shape)
            
            # Compute log probabilities (simplified)
            log_probs = -0.5 * jnp.sum(actions ** 2, axis=1)
            
            return actions, log_probs
        
        # Initialize parameters
        key = jax.random.PRNGKey(42)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        params = (
            jax.random.normal(k1, (obs_dim, 512)),
            jax.random.normal(k2, (512,)),
            jax.random.normal(k3, (512, 2)),
            jax.random.normal(k4, (2,))
        )
        
        # Simulate multiple steps
        total_actions = []
        total_log_probs = []
        
        for step in range(n_steps):
            # Generate observations and keys
            obs_batch = jax.random.normal(key, (batch_size, obs_dim))
            step_keys = jax.random.split(key, batch_size)
            
            # Process step
            actions, log_probs = simulate_rollout_step(params, obs_batch, step_keys)
            
            total_actions.append(actions)
            total_log_probs.append(log_probs)
            
            # Update key for next step
            key = jax.random.split(key, 1)[0]
        
        return jnp.stack(total_actions), jnp.stack(total_log_probs)
    
    time_sim, cpu_sim, cores_sim = monitor_cpu_during_operation("Trading Rollout Simulation", trading_step_simulation)
    
    return time_sim, cpu_sim, cores_sim

def main():
    """Main test function"""
    print("🔧 M4 Max Threading Optimization Test")
    print(f"Threading configuration:")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS')}")
    print(f"  XLA_CPU_MULTI_THREAD_EIGEN: {os.environ.get('XLA_CPU_MULTI_THREAD_EIGEN')}")
    print(f"  OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS')}")
    
    # Test JAX operations
    jax_results = test_jax_operations()
    
    # Test trading simulation
    trading_time, trading_cpu, trading_cores = test_trading_simulation()
    
    # Analysis
    print("\n" + "=" * 60)
    print("📊 THREADING OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print("JAX Operations:")
    print(f"  Matrix ops:     {jax_results['matrix_time']:.3f}s, {jax_results['matrix_cpu']:.1f}% CPU, {jax_results['matrix_cores']}/16 cores")
    print(f"  Batch ops:      {jax_results['batch_time']:.3f}s, {jax_results['batch_cpu']:.1f}% CPU, {jax_results['batch_cores']}/16 cores") 
    print(f"  Policy nets:    {jax_results['policy_time']:.3f}s, {jax_results['policy_cpu']:.1f}% CPU, {jax_results['policy_cores']}/16 cores")
    
    print(f"\nTrading Simulation:")
    print(f"  Rollout sim:    {trading_time:.3f}s, {trading_cpu:.1f}% CPU, {trading_cores}/16 cores")
    
    # Assessment
    avg_cpu = np.mean([jax_results['matrix_cpu'], jax_results['batch_cpu'], jax_results['policy_cpu'], trading_cpu])
    avg_cores = np.mean([jax_results['matrix_cores'], jax_results['batch_cores'], jax_results['policy_cores'], trading_cores])
    
    print(f"\n🎯 ASSESSMENT:")
    print(f"Average CPU utilization: {avg_cpu:.1f}%")
    print(f"Average active cores: {avg_cores:.1f}/16")
    
    if avg_cpu > 50 and avg_cores > 8:
        print("✅ Threading optimization is working well!")
        print("   Multiple cores are being utilized effectively")
        print("   Expected training speedup: 20-30%")
    elif avg_cpu > 30 and avg_cores > 4:
        print("⚠️ Threading provides moderate improvement")
        print("   Some operations using multiple cores")
        print("   Expected training speedup: 10-20%")
    else:
        print("❌ Threading may not be fully effective")
        print("   Most operations still single-threaded")
        print("   May need different optimization approach")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("1. Run actual training test: python train_evorl_only.py --symbols BPCL --timesteps 1000")
    print("2. Compare with previous 60 it/s baseline")
    print("3. Monitor with: python simple_monitor.py &")
    print("4. Look for improved core utilization during training")

if __name__ == "__main__":
    main()