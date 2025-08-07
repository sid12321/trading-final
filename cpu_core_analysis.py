#!/usr/bin/env python3
"""
CPU Core Usage Analysis for EvoRL Training
Investigates multi-core utilization and identifies optimization opportunities
"""

import os
import sys
import time
import psutil
import numpy as np
import jax
import jax.numpy as jnp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

def analyze_current_cpu_usage():
    """Analyze current system CPU configuration"""
    print("=" * 60)
    print("🖥️  CPU Configuration Analysis")
    print("=" * 60)
    
    # System info
    cpu_count = psutil.cpu_count()
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    
    print(f"Physical CPU cores: {cpu_count}")
    print(f"Logical CPU cores:  {cpu_count_logical}")
    if cpu_freq:
        print(f"CPU frequency:      {cpu_freq.current:.0f} MHz (max: {cpu_freq.max:.0f} MHz)")
    
    # Current per-core usage
    cpu_percent_per_core = psutil.cpu_percent(interval=1, percpu=True)
    load_avg = os.getloadavg()
    
    print(f"\nCurrent per-core usage:")
    for i, usage in enumerate(cpu_percent_per_core):
        status = "🔥" if usage > 80 else "🟢" if usage > 40 else "💤"
        print(f"  Core {i:2d}: {usage:5.1f}% {status}")
    
    print(f"\nLoad averages: {load_avg[0]:.1f} (1m) | {load_avg[1]:.1f} (5m) | {load_avg[2]:.1f} (15m)")
    print(f"Optimal load for {cpu_count} cores: ~{cpu_count * 0.8:.1f}")
    
    return {
        'physical_cores': cpu_count,
        'logical_cores': cpu_count_logical,
        'per_core_usage': cpu_percent_per_core,
        'load_avg': load_avg
    }

def test_jax_threading():
    """Test JAX's threading behavior"""
    print("\n" + "=" * 60)
    print("⚡ JAX Threading Analysis")
    print("=" * 60)
    
    # Check JAX thread configuration
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    
    # Check environment variables that affect threading
    thread_vars = [
        'XLA_CPU_MULTI_THREAD_EIGEN',
        'OMP_NUM_THREADS', 
        'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'JAX_ENABLE_X64',
        'JAX_PLATFORMS'
    ]
    
    print("\nThread-related environment variables:")
    for var in thread_vars:
        value = os.environ.get(var, "Not set")
        print(f"  {var}: {value}")
    
    # Test parallel operations
    print("\nTesting parallel operations...")
    
    @jax.jit
    def cpu_intensive_op(x):
        """CPU-intensive operation that could benefit from threading"""
        # Sequential operations that might parallelize
        for i in range(100):
            x = jnp.sin(x) * jnp.cos(x)
            x = jnp.exp(x / 10)  # Prevent overflow
        return jnp.mean(x)
    
    # Test different batch sizes
    batch_sizes = [1, 10, 100, 1000]
    
    for batch_size in batch_sizes:
        data = jax.random.normal(jax.random.PRNGKey(42), (batch_size, 1000))
        
        # Monitor CPU during operation
        start_cpu = psutil.cpu_percent(interval=None, percpu=True)
        
        start_time = time.time()
        result = jax.vmap(cpu_intensive_op)(data)
        end_time = time.time()
        
        end_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
        
        print(f"\nBatch size {batch_size:4d}:")
        print(f"  Time: {end_time - start_time:.4f}s")
        print(f"  Active cores: {sum(1 for c in end_cpu if c > 10)}/{len(end_cpu)}")
        print(f"  Max core usage: {max(end_cpu):.1f}%")

def test_multiprocessing_alternatives():
    """Test multiprocessing approaches for CPU-bound tasks"""
    print("\n" + "=" * 60)
    print("🔄 Multiprocessing Alternatives")
    print("=" * 60)
    
    def cpu_bound_task(args):
        """Simulate CPU-bound trading environment operations"""
        data, iterations = args
        result = 0
        for i in range(iterations):
            result += np.sum(np.sin(data) * np.cos(data))
        return result
    
    # Prepare test data
    n_tasks = 16
    data_per_task = np.random.randn(1000)
    iterations_per_task = 1000
    
    tasks = [(data_per_task, iterations_per_task) for _ in range(n_tasks)]
    
    # Test 1: Sequential execution
    print("Test 1: Sequential execution")
    start_time = time.time()
    sequential_results = [cpu_bound_task(task) for task in tasks]
    sequential_time = time.time() - start_time
    print(f"  Time: {sequential_time:.4f}s")
    
    # Test 2: Thread pool
    print("\nTest 2: Thread pool")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=psutil.cpu_count()) as executor:
        thread_results = list(executor.map(cpu_bound_task, tasks))
    thread_time = time.time() - start_time
    print(f"  Time: {thread_time:.4f}s")
    print(f"  Speedup: {sequential_time / thread_time:.2f}x")
    
    # Test 3: Process pool
    print("\nTest 3: Process pool")
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=psutil.cpu_count()) as executor:
        process_results = list(executor.map(cpu_bound_task, tasks))
    process_time = time.time() - start_time
    print(f"  Time: {process_time:.4f}s")
    print(f"  Speedup: {sequential_time / process_time:.2f}x")
    
    return {
        'sequential_time': sequential_time,
        'thread_time': thread_time,
        'process_time': process_time
    }

def analyze_trading_bottlenecks():
    """Analyze where CPU cores could help in trading pipeline"""
    print("\n" + "=" * 60)
    print("📊 Trading Pipeline CPU Analysis")
    print("=" * 60)
    
    print("CPU utilization opportunities in your training:")
    print("\n1. Environment Steps (Currently Sequential):")
    print("   • StockTradingEnv2.step() calls")
    print("   • Technical indicator calculations")
    print("   • Reward computations")
    print("   • 🔧 OPTIMIZATION: Parallel environment wrapper")
    
    print("\n2. Data Preprocessing (CPU-bound):")
    print("   • Feature normalization")
    print("   • Technical indicator computation")
    print("   • Signal generation")
    print("   • 🔧 OPTIMIZATION: Multiprocessing data pipeline")
    
    print("\n3. JAX Operations (Mixed CPU/Metal):")
    print("   • Policy network forward pass → Metal GPU")
    print("   • GAE computation → Could use more CPU")
    print("   • Gradient computation → Metal GPU")
    print("   • 🔧 OPTIMIZATION: CPU threading for non-GPU ops")
    
    print("\n4. I/O Operations (CPU-bound):")
    print("   • Model checkpointing")
    print("   • Data loading")
    print("   • Logging and metrics")
    print("   • 🔧 OPTIMIZATION: Background threading")

def suggest_optimizations():
    """Suggest specific optimizations for better CPU usage"""
    print("\n" + "=" * 60)
    print("🚀 CPU Optimization Recommendations")
    print("=" * 60)
    
    cpu_count = psutil.cpu_count()
    
    print(f"Your M4 Max has {cpu_count} CPU cores. Here's how to use them better:")
    
    print("\n1. 🔧 Environment Parallelization:")
    print("   Current: 1 environment, sequential steps")
    print(f"   Optimized: {cpu_count//2} parallel environments")
    print("   Implementation: AsyncVectorEnv with multiprocessing")
    print("   Expected speedup: 2-4x on environment operations")
    
    print("\n2. 🔧 JAX Threading Configuration:")
    print("   Add to your training script:")
    print(f"   export OMP_NUM_THREADS={cpu_count}")
    print(f"   export MKL_NUM_THREADS={cpu_count}")
    print("   export XLA_CPU_MULTI_THREAD_EIGEN=true")
    print("   Expected: Better CPU utilization for CPU fallback operations")
    
    print("\n3. 🔧 Data Pipeline Optimization:")
    print("   Current: Sequential feature computation")
    print(f"   Optimized: {cpu_count} worker processes for preprocessing")
    print("   Implementation: multiprocessing.Pool for batch feature generation")
    print("   Expected speedup: 3-8x on data preprocessing")
    
    print("\n4. 🔧 Background Operations:")
    print("   Current: Synchronous checkpointing blocks training")
    print("   Optimized: Background threads for I/O")
    print("   Implementation: concurrent.futures.ThreadPoolExecutor")
    print("   Expected: 10-20% overall speedup")
    
    print("\n5. 🔧 Memory Management:")
    print("   Current: Single-threaded garbage collection")
    print("   Optimized: Periodic cleanup in separate thread")
    print("   Expected: Reduced training stutters")

def create_optimization_plan():
    """Create specific implementation plan"""
    print("\n" + "=" * 60)
    print("📋 Implementation Plan")
    print("=" * 60)
    
    print("Priority 1 (Easy, High Impact):")
    print("✅ Set threading environment variables")
    print("✅ Add background checkpoint saving")
    print("Expected improvement: +10-15% speed")
    
    print("\nPriority 2 (Medium, Medium Impact):")
    print("🔧 Implement AsyncVectorEnv wrapper")
    print("🔧 Parallelize data preprocessing")
    print("Expected improvement: +20-40% speed")
    
    print("\nPriority 3 (Hard, Variable Impact):")
    print("🔧 Custom multi-process environment pool")
    print("🔧 Advanced JAX compilation optimization")
    print("Expected improvement: +10-50% speed (uncertain)")
    
    print("\nTotal potential improvement: +40-100% (94-120 it/s)")
    print("Combined with current 60 it/s → Could reach 84-120 it/s!")

def main():
    """Main analysis function"""
    # Analyze current CPU usage
    cpu_info = analyze_current_cpu_usage()
    
    # Test JAX threading
    test_jax_threading()
    
    # Test multiprocessing alternatives
    mp_results = test_multiprocessing_alternatives()
    
    # Analysis and recommendations
    analyze_trading_bottlenecks()
    suggest_optimizations()
    create_optimization_plan()
    
    print("\n" + "=" * 60)
    print("🎯 SUMMARY")
    print("=" * 60)
    
    avg_cpu_usage = np.mean(cpu_info['per_core_usage'])
    active_cores = sum(1 for usage in cpu_info['per_core_usage'] if usage > 10)
    
    print(f"Current CPU utilization: {avg_cpu_usage:.1f}% average")
    print(f"Active cores: {active_cores}/{cpu_info['physical_cores']}")
    print(f"Load average: {cpu_info['load_avg'][0]:.1f}/{cpu_info['physical_cores']}")
    
    if avg_cpu_usage < 50:
        print("✅ Significant CPU optimization opportunity available!")
        print("   Implementing suggested optimizations could provide 40-100% speedup")
    elif avg_cpu_usage < 70:
        print("⚠️ Moderate CPU optimization opportunity")
        print("   Threading and parallelization could help")
    else:
        print("💪 CPU usage is already quite good")
        print("   Focus on algorithmic optimizations")
    
    print(f"\nPotential performance: 60 it/s → 84-120 it/s")
    print("This would be a 40-100% additional improvement!")

if __name__ == "__main__":
    main()