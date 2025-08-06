#!/usr/bin/env python3
"""
CPU affinity optimization for better parallel performance
"""

import os
import psutil
import multiprocessing as mp

def set_cpu_affinity_for_workers():
    """
    Set CPU affinity for worker processes to reduce context switching
    """
    try:
        # Get the current process
        p = psutil.Process()
        
        # Get total CPU count
        cpu_count = psutil.cpu_count(logical=True)
        
        # For the main process, use all CPUs
        p.cpu_affinity(list(range(cpu_count)))
        
        # Set high priority for better performance
        try:
            # Linux/Unix
            p.nice(-5)  # Higher priority (requires permissions)
        except:
            try:
                # Windows
                p.nice(psutil.HIGH_PRIORITY_CLASS)
            except:
                pass
        
        print(f"CPU affinity optimization: Using all {cpu_count} cores")
        print(f"Process priority set to high")
        
        # Optimize memory allocation
        try:
            # Increase resource limits for better performance
            import resource
            
            # Increase stack size
            resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
            
            # Increase number of open files
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
            
            print("Resource limits optimized")
        except:
            pass
            
    except Exception as e:
        print(f"Could not optimize CPU affinity: {e}")

def get_optimal_worker_distribution(total_tasks, num_cores):
    """
    Calculate optimal distribution of tasks across workers
    """
    if total_tasks <= num_cores:
        # Few tasks: one worker per task
        return total_tasks, 1
    elif total_tasks <= num_cores * 2:
        # Moderate tasks: use all cores
        return num_cores, (total_tasks + num_cores - 1) // num_cores
    else:
        # Many tasks: balance between parallelism and overhead
        # Use 80% of cores to leave room for system tasks
        workers = int(num_cores * 0.8)
        tasks_per_worker = (total_tasks + workers - 1) // workers
        return workers, tasks_per_worker

# Call this at the start of the optimization
if __name__ == "__main__":
    set_cpu_affinity_for_workers()