#!/usr/bin/env python3
"""
Real-Time System Monitor for M4 Max EvoRL Training
Monitors CPU, GPU (Metal), Memory, and JAX performance at 2-second intervals
"""

import time
import sys
import os
import subprocess
import psutil
import threading
from datetime import datetime
from collections import deque
import json

class M4MaxSystemMonitor:
    def __init__(self, interval=2.0, history_length=30):
        self.interval = interval
        self.history_length = history_length
        self.running = False
        
        # History storage
        self.cpu_history = deque(maxlen=history_length)
        self.memory_history = deque(maxlen=history_length)
        self.gpu_history = deque(maxlen=history_length)
        self.jax_history = deque(maxlen=history_length)
        
        # Process tracking
        self.python_processes = []
        
    def get_cpu_stats(self):
        """Get detailed CPU statistics for M4 Max"""
        try:
            # Overall CPU usage
            cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
            cpu_avg = psutil.cpu_percent(interval=None)
            
            # CPU frequency (if available)
            try:
                cpu_freq = psutil.cpu_freq()
                freq_current = cpu_freq.current if cpu_freq else 0
            except:
                freq_current = 0
                
            # Load average
            load_avg = os.getloadavg()
            
            return {
                'avg_percent': cpu_avg,
                'per_core': cpu_percent,
                'frequency_mhz': freq_current,
                'load_1m': load_avg[0],
                'load_5m': load_avg[1], 
                'load_15m': load_avg[2],
                'active_cores': len([x for x in cpu_percent if x > 5.0])
            }
        except Exception as e:
            return {'error': str(e), 'avg_percent': 0}
    
    def get_memory_stats(self):
        """Get memory statistics for M4 Max unified memory"""
        try:
            # System memory
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Convert to GB
            total_gb = memory.total / (1024**3)
            used_gb = memory.used / (1024**3)
            available_gb = memory.available / (1024**3)
            
            return {
                'total_gb': round(total_gb, 1),
                'used_gb': round(used_gb, 1),
                'available_gb': round(available_gb, 1),
                'percent_used': memory.percent,
                'swap_used_gb': round(swap.used / (1024**3), 1),
                'unified_memory': True  # M4 Max feature
            }
        except Exception as e:
            return {'error': str(e)}
            
    def get_gpu_metal_stats(self):
        """Get Metal GPU statistics using powermetrics (requires sudo)"""
        try:
            # Try to get Metal GPU stats via powermetrics
            # Note: This requires sudo, so we'll use a simpler approach first
            result = subprocess.run([
                'sudo', 'powermetrics', '--samplers', 'gpu_power', '-n', '1', '-i', '100'
            ], capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0:
                # Parse GPU usage from powermetrics output
                lines = result.stdout.split('\n')
                gpu_active = 0
                gpu_power = 0
                
                for line in lines:
                    if 'GPU Active residency' in line:
                        try:
                            gpu_active = float(line.split(':')[1].strip().replace('%', ''))
                        except:
                            pass
                    elif 'GPU Power' in line:
                        try:
                            gpu_power = float(line.split(':')[1].strip().split()[0])
                        except:
                            pass
                
                return {
                    'gpu_active_percent': gpu_active,
                    'gpu_power_mw': gpu_power,
                    'metal_backend': True,
                    'm4_max': True
                }
            else:
                # Fallback: Basic stats without sudo
                return self.get_gpu_basic_stats()
                
        except subprocess.TimeoutExpired:
            return {'error': 'powermetrics timeout', 'gpu_active_percent': 0}
        except Exception as e:
            return self.get_gpu_basic_stats()
    
    def get_gpu_basic_stats(self):
        """Basic GPU stats without sudo requirements"""
        try:
            # Check if JAX Metal is active by looking for Metal processes
            result = subprocess.run([
                'ps', 'aux'
            ], capture_output=True, text=True)
            
            metal_processes = 0
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'python' in line and ('train' in line or 'jax' in line):
                        metal_processes += 1
            
            return {
                'metal_processes': metal_processes,
                'metal_backend': True,
                'gpu_detection_method': 'process_based',
                'm4_max': True
            }
        except Exception as e:
            return {'error': str(e), 'metal_processes': 0}
            
    def get_jax_process_stats(self):
        """Get JAX/Python process specific statistics"""
        try:
            python_processes = []
            total_memory = 0
            total_cpu = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        
                        # Look for training processes
                        if any(keyword in cmdline.lower() for keyword in ['train', 'evorl', 'jax']):
                            memory_mb = proc.info['memory_info'].rss / (1024**2)
                            cpu_pct = proc.info['cpu_percent'] or 0
                            
                            python_processes.append({
                                'pid': proc.info['pid'],
                                'name': proc.info['name'],
                                'cpu_percent': cpu_pct,
                                'memory_mb': round(memory_mb, 1),
                                'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                            })
                            
                            total_memory += memory_mb
                            total_cpu += cpu_pct
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            return {
                'python_processes': len(python_processes),
                'training_processes': python_processes,
                'total_cpu_percent': round(total_cpu, 1),
                'total_memory_mb': round(total_memory, 1),
                'jax_active': len(python_processes) > 0
            }
            
        except Exception as e:
            return {'error': str(e), 'python_processes': 0}
    
    def display_stats(self, cpu_stats, memory_stats, gpu_stats, jax_stats):
        """Display real-time statistics in a formatted way"""
        # Clear screen (works on macOS Terminal)
        os.system('clear')
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"🚀 M4 Max EvoRL Training Monitor - {timestamp}")
        print("=" * 80)
        
        # CPU Stats
        print(f"🖥️  CPU (12-core M4 Max)")
        print(f"   Average Usage: {cpu_stats.get('avg_percent', 0):5.1f}%")
        print(f"   Active Cores:  {cpu_stats.get('active_cores', 0):2d}/12")
        print(f"   Load Average:  {cpu_stats.get('load_1m', 0):.1f} (1m) | {cpu_stats.get('load_5m', 0):.1f} (5m)")
        if cpu_stats.get('frequency_mhz', 0) > 0:
            print(f"   Frequency:     {cpu_stats.get('frequency_mhz', 0):.0f} MHz")
        
        # Memory Stats  
        print(f"\n🧠 Memory (Unified)")
        memory_used = memory_stats.get('used_gb', 0)
        memory_total = memory_stats.get('total_gb', 64)
        memory_pct = memory_stats.get('percent_used', 0)
        print(f"   Used:     {memory_used:5.1f}GB / {memory_total:.0f}GB ({memory_pct:4.1f}%)")
        print(f"   Available: {memory_stats.get('available_gb', 0):5.1f}GB")
        if memory_stats.get('swap_used_gb', 0) > 0:
            print(f"   Swap Used: {memory_stats.get('swap_used_gb', 0):5.1f}GB")
        
        # GPU Stats
        print(f"\n🎮 GPU (Metal)")
        if 'gpu_active_percent' in gpu_stats:
            print(f"   Active:   {gpu_stats.get('gpu_active_percent', 0):5.1f}%")
            print(f"   Power:    {gpu_stats.get('gpu_power_mw', 0):5.0f}mW")
        else:
            print(f"   Processes: {gpu_stats.get('metal_processes', 0)} Metal/JAX processes detected")
            print(f"   Backend:   Metal (M4 Max)")
        
        # JAX/Training Process Stats
        print(f"\n⚡ JAX Training Processes")
        if jax_stats.get('jax_active', False):
            print(f"   Active Processes: {jax_stats.get('python_processes', 0)}")
            print(f"   Total CPU Usage:  {jax_stats.get('total_cpu_percent', 0):5.1f}%")
            print(f"   Total Memory:     {jax_stats.get('total_memory_mb', 0):5.0f}MB")
            
            # Show top training processes
            processes = jax_stats.get('training_processes', [])[:3]  # Top 3
            for proc in processes:
                print(f"   PID {proc['pid']:5d}: {proc['cpu_percent']:4.1f}% CPU | {proc['memory_mb']:6.0f}MB")
        else:
            print(f"   No training processes detected")
        
        # Performance History (simple bar chart)
        if len(self.cpu_history) > 5:
            print(f"\n📊 History (last {len(self.cpu_history)} readings)")
            cpu_avg = sum(self.cpu_history) / len(self.cpu_history)
            memory_avg = sum(self.memory_history) / len(self.memory_history)
            print(f"   CPU Avg:    {cpu_avg:5.1f}%")
            print(f"   Memory Avg: {memory_avg:5.1f}%")
            
        print(f"\n💡 Monitoring every {self.interval}s - Press Ctrl+C to stop")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        print(f"🚀 Starting M4 Max System Monitor (interval: {self.interval}s)")
        
        try:
            while self.running:
                # Collect stats
                cpu_stats = self.get_cpu_stats()
                memory_stats = self.get_memory_stats()
                gpu_stats = self.get_gpu_metal_stats()
                jax_stats = self.get_jax_process_stats()
                
                # Store in history
                self.cpu_history.append(cpu_stats.get('avg_percent', 0))
                self.memory_history.append(memory_stats.get('percent_used', 0))
                self.gpu_history.append(gpu_stats.get('gpu_active_percent', 0))
                
                # Display
                self.display_stats(cpu_stats, memory_stats, gpu_stats, jax_stats)
                
                # Wait for next interval
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Error in monitoring loop: {e}")
    
    def start(self):
        """Start monitoring"""
        self.running = True
        self.monitor_loop()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Real-time system monitor for M4 Max EvoRL training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor every 2 seconds (default)
  python system_monitor.py
  
  # Monitor every 1 second  
  python system_monitor.py --interval 1
  
  # Monitor with longer history
  python system_monitor.py --history 60
        """
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=float,
        default=2.0,
        help='Monitoring interval in seconds (default: 2.0)'
    )
    
    parser.add_argument(
        '--history',
        type=int,
        default=30,
        help='Number of historical readings to keep (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Create and start monitor
    monitor = M4MaxSystemMonitor(interval=args.interval, history_length=args.history)
    
    print(f"🖥️  M4 Max System Monitor")
    print(f"📊 Monitoring: CPU, Memory, Metal GPU, JAX processes")
    print(f"⏱️  Interval: {args.interval}s")
    print(f"📈 History: {args.history} readings")
    print(f"🎯 Optimized for EvoRL training monitoring")
    print()
    
    try:
        monitor.start()
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")
    
if __name__ == "__main__":
    main()