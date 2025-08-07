#!/usr/bin/env python3
"""
Simple Real-Time Monitor for M4 Max EvoRL Training
Lightweight version without sudo requirements
"""

import time
import os
import psutil
from datetime import datetime

def clear_screen():
    """Clear terminal screen"""
    os.system('clear')

def get_bar(percentage, width=20):
    """Create a progress bar"""
    filled = int(percentage * width / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percentage:5.1f}%"

def monitor_system(interval=2.0):
    """Simple monitoring loop"""
    print("🚀 Starting Simple M4 Max Monitor...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            clear_screen()
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"🖥️  M4 Max EvoRL Monitor - {timestamp}")
            print("=" * 60)
            
            # CPU Stats
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            load_avg = os.getloadavg()
            
            print(f"🔥 CPU ({cpu_count} cores)")
            print(f"   Usage: {get_bar(cpu_percent)}")
            print(f"   Load:  {load_avg[0]:.1f} (1m) | {load_avg[1]:.1f} (5m)")
            
            # Memory Stats
            memory = psutil.virtual_memory()
            memory_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            print(f"\n🧠 Memory (Unified)")
            print(f"   Usage: {get_bar(memory.percent)}")
            print(f"   Used:  {memory_gb:.1f}GB / {memory_total_gb:.0f}GB")
            
            # Process Stats
            python_procs = []
            total_cpu = 0
            total_memory = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if any(keyword in cmdline.lower() for keyword in ['train', 'evorl', 'jax']):
                            cpu_pct = proc.info['cpu_percent'] or 0
                            mem_mb = proc.info['memory_info'].rss / (1024**2)
                            
                            python_procs.append({
                                'pid': proc.info['pid'],
                                'cpu': cpu_pct,
                                'memory': mem_mb,
                                'cmd': cmdline.split()[-1] if cmdline else 'python'
                            })
                            
                            total_cpu += cpu_pct
                            total_memory += mem_mb
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Training Process Stats
            print(f"\n⚡ Training Processes ({len(python_procs)})")
            if python_procs:
                print(f"   Total CPU: {total_cpu:5.1f}%")
                print(f"   Total RAM: {total_memory:6.0f}MB")
                
                # Show top 3 processes
                python_procs.sort(key=lambda x: x['cpu'], reverse=True)
                for proc in python_procs[:3]:
                    print(f"   PID {proc['pid']:5d}: {proc['cpu']:4.1f}% | {proc['memory']:6.0f}MB | {proc['cmd']}")
            else:
                print("   No active training processes")
            
            # Metal GPU Status (simple detection)
            metal_active = len(python_procs) > 0
            print(f"\n🎮 Metal GPU")
            print(f"   Status: {'🟢 ACTIVE' if metal_active else '🔴 IDLE'}")
            print(f"   Backend: Apple Metal (M4 Max)")
            
            # Simple performance indicators
            if cpu_percent > 80:
                print(f"\n🔥 High CPU usage detected!")
            if memory.percent > 90:
                print(f"\n⚠️  High memory usage detected!")
            if total_cpu > 200:  # Multiple cores active
                print(f"\n🚀 Multi-core training active!")
                
            print(f"\n💡 Next update in {interval}s...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n\n👋 Monitor stopped")

if __name__ == "__main__":
    import sys
    
    interval = 2.0
    if len(sys.argv) > 1:
        try:
            interval = float(sys.argv[1])
        except ValueError:
            print("Usage: python simple_monitor.py [interval_seconds]")
            sys.exit(1)
    
    print(f"Simple M4 Max Monitor - {interval}s intervals")
    monitor_system(interval)