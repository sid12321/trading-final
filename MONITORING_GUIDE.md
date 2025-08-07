# 📊 M4 Max System Monitoring Guide

## **Real-Time Monitoring During EvoRL Training**

I've created two monitoring scripts to track your M4 Max performance during EvoRL training:

---

## 🚀 **Quick Start**

### **Simple Monitor (Recommended)**
```bash
# Activate environment
source /Users/skumar81/.virtualenvs/trading-final/bin/activate

# Start simple monitoring (2s intervals)
python simple_monitor.py

# Custom interval (e.g., 1 second)
python simple_monitor.py 1
```

### **Advanced Monitor (Full Features)**
```bash
# Full monitoring with GPU power metrics (requires sudo for GPU stats)
python system_monitor.py

# Custom interval and history
python system_monitor.py --interval 1 --history 60
```

---

## 📺 **What You'll See**

### **Real-Time Dashboard:**
```
🖥️  M4 Max EvoRL Monitor - 21:45:30
============================================================
🔥 CPU (12 cores)
   Usage: [████████████░░░░░░░░] 60.2%
   Load:  8.1 (1m) | 6.4 (5m)

🧠 Memory (Unified)
   Usage: [██████████████░░░░░░] 71.5%
   Used:  45.8GB / 64GB

⚡ Training Processes (3)
   Total CPU: 180.5%
   Total RAM: 12800MB
   PID 45231: 65.2% | 4200MB | train_evorl_only.py
   PID 45248: 58.1% | 3800MB | python
   PID 45265: 57.2% | 4800MB | python

🎮 Metal GPU
   Status: 🟢 ACTIVE
   Backend: Apple Metal (M4 Max)

💡 Next update in 2s...
```

---

## 📈 **Key Metrics to Watch**

### **🎯 Optimal Training Performance:**
- **CPU Usage**: 60-80% (good utilization without bottleneck)
- **Memory Usage**: 70-85% (utilizing M4 Max unified memory)
- **Training Processes**: 2-4 active Python processes
- **Metal GPU**: 🟢 ACTIVE status

### **⚠️ Performance Issues:**
- **CPU > 95%**: Potential CPU bottleneck
- **Memory > 95%**: Risk of swapping/OOM
- **No Training Processes**: Training not running
- **Metal GPU IDLE**: GPU not being utilized

---

## 🔧 **Usage Scenarios**

### **1. Monitor During Training**
```bash
# Terminal 1: Start monitoring
python simple_monitor.py

# Terminal 2: Start training
python train_evorl_only.py --symbols BPCL --timesteps 50000
```

### **2. Performance Optimization**
```bash
# Monitor with 1-second intervals for detailed analysis
python simple_monitor.py 1

# Watch for:
# - CPU spikes during batch processing
# - Memory allocation patterns
# - Process CPU distribution
```

### **3. Debugging Issues**
```bash
# Advanced monitor for detailed diagnostics
python system_monitor.py --interval 1

# Provides:
# - Per-core CPU usage
# - Process-level statistics
# - Historical averages
# - Metal GPU power consumption (with sudo)
```

---

## 📊 **Interpreting the Data**

### **CPU Utilization:**
- **< 50%**: Underutilized (consider larger batch sizes)
- **50-80%**: Optimal range for training
- **80-95%**: High utilization (good performance)
- **> 95%**: Potential bottleneck (reduce parallel processes)

### **Memory Usage (M4 Max Unified):**
- **< 60%**: Conservative usage
- **60-85%**: Optimal for large batch training
- **85-95%**: High usage (monitor for stability)
- **> 95%**: Risk of system slowdown

### **Training Process Count:**
- **1 Process**: Single-threaded training
- **2-4 Processes**: Multi-process training (optimal)
- **> 10 Processes**: Possible resource contention

---

## 🎮 **Metal GPU Monitoring**

### **Simple Detection:**
- **🟢 ACTIVE**: JAX Metal backend is running
- **🔴 IDLE**: No Metal operations detected

### **Advanced (with sudo):**
- **GPU Active %**: Metal GPU utilization percentage
- **GPU Power**: Power consumption in milliwatts
- **Real-time metrics**: Actual hardware utilization

```bash
# Enable advanced GPU monitoring (requires password)
sudo python system_monitor.py
```

---

## 💡 **Pro Tips**

### **Multi-Terminal Setup:**
```bash
# Terminal 1: System monitoring
python simple_monitor.py

# Terminal 2: Training with progress
python train_evorl_only.py --symbols BPCL --verbose

# Terminal 3: Resource monitoring
htop  # Alternative system view
```

### **Performance Optimization:**
1. **Watch CPU**: Should be 60-80% during training
2. **Monitor Memory**: Aim for 70-85% utilization
3. **Check Process Count**: 2-4 training processes optimal
4. **Verify Metal**: GPU status should be ACTIVE

### **Troubleshooting:**
- **Low CPU usage**: Increase batch size or environments
- **High memory**: Reduce batch size if approaching 95%
- **No GPU activity**: Check JAX Metal configuration
- **Multiple Python processes**: Normal for parallel training

---

## 🚀 **Ready Commands**

```bash
# Quick monitoring while training
python simple_monitor.py &
python train_evorl_only.py --symbols BPCL --test-days 30

# Advanced monitoring
python system_monitor.py --interval 1 &
python train_evorl_only.py --symbols BPCL HDFCLIFE --timesteps 100000

# Stop monitoring
# Press Ctrl+C in the monitoring terminal
```

---

## 📱 **Example Output During Optimized Training**

```
🖥️  M4 Max EvoRL Monitor - 21:46:15
============================================================
🔥 CPU (12 cores)
   Usage: [████████████████░░░░] 82.1%  ← Excellent utilization
   Load:  9.2 (1m) | 8.1 (5m)

🧠 Memory (Unified)
   Usage: [████████████████░░░░] 78.5%  ← Good M4 Max usage
   Used:  50.2GB / 64GB

⚡ Training Processes (4)                ← Multi-process training
   Total CPU: 285.2%                    ← Multi-core utilization
   Total RAM: 18600MB                   ← Large batch processing
   PID 45231: 95.1% | 6200MB | train_evorl_only.py
   PID 45248: 88.3% | 4800MB | python
   PID 45265: 52.8% | 3900MB | python
   PID 45280: 49.0% | 3700MB | python

🎮 Metal GPU
   Status: 🟢 ACTIVE                    ← JAX Metal working
   Backend: Apple Metal (M4 Max)

🚀 Multi-core training active!          ← Performance indicator
```

**This shows optimal performance**: High CPU utilization, good memory usage, active Metal backend, and multi-process training! 🎉

---

## 🎯 **Monitor Your 60 it/s Performance**

Use these monitors to verify your **33% performance improvement** and watch your M4 Max efficiently train EvoRL models at **60 iterations/second**! 💹