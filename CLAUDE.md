# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **GPU-only** algorithmic trading system built with EvoRL (Evolution-based Reinforcement Learning) using pure JAX implementation. The system features hyperparameter optimization via MCMC, real-time tick data processing, and a complete pipeline for training, evaluation, and deployment.

**Core Technologies**: Python, JAX (GPU-only), EvoRL, KiteConnect API, TA-Lib, MCMC optimization

**IMPORTANT**: This project uses ONLY EvoRL with JAX for GPU acceleration. SB3 (Stable-Baselines3) has been deprecated and should NOT be used as it's not GPU-friendly.

### Instructions

1. Do not implement fall backs if you're not able to achieve the primary goal, without getting approval from the user. 

2. Check on the requested initial list of user requests in the prompt, and ensure you've done everything in the list to the best of your ability. 

3. If you're unable to solve the problem, ask the user for additional information or clarifications, or tell them that you're not able to solve the problem the way they intended. It's alright to admit, rather than implement something else, or something wrongly. 

## Development Commands

### Environment Setup
```bash
# Install dependencies (requires TA-Lib system library)
# Ubuntu/Debian: sudo apt-get install libta-lib-dev
# macOS: brew install ta-lib
pip install -r requirements.txt

# Install JAX with GPU support
pip install --upgrade 'jax[cuda12_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Virtual environment is in venv/ - activate if needed
source venv/bin/activate
```

### Training with EvoRL (GPU-Only)
```bash
# Main training command - Pure EvoRL/JAX implementation
python train_evorl_only.py --symbols BPCL HDFCLIFE --test-days 30

# Quick test run
python train_evorl_only.py --symbols BPCL --timesteps 1000 --test-days 10

# Full training with deployment
python train_evorl_only.py --symbols BPCL HDFCLIFE TATASTEEL --timesteps 3000000 --test-days 90 --deploy

# Train with custom date split
python train_evorl_only.py --symbols BPCL --train-end-date 2024-01-01 --test-days 60

# Skip data preprocessing (use existing features)
python train_evorl_only.py --symbols BPCL --no-preprocessing --test-days 30
```

### Testing
```bash
# Test JAX GPU setup
python test_gpu_setup.py

# Test EvoRL implementation
python test_evorl_performance.py

# Test posterior analysis compatibility
python test_evorl_posterior_integration.py

# Run specific test suites
pytest tests/test_bounded_entropy.py -v
pytest tests/test_trading_improvements.py -v
```

### Data Preparation
```bash
# Prepare data (requires Kite login)
python data_preparation.py

# Generate optimized signals
python optimized_signal_generator_cpu.py
```

### Hyperparameter Optimization
```bash
# MCMC optimization with EvoRL
python hyperparameter_optimizer.py

# Quick test mode
python run_hyperparameter_tuning.py --quick-test
```

### Live Trading
```bash
# Deploy trained EvoRL models for live trading
python trader.py  # Requires Kite authentication
```

## Typical Workflow

### 1. Initial Setup
```bash
# Test JAX GPU setup
python test_gpu_setup.py

# Should show:
# ✅ JAX GPU initialized: cuda:0
```

### 2. Data Preparation
```bash
# Prepare trading data
python data_preparation.py
```

### 3. Training with Test Evaluation
```bash
# Train and evaluate on test period
python train_evorl_only.py --symbols BPCL HDFCLIFE --test-days 60

# Output includes:
# - Training metrics
# - Test period performance (Sharpe, returns, drawdown)
# - Deployment-ready models
```

### 4. Deploy for Live Trading
```bash
# Save models for deployment
python train_evorl_only.py --symbols BPCL --deploy

# Use in live trading
python trader.py
```

## Architecture (EvoRL-Only)

### Core Components

**EvoRL PPO Implementation (`evorl_ppo_trainer.py`)**
- Pure JAX implementation with continuous actions
- GPU-optimized neural networks (512, 256, 128 hidden dims)
- Generalized Advantage Estimation (GAE)
- No CPU/GPU transfer overhead

**Complete Pipeline (`evorl_complete_pipeline.py`)**
- Training: GPU-accelerated PPO training
- Testing: Automatic performance evaluation on holdout period
- Deployment: Real-time trading decisions with confidence scores

**Main Training Script (`train_evorl_only.py`)**
- Clean CLI interface
- No SB3 dependencies
- Automatic JAX GPU initialization
- Complete workflow integration

**Parameters (`parameters.py`)**
- Central configuration for all hyperparameters
- GPU-optimized batch sizes and learning rates
- Symbol lists and trading parameters

### Data Processing

**Signal Generation**
- Technical indicators: MACD, RSI, Bollinger Bands, Stochastic
- Market regime detection
- Quantile-based feature normalization
- Lag-based return features

**Feature Engineering**
- 110+ technical features
- GPU-friendly feature vectors
- Optimized for JAX processing

## Key Files

**Core EvoRL Implementation**
- `evorl_ppo_trainer.py` - Pure JAX PPO implementation
- `evorl_complete_pipeline.py` - Full training/testing/deployment pipeline
- `train_evorl_only.py` - Main training script (use this!)
- `evorl_sb3_compatibility.py` - Compatibility bridge for posterior analysis

**Configuration**
- `parameters.py` - Central configuration
- `requirements.txt` - Python dependencies

**Environment & Data**
- `StockTradingEnv2.py` - Trading environment
- `lib.py`, `common.py` - Utility functions
- `data_preparation.py` - Data preprocessing
- `optimized_signal_generator_cpu.py` - Feature generation

**Trading**
- `trader.py` - Live trading implementation
- `kitelogin.py` - Kite authentication

**Testing & Monitoring**
- `test_evorl_performance.py` - Performance testing
- `monitor_training.py` - Training monitor

**Models & Data**
- `models/` - Trained EvoRL models (*.pkl) and normalizers
- `traindata/` - Training datasets
- `tmp/` - Temporary files and checkpoints

## Performance Optimization

### GPU Settings (RTX 4080)
- Batch size: 512 (optimal for 12GB VRAM)
- Parallel environments: 32-64
- JIT compilation: Enabled for all critical functions
- Memory management: Automatic with JAX

### JAX Configuration
```bash
# Set before running
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Training Performance Metrics

The EvoRL implementation provides comprehensive performance evaluation:

```python
{
    'total_return': 0.15,      # 15% return on test period
    'sharpe_ratio': 1.85,      # Risk-adjusted performance
    'sortino_ratio': 2.34,     # Downside risk-adjusted
    'max_drawdown': -0.08,     # Maximum portfolio loss
    'win_rate': 0.58,          # Percentage of winning days
    'calmar_ratio': 1.88,      # Return/MaxDrawdown ratio
    'num_trades': 145          # Trading activity
}
```

## Deployment

After training, models can be deployed for real-time trading:

```python
# Example deployment usage
from evorl_complete_pipeline import EvoRLCompletePipeline

pipeline = EvoRLCompletePipeline()
decision = pipeline.deploy_model('BPCL', realtime_features)

# Returns:
{
    'action': 'BUY',           # BUY/SELL/HOLD
    'confidence': 0.85,        # Model confidence
    'position_size': 0.75,     # Recommended position
    'timestamp': '2024-01-15'
}
```

## Common Issues and Solutions

### JAX GPU Not Detected
```bash
# Reinstall JAX with CUDA support
pip uninstall -y jax jaxlib
pip install --upgrade 'jax[cuda12_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify
python -c "import jax; print(jax.devices())"
```

### Out of Memory
- Reduce batch_size in parameters.py
- Reduce N_ENVS (parallel environments)
- Use gradient accumulation if needed

### Performance Optimization
- Ensure JIT compilation is working
- Check GPU utilization with nvidia-smi
- Verify no CPU/GPU transfers in training loop

## Migration from SB3

**IMPORTANT**: `train.py` has been deprecated and will show a warning directing to use `train_evorl_only.py`. 

All functionality has been migrated to EvoRL:
- ✅ Training → `train_evorl_only.py`
- ✅ Testing → Automatic test period evaluation
- ✅ Deployment → Built into pipeline
- ✅ GPU acceleration → Pure JAX (5-10x faster)

## Summary

This project is now a **pure EvoRL/JAX implementation** optimized for GPU-only execution. The system provides:

1. **Maximum Performance**: 5-10x faster than SB3
2. **Complete Pipeline**: Training, testing, and deployment
3. **GPU Optimization**: >90% GPU utilization
4. **No Legacy Dependencies**: Pure JAX, no SB3

Always use `train_evorl_only.py` for training - it's the only supported method for scalable GPU performance.