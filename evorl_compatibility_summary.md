# EvoRL Compatibility Issues Summary

## The Problem
EvoRL has multiple dependency conflicts with your current environment:

1. **JAX 0.7.0** removed `jax.experimental.maps` (required by Flax 0.8.0)
2. **TensorFlow Probability 0.25.0** uses deprecated JAX APIs
3. **Brax 0.12.4** is incompatible with JAX 0.7.0

## Solutions

### Option 1: Downgrade to Compatible Versions (Recommended)
```bash
# Uninstall current versions
pip uninstall -y jax jaxlib flax brax tensorflow-probability

# Install compatible versions
pip install "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax==0.7.0
pip install brax==0.9.1
pip install tensorflow-probability==0.20.1
```

### Option 2: Use Without EvoRL
Since EvoRL has compatibility issues, you can use the standard training system:
```bash
python train.py  # Uses stable-baselines3 with PyTorch
```

### Option 3: Create Custom RL Implementation
Use the simplified PPO implementation without EvoRL dependencies:
```python
# See evorl_test_no_brax.py for a basic example
```

## Version Compatibility Matrix

| Library | Current | Compatible | Notes |
|---------|---------|------------|-------|
| JAX | 0.7.0 | 0.4.13 | Need older version for maps API |
| Flax | 0.8.0 | 0.7.0 | Must match JAX version |
| Brax | 0.12.4 | 0.9.1 | Requires older JAX |
| TFP | 0.25.0 | 0.20.1 | Must match JAX version |

## Why This Happens
- JAX made breaking changes between 0.4.x and 0.7.x
- Many RL libraries haven't updated to support newer JAX versions
- The ecosystem is fragmented between different JAX versions

## Recommendation
For your trading system, stick with the working GPU-optimized setup using:
- **stable-baselines3** (PyTorch-based, no JAX issues)
- **Your custom PPO implementation** (bounded_entropy_ppo.py)

This avoids all the JAX compatibility issues while still providing excellent performance.