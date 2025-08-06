# Compatible Versions for CUDA 11.8 + PyTorch 2.7.1

## System Requirements
- **Python**: 3.12.7
- **PyTorch**: 2.7.1+cu118
- **CUDA**: 11.8
- **cuDNN**: 9.1.0.70

## Compatible Package Versions

### For Python 3.12 (Recommended)
- **NumPy**: 1.26.0+ (required for Python 3.12)
- **JAX**: 0.4.23 (supports both Python 3.12 and CUDA 11)
- **Jaxlib**: 0.4.23+cuda11.cudnn86
- **SBX-RL**: 0.18.0
- **Chex**: 0.1.85
- **Optax**: 0.1.8
- **Flax**: 0.8.0
- **tensorflow-probability**: 0.23.0

### For Python 3.11 and below
- **NumPy**: 1.24.3
- **JAX**: 0.4.13
- **Jaxlib**: 0.4.13+cuda11.cudnn86
- **SBX-RL**: 0.12.0
- **Chex**: 0.1.82
- **Optax**: 0.1.7
- **Flax**: 0.7.0
- **tensorflow-probability**: 0.20.1

### CUDA Libraries
- **nvidia-cudnn-cu11**: 8.6.0.163

## Installation

Run the installation script:
```bash
chmod +x install_compatible_jax_sbx.py
python install_compatible_jax_sbx.py
```

## Important Notes

1. **NumPy Version**: NumPy 1.24.3 is critical. Newer versions (1.25+) have breaking changes that are incompatible with JAX 0.4.13.

2. **JAX Version**: JAX 0.4.13 is the last version with robust CUDA 11 support. Newer versions (0.4.14+) primarily support CUDA 12.

3. **Environment Variables**: Required for JAX GPU detection:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   export JAX_PLATFORMS=cuda
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export JAX_ENABLE_X64=true
   export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.8
   ```

4. **Version Conflicts**: If you see dependency conflicts during installation, it's usually due to other packages requiring different NumPy versions. The training system should still work with PyTorch GPU even if JAX falls back to CPU.

## Verification

After installation, verify GPU detection:
```bash
python -c "import jax; print('JAX devices:', jax.devices())"
```

Expected output should show `CudaDevice` or `GpuDevice`.