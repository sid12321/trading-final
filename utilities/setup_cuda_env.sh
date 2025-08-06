#!/bin/bash
# CUDA 12 Environment Setup for JAX and PyTorch
# Run with: source setup_cuda_env.sh

echo "Setting up CUDA 12 environment..."

# CUDA 12.6 paths
export CUDA_VERSION=12.6
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# JAX CUDA settings
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

# Force JAX to ignore cuDNN version mismatch (if needed)
export TF_CPP_MIN_LOG_LEVEL=1
export JAX_SKIP_SLOW_TESTS=true

# PyTorch settings  
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 4080 architecture
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Additional CUDA settings
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

echo "CUDA 12 environment configured!"
echo "CUDA Home: $CUDA_HOME"
echo "JAX Platform: $JAX_PLATFORMS"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"