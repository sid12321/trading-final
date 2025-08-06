#!/bin/bash
# Script to set up JAX GPU environment variables

echo "Setting up JAX GPU environment variables..."

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=/home/sid12321/Desktop/Trading-Final/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

echo "Environment variables set:"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  JAX_PLATFORMS=$JAX_PLATFORMS"
echo "  XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE"
echo "  LD_LIBRARY_PATH includes cudnn libs"

echo ""
echo "To use JAX with GPU, run your scripts after sourcing this file:"
echo "  source setup_jax_gpu_env.sh"
echo "  python train.py"