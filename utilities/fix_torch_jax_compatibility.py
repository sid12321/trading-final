#!/usr/bin/env python3
"""
Fix PyTorch and JAX CUDA compatibility with cuDNN 8.9
Run with: python fix_torch_jax_compatibility.py
"""

import subprocess
import sys
import os

def run_cmd(cmd, description):
    """Run command and handle errors"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    """Main fix routine"""
    print("=== PyTorch + JAX CUDA Compatibility Fix ===")
    
    # Activate venv
    venv_path = "/Users/skumar81/Desktop/Personal/trading-final/venv/bin/activate"
    if not os.path.exists(venv_path):
        print("❌ Virtual environment not found!")
        return
    
    # Step 1: Clean up all conflicting packages
    print("\n📦 Step 1: Cleaning up conflicting packages...")
    cleanup_cmds = [
        f"source {venv_path} && pip uninstall -y torch torchvision torchaudio",
        f"source {venv_path} && pip uninstall -y nvidia-cudnn-cu11 nvidia-cudnn-cu12",
        f"source {venv_path} && pip uninstall -y triton"
    ]
    
    for cmd in cleanup_cmds:
        run_cmd(cmd, "Removing packages")
    
    # Step 2: Install cuDNN 8.9 (JAX compatible)
    print("\n🔧 Step 2: Installing cuDNN 8.9...")
    run_cmd(f"source {venv_path} && pip install nvidia-cudnn-cu12==8.9.7.29", 
            "Installing cuDNN 8.9")
    
    # Step 3: Install PyTorch that works with cuDNN 8.9
    print("\n🔧 Step 3: Installing compatible PyTorch...")
    
    # Try PyTorch 2.3.1 which is more compatible with cuDNN 8.9
    pytorch_cmd = f"""source {venv_path} && pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121"""
    
    if not run_cmd(pytorch_cmd, "Installing PyTorch 2.3.1"):
        # Fallback: Install without specific cuDNN requirements
        print("\n⚠️  Trying fallback PyTorch installation...")
        fallback_cmd = f"source {venv_path} && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        run_cmd(fallback_cmd, "Installing PyTorch fallback")
    
    # Step 4: Test installations
    print("\n🧪 Step 4: Testing installations...")
    
    test_script = f"""
source {venv_path} && python -c "
print('=== Testing JAX ===')
try:
    import jax
    import jaxlib
    print(f'JAX version: {{jax.__version__}}')
    print(f'JAXlib version: {{jaxlib.__version__}}')
    print(f'JAX backend: {{jax.default_backend()}}')
    print(f'JAX devices: {{jax.devices()}}')
    
    import jax.numpy as jnp
    x = jnp.array([1, 2, 3])
    y = jnp.sum(x)
    print(f'JAX test: {{x}} -> sum = {{y}}')
    print('✅ JAX working!')
except Exception as e:
    print(f'❌ JAX failed: {{e}}')

print('/n=== Testing PyTorch ===')
try:
    import torch
    print(f'PyTorch version: {{torch.__version__}}')
    print(f'CUDA available: {{torch.cuda.is_available()}}')
    if torch.cuda.is_available():
        print(f'CUDA version: {{torch.version.cuda}}')
        print(f'GPU device: {{torch.cuda.get_device_name(0)}}')
        
        # Test GPU tensor
        x = torch.randn(2, 3).cuda()
        y = torch.sum(x)
        print(f'PyTorch GPU test: device={{x.device}}, sum={{y.item():.2f}}')
        print('✅ PyTorch CUDA working!')
    else:
        print('❌ PyTorch CUDA not available')
except Exception as e:
    print(f'❌ PyTorch failed: {{e}}')
"
"""
    
    run_cmd(test_script, "Testing installations")
    
    # Step 5: Create optimized environment script
    print("\n📝 Step 5: Creating environment script...")
    
    env_script = """#!/bin/bash
# Optimized CUDA 12 + cuDNN 8.9 Environment
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# JAX settings for cuDNN 8.9
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 4080

# CUDA settings
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

echo "🚀 CUDA 12 + cuDNN 8.9 environment ready!"
echo "JAX Backend: $(python -c 'import jax; print(jax.default_backend())' 2>/dev/null || echo 'Not available')"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Not available')"
"""
    
    with open("/Users/skumar81/Desktop/Personal/trading-final/cuda_env_optimized.sh", "w") as f:
        f.write(env_script)
    
    os.chmod("/Users/skumar81/Desktop/Personal/trading-final/cuda_env_optimized.sh", 0o755)
    
    print("\n✅ Fix completed!")
    print("\n📋 Next steps:")
    print("1. Run: source cuda_env_optimized.sh")
    print("2. Run: source venv/bin/activate") 
    print("3. Test: python test_gpu_setup.py")

if __name__ == "__main__":
    main()