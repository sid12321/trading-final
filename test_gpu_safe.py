#!/usr/bin/env python3
"""
Safe GPU test that handles segmentation faults and library conflicts
"""

import os
import sys
import subprocess

def test_pytorch_gpu():
    """Test PyTorch GPU separately"""
    print("🔍 Testing PyTorch GPU Setup")
    print("-" * 40)
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.current_device()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            
            # Memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU memory: {total_memory:.1f}GB")
            
            # Test computation
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x)
            print("✅ GPU matrix multiplication test passed")
            return True
        else:
            print("❌ CUDA not available")
            return False
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False

def test_jax_in_subprocess():
    """Test JAX in a subprocess to avoid segfaults"""
    print("\n🔍 Testing JAX GPU Setup (in subprocess)")
    print("-" * 40)
    
    # Create a test script
    test_script = """
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

try:
    import jax
    print(f"JAX version: {jax.__version__}")
    devices = jax.devices()
    backend = jax.default_backend()
    print(f"JAX backend: {backend}")
    print(f"JAX devices: {devices}")
    
    if 'gpu' in backend.lower() or any('cuda' in str(d).lower() for d in devices):
        print("✅ JAX GPU detected")
        # Test computation
        x = jax.numpy.ones((100, 100))
        y = jax.numpy.dot(x, x)
        print(f"✅ JAX GPU operation test passed (result: {y[0,0]})")
    else:
        print("⚠️ JAX using CPU backend")
except Exception as e:
    print(f"❌ JAX error: {e}")
    import traceback
    traceback.print_exc()
"""
    
    # Write test script
    with open('_test_jax_gpu.py', 'w') as f:
        f.write(test_script)
    
    # Run in subprocess
    try:
        result = subprocess.run(
            [sys.executable, '_test_jax_gpu.py'],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Clean up
        os.remove('_test_jax_gpu.py')
        
        return result.returncode == 0 and 'GPU detected' in result.stdout
    except subprocess.TimeoutExpired:
        print("❌ JAX test timed out")
        return False
    except Exception as e:
        print(f"❌ Subprocess error: {e}")
        return False

def test_sbx_import():
    """Test SBX import"""
    print("\n🔍 Testing SBX Import")
    print("-" * 40)
    try:
        import sbx
        print("✅ SBX imported successfully")
        print(f"SBX version: {sbx.__version__}")
        
        # Try importing PPO
        from sbx import PPO
        print("✅ SBX PPO available")
        return True
    except Exception as e:
        print(f"❌ SBX import failed: {e}")
        return False

def check_cuda_environment():
    """Check CUDA environment setup"""
    print("\n🔍 Checking CUDA Environment")
    print("-" * 40)
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi working:")
            print(f"   {result.stdout.strip()}")
    except:
        print("❌ nvidia-smi not found")
    
    # Check CUDA paths
    cuda_paths = [
        '/usr/local/cuda',
        '/usr/local/cuda-11.8',
        '/usr/local/cuda-12.0'
    ]
    
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"✅ Found CUDA installation: {path}")
            
    # Check environment variables
    env_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value[:50]}..." if len(value) > 50 else f"   {var}: {value}")

def main():
    print("🧪 Safe GPU Setup Test")
    print("=" * 50)
    
    # Test each component
    pytorch_ok = test_pytorch_gpu()
    jax_ok = test_jax_in_subprocess()
    sbx_ok = test_sbx_import()
    check_cuda_environment()
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"   PyTorch GPU: {'✅ Working' if pytorch_ok else '❌ Not working'}")
    print(f"   JAX GPU: {'✅ Working' if jax_ok else '⚠️ Not working (CPU fallback available)'}")
    print(f"   SBX: {'✅ Available' if sbx_ok else '❌ Not available'}")
    
    if pytorch_ok:
        print("\n✅ You can train with GPU acceleration using:")
        print("   python train.py --device cuda")
    
    if not jax_ok:
        print("\n⚠️ JAX GPU not working. Possible fixes:")
        print("   1. Run: python install_jax_cuda11_py312.py")
        print("   2. Source environment: source setup_jax_env.sh")
        print("   3. Use PyTorch GPU mode: python train.py --device cuda")

if __name__ == "__main__":
    main()