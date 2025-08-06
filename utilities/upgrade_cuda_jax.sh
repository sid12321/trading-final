#!/bin/bash

# CUDA 12 and JAX 0.4.26 Installation Script
# Run with: bash upgrade_cuda_jax.sh

set -e  # Exit on any error

echo "=== CUDA 12 and JAX 0.4.26 Installation Script ==="
echo "Starting at: $(date)"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print colored output
print_status() {
    echo -e "\n\033[1;34m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\n\033[1;31m[ERROR]\033[0m $1"
}

print_success() {
    echo -e "\n\033[1;32m[SUCCESS]\033[0m $1"
}

# Check current status
print_status "Checking current system status..."
echo "Current CUDA driver version:"
nvidia-smi | grep "CUDA Version" || echo "nvidia-smi not available"

echo -e "\nCurrent CUDA toolkit version:"
nvcc --version 2>/dev/null || echo "nvcc not found"

echo -e "\nCurrent JAX status:"
python -c "import jax; print('JAX version:', jax.__version__)" 2>/dev/null || echo "JAX not working"

# Step 1: Install CUDA 12 toolkit
print_status "Step 1: Installing CUDA 12 toolkit..."

# Remove old CUDA toolkit if present
print_status "Removing old CUDA installations..."
sudo apt-get remove --purge nvidia-* -y 2>/dev/null || true
sudo apt-get remove --purge cuda-* -y 2>/dev/null || true
sudo apt-get autoremove -y 2>/dev/null || true

# Download and install CUDA 12.6
print_status "Downloading CUDA 12.6 installer..."
cd /tmp
wget -q --show-progress https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run

print_status "Installing CUDA 12.6 (this may take 10-15 minutes)..."
sudo chmod +x cuda_12.6.3_560.35.05_linux.run
sudo ./cuda_12.6.3_560.35.05_linux.run --silent --toolkit --samples --no-opengl-libs

# Add CUDA to PATH
print_status "Setting up CUDA environment variables..."
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' | sudo tee -a /etc/environment
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/environment

# Source for current session
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

print_success "CUDA 12.6 installation completed!"

# Verify CUDA installation
print_status "Verifying CUDA installation..."
nvcc --version

# Step 2: Install JAX and JAXlib 0.4.26 with CUDA 12 support
print_status "Step 2: Installing JAX and JAXlib 0.4.26 with CUDA 12 support..."

# Navigate to project directory
cd /Users/skumar81/Desktop/Personal/trading-final

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    print_status "Activating virtual environment..."
    source venv/bin/activate
fi

# Uninstall existing JAX installations
print_status "Removing existing JAX installations..."
pip uninstall -y jax jaxlib 2>/dev/null || true

# Install JAX 0.4.26 with CUDA 12 support
print_status "Installing JAX 0.4.26 with CUDA 12 support..."
pip install --upgrade "jax[cuda12_pip]==0.4.26" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Alternative method if the above fails
if ! python -c "import jax; import jaxlib; print('JAX version:', jax.__version__); print('JAXlib version:', jaxlib.__version__)" 2>/dev/null; then
    print_status "Primary installation failed, trying alternative method..."
    pip install jax==0.4.26 jaxlib==0.4.26+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

print_success "JAX installation completed!"

# Step 3: Verify installation
print_status "Step 3: Verifying JAX CUDA installation..."

# Test JAX CUDA detection
python << 'EOF'
import jax
import jaxlib
print(f"JAX version: {jax.__version__}")
print(f"JAXlib version: {jaxlib.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")

# Test CUDA functionality
try:
    import jax.numpy as jnp
    x = jnp.array([1, 2, 3])
    y = jnp.sum(x)
    print(f"JAX computation test: {x} -> sum = {y}")
    print("✅ JAX is working correctly!")
except Exception as e:
    print(f"❌ JAX test failed: {e}")
EOF

# Step 4: Update PyTorch if needed
print_status "Step 4: Updating PyTorch for CUDA 12 compatibility..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Test PyTorch CUDA
python << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print("✅ PyTorch CUDA is working!")
else:
    print("❌ PyTorch CUDA not available")
EOF

# Step 5: Set environment variables for optimal performance
print_status "Step 5: Setting up optimal environment variables..."

# Create environment setup script
cat > setup_cuda_env.sh << 'EOF'
#!/bin/bash
# CUDA 12 Environment Setup
export CUDA_VERSION=12.6
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# JAX CUDA settings
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# PyTorch settings
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 4080 architecture

echo "CUDA 12 environment configured!"
EOF

chmod +x setup_cuda_env.sh

print_success "Environment setup script created: setup_cuda_env.sh"

# Final verification
print_status "Final system verification..."
echo "=== System Status ==="
echo "CUDA Driver: $(nvidia-smi | grep "CUDA Version" | awk '{print $9}')"
echo "CUDA Toolkit: $(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)"
echo "JAX version: $(python -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'Not available')"
echo "JAXlib version: $(python -c 'import jaxlib; print(jaxlib.__version__)' 2>/dev/null || echo 'Not available')"
echo "JAX backend: $(python -c 'import jax; print(jax.default_backend())' 2>/dev/null || echo 'Not available')"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Not available')"

print_success "Installation completed! Remember to:"
print_success "1. Restart your terminal or run: source setup_cuda_env.sh"
print_success "2. Test the installation with: python test_gpu_setup.py"

echo -e "\nInstallation finished at: $(date)"