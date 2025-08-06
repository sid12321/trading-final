"""
Dynamic path configuration for trading-final
Automatically detects the correct base path regardless of operating system
"""

import os
import sys
from pathlib import Path

def get_project_root():
    """
    Get the project root directory dynamically
    Works on both Linux and macOS
    """
    # Method 1: Use __file__ to find the current script location
    current_file = Path(__file__).resolve()
    
    # Look for key files that indicate we're in the trading-final directory
    key_files = ['train_evorl_only.py', 'parameters.py', 'common.py']
    
    # Start from current file location and go up until we find the project root
    current_dir = current_file.parent
    for _ in range(5):  # Limit search to 5 levels up
        if all((current_dir / file).exists() for file in key_files):
            return str(current_dir)
        current_dir = current_dir.parent
    
    # Method 2: Check common locations
    common_locations = [
        "/Users/skumar81/Desktop/Personal/trading-final",  # macOS
        "/home/sid12321/Desktop/Trading-Final",            # Linux (old)
        "/home/sid12321/Desktop/Personal/trading-final",   # Linux (new)
    ]
    
    for location in common_locations:
        if Path(location).exists() and all(Path(location, file).exists() for file in key_files):
            return location
    
    # Method 3: Use environment variable if set
    if 'TRADING_FINAL_PATH' in os.environ:
        env_path = os.environ['TRADING_FINAL_PATH']
        if Path(env_path).exists():
            return env_path
    
    # Fallback: current working directory
    cwd = Path.cwd()
    if all((cwd / file).exists() for file in key_files):
        return str(cwd)
    
    # If all else fails, use the detected path
    return "/Users/skumar81/Desktop/Personal/trading-final"

# Global basepath variable
basepath = get_project_root()

def setup_project_environment():
    """Setup the project environment with correct paths"""
    global basepath
    
    # Add project root to Python path
    if basepath not in sys.path:
        sys.path.insert(0, basepath)
    
    # Change to project directory
    try:
        os.chdir(basepath)
    except OSError as e:
        print(f"Warning: Could not change to directory {basepath}: {e}")
    
    # Create necessary directories
    directories = [
        'models',
        'tmp/sb3_log',
        'tmp/checkpoints', 
        'tmp/tensorboard_logs',
        'traindata'
    ]
    
    for directory in directories:
        full_path = Path(basepath) / directory
        full_path.mkdir(parents=True, exist_ok=True)
    
    return basepath

if __name__ == "__main__":
    print(f"Project root: {get_project_root()}")
    print(f"Setting up environment...")
    setup_project_environment()
    print("✅ Environment setup complete")
