#!/usr/bin/env python3
"""
Path update script for trading-final migration from Linux to macOS
Updates all hardcoded paths to use the correct macOS directory structure
"""

import os
import re
import glob
from pathlib import Path

# Path mappings
OLD_PATH = "/Users/skumar81/Desktop/Personal/trading-final"
NEW_PATH = "/Users/skumar81/Desktop/Personal/trading-final"

def update_file_paths(file_path):
    """Update paths in a single file"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    original_content = content
    changes_made = []
    
    # Replace the main hardcoded path
    if OLD_PATH in content:
        content = content.replace(OLD_PATH, NEW_PATH)
        changes_made.append(f"Updated basepath: {OLD_PATH} -> {NEW_PATH}")
    
    # Update sys.path.append calls
    pattern1 = rf"sys\.path\.append\('{re.escape(OLD_PATH)}'\)"
    replacement1 = f"sys.path.append('{NEW_PATH}')"
    if re.search(pattern1, content):
        content = re.sub(pattern1, replacement1, content)
        changes_made.append("Updated sys.path.append")
    
    # Update sys.path.insert calls
    pattern2 = rf"sys\.path\.insert\(0,\s*'{re.escape(OLD_PATH)}'\)"
    replacement2 = f"sys.path.insert(0, '{NEW_PATH}')"
    if re.search(pattern2, content):
        content = re.sub(pattern2, replacement2, content)
        changes_made.append("Updated sys.path.insert")
    
    # Update os.chdir calls
    pattern3 = rf"os\.chdir\('{re.escape(OLD_PATH)}'\)"
    replacement3 = f"os.chdir('{NEW_PATH}')"
    if re.search(pattern3, content):
        content = re.sub(pattern3, replacement3, content)
        changes_made.append("Updated os.chdir")
    
    # Update basepath assignments
    pattern4 = rf"basepath\s*=\s*'{re.escape(OLD_PATH)}'"
    replacement4 = f"basepath = '{NEW_PATH}'"
    if re.search(pattern4, content):
        content = re.sub(pattern4, replacement4, content)
        changes_made.append("Updated basepath assignment")
    
    # Update venv path for utilities
    venv_pattern = rf"venv_path\s*=\s*\"{re.escape(OLD_PATH)}/venv/bin/activate\""
    venv_replacement = f'venv_path = "{NEW_PATH}/venv/bin/activate"'
    if re.search(venv_pattern, content):
        content = re.sub(venv_pattern, venv_replacement, content)
        changes_made.append("Updated venv path")
    
    # Update shell script paths
    shell_pattern = rf'\"?{re.escape(OLD_PATH)}/[^\"]*\.sh\"?'
    def shell_replacement(match):
        old_full_path = match.group(0).strip('"')
        filename = os.path.basename(old_full_path)
        return f'"{NEW_PATH}/{filename}"'
    
    if re.search(shell_pattern, content):
        content = re.sub(shell_pattern, shell_replacement, content)
        changes_made.append("Updated shell script paths")
    
    # Save the file if changes were made
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated {file_path}")
            for change in changes_made:
                print(f"   - {change}")
            return True
        except Exception as e:
            print(f"❌ Error writing {file_path}: {e}")
            return False
    else:
        return False

def update_paths_in_directory(directory):
    """Update paths in all Python files in the directory"""
    
    directory = Path(directory)
    updated_files = []
    
    # Find all Python files
    python_files = list(directory.rglob("*.py"))
    
    print(f"Found {len(python_files)} Python files to process...")
    print()
    
    for py_file in python_files:
        # Skip certain directories
        if any(skip_dir in str(py_file) for skip_dir in ['__pycache__', '.git', 'venv', 'env']):
            continue
            
        if update_file_paths(str(py_file)):
            updated_files.append(str(py_file))
    
    return updated_files

def create_dynamic_path_solution():
    """Create a more robust path detection solution"""
    
    # Create a new common_paths.py file for dynamic path detection
    common_paths_content = '''"""
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
        "/Users/skumar81/Desktop/Personal/trading-final",            # Linux (old)
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
'''
    
    with open("/Users/skumar81/Desktop/Personal/trading-final/common_paths.py", 'w') as f:
        f.write(common_paths_content)
    
    print("✅ Created common_paths.py for dynamic path detection")

def main():
    """Main function to update all paths"""
    
    trading_final_dir = "/Users/skumar81/Desktop/Personal/trading-final"
    
    if not os.path.exists(trading_final_dir):
        print(f"❌ Directory {trading_final_dir} does not exist!")
        return
    
    print("🔄 Updating file paths in trading-final...")
    print(f"Changing {OLD_PATH} → {NEW_PATH}")
    print("=" * 60)
    
    # Update paths in all Python files
    updated_files = update_paths_in_directory(trading_final_dir)
    
    # Create dynamic path solution
    create_dynamic_path_solution()
    
    print("=" * 60)
    print(f"✅ Path update complete!")
    print(f"Updated {len(updated_files)} files:")
    
    for file_path in updated_files:
        rel_path = os.path.relpath(file_path, trading_final_dir)
        print(f"   - {rel_path}")
    
    print()
    print("💡 Recommendations:")
    print("1. Test the updated paths by running: python3 train_evorl_only.py --help")
    print("2. Consider using the new common_paths.py for dynamic path detection")
    print("3. Set environment variable: export TRADING_FINAL_PATH=/Users/skumar81/Desktop/Personal/trading-final")

if __name__ == "__main__":
    main()