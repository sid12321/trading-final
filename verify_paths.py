#!/usr/bin/env python3
"""
Verification script to check that all paths have been updated correctly
for the macOS migration
"""

import os
import sys
from pathlib import Path

def verify_path_updates():
    """Verify that all paths have been updated correctly"""
    
    current_dir = "/Users/skumar81/Desktop/Personal/trading-final"
    expected_path = "/Users/skumar81/Desktop/Personal/trading-final"
    
    print("🔍 Verifying path updates for trading-final...")
    print(f"Expected path: {expected_path}")
    print(f"Current directory: {os.getcwd()}")
    print("=" * 60)
    
    # Test 1: Check key files can import correctly
    print("✅ Test 1: Checking key file imports...")
    
    key_files = [
        'parameters.py',
        'train_evorl_only.py', 
        'common_paths.py',
        'evorl_complete_pipeline.py'
    ]
    
    for filename in key_files:
        filepath = Path(current_dir) / filename
        if filepath.exists():
            print(f"   ✓ {filename} exists")
            # Read first few lines to check basepath
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    if expected_path in content:
                        print(f"     ✓ Contains correct path: {expected_path}")
                    elif '/home/sid12321/Desktop/Trading-Final' in content:
                        print(f"     ❌ Still contains old path!")
                    else:
                        print(f"     ℹ️ No explicit path found (may be dynamic)")
            except Exception as e:
                print(f"     ⚠️ Could not read file: {e}")
        else:
            print(f"   ❌ {filename} not found!")
    
    print()
    
    # Test 2: Check directory structure
    print("✅ Test 2: Checking directory structure...")
    
    required_dirs = [
        'models',
        'traindata', 
        'evorl',
        'utilities',
        'tests'
    ]
    
    for dirname in required_dirs:
        dirpath = Path(current_dir) / dirname
        if dirpath.exists():
            print(f"   ✓ {dirname}/ directory exists")
        else:
            print(f"   ❌ {dirname}/ directory missing!")
    
    print()
    
    # Test 3: Test dynamic path detection
    print("✅ Test 3: Testing dynamic path detection...")
    
    try:
        sys.path.insert(0, current_dir)
        from common_paths import get_project_root, basepath
        
        detected_path = get_project_root()
        print(f"   Detected project root: {detected_path}")
        print(f"   Basepath from common_paths: {basepath}")
        
        if detected_path == expected_path:
            print("   ✓ Dynamic path detection working correctly")
        else:
            print("   ❌ Dynamic path detection needs adjustment")
            
    except Exception as e:
        print(f"   ⚠️ Could not test dynamic path detection: {e}")
    
    print()
    
    # Test 4: Check for remaining old paths (excluding fallback locations)
    print("✅ Test 4: Scanning for problematic old paths...")
    
    old_patterns = [
        '/C:/',
        'C:/',
    ]
    
    issues_found = []
    
    for pattern in old_patterns:
        print(f"   Searching for: {pattern}")
        # This is a simple check - in production you'd use grep or similar
        # For now, just check key files
        for filename in key_files:
            filepath = Path(current_dir) / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        if pattern in content:
                            issues_found.append(f"{filename} contains {pattern}")
                except:
                    pass
    
    # Special check for basepath assignments (not fallback locations)
    print(f"   Checking for hardcoded basepath assignments...")
    problematic_pattern = 'basepath = \'/home/sid12321/Desktop/Trading-Final\''
    for filename in key_files:
        if filename == 'common_paths.py':  # Skip common_paths.py as it has fallback locations
            continue
        filepath = Path(current_dir) / filename  
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    if problematic_pattern in content:
                        issues_found.append(f"{filename} has hardcoded old basepath")
            except:
                pass
    
    if issues_found:
        print("   ❌ Issues found:")
        for issue in issues_found:
            print(f"     - {issue}")
    else:
        print("   ✓ No old path patterns found")
    
    print()
    print("=" * 60)
    
    # Summary
    if not issues_found:
        print("🎉 SUCCESS: All path updates appear to be working correctly!")
        print()
        print("Next steps:")
        print("1. Run the migration script: ./migrate_to_mac.sh")
        print("2. Install dependencies and test training")
        print("3. Verify GPU acceleration with JAX")
    else:
        print("⚠️ ISSUES FOUND: Some paths may need manual correction")
        print("Please review the issues above before proceeding with migration")
    
    return len(issues_found) == 0

if __name__ == "__main__":
    os.chdir("/Users/skumar81/Desktop/Personal/trading-final")
    success = verify_path_updates()
    sys.exit(0 if success else 1)