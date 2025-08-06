#!/usr/bin/env python3
"""
Patch for jax.experimental.maps compatibility
This adds a dummy maps module to make old code work with new JAX
"""

import sys
import os

def create_maps_compatibility():
    """Create a compatibility shim for jax.experimental.maps"""
    
    # Create the patch file
    patch_content = '''"""
Compatibility shim for jax.experimental.maps
This module was removed in JAX 0.4.14+ but some libraries still expect it
"""

import jax
from jax import pmap, vmap
from typing import Any, Callable

# Provide compatibility aliases
Mesh = None  # Mesh was in maps but not critical for basic usage

# xmap was deprecated and removed, provide a fallback
def xmap(*args, **kwargs):
    """Fallback for removed xmap function"""
    raise NotImplementedError(
        "xmap has been removed from JAX. "
        "Consider using pmap or vmap instead, "
        "or downgrade to jax<0.4.14"
    )

# Provide other common attributes that might be expected
serial = None
axis_index = None

# Make this module appear to have the removed content
__all__ = ['Mesh', 'xmap', 'serial', 'axis_index']
'''
    
    # Find JAX installation
    try:
        import jax
        jax_path = os.path.dirname(jax.__file__)
        experimental_path = os.path.join(jax_path, 'experimental')
        
        if os.path.exists(experimental_path):
            maps_file = os.path.join(experimental_path, 'maps.py')
            
            print(f"Creating compatibility shim at: {maps_file}")
            
            # Backup existing file if it exists
            if os.path.exists(maps_file):
                import shutil
                shutil.copy(maps_file, maps_file + '.backup')
                print(f"Backed up existing file to: {maps_file}.backup")
            
            # Write the compatibility shim
            with open(maps_file, 'w') as f:
                f.write(patch_content)
            
            print("✅ Compatibility shim created successfully!")
            
            # Test the import
            try:
                from jax.experimental import maps
                print("✅ Import test successful!")
                return True
            except Exception as e:
                print(f"❌ Import test failed: {e}")
                return False
        else:
            print(f"❌ JAX experimental path not found: {experimental_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating compatibility shim: {e}")
        return False

def main():
    print("=" * 60)
    print("JAX Maps Compatibility Patcher")
    print("=" * 60)
    
    print("\nThis will create a compatibility shim for jax.experimental.maps")
    print("This is a temporary fix for libraries expecting the old API")
    
    response = input("\nCreate compatibility shim? (y/n): ")
    
    if response.lower() == 'y':
        if create_maps_compatibility():
            print("\n✅ Patch applied successfully!")
            print("\nNow try running your evorl test again:")
            print("  python evorltest.py")
        else:
            print("\n❌ Patch failed!")
            print("\nAlternative solutions:")
            print("1. Run: python fix_evorl_dependencies.py")
            print("2. Use: python evorl_test_no_brax.py")
    else:
        print("\nAlternative solutions:")
        print("1. Run: python fix_evorl_dependencies.py")
        print("2. Use: python evorl_test_no_brax.py")

if __name__ == "__main__":
    main()