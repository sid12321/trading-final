#!/usr/bin/env python3
"""
Monkey patch for chex compatibility with JAX 0.4.26
This adds the missing KeyArray attribute to jax.random
"""

import jax
import jax.random

# Add the missing KeyArray attribute
if not hasattr(jax.random, 'KeyArray'):
    # In JAX 0.4.26, PRNGKey is just jax.Array
    jax.random.KeyArray = jax.Array
    print("✅ Patched jax.random.KeyArray for chex compatibility")

# Also patch any other missing attributes that might be needed
if not hasattr(jax.random, 'key'):
    jax.random.key = jax.random.PRNGKey
    print("✅ Patched jax.random.key for compatibility")

print(f"JAX version: {jax.__version__}")
print(f"JAX random KeyArray: {jax.random.KeyArray}")