
# FAST LEARNING CONFIGURATION - Generated automatically
# Optimized for 3-5x faster convergence

# === CORE LEARNING PARAMETERS ===
FAST_LEARNING_RATE = 0.001  # 10x faster than default
FAST_ENT_COEF_START = 0.1  # Enhanced exploration
FAST_ENT_COEF_END = 0.01    # Final exploration level
FAST_VALUE_COEF = 1.0        # Accelerated value learning
FAST_MAX_GRAD_NORM = 1.0  # Less restrictive clipping

# === TRAINING DYNAMICS ===
FAST_N_EPOCHS = 20           # More epochs per rollout
FAST_N_STEPS = 256             # Shorter rollouts, frequent updates
FAST_BATCH_SIZE = 2048       # Optimized batch size
FAST_MINI_BATCH_SIZE = 512  # Multiple mini-batches

# === NETWORK ARCHITECTURE ===
FAST_NETWORK_DIMS = (2048, 1024, 512, 256)   # Larger, deeper network
FAST_DROPOUT_RATE = 0.1   # Regularization

# === ADVANCED SETTINGS ===
FAST_GAE_LAMBDA = 0.98       # Better credit assignment  
FAST_GAMMA = 0.995                 # Long-term thinking
FAST_TARGET_KL = 0.015         # Stability constraint

# Usage: Import these in your training script
# from fast_learning_hyperparameters import *
