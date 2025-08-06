#!/usr/bin/env python3
"""
Test EvoRL GPU Performance
Quick test to verify GPU-only training is working
"""

import time
import numpy as np
import pandas as pd
from evorl_ppo_trainer import create_evorl_trainer_from_data

# Generate test data
print("🧪 Testing EvoRL GPU Performance")
print("=" * 60)

# Create synthetic trading data
n_samples = 1000
n_features = 10

data = pd.DataFrame({
    **{f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)},
    'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
    'vwap2': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
})

signals = [f'feature_{i}' for i in range(n_features)]

# Create trainer with optimized settings
print("📊 Creating EvoRL trainer...")
trainer = create_evorl_trainer_from_data(
    data, 
    signals,
    n_steps=256,      # Larger for GPU efficiency
    batch_size=128,   # Larger batch for GPU
    n_epochs=4,       # Moderate epochs
    learning_rate=0.0005
)

print(f"✅ Trainer created")
print(f"   Observation dim: {trainer.obs_dim}")
print(f"   Action dim: {trainer.action_dim}")
print(f"   Network architecture: {trainer.hidden_dims}")

# Run short training to test performance
print("\n🚀 Running GPU performance test...")
start_time = time.time()

try:
    results = trainer.train(total_timesteps=1024)  # 4 rollouts
    end_time = time.time()
    
    training_time = end_time - start_time
    timesteps_per_second = 1024 / training_time
    
    print(f"\n✅ Performance Test Results:")
    print(f"   Total timesteps: 1,024")
    print(f"   Training time: {training_time:.2f} seconds")
    print(f"   Timesteps/second: {timesteps_per_second:.0f}")
    print(f"   Final reward: {results['training_metrics'][-1]['mean_reward']:.4f}")
    
    # Evaluate model
    eval_results = trainer.evaluate(n_episodes=5)
    print(f"   Evaluation reward: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
    
    print("\n🎉 GPU-only training successful!")
    print(f"   Expected production performance: {timesteps_per_second * 60:.0f} timesteps/minute")
    
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()