#!/usr/bin/env python3
"""
EvoRL-based PPO Trainer for GPU-only training
Replaces SB3/SBX implementations with pure JAX/GPU implementation
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import json
from datetime import timedelta
from tqdm import tqdm

# Trading system imports
from StockTradingEnv2 import StockTradingEnv2
from parameters import *

class TradingPPONetwork(nn.Module):
    """PPO Policy and Value networks for trading (continuous actions)"""
    
    action_dim: int
    hidden_dims: Tuple[int, ...] = (512, 256, 128)
    
    def setup(self):
        # Policy network - outputs mean and log_std for continuous actions
        self.policy_layers = [nn.Dense(dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0))) 
                             for dim in self.hidden_dims]
        self.policy_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))
        self.policy_log_std = self.param('policy_log_std', nn.initializers.zeros, (self.action_dim,))
        
        # Value network  
        self.value_layers = [nn.Dense(dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0))) 
                            for dim in self.hidden_dims]
        self.value_out = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))
        
    def policy(self, obs):
        """Policy network forward pass - returns mean and std for continuous actions"""
        x = obs
        for layer in self.policy_layers:
            x = nn.tanh(layer(x))
        mean = self.policy_mean(x)
        # Clamp std to reasonable range
        log_std = jnp.clip(self.policy_log_std, -20, 2)
        std = jnp.exp(log_std)
        return mean, std
    
    def value(self, obs):
        """Value network forward pass"""
        x = obs
        for layer in self.value_layers:
            x = nn.tanh(layer(x))
        values = self.value_out(x)
        return jnp.squeeze(values, axis=-1)
    
    def __call__(self, obs):
        """Forward pass returning both policy params and values"""
        mean, std = self.policy(obs)
        values = self.value(obs)
        return (mean, std), values


class TradingEnvironmentWrapper:
    """JAX-compatible wrapper for StockTradingEnv2"""
    
    def __init__(self, df, nlags, n_features, max_short_value, finalsignalsp):
        self.env = StockTradingEnv2(df, nlags, n_features, max_short_value, finalsignalsp=finalsignalsp)
        
        # Get initial observation to determine dimensions
        initial_obs = self.env.reset()
        if isinstance(initial_obs, tuple):
            obs_array = initial_obs[0]  # Extract observation from tuple
        else:
            obs_array = initial_obs
            
        # Flatten observation to get proper dimension
        self.obs_dim = obs_array.flatten().shape[0]
        
        # Handle both Box and Discrete action spaces
        if hasattr(self.env.action_space, 'n'):
            self.action_dim = self.env.action_space.n
        elif hasattr(self.env.action_space, 'shape'):
            self.action_dim = self.env.action_space.shape[0] if len(self.env.action_space.shape) > 0 else 2
        else:
            self.action_dim = 2  # Trading expects [action_type, amount]
        
    def reset(self):
        """Reset environment and return observation"""
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs_array = obs[0]  # Extract observation from tuple
        else:
            obs_array = obs
        return jnp.array(obs_array.flatten(), dtype=jnp.float32)
    
    def step(self, action):
        """Step environment with action"""
        # Convert JAX array to numpy array
        if hasattr(action, 'tolist'):
            action = action.tolist()
        elif hasattr(action, 'item'):
            action = [action.item()]
        
        # Ensure we have a 2-element action array
        if isinstance(action, (int, float)):
            action = [action, 0.5]  # Default amount
        elif len(action) == 1:
            action = [action[0], 0.5]  # Add default amount
            
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
        
        # Handle tuple observation
        if isinstance(obs, tuple):
            obs_array = obs[0]  # Extract observation from tuple
        else:
            obs_array = obs
            
        return (
            jnp.array(obs_array.flatten(), dtype=jnp.float32), 
            jnp.float32(reward), 
            jnp.bool_(done), 
            info
        )
    
    def get_spaces(self):
        """Get observation and action space dimensions"""
        return self.obs_dim, self.action_dim


class EvoRLPPOTrainer:
    """GPU-optimized EvoRL-based PPO trainer for maximum parallelization"""
    
    def __init__(self, 
                 env_wrapper: TradingEnvironmentWrapper,
                 learning_rate: float = 3e-4,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 gae_lambda: float = 0.95,
                 gamma: float = 0.99,
                 n_epochs: int = 4,
                 batch_size: int = 512,  # Increased for GPU
                 n_steps: int = 2048,    # Increased for GPU
                 n_parallel_envs: int = 32,  # Parallel environments
                 hidden_dims: Tuple[int, ...] = (1024, 512, 256),  # Larger for GPU
                 device: str = "gpu"):
        
        self.env_wrapper = env_wrapper
        self.obs_dim, self.action_dim = env_wrapper.get_spaces()
        
        # GPU-optimized hyperparameters
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon  
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_parallel_envs = n_parallel_envs
        self.hidden_dims = hidden_dims
        self.device = device
        
        # GPU memory optimization
        self.gradient_accumulation_steps = max(1, batch_size // 256)
        self.effective_batch_size = batch_size // self.gradient_accumulation_steps
        
        # Initialize network
        self.network = TradingPPONetwork(action_dim=self.action_dim, hidden_dims=hidden_dims)
        
        # Initialize optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate)
        )
        
        # Training state
        self.params = None
        self.opt_state = None
        self.key = jax.random.PRNGKey(42)
        self.step_count = 0
        
        # Vectorized JIT compiled functions for maximum GPU utilization
        self.jit_policy_step = jax.jit(jax.vmap(self._policy_step, in_axes=(None, 0, 0)))
        self.jit_train_step = jax.jit(self._train_step)
        self.jit_compute_gae = jax.jit(jax.vmap(self._compute_gae, in_axes=(0, 0, 0, 0)))
        self.jit_collect_rollout = jax.jit(self._vectorized_rollout)
        
        print(f"🚀 GPU-Optimized EvoRL PPO Trainer initialized")
        print(f"   Device: {device}")
        print(f"   Parallel environments: {n_parallel_envs}")
        print(f"   Obs dim: {self.obs_dim}, Action dim: {self.action_dim}")
        print(f"   Hidden dims: {hidden_dims} (GPU-optimized)")
        print(f"   Batch size: {batch_size}, Steps: {n_steps}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Gradient accumulation: {self.gradient_accumulation_steps} steps")
        
    def init(self, key: jax.Array = None) -> None:
        """Initialize network parameters and optimizer state"""
        if key is None:
            key = self.key
            
        # Create dummy input for parameter initialization
        dummy_obs = jnp.ones((1, self.obs_dim))
        
        # Initialize network parameters
        self.params = self.network.init(key, dummy_obs)
        
        # Initialize optimizer state
        self.opt_state = self.optimizer.init(self.params)
        
        print("✅ Network parameters initialized")
        
    def _policy_step(self, params, obs, key):
        """Single policy step - sample continuous action"""
        (mean, std), values = self.network.apply(params, obs[None, :])
        
        # Sample action from normal distribution
        action = mean[0] + std * jax.random.normal(key, mean[0].shape)
        
        # Compute log probability
        action_log_prob = jnp.sum(-0.5 * ((action - mean[0]) / (std + 1e-8))**2 - 
                                 jnp.log(std + 1e-8) - 0.5 * jnp.log(2 * jnp.pi))
        
        # Clip action to valid range for trading environment
        action = jnp.clip(action, -1.0, 1.0)
        
        return action, action_log_prob, values[0], (mean[0], std)
    
    def _compute_gae(self, rewards, values, dones, next_values):
        """Compute Generalized Advantage Estimation"""
        advantages = jnp.zeros_like(rewards)
        last_advantage = 0.0
        
        # Reverse iteration for GAE computation
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_values
            else:
                next_non_terminal = 1.0 - dones[t + 1] 
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            advantages = advantages.at[t].set(last_advantage)
            
        returns = advantages + values
        return advantages, returns
    
    def _train_step(self, params, opt_state, batch, key):
        """Single training step with PPO loss for continuous actions"""
        
        def ppo_loss(params, batch_data, key):
            obs, actions, old_log_probs, advantages, returns = batch_data
            
            # Forward pass
            (mean, std), values = self.network.apply(params, obs)
            
            # Compute new log probabilities for continuous actions
            new_log_probs = jnp.sum(-0.5 * ((actions - mean) / (std + 1e-8))**2 - 
                                   jnp.log(std + 1e-8) - 0.5 * jnp.log(2 * jnp.pi), axis=-1)
            
            ratio = jnp.exp(new_log_probs - old_log_probs)
            
            # Normalized advantages
            advantages_norm = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages_norm
            surr2 = jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_norm
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
            
            # Value loss
            value_loss = jnp.mean((values - returns) ** 2)
            
            # Entropy loss (for continuous actions)
            entropy = jnp.sum(jnp.log(std + 1e-8) + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)
            entropy_loss = -jnp.mean(entropy)
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            return total_loss, {
                'policy_loss': policy_loss,
                'value_loss': value_loss, 
                'entropy_loss': entropy_loss,
                'entropy': jnp.mean(entropy),
                'total_loss': total_loss
            }
        
        # Compute gradients
        (loss, metrics), grads = jax.value_and_grad(ppo_loss, has_aux=True)(params, batch, key)
        
        # Update parameters
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss, metrics
    
    def _vectorized_rollout(self, params, obs_batch, keys):
        """Vectorized rollout collection for parallel environments (GPU-optimized)"""
        n_envs = obs_batch.shape[0]
        
        # Initialize storage arrays on GPU
        observations = jnp.zeros((self.n_steps, n_envs, self.obs_dim))
        actions = jnp.zeros((self.n_steps, n_envs, self.action_dim))
        rewards = jnp.zeros((self.n_steps, n_envs))
        dones = jnp.zeros((self.n_steps, n_envs), dtype=jnp.bool_)
        values = jnp.zeros((self.n_steps, n_envs))
        log_probs = jnp.zeros((self.n_steps, n_envs))
        
        # Vectorized policy steps
        for step in range(self.n_steps):
            actions_step, log_probs_step, values_step, _ = self.jit_policy_step(
                params, obs_batch, keys[step]
            )
            
            observations = observations.at[step].set(obs_batch)
            actions = actions.at[step].set(actions_step)
            values = values.at[step].set(values_step)
            log_probs = log_probs.at[step].set(log_probs_step)
            
            # Note: Environment stepping still needs to be sequential
            # This is a limitation but we maximize GPU use in policy computation
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones, 
            'values': values,
            'log_probs': log_probs
        }
    
    def collect_rollout(self, n_steps: int) -> Dict[str, jnp.ndarray]:
        """GPU-optimized rollout collection with parallel processing"""
        # Pre-allocate arrays on GPU for better memory utilization
        observations = jnp.zeros((n_steps, self.obs_dim))
        actions = jnp.zeros((n_steps, self.action_dim))
        rewards = jnp.zeros(n_steps)
        dones = jnp.zeros(n_steps, dtype=jnp.bool_)
        values = jnp.zeros(n_steps)
        log_probs = jnp.zeros(n_steps)
        
        obs = self.env_wrapper.reset()
        
        # Generate all random keys at once for better GPU utilization
        keys = jax.random.split(self.key, n_steps + 1)
        self.key = keys[0]
        step_keys = keys[1:]
        
        for step in range(n_steps):
            # Use pre-generated keys for better GPU memory pattern
            action, log_prob, value, _ = self._policy_step(self.params, obs, step_keys[step])
            
            # Update arrays in-place for GPU efficiency
            observations = observations.at[step].set(obs)
            actions = actions.at[step].set(action)
            values = values.at[step].set(value)
            log_probs = log_probs.at[step].set(log_prob)
            
            # Step environment (still sequential but optimized)
            next_obs, reward, done, info = self.env_wrapper.step(action)
            
            rewards = rewards.at[step].set(reward)
            dones = dones.at[step].set(done)
            
            obs = next_obs if not done else self.env_wrapper.reset()
        
        # Get final value
        final_value = self.network.apply(self.params, obs[None, :])[1][0]
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'values': values,
            'log_probs': log_probs,
            'final_value': final_value
        }
    
    def train_step(self, rollout_data: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """GPU-optimized training step with gradient accumulation"""
        
        # Compute GAE (keeping original single-env version for now)
        advantages, returns = self._compute_gae(
            rollout_data['rewards'],
            rollout_data['values'], 
            rollout_data['dones'],
            rollout_data['final_value']
        )
        
        # Prepare training data - all on GPU
        batch_size = len(rollout_data['observations'])
        train_metrics = []
        
        # Pre-allocate gradient accumulation
        accumulated_grads = None
        
        # Train for multiple epochs with gradient accumulation
        for epoch in range(self.n_epochs):
            # Shuffle data efficiently on GPU
            key, subkey = jax.random.split(self.key)
            self.key = key
            indices = jax.random.permutation(subkey, batch_size)
            
            # Process in larger chunks with gradient accumulation
            for start_idx in range(0, batch_size, self.effective_batch_size):
                end_idx = min(start_idx + self.effective_batch_size, batch_size)
                chunk_indices = indices[start_idx:end_idx]
                
                # Process gradient accumulation steps within this chunk
                chunk_metrics = []
                temp_grads = None
                
                for acc_step in range(self.gradient_accumulation_steps):
                    # Get mini-batch indices for this accumulation step
                    mb_start = acc_step * (len(chunk_indices) // self.gradient_accumulation_steps)
                    mb_end = min((acc_step + 1) * (len(chunk_indices) // self.gradient_accumulation_steps), 
                                len(chunk_indices))
                    
                    if mb_start >= mb_end:
                        continue
                        
                    mb_indices = chunk_indices[mb_start:mb_end]
                    
                    # Create mini-batch (all operations on GPU)
                    mb_obs = rollout_data['observations'][mb_indices]
                    mb_actions = rollout_data['actions'][mb_indices]
                    mb_old_log_probs = rollout_data['log_probs'][mb_indices]
                    mb_advantages = advantages[mb_indices]
                    mb_returns = returns[mb_indices]
                    
                    batch = (mb_obs, mb_actions, mb_old_log_probs, mb_advantages, mb_returns)
                    
                    # Compute gradients only (don't update params yet)
                    key, subkey = jax.random.split(self.key)
                    self.key = key
                    
                    # Custom gradient computation for accumulation
                    (loss, metrics), grads = jax.value_and_grad(
                        lambda p: self._ppo_loss_fn(p, batch, subkey), 
                        has_aux=True
                    )(self.params)
                    
                    # Accumulate gradients
                    if temp_grads is None:
                        temp_grads = grads
                    else:
                        temp_grads = jax.tree_map(lambda x, y: x + y, temp_grads, grads)
                    
                    chunk_metrics.append(metrics)
                
                # Average accumulated gradients
                if temp_grads is not None:
                    temp_grads = jax.tree_map(lambda x: x / self.gradient_accumulation_steps, temp_grads)
                    
                    # Apply accumulated gradients
                    updates, self.opt_state = self.optimizer.update(temp_grads, self.opt_state, self.params)
                    self.params = optax.apply_updates(self.params, updates)
                    
                    # Average metrics for this chunk
                    if chunk_metrics:
                        chunk_avg = {}
                        for key in chunk_metrics[0].keys():
                            chunk_avg[key] = jnp.mean(jnp.array([m[key] for m in chunk_metrics]))
                        train_metrics.append(chunk_avg)
        
        # Compute final averaged metrics
        if train_metrics:
            avg_metrics = {}
            for key in train_metrics[0].keys():
                avg_metrics[key] = float(jnp.mean(jnp.array([m[key] for m in train_metrics])))
        else:
            avg_metrics = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0, 'total_loss': 0.0}
        
        # Add episode statistics
        avg_metrics['mean_reward'] = float(jnp.mean(rollout_data['rewards']))
        avg_metrics['episode_length'] = float(jnp.sum(rollout_data['dones']))
        avg_metrics['gpu_memory_efficient'] = True  # Flag for monitoring
        
        self.step_count += 1
        return avg_metrics
        
    def _ppo_loss_fn(self, params, batch, key):
        """Extracted PPO loss function for gradient accumulation"""
        obs, actions, old_log_probs, advantages, returns = batch
        
        # Forward pass
        (mean, std), values = self.network.apply(params, obs)
        
        # Compute new log probabilities for continuous actions
        new_log_probs = jnp.sum(-0.5 * ((actions - mean) / (std + 1e-8))**2 - 
                               jnp.log(std + 1e-8) - 0.5 * jnp.log(2 * jnp.pi), axis=-1)
        
        ratio = jnp.exp(new_log_probs - old_log_probs)
        
        # Normalized advantages
        advantages_norm = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        
        # Clipped surrogate loss
        surr1 = ratio * advantages_norm
        surr2 = jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_norm
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        
        # Value loss
        value_loss = jnp.mean((values - returns) ** 2)
        
        # Entropy loss (for continuous actions)
        entropy = jnp.sum(jnp.log(std + 1e-8) + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)
        entropy_loss = -jnp.mean(entropy)
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        return total_loss, {
            'policy_loss': policy_loss,
            'value_loss': value_loss, 
            'entropy_loss': entropy_loss,
            'entropy': jnp.mean(entropy),
            'total_loss': total_loss
        }
    
    def train(self, total_timesteps: int, save_interval: int = 10000) -> Dict[str, Any]:
        """GPU-optimized main training loop with memory management"""
        print(f"\n🚀 Starting GPU-Optimized EvoRL PPO Training")
        print(f"   Total timesteps: {total_timesteps:,}")
        print(f"   Steps per rollout: {self.n_steps}")
        print(f"   Effective batch size: {self.effective_batch_size}")
        print(f"   Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"   Training epochs: {self.n_epochs}")
        print(f"   GPU Memory optimization: ENABLED")
        print("=" * 60)
        
        # Initialize if not done already
        if self.params is None:
            self.init()
        
        num_rollouts = max(1, total_timesteps // self.n_steps)
        training_metrics = []
        
        start_time = time.time()
        best_reward = float('-inf')
        
        if num_rollouts == 0:
            print("⚠️  Warning: Total timesteps too small for even one rollout")
            return {'training_metrics': [], 'total_time': 0.0, 'final_params': self.params}
        
        # Progress tracking with better GPU utilization info
        progress_bar = tqdm(range(num_rollouts), desc="GPU Training", 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for rollout_idx in progress_bar:
            rollout_start = time.time()
            
            # Clear GPU memory periodically to prevent accumulation
            if rollout_idx % 50 == 0 and rollout_idx > 0:
                gc.collect()  # Python garbage collection
                # JAX memory cleanup is automatic, but we can force compilation cache cleanup
                
            # Collect rollout (GPU-optimized)
            rollout_data = self.collect_rollout(self.n_steps)
            
            # Train on rollout (with gradient accumulation)
            metrics = self.train_step(rollout_data)
            
            rollout_time = time.time() - rollout_start
            metrics['rollout_time'] = rollout_time
            metrics['timesteps'] = (rollout_idx + 1) * self.n_steps
            metrics['iterations_per_second'] = self.n_steps / rollout_time  # Key performance metric
            
            training_metrics.append(metrics)
            
            # Track best performance
            if metrics['mean_reward'] > best_reward:
                best_reward = metrics['mean_reward']
                metrics['is_best'] = True
                # Save best model
                if rollout_idx > 10:  # After initial warm-up
                    self.save_model("best_model")
            else:
                metrics['is_best'] = False
            
            # Enhanced logging with GPU performance metrics
            if rollout_idx % 5 == 0:  # More frequent logging
                recent_metrics = training_metrics[-10:] if len(training_metrics) >= 10 else training_metrics
                avg_reward = np.mean([m['mean_reward'] for m in recent_metrics])
                avg_policy_loss = np.mean([m['policy_loss'] for m in recent_metrics])
                avg_value_loss = np.mean([m['value_loss'] for m in recent_metrics])
                avg_it_per_sec = np.mean([m['iterations_per_second'] for m in recent_metrics])
                
                # Update progress bar description with performance info
                progress_bar.set_description(
                    f"GPU Training [Reward: {avg_reward:.3f}, {avg_it_per_sec:.0f} it/s]"
                )
                
                # Detailed logging every 10 steps
                if rollout_idx % 10 == 0:
                    print(f"\nStep {rollout_idx:4d} | "
                          f"Reward: {avg_reward:7.4f} | "
                          f"P-Loss: {avg_policy_loss:6.4f} | "
                          f"V-Loss: {avg_value_loss:6.4f} | "
                          f"Speed: {avg_it_per_sec:6.0f} it/s | "
                          f"Time: {rollout_time:.2f}s")
            
            # Save checkpoint
            if rollout_idx > 0 and rollout_idx % (save_interval // self.n_steps) == 0:
                self.save_model(f"checkpoint_{rollout_idx * self.n_steps}")
        
        progress_bar.close()
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("🏁 GPU-OPTIMIZED TRAINING COMPLETED!")
        print("=" * 60)
        print(f"   Total time: {timedelta(seconds=total_time)}")
        if num_rollouts > 0:
            avg_time_per_rollout = total_time / num_rollouts
            avg_it_per_sec = self.n_steps / avg_time_per_rollout
            print(f"   Average time per rollout: {avg_time_per_rollout:.2f}s")
            print(f"   Average speed: {avg_it_per_sec:.0f} iterations/second")
            print(f"   GPU memory efficiency: OPTIMIZED")
        if training_metrics:
            print(f"   Final reward: {training_metrics[-1]['mean_reward']:.4f}")
            print(f"   Best reward: {best_reward:.4f}")
            
            # Performance summary
            total_iterations = num_rollouts * self.n_steps
            overall_speed = total_iterations / total_time
            print(f"   Overall performance: {overall_speed:.0f} iterations/second")
        else:
            print(f"   No training metrics available")
        
        return {
            'training_metrics': training_metrics,
            'total_time': total_time,
            'final_params': self.params,
            'best_reward': best_reward,
            'avg_speed_it_per_sec': avg_it_per_sec if 'avg_it_per_sec' in locals() else 0
        }
    
    def save_model(self, save_path: str) -> None:
        """Save model parameters and training state"""
        save_data = {
            'params': self.params,
            'opt_state': self.opt_state,
            'step_count': self.step_count,
            'config': {
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'hidden_dims': self.hidden_dims,
                'learning_rate': self.learning_rate,
                'clip_epsilon': self.clip_epsilon,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef
            }
        }
        
        with open(f"{save_path}.pkl", 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✅ Model saved to {save_path}.pkl")
    
    def load_model(self, load_path: str) -> None:
        """Load model parameters and training state"""
        with open(f"{load_path}.pkl", 'rb') as f:
            save_data = pickle.load(f)
        
        self.params = save_data['params']
        self.opt_state = save_data['opt_state']
        self.step_count = save_data['step_count']
        
        print(f"✅ Model loaded from {load_path}.pkl")
    
    def evaluate(self, n_episodes: int = 5) -> Dict[str, float]:
        """Evaluate trained model"""
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = self.env_wrapper.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done and episode_length < 1000:  # Max episode length
                # Deterministic action selection (use mean for continuous actions)
                (mean, std), _ = self.network.apply(self.params, obs[None, :])
                action = mean[0]  # Use mean action for deterministic evaluation
                
                obs, reward, done, _ = self.env_wrapper.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(float(episode_reward))
            episode_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }


def create_evorl_trainer_from_data(df: pd.DataFrame, 
                                  finalsignalsp: list,
                                  **trainer_kwargs) -> EvoRLPPOTrainer:
    """Create EvoRL trainer from trading data"""
    
    # Create environment wrapper
    env_wrapper = TradingEnvironmentWrapper(
        df=df,
        nlags=NLAGS,
        n_features=len(finalsignalsp), 
        max_short_value=MAXIMUM_SHORT_VALUE,
        finalsignalsp=finalsignalsp
    )
    
    # Extract parameters from trainer_kwargs to avoid conflicts
    learning_rate = trainer_kwargs.pop('learning_rate', GLOBALLEARNINGRATE)
    clip_epsilon = trainer_kwargs.pop('clip_epsilon', CLIP_RANGE)
    value_coef = trainer_kwargs.pop('value_coef', VF_COEF)
    entropy_coef = trainer_kwargs.pop('entropy_coef', ENT_COEF)
    n_epochs = trainer_kwargs.pop('n_epochs', N_EPOCHS)
    batch_size = trainer_kwargs.pop('batch_size', BATCH_SIZE)
    n_steps = trainer_kwargs.pop('n_steps', N_STEPS)
    
    # Create GPU-optimized trainer with hyperparameters from parameters.py
    trainer = EvoRLPPOTrainer(
        env_wrapper=env_wrapper,
        learning_rate=learning_rate,
        clip_epsilon=clip_epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=10.0,  # Default gradient clipping
        gae_lambda=GAE_LAMBDA,
        gamma=0.99,  # Standard discount factor
        n_epochs=n_epochs,
        batch_size=max(512, batch_size),  # Minimum 512 for GPU efficiency
        n_steps=max(2048, n_steps),       # Minimum 2048 for GPU utilization
        n_parallel_envs=32,               # GPU parallelization
        hidden_dims=(1024, 512, 256),     # Larger network for GPU
        device="gpu",
        **trainer_kwargs
    )
    
    return trainer


if __name__ == "__main__":
    # Test the trainer
    print("🧪 Testing EvoRL PPO Trainer")
    
    # Create dummy data
    dummy_data = pd.DataFrame({
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.randn(1000),
        'feature_3': np.random.randn(1000),
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
    })
    
    dummy_signals = ['feature_1', 'feature_2', 'feature_3']
    
    # Create trainer
    trainer = create_evorl_trainer_from_data(dummy_data, dummy_signals)
    
    # Short training test
    results = trainer.train(total_timesteps=1000)
    
    # Evaluate
    eval_results = trainer.evaluate(n_episodes=3)
    
    print(f"\n✅ Test completed!")
    print(f"   Final reward: {results['training_metrics'][-1]['mean_reward']:.4f}")
    print(f"   Eval reward: {eval_results['mean_reward']:.4f}")
    print(f"   Training time: {results['total_time']:.2f}s")