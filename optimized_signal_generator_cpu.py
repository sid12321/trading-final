"""
CPU-optimized signal generator for better performance
Switches to CPU-only processing to avoid GPU thread contention
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import warnings
import os
import time

warnings.filterwarnings('ignore')

# Force CPU mode
DEVICE = "cpu"
NUM_CORES = min(mp.cpu_count() - 2, 28)  # Leave 2 cores for system
print(f"Using CPU optimization with {NUM_CORES} cores")

# Quantile cache for performance
QUANTILE_CACHE = {}

def generate_signals_from_quantiles_fast(
    values: np.ndarray, 
    lower_quantile: float = 0.3,
    upper_quantile: float = 0.7,
    signal_strength: float = 1.0,
    lookback: int = 0
) -> np.ndarray:
    """
    Fast quantile-based signal generation (CPU optimized)
    """
    # Apply smoothing if lookback > 0
    if lookback > 0 and len(values) > lookback:
        # Use efficient convolution for smoothing
        kernel = np.ones(lookback) / lookback
        smoothed_values = np.convolve(values, kernel, mode='same')
    else:
        smoothed_values = values
    
    # Calculate quantiles efficiently
    valid_values = smoothed_values[~np.isnan(smoothed_values)]
    if len(valid_values) > 0:
        lq = np.quantile(valid_values, lower_quantile, method='linear')
        uq = np.quantile(valid_values, upper_quantile, method='linear')
    else:
        lq, uq = 0.0, 0.0
    
    # Vectorized signal generation
    signal = np.zeros_like(smoothed_values, dtype=np.float32)
    signal[smoothed_values <= lq] = signal_strength
    signal[smoothed_values >= uq] = -signal_strength
    signal[np.isnan(smoothed_values)] = 0
    
    return signal

def simulate_trades_on_day_vectorized(
    vwap: np.ndarray, 
    action: np.ndarray, 
    transaction_cost: float = 0.001
) -> Tuple[float, float]:
    """
    Vectorized trading simulation (CPU optimized)
    """
    if len(vwap) < 2:
        return 0.0, 0.0
    
    vwap = np.asarray(vwap, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    
    # Vectorized position and trade calculation
    positions = np.clip(np.cumsum(action), -1, 1)
    trades = np.diff(np.concatenate([[0], positions]))
    
    # Calculate returns efficiently
    returns = np.diff(vwap) / vwap[:-1]
    position_returns = positions[:-1] * returns
    
    # Transaction costs
    trade_costs = np.abs(trades[1:]) * transaction_cost
    
    # Total PnL
    pnl = np.sum(position_returns - trade_costs)
    final_position = positions[-1]
    
    return float(pnl), float(final_position)

def adaptive_simulated_annealing_cpu(
    signal_data: np.ndarray,
    vwap_data: np.ndarray,
    dates: np.ndarray,
    max_iterations: int = 200,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.95
) -> Tuple[Dict, float]:
    """
    CPU-optimized simulated annealing
    """
    # Initialize parameters
    best_params = {
        'lower_quantile': np.random.uniform(0.1, 0.4),
        'upper_quantile': np.random.uniform(0.6, 0.9),
        'signal_strength': np.random.uniform(0.5, 2.0),
        'lookback': np.random.randint(0, 20)
    }
    
    # Generate initial signals
    signals = generate_signals_from_quantiles_fast(
        signal_data, 
        best_params['lower_quantile'],
        best_params['upper_quantile'],
        best_params['signal_strength'],
        best_params['lookback']
    )
    
    # Calculate initial PnL
    unique_dates = np.unique(dates)
    best_pnl = 0.0
    
    for date in unique_dates:
        mask = dates == date
        if np.sum(mask) > 1:
            pnl, _ = simulate_trades_on_day_vectorized(
                vwap_data[mask], 
                signals[mask]
            )
            best_pnl += pnl
    
    # Simulated annealing optimization
    current_params = best_params.copy()
    current_pnl = best_pnl
    temperature = initial_temp
    
    for iteration in range(max_iterations):
        # Generate neighbor solution
        neighbor_params = current_params.copy()
        param_to_change = np.random.choice(['lower_quantile', 'upper_quantile', 'signal_strength', 'lookback'])
        
        if param_to_change == 'lower_quantile':
            neighbor_params['lower_quantile'] = np.clip(
                current_params['lower_quantile'] + np.random.normal(0, 0.05),
                0.1, 0.4
            )
        elif param_to_change == 'upper_quantile':
            neighbor_params['upper_quantile'] = np.clip(
                current_params['upper_quantile'] + np.random.normal(0, 0.05),
                0.6, 0.9
            )
        elif param_to_change == 'signal_strength':
            neighbor_params['signal_strength'] = np.clip(
                current_params['signal_strength'] + np.random.normal(0, 0.2),
                0.5, 2.0
            )
        else:  # lookback
            neighbor_params['lookback'] = np.clip(
                int(current_params['lookback'] + np.random.randint(-2, 3)),
                0, 20
            )
        
        # Generate new signals
        neighbor_signals = generate_signals_from_quantiles_fast(
            signal_data,
            neighbor_params['lower_quantile'],
            neighbor_params['upper_quantile'],
            neighbor_params['signal_strength'],
            neighbor_params['lookback']
        )
        
        # Calculate neighbor PnL
        neighbor_pnl = 0.0
        for date in unique_dates:
            mask = dates == date
            if np.sum(mask) > 1:
                pnl, _ = simulate_trades_on_day_vectorized(
                    vwap_data[mask],
                    neighbor_signals[mask]
                )
                neighbor_pnl += pnl
        
        # Accept or reject
        delta = neighbor_pnl - current_pnl
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current_params = neighbor_params
            current_pnl = neighbor_pnl
            
            if current_pnl > best_pnl:
                best_params = current_params.copy()
                best_pnl = current_pnl
        
        # Cool down
        temperature *= cooling_rate
    
    return best_params, best_pnl

def optimize_single_signal_cpu(task_data):
    """
    Single signal optimization for CPU processing
    """
    var, signalmultiplier, signal_data, vwap_data, dates = task_data
    
    try:
        # Apply signal multiplier
        adjusted_signal_data = signalmultiplier * signal_data
        
        # Run optimization
        best_params, best_pnl = adaptive_simulated_annealing_cpu(
            adjusted_signal_data, vwap_data, dates,
            max_iterations=200  # Reduced iterations for speed
        )
        
        return {
            'success': True,
            'var': var,
            'signalmultiplier': signalmultiplier,
            'best_params': best_params,
            'best_pnl': best_pnl
        }
    except Exception as e:
        return {
            'success': False,
            'var': var,
            'signalmultiplier': signalmultiplier,
            'error': str(e)
        }

def generate_optimized_signals_cpu(
    mldf: pd.DataFrame,
    signalcolumns: List[str],
    use_parallel: bool = True,
    batch_size: int = 20
) -> Tuple[Dict, pd.DataFrame]:
    """
    CPU-optimized signal generation with efficient parallelization
    """
    optimized_signals = {}
    
    # Filter valid signals
    valid_signals = [col for col in signalcolumns if col in mldf.columns]
    print(f"Optimizing {len(valid_signals)} signals using CPU-optimized processing...")
    
    # Prepare optimization tasks
    optimization_tasks = []
    
    for var in valid_signals:
        signal_data = mldf[var].values
        vwap_data = mldf['vwap'].values
        dates = mldf['date'].values
        
        # Skip if insufficient data
        if len(signal_data) < 100 or len(np.unique(dates)) < 5:
            continue
        
        # Check signal variance
        signal_variance = np.var(signal_data[~np.isnan(signal_data)])
        if signal_variance < 1e-6:
            continue
        
        # Add tasks for both multipliers
        for signalmultiplier in [1, -1]:
            optimization_tasks.append((
                var, signalmultiplier, signal_data, vwap_data, dates
            ))
    
    print(f"Created {len(optimization_tasks)} optimization tasks")
    print(f"Using {NUM_CORES} CPU cores with batch size {batch_size}")
    
    # Process tasks in parallel batches
    results = []
    start_time = time.time()
    
    if use_parallel and len(optimization_tasks) > 1:
        # Process in batches to avoid memory issues
        for batch_start in range(0, len(optimization_tasks), batch_size):
            batch_end = min(batch_start + batch_size, len(optimization_tasks))
            batch_tasks = optimization_tasks[batch_start:batch_end]
            
            batch_num = batch_start // batch_size + 1
            total_batches = (len(optimization_tasks) + batch_size - 1) // batch_size
            print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_tasks)} tasks)")
            
            # Use ProcessPoolExecutor for true parallelism
            with ProcessPoolExecutor(max_workers=min(NUM_CORES, len(batch_tasks))) as executor:
                # Submit all tasks in batch
                future_to_task = {
                    executor.submit(optimize_single_signal_cpu, task): task 
                    for task in batch_tasks
                }
                
                # Process completed tasks
                completed = 0
                for future in as_completed(future_to_task):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    # Progress update every 10 tasks
                    if completed % 10 == 0:
                        elapsed = time.time() - start_time
                        total_completed = batch_start + completed
                        progress = total_completed / len(optimization_tasks) * 100
                        eta = elapsed / total_completed * (len(optimization_tasks) - total_completed)
                        print(f"  Completed {total_completed}/{len(optimization_tasks)} tasks "
                              f"({progress:.1f}%) - ETA: {eta:.0f}s")
    else:
        # Sequential processing for small tasks
        for task in optimization_tasks:
            result = optimize_single_signal_cpu(task)
            results.append(result)
    
    # Process results
    successful_results = [r for r in results if r['success']]
    print(f"\nOptimization complete: {len(successful_results)}/{len(results)} successful")
    print(f"Total time: {time.time() - start_time:.1f}s")
    
    # Build optimized signals dictionary
    for result in successful_results:
        var = result['var']
        signalmultiplier = result['signalmultiplier']
        key = f"{var}_mult{signalmultiplier}"
        
        optimized_signals[key] = {
            'params': result['best_params'],
            'pnl': result['best_pnl'],
            'signal': generate_signals_from_quantiles_fast(
                signalmultiplier * mldf[var].values,
                **result['best_params']
            )
        }
    
    # Create enhanced DataFrame
    enhanced_df = mldf.copy()
    for key, data in optimized_signals.items():
        enhanced_df[f"opt_{key}"] = data['signal']
    
    return optimized_signals, enhanced_df

if __name__ == "__main__":
    # Test with sample data
    print("CPU-optimized signal generator ready")
    print(f"Configuration: {NUM_CORES} CPU cores, batch processing")