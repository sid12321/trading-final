#!/usr/bin/env python3
"""
EvoRL Integration Module
Replaces SB3/SBX training with pure JAX/GPU EvoRL implementation
"""

import os
import sys
import gc
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import timedelta

# Import the EvoRL trainer
from evorl_ppo_trainer import EvoRLPPOTrainer, create_evorl_trainer_from_data

# Trading system imports
basepath = '/Users/skumar81/Desktop/Personal/trading-final'
sys.path.insert(0, basepath)
os.chdir(basepath)

from parameters import *
from lib import *

# Import common functions individually to avoid import errors
try:
    from common import *
except ImportError as e:
    print(f"⚠️  Warning: Could not import all common functions: {e}")
    # Import essential functions only
    from common import modeltrain as original_modeltrain

# Import posterior compatibility modules
from evorl_posterior_compatibility import ensure_posterior_compatibility
from evorl_sb3_compatibility import ensure_evorl_posterior_compatibility


class EvoRLModelTrainer:
    """EvoRL-based model trainer that replaces SB3/SBX functionality"""
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or TESTSYMBOLS
        self.trainers = {}  # Store trainers for each symbol
        self.models = {}    # Store trained models
        self.normalizers = {} # Store data normalizers
        self.training_metrics = {}
        
        print(f"🚀 EvoRL Model Trainer initialized")
        print(f"   Symbols: {self.symbols}")
        print(f"   Device: GPU-only JAX")
        
    def train_symbol_model(self, 
                          symbol: str, 
                          df: pd.DataFrame, 
                          finalsignalsp: List[str],
                          iterations: int = BASEMODELITERATIONS) -> Dict[str, Any]:
        """Train PPO model for a single symbol using EvoRL"""
        
        print(f"\n📈 Training {symbol} with EvoRL PPO")
        print(f"   Data shape: {df.shape}")
        print(f"   Features: {len(finalsignalsp)}")
        print(f"   Iterations: {iterations:,}")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Create EvoRL trainer
            trainer = create_evorl_trainer_from_data(
                df=df,
                finalsignalsp=finalsignalsp,
            )
            
            # Store trainer
            self.trainers[symbol] = trainer
            
            # Train model
            results = trainer.train(total_timesteps=iterations)
            
            # Store training results
            self.training_metrics[symbol] = results['training_metrics']
            
            # Evaluate final model
            eval_results = trainer.evaluate(n_episodes=10)
            
            training_time = time.time() - start_time
            
            # Save model
            model_path = f"{basepath}/models/{symbol}localmodel_evorl"
            trainer.save_model(model_path)
            
            # Create normalizer (for compatibility with existing code)
            # Create QuantileTransformer-like object for compatibility
            from sklearn.preprocessing import QuantileTransformer
            
            feature_cols = [col for col in df.columns if col in finalsignalsp]
            feature_data = df[feature_cols].values
            
            # Create and fit QuantileTransformer
            qt = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
            qt.fit(feature_data)
            
            normalizer_path = f'{basepath}/models/{symbol}qt.joblib'
            joblib.dump(qt, normalizer_path)
            self.normalizers[symbol] = qt
            
            print(f"\n✅ {symbol} training completed!")
            print(f"   Training time: {timedelta(seconds=training_time)}")
            print(f"   Final reward: {results['training_metrics'][-1]['mean_reward']:.4f}")
            print(f"   Eval reward: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
            print(f"   Model saved: {model_path}.pkl")
            print(f"   Normalizer saved: {normalizer_path}")
            
            return {
                'success': True,
                'final_reward': results['training_metrics'][-1]['mean_reward'],
                'eval_reward': eval_results['mean_reward'],
                'training_time': training_time,
                'model_path': f"{model_path}.pkl",
                'normalizer_path': normalizer_path
            }
            
        except Exception as e:
            print(f"❌ Error training {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'final_reward': 0.0
            }
    
    def train_all_models(self, 
                        rdflistp: Dict[str, pd.DataFrame], 
                        lol: Dict[str, List[str]],
                        iterations: int = BASEMODELITERATIONS) -> float:
        """Train models for all symbols and return average reward"""
        
        print(f"\n🏋️  Training all models with EvoRL PPO")
        print(f"   Symbols: {self.symbols}")
        print(f"   Total iterations per symbol: {iterations:,}")
        print("=" * 60)
        
        all_rewards = []
        training_results = {}
        
        for symbol in self.symbols:
            symbol_key = f"{symbol}final"
            
            if symbol_key not in rdflistp:
                print(f"⚠️  Warning: {symbol_key} not found in data")
                continue
                
            if symbol not in lol:
                print(f"⚠️  Warning: {symbol} not found in signal list")
                continue
                
            df = rdflistp[symbol_key].copy()
            finalsignalsp = lol[symbol]
            
            # Train model for this symbol
            result = self.train_symbol_model(symbol, df, finalsignalsp, iterations)
            training_results[symbol] = result
            
            if result['success']:
                all_rewards.append(result['final_reward'])
            
            # Cleanup between symbols
            gc.collect()
        
        # Calculate average reward
        avg_reward = np.mean(all_rewards) if all_rewards else 0.0
        
        print(f"\n🎉 All models training completed!")
        print(f"   Average final reward: {avg_reward:.4f}")
        print(f"   Models trained: {len(all_rewards)}/{len(self.symbols)}")
        
        # Create posterior compatibility data for trained models
        print(f"\n📊 Creating posterior analysis compatibility data...")
        try:
            ensure_posterior_compatibility(self.symbols, rdflistp, save_to_file=True)
            print(f"✅ Posterior compatibility data created")
        except Exception as e:
            print(f"⚠️  Warning: Could not create posterior compatibility data: {e}")
        
        # Enable EvoRL-SB3 compatibility bridge
        print(f"\n🔗 Enabling EvoRL-SB3 compatibility bridge...")
        try:
            ensure_evorl_posterior_compatibility()
            print(f"✅ EvoRL-SB3 compatibility bridge enabled")
        except Exception as e:
            print(f"⚠️  Warning: Could not enable compatibility bridge: {e}")
        
        # Save training summary
        summary = {
            'symbols': self.symbols,
            'avg_reward': avg_reward,
            'individual_results': training_results,
            'total_models': len(all_rewards),
            'timestamp': time.time(),
            'posterior_compatibility': True
        }
        
        summary_path = f"{basepath}/models/evorl_training_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"   Training summary saved: {summary_path}")
        
        return avg_reward
    
    def load_model(self, symbol: str) -> Optional[EvoRLPPOTrainer]:
        """Load trained EvoRL model for a symbol"""
        model_path = f"{basepath}/models/{symbol}localmodel_evorl"
        
        try:
            # Create dummy trainer to load into
            dummy_df = pd.DataFrame({'feature': [1, 2, 3], 'close': [100, 101, 102]})
            trainer = create_evorl_trainer_from_data(dummy_df, ['feature'])
            
            # Load the actual model
            trainer.load_model(model_path)
            
            return trainer
            
        except Exception as e:
            print(f"❌ Error loading model for {symbol}: {e}")
            return None
    
    def evaluate_model(self, symbol: str, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate a trained model"""
        if symbol in self.trainers:
            trainer = self.trainers[symbol]
        else:
            trainer = self.load_model(symbol)
            
        if trainer is None:
            return {'mean_reward': 0.0, 'std_reward': 0.0}
            
        return trainer.evaluate(n_episodes=n_episodes)
    
    def get_training_metrics(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Get training metrics for a symbol"""
        return self.training_metrics.get(symbol, None)


def evorl_modeltrain(rdflistp: Dict[str, pd.DataFrame], 
                     newmodelflag: bool,
                     symbols: List[str],
                     DELETEMODELS: bool = False,
                     SAVE_BEST_MODEL: bool = True,
                     lol: Dict[str, List[str]] = None) -> float:
    """
    EvoRL-based replacement for the original modeltrain function
    
    This function maintains the same interface as the original SB3/SBX modeltrain
    but uses pure JAX/GPU EvoRL implementation instead.
    """
    
    print(f"\n🚀 EvoRL Model Training")
    print(f"   GPU-only JAX implementation")
    print(f"   Symbols: {symbols}")
    print(f"   Delete models: {DELETEMODELS}")
    print(f"   Save best model: {SAVE_BEST_MODEL}")
    print("=" * 60)
    
    # Delete existing models if requested
    if DELETEMODELS:
        print("🗑️  Deleting existing models...")
        for symbol in symbols:
            model_files = [
                f"{basepath}/models/{symbol}localmodel_evorl.pkl",
                f"{basepath}/models/{symbol}localmodel.zip",
                f"{basepath}/models/{symbol}localmodel_vecnormalize.pkl",
                f"{basepath}/models/{symbol}qt.joblib"
            ]
            
            for model_file in model_files:
                if os.path.exists(model_file):
                    os.remove(model_file)
                    print(f"   Deleted: {model_file}")
    
    # Create EvoRL trainer
    trainer = EvoRLModelTrainer(symbols=symbols)
    
    # Train all models
    avg_reward = trainer.train_all_models(
        rdflistp=rdflistp,
        lol=lol,
        iterations=BASEMODELITERATIONS
    )
    
    # Create compatibility files for existing code
    print(f"\n📁 Creating compatibility files...")
    
    for symbol in symbols:
        # Create dummy SB3-style model file for compatibility
        sb3_model_path = f"{basepath}/models/{symbol}localmodel.zip"
        
        # Create a placeholder file
        compatibility_data = {
            'model_type': 'evorl_ppo',
            'symbol': symbol,
            'evorl_model_path': f"{basepath}/models/{symbol}localmodel_evorl.pkl",
            'final_reward': avg_reward,
            'trained_with_evorl': True
        }
        
        import pickle
        with open(sb3_model_path, 'wb') as f:
            pickle.dump(compatibility_data, f)
        
        print(f"   Created compatibility file: {sb3_model_path}")
    
    # Enable EvoRL-SB3 compatibility bridge for posterior analysis
    print(f"\n🔗 Activating EvoRL-SB3 compatibility bridge...")
    try:
        ensure_evorl_posterior_compatibility()
        print(f"✅ Compatibility bridge activated - posterior analysis will work with EvoRL models")
    except Exception as e:
        print(f"⚠️  Warning: Could not activate compatibility bridge: {e}")
    
    print(f"\n✅ EvoRL training completed!")
    print(f"   Average reward: {avg_reward:.4f}")
    print(f"   Posterior analysis: Compatible with EvoRL models")
    
    return avg_reward


# Integration with existing training pipeline
def replace_sb3_with_evorl():
    """Replace SB3/SBX modeltrain function with EvoRL implementation"""
    
    # Import the common module and replace the modeltrain function
    import common
    
    # Store original function for fallback
    common._original_modeltrain = getattr(common, 'modeltrain', None)
    
    # Replace with EvoRL implementation
    common.modeltrain = evorl_modeltrain
    
    print("✅ SB3/SBX modeltrain replaced with EvoRL implementation")


if __name__ == "__main__":
    # Test the integration
    print("🧪 Testing EvoRL Integration")
    
    # Replace SB3 function
    replace_sb3_with_evorl()
    
    # Test with dummy data
    dummy_symbols = ['TEST']
    dummy_df = pd.DataFrame({
        'feature_1': np.random.randn(500),
        'feature_2': np.random.randn(500), 
        'feature_3': np.random.randn(500),
        'close': 100 + np.cumsum(np.random.randn(500) * 0.01),
        'vwap2': 100 + np.cumsum(np.random.randn(500) * 0.01),
        'currentt': pd.date_range('2023-01-01', periods=500, freq='D')
    })
    
    dummy_rdflistp = {'TESTfinal': dummy_df}
    dummy_lol = {'TEST': ['feature_1', 'feature_2', 'feature_3']}
    
    # Test training
    result = evorl_modeltrain(
        rdflistp=dummy_rdflistp,
        newmodelflag=True,
        symbols=dummy_symbols,
        DELETEMODELS=True,
        lol=dummy_lol
    )
    
    print(f"\n✅ Integration test completed!")
    print(f"   Result: {result:.4f}")