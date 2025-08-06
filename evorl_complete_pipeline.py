#!/usr/bin/env python3
"""
EvoRL Complete Pipeline - GPU-Only Implementation
Includes: Training, Test Period Assessment, and Deployment
NO SB3 dependencies - Pure JAX/GPU implementation
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import joblib

# Setup paths
basepath = '/Users/skumar81/Desktop/Personal/trading-final'
sys.path.insert(0, basepath)
os.chdir(basepath)

# Import parameters and utilities
from parameters import *
from lib import *
from evorl_ppo_trainer import EvoRLPPOTrainer, create_evorl_trainer_from_data
from StockTradingEnv2 import StockTradingEnv2 as StockTradingEnv


class EvoRLCompletePipeline:
    """Complete EvoRL pipeline with training, testing, and deployment"""
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or TESTSYMBOLS
        self.trainers = {}
        self.test_results = {}
        self.deployment_models = {}
        
        print(f"🚀 EvoRL Complete Pipeline Initialized")
        print(f"   Pure JAX/GPU implementation - NO SB3")
        print(f"   Symbols: {self.symbols}")
        
    def train_and_evaluate(self, 
                          rdflistp: Dict[str, pd.DataFrame],
                          lol: Dict[str, List[str]],
                          train_end_date: str = None,
                          test_days: int = 30) -> Dict[str, Any]:
        """
        Train models and evaluate on test period
        
        Args:
            rdflistp: Dictionary of dataframes for each symbol
            lol: Dictionary of feature lists for each symbol
            train_end_date: End date for training (format: 'YYYY-MM-DD')
            test_days: Number of days for test period evaluation
        """
        
        print(f"\n📊 EvoRL Training and Evaluation Pipeline")
        print(f"   Test period: {test_days} days")
        print("=" * 60)
        
        all_results = {}
        
        for symbol in self.symbols:
            symbol_key = f"{symbol}final"
            
            if symbol_key not in rdflistp:
                print(f"⚠️  Warning: {symbol_key} not found in data")
                continue
                
            if symbol not in lol:
                print(f"⚠️  Warning: {symbol} not found in signal list")
                continue
            
            # Get data and features
            df_full = rdflistp[symbol_key].copy()
            finalsignalsp = lol[symbol]
            
            # Split data into train and test periods
            if train_end_date:
                train_end_idx = df_full[df_full['currentt'] <= pd.to_datetime(train_end_date)].index[-1]
            else:
                # Use last test_days for testing
                train_end_idx = len(df_full) - test_days
            
            df_train = df_full.iloc[:train_end_idx].copy()
            df_test = df_full.iloc[train_end_idx:].copy()
            
            print(f"\n📈 Processing {symbol}")
            print(f"   Total data: {len(df_full)} rows")
            print(f"   Train data: {len(df_train)} rows (up to index {train_end_idx})")
            print(f"   Test data: {len(df_test)} rows")
            print("-" * 40)
            
            # Train model
            train_results = self._train_symbol(symbol, df_train, finalsignalsp)
            
            if train_results['success']:
                # Evaluate on test period
                test_results = self._evaluate_test_period(
                    symbol, df_test, finalsignalsp, train_results['trainer']
                )
                
                all_results[symbol] = {
                    'train': train_results,
                    'test': test_results,
                    'metrics': self._calculate_performance_metrics(test_results)
                }
                
                # Store for deployment
                self.deployment_models[symbol] = train_results['trainer']
                
                print(f"\n✅ {symbol} Complete:")
                print(f"   Train reward: {train_results['final_reward']:.4f}")
                print(f"   Test return: {test_results['total_return']:.2%}")
                print(f"   Sharpe ratio: {test_results['sharpe_ratio']:.2f}")
                print(f"   Max drawdown: {test_results['max_drawdown']:.2%}")
            else:
                all_results[symbol] = {
                    'train': train_results,
                    'test': None,
                    'error': train_results.get('error', 'Training failed')
                }
        
        # Generate summary report
        self._generate_summary_report(all_results)
        
        return all_results
    
    def _train_symbol(self, symbol: str, df: pd.DataFrame, 
                     finalsignalsp: List[str]) -> Dict[str, Any]:
        """Train EvoRL model for a single symbol"""
        
        print(f"🏋️  Training {symbol} with EvoRL (GPU-only)")
        start_time = time.time()
        
        try:
            # Create EvoRL trainer
            trainer = create_evorl_trainer_from_data(
                df=df,
                finalsignalsp=finalsignalsp,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                learning_rate=GLOBALLEARNINGRATE,
                n_epochs=N_EPOCHS
            )
            
            # Store trainer
            self.trainers[symbol] = trainer
            
            # Train model
            results = trainer.train(total_timesteps=BASEMODELITERATIONS)
            
            # Save model
            model_path = f"{basepath}/models/{symbol}localmodel_evorl"
            trainer.save_model(model_path)
            
            # Create normalizer for compatibility
            from sklearn.preprocessing import QuantileTransformer
            feature_cols = [col for col in df.columns if col in finalsignalsp]
            feature_data = df[feature_cols].values
            
            qt = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
            qt.fit(feature_data)
            
            normalizer_path = f'{basepath}/models/{symbol}qt_evorl.joblib'
            joblib.dump(qt, normalizer_path)
            
            training_time = time.time() - start_time
            
            return {
                'success': True,
                'trainer': trainer,
                'final_reward': results['training_metrics'][-1]['mean_reward'],
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
                'error': str(e)
            }
    
    def _evaluate_test_period(self, symbol: str, df_test: pd.DataFrame,
                             finalsignalsp: List[str], 
                             trainer: EvoRLPPOTrainer) -> Dict[str, Any]:
        """Evaluate model performance on test period"""
        
        print(f"📊 Evaluating {symbol} on test period...")
        
        # Create test environment
        test_env = StockTradingEnv(
            df=df_test,
            nlags=NLAGS,
            maximum_short_value=MAXIMUM_SHORT_VALUE,
            cost_per_trade=COST_PER_TRADE,
            finalsignalsp=finalsignalsp
        )
        
        # Run evaluation
        obs = test_env.reset()
        
        # Track metrics
        positions = []
        returns = []
        actions_taken = []
        portfolio_values = [INITIAL_ACCOUNT_BALANCE]
        
        done = False
        step = 0
        
        while not done and step < len(df_test) - 1:
            # Get action from trained model
            obs_jax = jnp.array(obs, dtype=jnp.float32)
            (mean, std), _ = trainer.network.apply(trainer.params, obs_jax[None, :])
            action = np.array(mean[0])  # Use mean for deterministic evaluation
            
            # Step environment
            next_obs, reward, done, info = test_env.step(action)
            
            # Record metrics
            actions_taken.append(action)
            positions.append(info.get('position', 0))
            returns.append(info.get('step_return', 0))
            portfolio_values.append(info.get('total_value', portfolio_values[-1]))
            
            obs = next_obs
            step += 1
        
        # Calculate performance metrics
        returns_array = np.array(returns)
        portfolio_array = np.array(portfolio_values[1:])  # Exclude initial value
        
        # Calculate key metrics
        total_return = (portfolio_values[-1] - INITIAL_ACCOUNT_BALANCE) / INITIAL_ACCOUNT_BALANCE
        
        # Daily returns
        daily_returns = np.diff(portfolio_array) / portfolio_array[:-1]
        
        # Sharpe ratio (annualized)
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        winning_days = np.sum(daily_returns > 0)
        total_days = len(daily_returns)
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'portfolio_values': portfolio_values,
            'positions': positions,
            'actions': actions_taken,
            'daily_returns': daily_returns.tolist(),
            'final_value': portfolio_values[-1],
            'num_trades': np.sum(np.abs(np.diff(positions)) > 0.1)
        }
    
    def _calculate_performance_metrics(self, test_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate additional performance metrics"""
        
        if not test_results:
            return {}
        
        daily_returns = np.array(test_results['daily_returns'])
        
        # Sortino ratio (downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            sortino_ratio = np.sqrt(252) * np.mean(daily_returns) / downside_std if downside_std > 0 else 0
        else:
            sortino_ratio = float('inf')  # No negative returns
        
        # Calmar ratio
        annual_return = test_results['total_return'] * (252 / len(daily_returns)) if len(daily_returns) > 0 else 0
        calmar_ratio = annual_return / abs(test_results['max_drawdown']) if test_results['max_drawdown'] != 0 else 0
        
        return {
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'avg_daily_return': np.mean(daily_returns),
            'volatility': np.std(daily_returns) * np.sqrt(252),  # Annualized
            'skewness': self._calculate_skewness(daily_returns),
            'kurtosis': self._calculate_kurtosis(daily_returns)
        }
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        if len(returns) < 3:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(((returns - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns"""
        if len(returns) < 4:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(((returns - mean) / std) ** 4) - 3  # Excess kurtosis
    
    def _generate_summary_report(self, all_results: Dict[str, Any]) -> None:
        """Generate comprehensive summary report"""
        
        print("\n" + "=" * 80)
        print("📊 EVORL PERFORMANCE SUMMARY REPORT")
        print("=" * 80)
        
        summary_data = []
        
        for symbol, results in all_results.items():
            if results['test'] is not None:
                test_data = results['test']
                metrics = results['metrics']
                
                summary_data.append({
                    'Symbol': symbol,
                    'Train Reward': results['train']['final_reward'],
                    'Test Return': f"{test_data['total_return']:.2%}",
                    'Sharpe': f"{test_data['sharpe_ratio']:.2f}",
                    'Sortino': f"{metrics['sortino_ratio']:.2f}",
                    'Max DD': f"{test_data['max_drawdown']:.2%}",
                    'Win Rate': f"{test_data['win_rate']:.2%}",
                    'Trades': test_data['num_trades']
                })
        
        if summary_data:
            # Print table
            df_summary = pd.DataFrame(summary_data)
            print(df_summary.to_string(index=False))
            
            # Save report
            report_path = f"{basepath}/models/evorl_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(report_path, 'wb') as f:
                pickle.dump({
                    'summary': all_results,
                    'timestamp': datetime.now(),
                    'symbols': self.symbols
                }, f)
            
            print(f"\n✅ Report saved to: {report_path}")
    
    def deploy_model(self, symbol: str, realtime_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Deploy trained model for real-time trading decisions
        
        Args:
            symbol: Symbol to trade
            realtime_data: Current market data with required features
            
        Returns:
            Trading decision with action and confidence
        """
        
        if symbol not in self.deployment_models:
            raise ValueError(f"No trained model found for {symbol}")
        
        trainer = self.deployment_models[symbol]
        
        # Prepare observation
        # Assuming realtime_data has the required features
        obs = realtime_data.values.flatten()
        obs_jax = jnp.array(obs, dtype=jnp.float32)
        
        # Get action from model
        (mean, std), value = trainer.network.apply(trainer.params, obs_jax[None, :])
        
        action = np.array(mean[0])
        confidence = 1.0 / (1.0 + np.array(std[0]).mean())  # Higher confidence = lower std
        
        # Interpret action
        position_change = float(action[0])  # -1 to 1
        position_size = float(action[1]) if len(action) > 1 else 0.5  # 0 to 1
        
        decision = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'action': 'BUY' if position_change > 0.1 else 'SELL' if position_change < -0.1 else 'HOLD',
            'position_change': position_change,
            'position_size': position_size,
            'confidence': confidence,
            'model_value': float(value[0]),
            'raw_action': action.tolist()
        }
        
        return decision
    
    def save_deployment_models(self) -> None:
        """Save all trained models for deployment"""
        
        deployment_path = f"{basepath}/models/evorl_deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(deployment_path, exist_ok=True)
        
        for symbol, trainer in self.deployment_models.items():
            trainer.save_model(f"{deployment_path}/{symbol}_model")
        
        # Save metadata
        metadata = {
            'symbols': list(self.deployment_models.keys()),
            'training_date': datetime.now(),
            'parameters': {
                'n_steps': N_STEPS,
                'batch_size': BATCH_SIZE,
                'learning_rate': GLOBALLEARNINGRATE,
                'n_epochs': N_EPOCHS
            }
        }
        
        with open(f"{deployment_path}/metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Deployment models saved to: {deployment_path}")


def main():
    """Main function to run complete EvoRL pipeline"""
    
    print("🚀 EvoRL Complete Pipeline - GPU Only Implementation")
    print("=" * 60)
    
    # Initialize JAX GPU
    from jax_gpu_init import init_jax_gpu
    init_jax_gpu()
    
    # Create pipeline
    pipeline = EvoRLCompletePipeline(symbols=['BPCL'])  # Start with one symbol for testing
    
    # Create dummy data for testing
    # In production, this would be your actual data loading
    dummy_data = {}
    dummy_signals = {}
    
    for symbol in pipeline.symbols:
        df = pd.DataFrame({
            'currentt': pd.date_range('2023-01-01', periods=1000, freq='D'),
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
            'vwap2': 100 + np.cumsum(np.random.randn(1000) * 0.01),
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.randn(1000),
            'feature_3': np.random.randn(1000),
        })
        
        dummy_data[f"{symbol}final"] = df
        dummy_signals[symbol] = ['feature_1', 'feature_2', 'feature_3']
    
    # Run complete pipeline
    results = pipeline.train_and_evaluate(
        rdflistp=dummy_data,
        lol=dummy_signals,
        test_days=50  # Last 50 days for testing
    )
    
    # Test deployment
    if 'BPCL' in pipeline.deployment_models:
        # Simulate real-time data
        realtime_features = pd.DataFrame({
            'feature_1': [0.5],
            'feature_2': [-0.3],
            'feature_3': [0.1]
        })
        
        decision = pipeline.deploy_model('BPCL', realtime_features)
        
        print(f"\n📈 Deployment Test:")
        print(f"   Decision: {decision['action']}")
        print(f"   Confidence: {decision['confidence']:.2%}")
        print(f"   Position size: {decision['position_size']:.2%}")
    
    # Save deployment models
    pipeline.save_deployment_models()
    
    print("\n✅ EvoRL Pipeline Complete!")


if __name__ == "__main__":
    main()