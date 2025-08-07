#!/usr/bin/env python3
"""
train_evorl_only.py - Pure EvoRL Training Script
NO SB3 dependencies - GPU-only implementation with JAX
Includes: Training, Test Period Evaluation, and Deployment
"""

import os
import sys
import argparse
import warnings
from datetime import datetime

# OPTIMIZATION: Configure threading for M4 Max 16 CPU cores BEFORE any imports
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['XLA_CPU_MULTI_THREAD_EIGEN'] = 'true'
os.environ['JAX_ENABLE_X64'] = 'false'
os.environ['OPENBLAS_NUM_THREADS'] = '16'

# Setup paths and suppress warnings
basepath = '/Users/skumar81/Desktop/Personal/trading-final'
sys.path.insert(0, basepath)
os.chdir(basepath)
warnings.filterwarnings('ignore')

# Initialize JAX GPU FIRST
from jax_gpu_init import init_jax_gpu, get_jax_status, is_gpu_available

# Import Metal compatibility layer for M4 Max
from jax_metal_compat import setup_jax_for_metal, patch_jax_for_metal, restore_jax_functions

# Import EvoRL complete pipeline
from evorl_complete_pipeline import EvoRLCompletePipeline

# Import data loading utilities
from parameters import *
from lib import *
import pandas as pd
import numpy as np


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Pure EvoRL GPU Training - NO SB3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single symbol with test evaluation
  python train_evorl_only.py --symbols BPCL --test-days 30
  
  # Train multiple symbols
  python train_evorl_only.py --symbols BPCL HDFCLIFE TATASTEEL --test-days 60
  
  # Full training with custom iterations
  python train_evorl_only.py --timesteps 100000 --test-days 90
  
  # Train and deploy
  python train_evorl_only.py --deploy --test-days 30
        """
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=None,
        help='Symbols to train (default: use TESTSYMBOLS from parameters.py)'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help=f'Total training timesteps (default: {BASEMODELITERATIONS})'
    )
    
    parser.add_argument(
        '--test-days',
        type=int,
        default=30,
        help='Number of days for test period evaluation (default: 30)'
    )
    
    parser.add_argument(
        '--no-preprocessing',
        action='store_true',
        help='Skip data preprocessing (use existing data)'
    )
    
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Save models for deployment after training'
    )
    
    parser.add_argument(
        '--train-end-date',
        type=str,
        default=None,
        help='End date for training period (format: YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def load_data(symbols, no_preprocessing=False):
    """Load data for training"""
    
    print("\n📊 Loading Data")
    print("-" * 30)
    
    # Initialize storage
    rdflistp = {}
    lol = {}
    
    # If not skipping preprocessing, run fresh data pull
    if not no_preprocessing:
        print("🔄 Fetching fresh data from Kite...")
        try:
            from evorl_data_preparation import fetch_fresh_data_from_kite, validate_existing_data
            
            # Check if we have recent data
            if validate_existing_data(symbols):
                print("   ✅ Recent data files found, skipping fresh data pull")
            else:
                print("   📡 Fetching fresh data from Kite...")
                success = fetch_fresh_data_from_kite(symbols)
                if success:
                    print("   ✅ Fresh data preprocessing completed")
                else:
                    print("   ⚠️  Fresh data pull failed, using existing data")
                
        except Exception as e:
            print(f"   ⚠️  Data preprocessing error: {e}")
            print("   Falling back to existing data files")
    else:
        print("📁 Using existing data files (--no-preprocessing)")
    
    # Load data for each symbol
    for symbol in symbols:
        try:
            # Try different data file patterns
            possible_paths = [
                f"{basepath}/traindata/finalmldf{symbol}.csv",
                f"{basepath}/traindata/finalmldf2{symbol}.csv", 
                f"{basepath}/traindata/mldf{symbol}.csv",
                f"{basepath}/traindata/data_{symbol}.csv"
            ]
            
            df = None
            for df_path in possible_paths:
                if os.path.exists(df_path):
                    df = pd.read_csv(df_path)
                    print(f"✓ Loaded {symbol}: {len(df)} rows from {os.path.basename(df_path)}")
                    break
            
            if df is None:
                print(f"❌ No data file found for {symbol}")
                print(f"   Please run data preprocessing first or ensure data files exist")
                continue
            
            # Ensure required columns for signal generator
            if 'vwap2' in df.columns and 'vwap' not in df.columns:
                df['vwap'] = df['vwap2']
            elif 'vwap' not in df.columns:
                df['vwap'] = df['close']  # Fallback to close price
                
            if 'currentt' in df.columns and 'date' not in df.columns:
                df['date'] = pd.to_datetime(df['currentt']).dt.date
            elif 'date' not in df.columns:
                df['date'] = pd.date_range('2022-01-01', periods=len(df), freq='D').date
            
            # Extract features from the loaded data
            print(f"   Extracting features for {symbol}...")
            
            # Clean up dataframe 
            if "Unnamed: 0" in df.columns:
                df = df.drop(["Unnamed: 0"], axis=1)
            if 't' in df.columns:
                df = df.drop(['t'], axis=1)
            
            # Ensure datetime column
            if 'currentt' in df.columns:
                df['currentt'] = pd.to_datetime(df['currentt'])
                df['currentdate'] = df['currentt'].dt.date
            
            # Drop last incomplete row
            df = df.head(len(df) - 1)
            
            # Extract feature columns (everything except time and price columns)
            exclude_cols = ['currentt', 'currento', 'currentdate', 'vwap2', 'vwap', 'date', 'close', 'open', 'high', 'low', 'volume']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            if not feature_cols:
                print(f"   ❌ No valid features found for {symbol}")
                print(f"   Data file may be incomplete or improperly formatted")
                continue
            
            lol[symbol] = feature_cols
            print(f"   ✅ Extracted {len(feature_cols)} features for {symbol}")
            
            # Store in expected format
            rdflistp[f"{symbol}final"] = df
            
        except Exception as e:
            print(f"❌ Error loading {symbol}: {e}")
            continue
    
    print(f"\n✅ Data loaded for {len(rdflistp)} symbols")
    return rdflistp, lol


def main():
    """Main training function"""
    args = parse_arguments()
    
    print("🚀 Pure EvoRL GPU Training Pipeline")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    print("NO SB3 - Pure JAX/GPU Implementation")
    print("=" * 60)
    
    # Initialize JAX GPU with Metal compatibility
    print("\n🔥 Initializing JAX GPU...")
    
    # Setup Metal compatibility first (for M4 Max)
    has_metal = setup_jax_for_metal()
    if has_metal:
        print("🍎 Applying Metal compatibility patches...")
        patch_jax_for_metal()
    
    # Initialize JAX GPU
    init_jax_gpu()
    jax_status = get_jax_status()
    
    if not jax_status.get('has_gpu', False):
        print("❌ ERROR: JAX GPU backend not available!")
        print("   EvoRL requires GPU for optimal performance")
        print("   Please check your JAX installation")
        print(f"   Current backend: {jax_status.get('backend', 'unknown')}")
        print(f"   Available devices: {jax_status.get('devices', [])}")
        return 1
    
    print(f"✅ JAX GPU initialized: {jax_status['devices'][0]}")
    if jax_status.get('gpu_type'):
        print(f"   GPU Type: {jax_status['gpu_type']}")
    if jax_status.get('gpu_memory'):
        print(f"   GPU Memory: {jax_status['gpu_memory']}")
    
    # Get symbols
    symbols = args.symbols or TESTSYMBOLS
    print(f"\n📈 Training symbols: {symbols}")
    
    # Override timesteps if provided
    if args.timesteps:
        global BASEMODELITERATIONS
        BASEMODELITERATIONS = args.timesteps
        print(f"   Training timesteps: {BASEMODELITERATIONS:,}")
    
    # Load data
    rdflistp, lol = load_data(symbols, args.no_preprocessing)
    
    if not rdflistp:
        print("❌ No data loaded, exiting")
        return 1
    
    # Create EvoRL pipeline
    print("\n🏗️  Creating EvoRL Pipeline...")
    pipeline = EvoRLCompletePipeline(symbols=symbols)
    
    # Train and evaluate
    print("\n🚀 Starting Training and Evaluation...")
    results = pipeline.train_and_evaluate(
        rdflistp=rdflistp,
        lol=lol,
        train_end_date=args.train_end_date,
        test_days=args.test_days
    )
    
    # Save deployment models if requested
    if args.deploy:
        print("\n💾 Saving deployment models...")
        pipeline.save_deployment_models()
        
        # Test deployment with sample data
        print("\n🧪 Testing deployment...")
        for symbol in symbols[:1]:  # Test first symbol
            if symbol in pipeline.deployment_models:
                # Create sample real-time data
                sample_features = lol[symbol][:5]  # Use first 5 features
                realtime_data = pd.DataFrame({
                    feat: [np.random.randn()] for feat in sample_features
                })
                
                decision = pipeline.deploy_model(symbol, realtime_data)
                
                print(f"\n📊 Deployment test for {symbol}:")
                print(f"   Action: {decision['action']}")
                print(f"   Confidence: {decision['confidence']:.2%}")
                print(f"   Position: {decision['position_size']:.2%}")
    
    # Cleanup Metal compatibility patches
    if has_metal:
        print("🔧 Restoring original JAX functions...")
        restore_jax_functions()
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ EVORL TRAINING COMPLETE")
    print("=" * 60)
    print(f"End time: {datetime.now()}")
    
    # Summary statistics
    successful_symbols = sum(1 for s, r in results.items() if r['train']['success'])
    print(f"\nSummary:")
    print(f"  • Symbols trained: {successful_symbols}/{len(symbols)}")
    print(f"  • Test period: {args.test_days} days")
    print(f"  • GPU backend: {jax_status['backend']}")
    print(f"  • Pure JAX implementation: YES")
    print(f"  • SB3 dependencies: NONE")
    
    if successful_symbols > 0:
        avg_test_return = np.mean([r['test']['total_return'] for s, r in results.items() 
                                   if r['train']['success'] and r['test']])
        print(f"  • Average test return: {avg_test_return:.2%}")
    
    print("\n🎉 Pipeline execution successful!")
    return 0


if __name__ == "__main__":
    exit(main())