#!/usr/bin/env python3
"""
train_evorl.py - EvoRL-based training script for GPU-only execution

This script replaces SB3/SBX with pure JAX/GPU EvoRL implementation for:
- Massive GPU utilization improvement
- Better parallelization
- Pure JAX compilation benefits
- Elimination of CPU/GPU transfer bottlenecks
"""

import os
import sys
import warnings
import argparse
from datetime import datetime

# Initialize JAX GPU support BEFORE any other imports
basepath = '/Users/skumar81/Desktop/Personal/trading-final'
sys.path.insert(0, basepath)
os.chdir(basepath)

# Import JAX GPU init first
from jax_gpu_init import init_jax_gpu, get_jax_status

# Suppress warnings
warnings.filterwarnings('ignore')

# Import EvoRL integration
from evorl_integration import replace_sb3_with_evorl, EvoRLModelTrainer

# Import existing training infrastructure
from parameters import *
from model_trainer import ModelTrainer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train PPO Trading Model with EvoRL')
    
    parser.add_argument(
        '--device', 
        choices=['gpu'], 
        default='gpu',
        help='Device to use for training (EvoRL supports GPU only)'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=None,
        help='Symbols to train on (default: use TESTSYMBOLS from parameters)'
    )
    
    parser.add_argument(
        '--no-preprocessing',
        action='store_true',
        help='Skip data preprocessing (use existing data)'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Override training timesteps (default: use BASEMODELITERATIONS)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--test-run',
        action='store_true',
        help='Run with reduced timesteps for testing'
    )
    
    parser.add_argument(
        '--enable-posterior',
        action='store_true',
        help='Enable posterior analysis (now compatible with EvoRL)'
    )
    
    return parser.parse_args()


def main():
    """Main training function with EvoRL"""
    args = parse_arguments()
    
    # Initialize JAX GPU
    print("🚀 Initializing JAX GPU for EvoRL...")
    init_jax_gpu()
    jax_status = get_jax_status()
    
    if jax_status['backend'] != 'gpu':
        print("❌ Error: EvoRL requires GPU backend")
        print("   Please ensure CUDA and JAX GPU are properly installed")
        return 1
    
    # Replace SB3/SBX with EvoRL
    print("🔄 Replacing SB3/SBX with EvoRL implementation...")
    replace_sb3_with_evorl()
    
    # Print system information
    print("\n🚀 EvoRL PPO Trading Model Training")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    print(f"Backend: {jax_status['backend']}")
    print(f"Devices: {jax_status['devices']}")
    print(f"Device count: {jax_status['device_count']}")
    
    if jax_status['backend'] == 'gpu':
        print(f"GPU memory: {jax_status.get('gpu_memory', 'Unknown')}")
    
    # Override timesteps for test run
    if args.test_run:
        print("🧪 Test run mode - using reduced timesteps")
        import parameters
        original_iterations = parameters.BASEMODELITERATIONS
        parameters.BASEMODELITERATIONS = 5000  # Reduced for testing
    
    # Override timesteps if specified
    if args.timesteps:
        import parameters
        original_iterations = parameters.BASEMODELITERATIONS
        parameters.BASEMODELITERATIONS = args.timesteps
        print(f"Training timesteps override: {args.timesteps:,}")
    
    print("=" * 60)
    
    try:
        # Create trainer instance (using existing infrastructure)
        symbols = args.symbols if args.symbols else TESTSYMBOLS
        trainer = ModelTrainer(symbols=symbols)
        
        if args.verbose:
            print(f"Training symbols: {symbols}")
        
        # Step 1: Data preprocessing (unless skipped)
        if not args.no_preprocessing:
            print("\n📊 Step 1: Data Preprocessing")
            print("-" * 30)
            processed_data, _ = trainer.preprocess_data()
            
            if processed_data is None:
                print("⚠️  Preprocessing failed, trying to load existing data...")
            
            # Always load historical data after preprocessing
            print("Loading processed data...")
            trainer.load_historical_data()
        else:
            print("\n📊 Step 1: Loading Existing Data")
            print("-" * 30)
            trainer.load_historical_data()
        
        # Step 2: Signal extraction
        print("\n🔍 Step 2: Signal Extraction")
        print("-" * 30)
        globalsignals = trainer.extract_signals()
        
        if globalsignals is None:
            print("❌ Signal extraction failed!")
            return 1
        
        # Step 3: EvoRL Model training
        print("\n🏋️  Step 3: EvoRL Model Training (GPU-only)")
        print("-" * 30)
        print("🚀 Using pure JAX/GPU EvoRL implementation")
        print("   ✅ Massive GPU utilization improvement")
        print("   ✅ Elimination of CPU/GPU transfer bottlenecks")
        print("   ✅ Pure JAX compilation benefits")
        print("")
        
        # Train models using EvoRL (the modeltrain function has been replaced)
        reward = trainer.train_models_with_params()
        
        # Restore original timesteps if modified
        if args.timesteps or args.test_run:
            import parameters
            parameters.BASEMODELITERATIONS = original_iterations
        
        # Results
        print("\n✅ EvoRL Training Results")
        print("-" * 30)
        if reward is not None:
            print(f"Final reward: {reward:.4f}")
            print("🚀 GPU-accelerated training completed successfully!")
        else:
            print("Training completed with warnings")
        
        # Step 4: Generate posterior analysis (optional with EvoRL compatibility)
        if args.enable_posterior:
            print("\n📈 Step 4: Posterior Analysis (EvoRL Compatible)")
            print("-" * 30)
            try:
                # Import the posterior analysis functionality
                from common import generateposterior
                
                # Run posterior analysis with EvoRL-SB3 compatibility bridge
                print("🔗 Using EvoRL-SB3 compatibility bridge for posterior analysis")
                generateposterior()
                print("✅ Posterior analysis completed successfully with EvoRL models")
            except Exception as e:
                print(f"❌ Posterior analysis failed: {e}")
                print("   Compatibility bridge may need adjustment")
                import traceback
                traceback.print_exc()
        else:
            print("\n📈 Step 4: Posterior Analysis (Disabled)")
            print("-" * 30)
            print("Posterior analysis disabled (use --enable-posterior to enable)")
            print("💡 EvoRL models now support posterior analysis via compatibility bridge")
        
        print(f"\n🎉 EvoRL training pipeline completed at {datetime.now()}")
        print("🚀 GPU-only training achieved maximum performance!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)