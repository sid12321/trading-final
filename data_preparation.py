#!/usr/bin/env python3
"""
Data Preparation for GPU Training
Handles all data loading, preprocessing, and signal extraction
"""

import os
import sys
import time
import joblib
import pandas as pd
from datetime import timedelta

# Add path for imports
sys.path.append('/home/sid12321/Desktop/Trading-Final')

from parameters import *
from lib import *
from common import *
import kitelogin

class DataPreparator:
    """Handles all data preparation for training"""
    
    def __init__(self, symbols=None):
        self.symbols = symbols or TESTSYMBOLS
        self.rdflistp = {}
        self.lol = {}
        self.qtnorm = {}
        
    def preprocess_live_data(self):
        """Download and preprocess live data from Kite"""
        print("=" * 60)
        print("LIVE DATA PREPROCESSING")
        print("=" * 60)
        
        if not PREPROCESS:
            print("Preprocessing disabled in parameters")
            return False
            
        print("Logging into Kite...")
        kite = kitelogin.login_to_kite()
        
        if kite:
            print("✓ Kite login successful")
            print("Starting data preprocessing...")
            start_time = time.time()
            
            preprocess(kite)
            
            elapsed = time.time() - start_time
            print(f"✓ Preprocessing completed in {timedelta(seconds=elapsed)}")
            return True
        else:
            print("✗ Failed to login to Kite")
            return False
    
    def load_historical_data(self):
        """Load historical training data"""
        print("\n" + "=" * 60)
        print("LOADING HISTORICAL DATA")
        print("=" * 60)
        
        start_time = time.time()
        
        for SYM in self.symbols:
            print(f"Loading {SYM}...")
            
            # Load data
            df_path = f"{basepath}/traindata/finalmldf{SYM}.csv"
            if not os.path.exists(df_path):
                print(f"✗ Data file not found: {df_path}")
                continue
                
            df = pd.read_csv(df_path)
            df = df.drop(['t'], axis=1).head(len(df) - 1)
            
            # Clean data
            if "Unnamed: 0" in df.columns:
                df = df.drop(["Unnamed: 0"], axis=1)
            df['currentt'] = pd.to_datetime(df['currentt'])
            df['currentdate'] = df['currentt'].dt.date
            
            # Extract signals
            finalsignalsp = df.columns[~df.columns.isin(['currentt', 'currento', 'currentdate', 'vwap2'])].tolist()
            
            self.rdflistp[SYM + 'final'] = df
            self.lol[SYM] = finalsignalsp
            
            print(f"  ✓ {SYM}: {len(df)} rows, {len(finalsignalsp)} signals")
        
        elapsed = time.time() - start_time
        print(f"\n✓ Data loading completed in {timedelta(seconds=elapsed)}")
        print(f"✓ Loaded {len(self.symbols)} symbols")
        
        return True
    
    def extract_and_optimize_signals(self):
        """Extract and optimize trading signals"""
        print("\n" + "=" * 60)
        print("SIGNAL EXTRACTION & OPTIMIZATION")
        print("=" * 60)
        
        start_time = time.time()
        
        # Extract basic signals
        globalsignals = self._extract_global_signals()
        
        # Apply CPU optimization if available
        try:
            from optimized_signal_generator_cpu import generate_optimized_signals_cpu
            print("Applying CPU-optimized signal generation...")
            
            main_symbol = 'BPCL'
            if main_symbol + 'final' in self.rdflistp:
                df = self.rdflistp[main_symbol + 'final'].copy()
                
                # Ensure required columns
                if 'vwap' not in df.columns and 'vwap2' in df.columns:
                    df['vwap'] = df['vwap2']
                if 'date' not in df.columns and 'currentdate' in df.columns:
                    df['date'] = df['currentdate']
                
                # Generate optimized signals
                optimized_signals, enhanced_df = generate_optimized_signals_cpu(
                    df, globalsignals, use_parallel=True, batch_size=20
                )
                
                # Update dataframe
                self.rdflistp[main_symbol + 'final'] = enhanced_df
                print(f"✓ Generated {len(optimized_signals)} optimized signal variants")
            
        except ImportError:
            print("CPU optimization not available, using standard signals")
        except Exception as e:
            print(f"CPU optimization failed: {e}, using standard signals")
        
        elapsed = time.time() - start_time
        print(f"✓ Signal processing completed in {timedelta(seconds=elapsed)}")
        
        return globalsignals
    
    def _extract_global_signals(self):
        """Extract signals common to all symbols"""
        print("Extracting global signals...")
        
        lolist = []
        for SYM in self.symbols:
            if SYM in self.lol:
                lolist.append(self.lol[SYM])
        
        if lolist:
            globalsignals = list(set.intersection(*[set(signals) for signals in lolist]))
            print(f"✓ Found {len(globalsignals)} global signals")
            return globalsignals
        else:
            print("✗ No signals found")
            return []
    
    def prepare_quantile_normalizers(self):
        """Prepare quantile normalizers for each symbol"""
        print("\n" + "=" * 60)
        print("PREPARING QUANTILE NORMALIZERS")
        print("=" * 60)
        
        start_time = time.time()
        
        for SYM in self.symbols:
            if SYM not in self.lol:
                continue
                
            print(f"Preparing normalizer for {SYM}...")
            
            try:
                # Load existing normalizer if available
                normalizer_path = f'{basepath}/models/{SYM}qt.joblib'
                if os.path.exists(normalizer_path):
                    self.qtnorm[SYM] = joblib.load(normalizer_path)
                    print(f"  ✓ Loaded existing normalizer")
                else:
                    print(f"  ! Normalizer not found, will be created during training")
                    
            except Exception as e:
                print(f"  ✗ Error loading normalizer: {e}")
        
        elapsed = time.time() - start_time
        print(f"✓ Normalizer preparation completed in {timedelta(seconds=elapsed)}")
    
    def save_prepared_data(self, output_dir="prepared_data"):
        """Save all prepared data for training"""
        print("\n" + "=" * 60)
        print("SAVING PREPARED DATA")
        print("=" * 60)
        
        os.makedirs(output_dir, exist_ok=True)
        start_time = time.time()
        
        # Save dataframes
        data_info = {}
        for key, df in self.rdflistp.items():
            # Try parquet first, fallback to pickle
            try:
                output_path = f"{output_dir}/{key}.parquet"
                df.to_parquet(output_path, compression='snappy')
                format_used = 'parquet'
            except ImportError:
                print(f"Parquet not available, using pickle for {key}")
                output_path = f"{output_dir}/{key}.pkl"
                df.to_pickle(output_path)
                format_used = 'pickle'
            
            data_info[key] = {
                'path': output_path,
                'format': format_used,
                'rows': len(df),
                'columns': len(df.columns)
            }
            print(f"✓ Saved {key}: {len(df)} rows ({format_used})")
        
        # Save signal lists
        joblib.dump(self.lol, f"{output_dir}/signal_lists.joblib")
        print(f"✓ Saved signal lists for {len(self.lol)} symbols")
        
        # Save normalizers
        if self.qtnorm:
            joblib.dump(self.qtnorm, f"{output_dir}/quantile_normalizers.joblib")
            print(f"✓ Saved {len(self.qtnorm)} normalizers")
        
        # Save metadata
        metadata = {
            'symbols': self.symbols,
            'data_info': data_info,
            'preparation_timestamp': time.time(),
            'parameters_snapshot': {
                'BATCH_SIZE': BATCH_SIZE,
                'N_STEPS': N_STEPS,
                'N_EPOCHS': N_EPOCHS,
                'GLOBALLEARNINGRATE': GLOBALLEARNINGRATE
            }
        }
        joblib.dump(metadata, f"{output_dir}/metadata.joblib")
        
        elapsed = time.time() - start_time
        print(f"✓ Data saving completed in {timedelta(seconds=elapsed)}")
        print(f"✓ All data saved to: {output_dir}/")
        
        return output_dir
    
    def run_full_preparation(self, save_data=True):
        """Run complete data preparation pipeline"""
        print("STARTING COMPLETE DATA PREPARATION PIPELINE")
        print("=" * 80)
        
        pipeline_start = time.time()
        
        # Step 1: Preprocess live data if enabled
        if PREPROCESS:
            self.preprocess_live_data()
        
        # Step 2: Load historical data
        if not self.load_historical_data():
            print("✗ Data loading failed")
            return False
        
        # Step 3: Extract and optimize signals
        globalsignals = self.extract_and_optimize_signals()
        if not globalsignals:
            print("✗ Signal extraction failed")
            return False
        
        # Step 4: Prepare normalizers
        self.prepare_quantile_normalizers()
        
        # Step 5: Save prepared data
        if save_data:
            output_dir = self.save_prepared_data()
        
        # Summary
        total_time = time.time() - pipeline_start
        print("\n" + "=" * 80)
        print("DATA PREPARATION COMPLETE")
        print("=" * 80)
        print(f"Total time: {timedelta(seconds=total_time)}")
        print(f"Symbols prepared: {len(self.symbols)}")
        print(f"Global signals: {len(globalsignals)}")
        
        if save_data:
            print(f"Data saved to: {output_dir}/")
            print("\nNext steps:")
            print("1. Run hyperparameter optimization: python hyperparameter_tuning.py")
            print("2. Run GPU training: python gpu_model_trainer.py")
        
        return True

def main():
    """Run data preparation"""
    preparator = DataPreparator()
    success = preparator.run_full_preparation()
    
    if success:
        print("\n✓ Data preparation successful!")
        return 0
    else:
        print("\n✗ Data preparation failed!")
        return 1

if __name__ == "__main__":
    exit(main())