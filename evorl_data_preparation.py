#!/usr/bin/env python3
"""
EvoRL Data Preparation - Clean implementation without SB3 dependencies
Handles fresh data fetching from Kite for the EvoRL pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Tuple, Optional

# Setup paths
basepath = '/home/sid12321/Desktop/Trading-Final'
sys.path.insert(0, basepath) 
os.chdir(basepath)

from parameters import *
from lib import *
import kitelogin


def fetch_fresh_data_from_kite(symbols: list) -> bool:
    """
    Fetch fresh data from Kite without SB3 dependencies
    
    Args:
        symbols: List of symbols to fetch data for
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    try:
        print("   Logging into Kite...")
        kite = kitelogin.login_to_kite()
        
        if not kite:
            print("   ❌ Kite login failed")
            return False
            
        print("   ✓ Kite login successful")
        print("   Downloading fresh market data...")
        
        # Calculate date range
        LATEST_DATE = date.today() - timedelta(days=1)
        FROM_DATE = LATEST_DATE - timedelta(days=HORIZONDAYS)
        
        print(f"   Fetching data from {FROM_DATE} to {LATEST_DATE}")
        
        # Get instruments
        nse = pd.DataFrame(kite.instruments("NSE"))
        nfo = pd.DataFrame(kite.instruments("NFO"))
        
        # Process each symbol
        for symbol in symbols:
            try:
                print(f"   Processing {symbol}...")
                
                # Get instrument token
                instrument = nse[nse['tradingsymbol'] == symbol]
                if instrument.empty:
                    print(f"   ⚠️  {symbol} not found in NSE instruments")
                    continue
                    
                instrument_token = instrument['instrument_token'].iloc[0]
                
                # Fetch historical data
                historical_data = kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=FROM_DATE,
                    to_date=LATEST_DATE,
                    interval="minute"
                )
                
                if not historical_data:
                    print(f"   ⚠️  No data received for {symbol}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(historical_data)
                
                # Basic processing
                df['currentt'] = pd.to_datetime(df['date'])
                df['currento'] = df['open']
                df['vwap2'] = (df['high'] + df['low'] + df['close']) / 3  # Simple VWAP approximation
                
                # Add basic technical indicators
                df = add_basic_technical_indicators(df)
                
                # Save to file
                output_file = f"{basepath}/traindata/finalmldf{symbol}.csv"
                df.to_csv(output_file, index=False)
                
                print(f"   ✅ {symbol}: {len(df)} rows saved to {output_file}")
                
            except Exception as e:
                print(f"   ❌ Error processing {symbol}: {e}")
                continue
        
        print("   ✅ Fresh data preprocessing completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Data preprocessing failed: {e}")
        return False


def add_basic_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical indicators to the dataframe
    
    Args:
        df: Input dataframe with OHLCV data
        
    Returns:
        DataFrame with technical indicators added
    """
    
    # Ensure we have the required columns
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        print("   ⚠️  Missing OHLCV columns, skipping technical indicators")
        return df
    
    try:
        # Basic price features
        df['hl'] = df['high'] - df['low']
        df['co'] = df['close'] - df['open']
        df['v'] = df['volume']
        
        # Simple moving averages
        for period in [5, 10, 20]:
            df[f'sma{period}'] = df['close'].rolling(window=period).mean()
            df[f'price_vs_sma{period}'] = df['close'] / df[f'sma{period}'] - 1
        
        # Simple RSI approximation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Price momentum features
        for lag in [1, 2, 3, 5, 7]:
            df[f'lret{lag}'] = df['close'].pct_change(lag)
        
        # Volatility features
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Simple Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_sma = df['close'].rolling(window=bb_period).mean()
        bb_std_dev = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = bb_sma + (bb_std_dev * bb_std)
        df['bb_lower'] = bb_sma - (bb_std_dev * bb_std)
        df['bb_middle'] = bb_sma
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(0)
        
        print(f"   ✅ Added technical indicators ({len(df.columns)} total columns)")
        
    except Exception as e:
        print(f"   ⚠️  Error adding technical indicators: {e}")
    
    return df


def validate_existing_data(symbols: list) -> bool:
    """
    Check if we have recent data files for the symbols
    
    Args:
        symbols: List of symbols to check
        
    Returns:
        bool: True if recent data exists for all symbols
    """
    
    all_exist = True
    
    for symbol in symbols:
        file_paths = [
            f"{basepath}/traindata/finalmldf{symbol}.csv",
            f"{basepath}/traindata/finalmldf2{symbol}.csv",
            f"{basepath}/traindata/mldf{symbol}.csv"
        ]
        
        symbol_exists = False
        for file_path in file_paths:
            if os.path.exists(file_path):
                # Check if file is recent (less than 1 day old)
                file_age = (date.today() - date.fromtimestamp(os.path.getmtime(file_path))).days
                if file_age <= 1:
                    print(f"   ✅ Recent data found for {symbol} (age: {file_age} days)")
                    symbol_exists = True
                    break
                else:
                    print(f"   ⚠️  Data for {symbol} is {file_age} days old")
        
        if not symbol_exists:
            print(f"   ❌ No recent data found for {symbol}")
            all_exist = False
    
    return all_exist


if __name__ == "__main__":
    # Test the data preparation
    test_symbols = ['BPCL']
    
    print("🧪 Testing EvoRL Data Preparation")
    print(f"Symbols: {test_symbols}")
    
    # Check existing data
    if validate_existing_data(test_symbols):
        print("✅ Recent data exists")
    else:
        print("⚠️  Fetching fresh data...")
        success = fetch_fresh_data_from_kite(test_symbols)
        if success:
            print("✅ Fresh data fetched successfully")
        else:
            print("❌ Failed to fetch fresh data")