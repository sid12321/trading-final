#SID: Implement a learning rate schedule 

# GPU/CPU hybrid configuration - optimized for RTX 4080 + i9-13950HX
import warnings
import os
warnings.filterwarnings('ignore')

# Initialize JAX GPU support FIRST before any other imports
from jax_gpu_init import get_jax_status

# First test if CUDA works, then configure accordingly
def test_cuda_available():
    """Test if CUDA is actually working"""
    try:
        import torch
        if torch.cuda.is_available():
            # Try a simple CUDA operation
            x = torch.tensor([1.0], device='cuda')
            _ = x + 1
            del x
            torch.cuda.empty_cache()
            return True
    except:
        pass
    return False

# Configure environment based on CUDA availability
if test_cuda_available():
    print("CUDA working, enabling GPU mode")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
    os.environ['JAX_PLATFORMS'] = 'cuda'
    # Reduce CPU threads for GPU priority
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    os.environ['NUMEXPR_NUM_THREADS'] = '8'
    os.environ['OPENBLAS_NUM_THREADS'] = '8'
else:
    print("CUDA not working, forcing CPU mode")
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA
    os.environ['JAX_PLATFORMS'] = 'cpu'  # Force JAX to use CPU
    # Optimize for CPU
    os.environ['OMP_NUM_THREADS'] = '28'
    os.environ['MKL_NUM_THREADS'] = '28'
    os.environ['NUMEXPR_NUM_THREADS'] = '28'
    os.environ['OPENBLAS_NUM_THREADS'] = '28'

# GPU memory management (only if CUDA is working)
if test_cuda_available():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Additional logging suppression
import logging
logging.getLogger().setLevel(logging.ERROR)
# Suppress JAX warnings specifically
logging.getLogger('jax._src.xla_bridge').setLevel(logging.CRITICAL)

basepath = '/Users/skumar81/Desktop/Personal/trading-final'
tmp_path = basepath+"/tmp/sb3_log/"
check_path = basepath+"/tmp/checkpoints/"
tensorboard_log_path = basepath+"/tmp/tensorboard_logs/"
METRICS_FILE = 'custom_metrics.txt'
METRICS_FILE_PATH = basepath + '/tmp/sb3_log/' + METRICS_FILE
CHECKPOINT_DIR = basepath + '/tmp/checkpoints/'

import torch
import multiprocessing as mp

# Detect optimal device with GPU priority
def detect_device():
    """Detect best available device (GPU or CPU)"""
    try:
        # Check if CUDA is available and working
        if torch.cuda.is_available():
            # Test GPU functionality with timeout
            test_tensor = torch.tensor([1.0], device='cuda')
            result = test_tensor + 1
            del test_tensor, result
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure GPU operations complete
            return "cuda", torch.cuda.device_count()
    except Exception as e:
        print(f"GPU detection failed ({e}), using CPU mode")
    return "cpu", 0

DEVICE, N_GPUS = detect_device()

if DEVICE == "cuda":
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
    if os.environ.get('TRADING_INIT_PRINTED') != '1':
        print(f"GPU mode enabled: {GPU_NAME} ({GPU_MEMORY:.1f}GB)")
        print(f"CPU: {mp.cpu_count()} cores available for data processing")
else:
    if os.environ.get('TRADING_INIT_PRINTED') != '1':
        print("CPU mode enabled")
        print(f"Using {mp.cpu_count()} CPU cores")

# GPU/CPU hybrid optimization settings
if DEVICE == "cuda":
    # GPU settings - optimize for RTX 4080
    N_CORES = 16  # OPTIMIZED: Reduce CPU-GPU sync overhead  # Use fewer CPU cores to avoid GPU contention
    N_ENVS = 32  # MEMORY-OPTIMIZED: Reduced for 12GB GPU  # OPTIMIZED: 4x increase for GPU throughput  # More environments for GPU (can handle more in parallel)
    SIGNAL_OPTIMIZATION_WORKERS = 8  # Match N_CORES  # Balance between GPU/CPU
    
    # Enable GPU optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    
    # Enable Flash Attention if available (RTX 4080 supports it)
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
else:
    # CPU settings - use most cores
    N_CORES = min(mp.cpu_count() - 4, 28)  # Leave some for system
    N_ENVS = N_CORES
    SIGNAL_OPTIMIZATION_WORKERS = N_CORES
    
    torch.set_num_threads(N_CORES)
    torch.set_num_interop_threads(N_CORES)

OMP_NUM_THREADS = N_CORES  # OpenMP threads

# Signal optimization parameters - GPU/CPU optimized
OPTIMIZATION_METHOD = 'simulated_annealing'  # or 'mcmc'
USE_GPU_ACCELERATION = (DEVICE == "cuda")  # Auto-detect
USE_PARALLEL_OPTIMIZATION = True
MAX_OPTIMIZATION_ITERATIONS = 3000 if DEVICE == "cuda" else 1500
MCMC_SAMPLES = 3000 if DEVICE == "cuda" else 1500

# Only print once to avoid spam in multiprocessing
if os.environ.get('TRADING_INIT_PRINTED') != '1':
    print(f"Device: {DEVICE}, CPU cores: {N_CORES}, Environments: {N_ENVS}")
    print(f"Signal optimization: {SIGNAL_OPTIMIZATION_WORKERS} workers, GPU acceleration: {USE_GPU_ACCELERATION}")
    os.environ['TRADING_INIT_PRINTED'] = '1'

POLICY_KWARGS = {
    'activation_fn': 'ReLU',  # Will be converted to torch.nn.ReLU in modeltrain
    'net_arch': {
        'pi': [256, 256],  # Simple, effective network for trading
        'vf': [256, 256]   # Simple, effective network for trading
    },
    'ortho_init': True  # Orthogonal initialization for better training
}

#Symbols
SYMLIST = ["BPCL","HDFCLIFE","BRITANNIA","HEROMOTOCO","INDUSINDBK","APOLLOHOSP","WIPRO","TATASTEEL","BHARTIARTL","ITC","HINDUNILVR","POWERGRID"]
TESTSYMBOLS = SYMLIST[:1] 

#Code scope
PREPROCESS = True
TRAINMODEL = True
NEWMODELFLAG = False
DETERMINISTIC = True
GENERATEPOSTERIOR = True
POSTERIORPLOTS = True
FAKE = True
GENOPTSIG = False

# Signal optimization controls - GPU/CPU optimized
SIGNAL_OPT_MAX_ITERATIONS = 500 if DEVICE == "cuda" else 250
SIGNAL_OPT_BATCH_SIZE = 8192  # MEMORY-OPTIMIZED: Reduced  # OPTIMIZED: 2x for GPU if DEVICE == "cuda" else 75  # Larger batches for GPU
SIGNAL_OPT_EARLY_STOP = 50  # Early stopping patience
SIGNAL_OPT_PRIORITY_SIGNALS = ['macd', 'rsi', 'bb_position', 'momentum', 'vwap']  # Optimize these first

# Thread settings optimized for device
if DEVICE == "cuda":
    # For GPU: use fewer CPU threads to avoid contention
    torch.set_num_threads(8)
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
else:
    # For CPU: use all available threads
    torch.set_num_threads(N_CORES)
    os.environ['OMP_NUM_THREADS'] = str(N_CORES)
    os.environ['MKL_NUM_THREADS'] = str(N_CORES)

#Base data scope
BENCHMARKHORIZON = 375*5
HORIZONDAYS = 60
INITIAL_ACCOUNT_BALANCE = 100000
TRAIN_MAX = 0.75
NLAGS = 5 #This is the TS consumed by the model 
MAXIMUM_SHORT_VALUE = INITIAL_ACCOUNT_BALANCE
TOPN = 0
MAXITERPOSTERIOR = 1 #Deterministic it's all the same 
NQUANTILES = 5
LAGS = [1,2,3,5,7,13,17,19,23] #Thes are lags taken in at each step
LAGCOLS = [f'lret{lag}' for lag in LAGS]
GENERICS = ['vwap', 'dv', 'c','o','h','l','v', 'co', 'scco', 'vscco', 'dvscco', \
'hl', 'vhl','opc', 'dvwap', 'd2vwap', 'ddv', 'd2dv', 'h5scco', 'h5vscco', 'h5dvscco','codv',\
'macd', 'macd_signal', 'macd_histogram', 'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',\
'rsi', 'rsi_oversold', 'rsi_overbought', 'stoch_k', 'stoch_d', 'atr', 'williams_r',\
'sma5', 'sma10', 'sma20', 'sma50', 'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20',\
'volume_sma', 'volume_ratio', 'price_volume', 'momentum', 'rate_of_change', 'volatility',\
'vol_spike', 'bear_signal', 'oversold_extreme', 'bb_squeeze', 'lower_highs', 'lower_lows',\
'support_break', 'hammer_pattern', 'volume_divergence', 'trend_alignment_bear', 'market_fear',\
'bull_signal', 'overbought_extreme', 'bb_expansion', 'higher_highs', 'higher_lows',\
'resistance_break', 'morning_star', 'volume_confirmation', 'trend_alignment_bull', 'market_greed',\
'pivot_high_3', 'pivot_low_3', 'pivot_high_5', 'pivot_low_5', 'pivot_high_7', 'pivot_low_7',\
'pivot_strength', 'local_max', 'local_min', 'swing_high', 'swing_low']
HISTORICAL = ['ndv', 'nmomentum']
QCOLS = ['q' + x for x in GENERICS + LAGCOLS] 
ALLSIGS = GENERICS + QCOLS + LAGCOLS + HISTORICAL

#PPO details
MAX_STEPS = 100  # Shorter episodes since we're training on entire dataset with daily liquidation
BASEMODELITERATIONS = 3000000  # More iterations for full dataset training
REDUCEDMCMCITERATIONS = BASEMODELITERATIONS//4
MEANREWARDTHRESHOLD = 0.2 #Corresponds to a more realistic 2% return for minute-based trading
BUYTHRESHOLD = 0.3
SELLTHRESHOLD = -0.3
COST_PER_TRADE = 0
MAXITERREPEAT = 1  # Fewer iterations since we train on full dataset
CLIP_RANGE = 0.2  # Standard clip range for policy updates
CLIP_RANGE_VF = 0.2  # Conservative value function clipping
VF_COEF = 0.25  # Standard value function coefficient for full dataset learning
VALUE_LOSS_BOUND = 1.0  # Bound value loss to [-1, 1] to prevent training instability
GAMMA = 0.99  # Standard discount factor for intraday trading with daily liquidation
RETRAIN = False
NORAD = True
ENTROPY_BOUND = 1.0  # Bound entropy loss to [-1, 1] to prevent training instability
MAX_GRAD_NORM = 0.25  # Gradient clipping for stability #SID
VERBOSITY = 0
TRAINREPS = 1
LOGFREQ = 1000  # OPTIMIZED: Less frequent logging  # Less frequent logging for long episodes
STATS_WINDOW_SIZE = 100  # Larger window for full dataset training stability

#To be Optimized - GPU/CPU-Optimized Trading System Settings
if DEVICE == "cuda":
    # GPU-optimized settings for RTX 4080
    GLOBALLEARNINGRATE = 0.0005  # OPTIMIZED: Adjusted for larger batches  # Higher LR for GPU
    N_EPOCHS = 10  # Standard epochs
    ENT_COEF = 0.01  # Slightly higher for GPU
    N_STEPS = 1024  # MEMORY-OPTIMIZED: Increased to maintain sample efficiency  # OPTIMIZED: Shorter for frequent updates  # Optimal for GPU memory
    TARGET_KL = 0.02  # Standard KL divergence
    GAE_LAMBDA = 0.95  # Higher for GPU
    BATCH_SIZE = 512  # MEMORY-OPTIMIZED: Reduced for memory  # OPTIMIZED: 4x for better GPU utilization  # Good for RTX 4080
    USE_SDE = True  # Enable SDE
    SDE_SAMPLE_FREQ = 4  # Sample new noise every 4 steps
else:
    # CPU-optimized settings
    GLOBALLEARNINGRATE = 0.0001  # Conservative for CPU
    N_EPOCHS = 10  # CPU-optimized epochs
    ENT_COEF = 0.005  # Standard entropy coefficient
    N_STEPS = 4096  # CPU-optimized rollout
    TARGET_KL = 0.02  # Standard KL divergence
    GAE_LAMBDA = 0.8  # Standard GAE lambda
    BATCH_SIZE = 1024  # MEMORY-OPTIMIZED: Reduced for memory  # OPTIMIZED: 4x for better GPU utilization  # CPU-optimized batch size
    USE_SDE = True  # Enable SDE
    SDE_SAMPLE_FREQ = 4  # Sample new noise every 4 steps

TOTAL_TIMESTEPS = N_STEPS * N_ENVS  # Total steps across all environments

# Learning Rate Scheduling Parameters
USE_LR_SCHEDULE = True  # Enable learning rate scheduling
INITIAL_LR = 1e-3  # Starting learning rate
FINAL_LR = 1e-4    # Final learning rate
LR_SCHEDULE_TYPE = "exponential"  # Options: "linear", "exponential", "cosine"

# Entropy Coefficient Scheduling Parameters
USE_ENT_SCHEDULE = True  # Enable entropy coefficient scheduling
INITIAL_ENT_COEF = 0.05  # Starting entropy coefficient
FINAL_ENT_COEF = 0.005   # Final entropy coefficient

# Target KL Scheduling Parameters
USE_TARGET_KL_SCHEDULE = True  # Enable target KL scheduling
INITIAL_TARGET_KL = 0.1  # Starting target KL
FINAL_TARGET_KL = 0.01    # Final target KL

# GPU/CPU optimization settings
USE_MIXED_PRECISION = (DEVICE == "cuda")  # Enable for GPU only
GPU_MEMORY_GROWTH = (DEVICE == "cuda")  # Dynamic memory allocation for GPU
OPTIMIZE_FOR_GPU = (DEVICE == "cuda")  # GPU-specific optimizations

# GPU-specific settings
if DEVICE == "cuda":
    # Memory optimization for RTX 4080 (12GB)
    GPU_MEMORY_FRACTION = 0.85  # MEMORY-OPTIMIZED: Conservative memory usage  # OPTIMIZED: Use more GPU memory  # Use 90% of GPU memory
    GRADIENT_ACCUMULATION_STEPS = 1  # Can increase if OOM
    MIXED_PRECISION_DTYPE = torch.float16  # FP16 for speed

signalhorizons = {x:y for x,y in zip(ALLSIGS,[1]*len(ALLSIGS))}
for col in QCOLS:
  signalhorizons[col] = 5 #You need at least 5 rows to compute the generic quantile (5 bars including)
  if col.startswith('h5'):
    signalhorizons[col] = 9 #5 bars + last bar has a tail of 4 - so it becomes 5 + 4 = 9
for lag in LAGS:
  signalhorizons['lret'+str(lag)] = lag + 1
  signalhorizons['qlret'+str(lag)] = lag + 5 #
MINHORIZON = max(signalhorizons.values()) + NLAGS + 1

# Kite API credentials - centralized configuration
API_KEY = "wh7m5jcdtj4g57oh"
API_SECRET = "2owm89v2qjd9mx4sngodejq9hdelfwxj"
USER_ID = "PN1089"
PASSWORD = "Pillowbat123$"
TOTP_KEY = "LC4GFGV5GZTYGFPXQRQ6J4YO2IU6ZL56"




# GPU/CPU Optimization Applied - 2025-07-24
HYBRID_OPTIMIZED = True
OPTIMIZATION_TIMESTAMP = '2025-07-27 19:06'
if DEVICE == "cuda":
    print(f"GPU Optimization (MEMORY-OPTIMIZED): {GPU_NAME} with {N_ENVS} environments")
else:
    print(f"CPU Optimization: {N_CORES} cores with {N_ENVS} environments")
