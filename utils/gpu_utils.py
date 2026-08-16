"""
Utility functions for GPU selection and common operations.
"""
import os
import socket
import tensorflow as tf
import nvidia_smi


def select_GPU(min_gpu_mem_frac=0.9):
    """
    Select a GPU with sufficient free memory.
    
    Args:
        min_gpu_mem_frac: Minimum fraction of free memory required
        
    Returns:
        use_gpu: Selected GPU index
        mem_free: Free memory on selected GPU
    """
    hostname = socket.gethostname()
    nvidia_smi.nvmlInit()
    device_count = nvidia_smi.nvmlDeviceGetCount()
    
    use_gpu = None
    mem_free = 0
    
    for device_index in range(device_count):
        # Skip GPU 1 on blackjack hostname (legacy config)
        if device_index == 1 and 'blackjack' in hostname:
            continue
            
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_index)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        
        print(f"GPU {device_index}:")
        print(f"  Total memory: {info.total / (1024**3):.2f} GB")
        print(f"  Free memory: {info.free / (1024**3):.2f} GB")
        print(f"  Used memory: {info.used / (1024**3):.2f} GB")
        
        if info.free > min_gpu_mem_frac * info.total:
            use_gpu = device_index
            mem_free = info.free
            os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpu)
            break
    
    nvidia_smi.nvmlShutdown()
    
    # Allow memory growth for the selected GPU
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"RuntimeError in GPU setup: {e}")
    
    if use_gpu is None:
        print("Warning: No GPU with sufficient memory found. Using first available GPU.")
        use_gpu = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    return use_gpu, mem_free


def set_seed(seed=0):
    """Set random seed for reproducibility."""
    import numpy as np
    np.random.seed(seed)
