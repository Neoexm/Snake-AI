"""
Automatic resource detection and scaling for optimal training throughput.

This module detects available GPU/CPU resources and recommends hyperparameters
to maximize hardware utilization without causing OOM errors.
"""

import os
import sys
import psutil
import warnings
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ResourceConfig:
    """Recommended training configuration based on available resources."""
    
    device: str
    n_envs: int
    n_steps: int
    batch_size: int
    num_workers: int
    use_amp: bool
    pin_memory: bool
    num_threads: Optional[int]
    gpu_info: Dict[str, Any]
    cpu_info: Dict[str, Any]
    
    def print_summary(self):
        """Print a human-readable summary of the configuration."""
        print("\n" + "="*60)
        print("AUTOSCALE RESOURCE CONFIGURATION")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Parallel Environments: {self.n_envs}")
        print(f"Steps per Rollout: {self.n_steps}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Dataloader Workers: {self.num_workers}")
        print(f"Mixed Precision (AMP): {self.use_amp}")
        print(f"Pin Memory: {self.pin_memory}")
        if self.num_threads:
            print(f"CPU Threads: {self.num_threads}")
        
        print("\n--- GPU Info ---")
        if self.gpu_info['available']:
            for key, val in self.gpu_info.items():
                if key != 'available':
                    print(f"  {key}: {val}")
        else:
            print("  No GPU detected")
        
        print("\n--- CPU Info ---")
        for key, val in self.cpu_info.items():
            print(f"  {key}: {val}")
        print("="*60 + "\n")


def detect_cuda() -> Dict[str, Any]:
    """
    Detect CUDA GPUs and their properties.
    
    Returns
    -------
    dict
        GPU information including count, memory, compute capability.
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {'available': False}
        
        gpu_count = torch.cuda.device_count()
        gpu_info = {
            'available': True,
            'count': gpu_count,
            'devices': []
        }
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                'name': props.name,
                'total_memory_gb': props.total_memory / (1024**3),
                'compute_capability': f"{props.major}.{props.minor}",
                'multi_processor_count': props.multi_processor_count,
            }
            gpu_info['devices'].append(device_info)
        
        # Primary GPU stats
        gpu_info['primary_memory_gb'] = gpu_info['devices'][0]['total_memory_gb']
        gpu_info['primary_name'] = gpu_info['devices'][0]['name']
        
        return gpu_info
        
    except ImportError:
        return {'available': False, 'error': 'torch not installed'}
    except Exception as e:
        return {'available': False, 'error': str(e)}


def detect_cpu() -> Dict[str, Any]:
    """
    Detect CPU properties.
    
    Returns
    -------
    dict
        CPU information including core count, memory, frequency.
    """
    cpu_info = {
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True),
        'total_memory_gb': psutil.virtual_memory().total / (1024**3),
        'available_memory_gb': psutil.virtual_memory().available / (1024**3),
    }
    
    # Try to get CPU frequency (may not work on all platforms)
    try:
        freq = psutil.cpu_freq()
        if freq:
            cpu_info['max_frequency_mhz'] = freq.max
    except:
        pass
    
    return cpu_info


def autoscale(
    max_utilization: bool = False,
    prefer_device: Optional[str] = None,
    override_n_envs: Optional[int] = None,
) -> ResourceConfig:
    """
    Automatically determine optimal training configuration.
    
    Parameters
    ----------
    max_utilization : bool
        If True, push resource usage more aggressively (risk of OOM).
    prefer_device : str, optional
        Force 'cuda', 'cpu', or 'auto' (default).
    override_n_envs : int, optional
        Manually override the number of parallel environments.
    
    Returns
    -------
    ResourceConfig
        Recommended configuration for training.
    """
    gpu_info = detect_cuda()
    cpu_info = detect_cpu()
    
    # Determine device
    if prefer_device == 'cpu':
        device = 'cpu'
        use_gpu = False
    elif prefer_device == 'cuda':
        if not gpu_info['available']:
            warnings.warn("CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
            use_gpu = False
        else:
            device = 'cuda'
            use_gpu = True
    else:  # auto
        use_gpu = gpu_info['available']
        device = 'cuda' if use_gpu else 'cpu'
    
    # Determine n_envs
    if override_n_envs is not None:
        n_envs = override_n_envs
    elif use_gpu:
        # Scale with GPU memory
        mem_gb = gpu_info['primary_memory_gb']
        if mem_gb >= 24:
            n_envs = 64 if max_utilization else 48
        elif mem_gb >= 16:
            n_envs = 48 if max_utilization else 32
        elif mem_gb >= 8:
            n_envs = 32 if max_utilization else 24
        else:
            n_envs = 16
    else:
        # CPU: use physical cores, but cap to avoid thrashing
        phys_cores = cpu_info['physical_cores']
        n_envs = min(phys_cores, 8) if not max_utilization else min(phys_cores, 16)
    
    # Determine n_steps (PPO rollout buffer size)
    # Larger for GPU (more memory), smaller for CPU
    if use_gpu:
        n_steps = 512 if max_utilization else 256
    else:
        n_steps = 256
    
    # Batch size: should divide (n_envs * n_steps) evenly
    total_samples = n_envs * n_steps
    if use_gpu:
        # Prefer larger batches on GPU
        batch_size = min(2048, total_samples // 2) if max_utilization else min(1024, total_samples // 4)
    else:
        batch_size = min(512, total_samples // 4)
    
    # Ensure batch_size divides total_samples
    if total_samples % batch_size != 0:
        # Adjust to nearest divisor
        for candidate in [2048, 1024, 512, 256, 128, 64]:
            if total_samples % candidate == 0:
                batch_size = candidate
                break
    
    # Dataloader workers (for experience collection)
    num_workers = 0  # SB3 doesn't use dataloader workers for envs
    
    # AMP (Automatic Mixed Precision) for CUDA
    use_amp = use_gpu and max_utilization
    
    # Pin memory for faster CPU->GPU transfer
    pin_memory = use_gpu
    
    # CPU threading
    num_threads = None
    if not use_gpu:
        num_threads = cpu_info['physical_cores']
        # Set PyTorch thread count
        try:
            import torch
            torch.set_num_threads(num_threads)
        except:
            pass
    
    # Enable CUDA optimizations
    if use_gpu:
        try:
            import torch
            torch.backends.cudnn.benchmark = True
        except:
            pass
    
    return ResourceConfig(
        device=device,
        n_envs=n_envs,
        n_steps=n_steps,
        batch_size=batch_size,
        num_workers=num_workers,
        use_amp=use_amp,
        pin_memory=pin_memory,
        num_threads=num_threads,
        gpu_info=gpu_info,
        cpu_info=cpu_info,
    )


if __name__ == "__main__":
    """CLI for testing resource detection."""
    print("Testing autoscale configuration...")
    
    config_auto = autoscale()
    config_auto.print_summary()
    
    if config_auto.gpu_info['available']:
        print("\nMax utilization config:")
        config_max = autoscale(max_utilization=True)
        config_max.print_summary()