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
from dataclasses import dataclass, field


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
    
    # New scalability parameters
    policy_width: int = 32
    policy_depth: int = 2
    max_n_envs: Optional[int] = None
    max_batch_size: Optional[int] = None
    precision: str = "fp32"  # "fp32" or "amp"
    compile_mode: str = "none"  # "none" or "default"
    env_processes: bool = True
    normalize_images: bool = False
    frame_stack: int = 1
    
    def print_summary(self):
        """Print a human-readable summary of the configuration."""
        print("\n" + "="*60)
        print("AUTOSCALE RESOURCE CONFIGURATION")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Parallel Environments: {self.n_envs}")
        print(f"Steps per Rollout: {self.n_steps}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Policy Width: {self.policy_width}")
        print(f"Policy Depth: {self.policy_depth}")
        print(f"Precision: {self.precision}")
        print(f"Compile Mode: {self.compile_mode}")
        print(f"Subprocess Envs: {self.env_processes}")
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
    override_policy_width: Optional[int] = None,
    override_policy_depth: Optional[int] = None,
    override_precision: Optional[str] = None,
    override_compile: Optional[str] = None,
    max_n_envs: Optional[int] = None,
    max_batch_size: Optional[int] = None,
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
    override_policy_width : int, optional
        Override policy network width.
    override_policy_depth : int, optional
        Override policy network depth.
    override_precision : str, optional
        Override precision mode ("fp32" or "amp").
    override_compile : str, optional
        Override compile mode ("none" or "default").
    max_n_envs : int, optional
        Maximum number of environments (hard cap).
    max_batch_size : int, optional
        Maximum batch size (hard cap).
    
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
    
    # Determine n_envs (PER-RANK in DDP mode, total in single-GPU mode)
    # In DDP: Each rank creates n_envs environments
    # Total environments = n_envs × world_size
    if override_n_envs is not None:
        n_envs = override_n_envs
    elif use_gpu:
        # Scale with GPU memory - more aggressive for high utilization
        mem_gb = gpu_info['primary_memory_gb']
        gpu_name = gpu_info.get('primary_name', '').lower()
        is_b200 = mem_gb >= 175 or 'b200' in gpu_name or 'blackwell' in gpu_name
        
        if is_b200:  # B200 specifically (192GB VRAM, ~10-15GB reserved = 177-182GB usable)
            # B200: Massive VRAM allows much larger env counts
            # DDP mode: 256 envs/GPU × 4 GPUs = 1024 total concurrent envs
            # Single-GPU: 512 envs for maximum throughput
            n_envs = 256 if max_utilization else 128
        elif mem_gb >= 80:  # H100 (80GB), MI300X
            # DDP mode: 64 envs/GPU × 4 GPUs = 256 total
            # Single-GPU: 256 envs on one GPU
            n_envs = 64 if max_utilization else 48
        elif mem_gb >= 40:  # A100 80GB
            n_envs = 48 if max_utilization else 32
        elif mem_gb >= 24:  # A100 40GB, RTX 4090
            n_envs = 32 if max_utilization else 24
        elif mem_gb >= 16:  # V100, RTX 3090
            n_envs = 24 if max_utilization else 16
        elif mem_gb >= 8:  # RTX 3070
            n_envs = 16 if max_utilization else 12
        else:
            n_envs = 8
    else:
        # CPU: scale with logical cores for Windows, capped
        logical_cores = cpu_info['logical_cores']
        phys_cores = cpu_info['physical_cores']
        if max_utilization:
            n_envs = min(logical_cores * 2, 64)
        else:
            n_envs = min(phys_cores, 16)
    
    # Apply hard cap if specified
    if max_n_envs is not None:
        n_envs = min(n_envs, max_n_envs)
    
    # Determine n_steps (PPO rollout buffer size)
    # Larger for GPU (more memory), smaller for CPU
    if use_gpu:
        mem_gb = gpu_info['primary_memory_gb']
        gpu_name = gpu_info.get('primary_name', '').lower()
        is_b200 = mem_gb >= 175 or 'b200' in gpu_name or 'blackwell' in gpu_name
        
        if is_b200:  # B200 - can handle very large rollouts
            n_steps = 2048 if max_utilization else 1024
        elif mem_gb >= 80:  # H100 - can handle larger rollouts
            n_steps = 1024 if max_utilization else 512
        elif mem_gb >= 40:
            n_steps = 512 if max_utilization else 256
        else:
            n_steps = 256
    else:
        n_steps = 256
    
    # Batch size: should divide (n_envs * n_steps) evenly
    total_samples = n_envs * n_steps
    if use_gpu:
        mem_gb = gpu_info['primary_memory_gb']
        gpu_name = gpu_info.get('primary_name', '').lower()
        is_b200 = mem_gb >= 175 or 'b200' in gpu_name or 'blackwell' in gpu_name
        
        # Prefer larger batches on GPU for better throughput
        # B200 with 256 envs × 2048 steps = 524288 samples/rank (max util)
        # B200 with 128 envs × 1024 steps = 131072 samples/rank (standard)
        if is_b200:  # B200 - can handle very large batch sizes
            batch_size = 32768 if max_utilization else 16384
        elif mem_gb >= 80:  # H100 - can handle 8192+ batch sizes
            batch_size = 8192 if max_utilization else 4096
        elif mem_gb >= 40:
            batch_size = 4096 if max_utilization else 2048
        else:
            batch_size = 2048 if max_utilization else 1024
    else:
        batch_size = min(1024, total_samples // 4)
    
    # Ensure batch_size doesn't exceed total_samples
    batch_size = min(batch_size, total_samples // 2)
    
    # Apply hard cap if specified
    if max_batch_size is not None:
        batch_size = min(batch_size, max_batch_size)
    
    # Ensure batch_size divides total_samples
    if total_samples % batch_size != 0:
        # Adjust to nearest divisor
        for candidate in [4096, 2048, 1024, 512, 256, 128, 64, 32]:
            if candidate <= batch_size and total_samples % candidate == 0:
                batch_size = candidate
                break
    
    # Dataloader workers (for experience collection)
    num_workers = 0  # SB3 doesn't use dataloader workers for envs
    
    # Policy network sizing based on GPU capability
    if override_policy_width is not None:
        policy_width = override_policy_width
    elif use_gpu:
        mem_gb = gpu_info['primary_memory_gb']
        gpu_name = gpu_info.get('primary_name', '').lower()
        is_b200 = mem_gb >= 175 or 'b200' in gpu_name or 'blackwell' in gpu_name
        
        if is_b200:  # B200 - can handle wider networks
            policy_width = 256 if max_utilization else 128
        elif mem_gb >= 80:  # H100
            policy_width = 128
        elif mem_gb >= 40 and max_utilization:
            policy_width = 128
        elif mem_gb >= 24:
            policy_width = 64 if max_utilization else 32
        else:
            policy_width = 32
    else:
        policy_width = 32
    
    if override_policy_depth is not None:
        policy_depth = override_policy_depth
    elif use_gpu:
        mem_gb = gpu_info['primary_memory_gb']
        gpu_name = gpu_info.get('primary_name', '').lower()
        is_b200 = mem_gb >= 175 or 'b200' in gpu_name or 'blackwell' in gpu_name
        
        if is_b200:  # B200 - can handle deeper networks
            policy_depth = 4 if max_utilization else 3
        elif max_utilization:
            policy_depth = 3
        else:
            policy_depth = 2
    else:
        policy_depth = 2
    
    # Precision mode
    if override_precision is not None:
        precision = override_precision
    elif use_gpu and max_utilization:
        precision = "amp"
    else:
        precision = "fp32"
    
    # AMP (Automatic Mixed Precision) for CUDA
    use_amp = precision == "amp"
    
    # Compile mode
    if override_compile is not None:
        compile_mode = override_compile
    else:
        compile_mode = "none"  # Conservative default
    
    # Pin memory for faster CPU->GPU transfer
    pin_memory = use_gpu
    
    # Use subprocess envs on Linux
    env_processes = sys.platform != 'win32'
    
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
        policy_width=policy_width,
        policy_depth=policy_depth,
        max_n_envs=max_n_envs,
        max_batch_size=max_batch_size,
        precision=precision,
        compile_mode=compile_mode,
        env_processes=env_processes,
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