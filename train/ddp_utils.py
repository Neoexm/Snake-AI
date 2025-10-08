"""
Distributed Data Parallel (DDP) utilities for multi-GPU training.

This module provides helper functions for wrapping models in DDP,
unwrapping for checkpointing, and managing distributed state.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Any


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank (0 if not distributed)."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes (1 if not distributed)."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed(backend: str = 'nccl') -> tuple[int, int, torch.device]:
    """
    Initialize distributed training process group.
    
    Parameters
    ----------
    backend : str
        Backend to use ('nccl' for GPU, 'gloo' for CPU)
    
    Returns
    -------
    rank : int
        Process rank
    world_size : int
        Total number of processes
    device : torch.device
        Device for this process
    """
    if not dist.is_available():
        raise RuntimeError("Distributed training not available")
    
    # Initialize process group
    dist.init_process_group(backend=backend)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device for this process
    if backend == 'nccl':
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')
    
    return rank, world_size, device


def cleanup_distributed():
    """Clean up distributed process group."""
    if is_distributed():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: Any,
    device_ids: Optional[list] = None,
    output_device: Optional[int] = None,
    find_unused_parameters: bool = False,
) -> DDP:
    """
    Wrap a model in DistributedDataParallel.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to wrap
    device_ids : list, optional
        Device IDs (default: [rank])
    output_device : int, optional
        Output device (default: rank)
    find_unused_parameters : bool
        Whether to find unused parameters
    
    Returns
    -------
    DDP
        Wrapped model
    """
    if not is_distributed():
        return model
    
    rank = get_rank()
    
    if device_ids is None:
        device_ids = [rank]
    if output_device is None:
        output_device = rank
    
    return DDP(
        model,
        device_ids=device_ids,
        output_device=output_device,
        broadcast_buffers=True,
        find_unused_parameters=find_unused_parameters,
    )


def unwrap_model_ddp(model: Any) -> Any:
    """
    Unwrap a DDP-wrapped model to get the underlying module.
    
    Parameters
    ----------
    model : Any
        Potentially DDP-wrapped model
    
    Returns
    -------
    Any
        Unwrapped model (module attribute if DDP, otherwise unchanged)
    """
    if isinstance(model, DDP):
        return model.module
    return model


def broadcast_model_parameters(model: Any, src: int = 0):
    """
    Broadcast model parameters from source rank to all ranks.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters to broadcast
    src : int
        Source rank (default: 0)
    """
    if not is_distributed():
        return
    
    # Unwrap if DDP
    unwrapped_model = unwrap_model_ddp(model)
    
    for param in unwrapped_model.parameters():
        dist.broadcast(param.data, src=src)


def all_reduce_dict(data: dict, op=dist.ReduceOp.SUM) -> dict:
    """
    All-reduce a dictionary of tensors across all ranks.
    
    Parameters
    ----------
    data : dict
        Dictionary of tensor values
    op : dist.ReduceOp
        Reduction operation (default: SUM)
    
    Returns
    -------
    dict
        Dictionary with reduced values
    """
    if not is_distributed():
        return data
    
    reduced_data = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            tensor = value.clone()
            dist.all_reduce(tensor, op=op)
            reduced_data[key] = tensor
        elif isinstance(value, (int, float)):
            tensor = torch.tensor([value], device=torch.cuda.current_device())
            dist.all_reduce(tensor, op=op)
            reduced_data[key] = tensor.item()
        else:
            reduced_data[key] = value
    
    return reduced_data


def barrier():
    """Synchronization barrier across all processes."""
    if is_distributed():
        dist.barrier()


def setup_nccl_env_for_b200():
    """
    Set up NCCL environment variables optimized for NVIDIA B200 GPUs.
    
    B200 features:
    - NVLink 5.0 (900 GB/s bandwidth)
    - InfiniBand support
    - PCIe Gen 5.0
    """
    os.environ.update({
        # Enable InfiniBand (disable only if not available)
        'NCCL_IB_DISABLE': '0',
        
        # Enable P2P (NVLink)
        'NCCL_P2P_DISABLE': '0',
        
        # Use NVLink for intra-node communication
        'NCCL_P2P_LEVEL': 'NVL',
        
        # Network interface (adjust for your system)
        'NCCL_SOCKET_IFNAME': 'eth0',
        
        # CUDA memory allocator configuration
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
        
        # Reduce malloc overhead
        'MALLOC_ARENA_MAX': '1',
        
        # Debug level (WARN for production, INFO for debugging)
        'NCCL_DEBUG': 'WARN',
        
        # Disable timeout for debugging (remove in production)
        # 'NCCL_TIMEOUT': '0',
    })


def print_distributed_info():
    """Print distributed training information (rank 0 only)."""
    if not is_main_process():
        return
    
    if is_distributed():
        print("\n" + "="*60)
        print("DISTRIBUTED TRAINING INFO")
        print("="*60)
        print(f"Backend: {dist.get_backend()}")
        print(f"World Size: {get_world_size()}")
        print(f"Rank: {get_rank()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        
        print("="*60 + "\n")
    else:
        print("\n⚠️  Running in single-GPU mode (not distributed)\n")