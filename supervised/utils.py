import os
import random
import time
import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any


def set_seed(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(mode=True, warn_only=True)
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False


def get_device(device_arg: str = 'auto') -> str:
    """Get the device to use for training."""
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_arg


def autoscale_workers(default: int = 4) -> int:
    """Autoscale number of dataloader workers based on CPU count."""
    try:
        cpu_count = os.cpu_count() or 4
        # Use half of available cores, capped at 8
        return min(cpu_count // 2, 8)
    except:
        return default


def save_json(data: Dict[str, Any], filepath: Path):
    """Save data to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


class TimeManager:
    """Manages time budget for training."""
    
    def __init__(self, budget_minutes: Optional[int] = None):
        self.budget_minutes = budget_minutes
        self.start_time = time.time()
    
    def is_expired(self) -> bool:
        """Check if time budget is expired."""
        if self.budget_minutes is None:
            return False
        elapsed_minutes = (time.time() - self.start_time) / 60
        return elapsed_minutes >= self.budget_minutes
    
    def elapsed_minutes(self) -> float:
        """Get elapsed time in minutes."""
        return (time.time() - self.start_time) / 60
    
    def remaining_minutes(self) -> Optional[float]:
        """Get remaining time in minutes."""
        if self.budget_minutes is None:
            return None
        return max(0, self.budget_minutes - self.elapsed_minutes())


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
