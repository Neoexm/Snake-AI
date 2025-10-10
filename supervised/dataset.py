import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any
import gc


class SnakeBCDataset(Dataset):
    """Dataset for behavior cloning with all data loaded in memory."""
    
    def __init__(self, root_dir: str, augment: bool = False, cache_dir: str = None):
        self.root_dir = Path(root_dir)
        self.augment = augment
        
        meta_path = self.root_dir / 'meta.json'
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        
        import json
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        
        self.shard_files = sorted(self.root_dir.glob('shard_*.pt'))
        if not self.shard_files:
            raise FileNotFoundError(f"No shard files found in {self.root_dir}")
        
        print(f"Loading {len(self.shard_files)} shards into memory...")
        all_obs = []
        all_actions = []
        
        for i, shard_file in enumerate(self.shard_files):
            data = torch.load(shard_file, map_location='cpu', weights_only=False)
            all_obs.append(data['observations'])
            all_actions.append(data['actions'])
            del data
            
            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1}/{len(self.shard_files)} shards")
        
        self.observations = torch.cat(all_obs, dim=0)
        self.actions = torch.cat(all_actions, dim=0)
        
        del all_obs, all_actions
        gc.collect()
        
        print(f"Loaded {len(self.observations)} samples into memory")
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx: int):
        obs = self.observations[idx]
        action = self.actions[idx]
        
        if self.augment and len(obs.shape) == 3:
            obs = self._augment_image(obs)
        
        return obs, action
    
    def _augment_image(self, obs):
        """Apply simple augmentations to image observations."""
        if torch.rand(1).item() > 0.5 and self.meta.get('allow_hflip', False):
            obs = torch.flip(obs, dims=[2])
        
        return obs


def create_dataloaders(
    data_dir: str,
    batch_size: int,
    val_split: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
    augment: bool = False,
    cache_dir: str = "F:/snake_bc_cache"
):
    """Create train and validation dataloaders."""
    
    dataset = SnakeBCDataset(data_dir, augment=augment, cache_dir=cache_dir)
    
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, dataset.meta
