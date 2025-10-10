import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from typing import Tuple


class SnakeCnnPolicy(nn.Module):
    """CNN policy for image observations."""
    
    def __init__(
        self,
        num_actions: int,
        in_channels: int = 3,
        img_size: Tuple[int, int] = (12, 12),
        width: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        h, w = img_size
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        pooled_h = h // 2
        pooled_w = w // 2
        flatten_dim = width * 2 * pooled_h * pooled_w
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_actions)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class SnakeMlpPolicy(nn.Module):
    """MLP policy for low-dimensional observations."""
    
    def __init__(
        self,
        num_actions: int,
        in_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_actions))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.flatten(1)
        return self.net(x)


def make_bc_model(obs_space: gym.Space, action_space: gym.Space, config: dict):
    """Create BC model based on observation space."""
    
    num_actions = action_space.n
    dropout = config.get('dropout', 0.1)
    
    if len(obs_space.shape) == 3:
        c, h, w = obs_space.shape
        width = config.get('cnn_width', 64)
        model = SnakeCnnPolicy(
            num_actions=num_actions,
            in_channels=c,
            img_size=(h, w),
            width=width,
            dropout=dropout
        )
    elif len(obs_space.shape) == 1:
        in_dim = obs_space.shape[0]
        hidden_dims = config.get('mlp_hidden_dims', (256, 256))
        model = SnakeMlpPolicy(
            num_actions=num_actions,
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unsupported observation shape: {obs_space.shape}")
    
    return model
