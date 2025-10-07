"""
Custom CNN feature extractor for small Snake game observations.

This CNN is designed for small grid sizes (8x8, 12x12, 16x16) where SB3's
default NatureCNN fails due to large kernel sizes (8x8, 4x4) being too big
for the input dimensions.
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SnakeTinyCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for small Snake grid observations.
    
    Designed for channel-first float32 inputs in [0, 1] with shape (3, H, W)
    where H and W are typically 8-16 pixels.
    
    Architecture:
    - Conv2d(3, 32, k=3, s=1, p=1) + ReLU
    - MaxPool2d(k=2, s=2)  # Reduces H,W by 2
    - Conv2d(32, 64, k=3, s=1, p=1) + ReLU
    - Flatten
    - Linear(64 * (H//2) * (W//2), features_dim) + ReLU
    
    For 12x12 input: 12x12 -> 6x6 after pooling -> 64*6*6=2304 features -> features_dim
    
    Parameters
    ----------
    observation_space : gym.Space
        The observation space (should be Box with shape (C, H, W))
    features_dim : int
        Number of output features (default: 256)
    
    Examples
    --------
    >>> from gymnasium import spaces
    >>> import torch
    >>> obs_space = spaces.Box(low=0, high=1, shape=(3, 12, 12), dtype=np.float32)
    >>> extractor = SnakeTinyCNN(obs_space, features_dim=256)
    >>> x = torch.randn(4, 3, 12, 12)  # Batch of 4
    >>> features = extractor(x)
    >>> features.shape
    torch.Size([4, 256])
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Assumes observation_space is Box with shape (C, H, W)
        assert len(observation_space.shape) == 3, \
            f"Expected 3D observation space (C, H, W), got {observation_space.shape}"
        
        n_input_channels = observation_space.shape[0]
        height = observation_space.shape[1]
        width = observation_space.shape[2]
        
        # Convolutional layers
        self.cnn = nn.Sequential(
            # First conv block: 3 -> 32 channels, same spatial size
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Pool: reduces spatial dimensions by 2
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block: 32 -> 64 channels, same spatial size
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Compute shape after convolutions
        # After MaxPool2d with k=2, s=2: H -> H//2, W -> W//2
        pooled_height = height // 2
        pooled_width = width // 2
        n_flatten = 64 * pooled_height * pooled_width
        
        # Linear layer to produce final features
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.
        
        Parameters
        ----------
        observations : torch.Tensor
            Batch of observations with shape (batch_size, C, H, W)
            Values should be float32 in range [0, 1] (already normalized)
        
        Returns
        -------
        torch.Tensor
            Extracted features with shape (batch_size, features_dim)
        """
        # Pass through CNN
        x = self.cnn(observations)
        
        # Flatten and pass through linear layer
        x = self.linear(x)
        
        return x