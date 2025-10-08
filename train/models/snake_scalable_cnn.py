"""
Scalable CNN feature extractor for Snake game observations.

Provides configurable width and depth for tuning to saturate GPU/CPU resources.
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SnakeScalableCNN(BaseFeaturesExtractor):
    """
    Scalable CNN feature extractor with configurable width and depth.
    
    Designed for channel-first float32 inputs in [0, 1] with shape (3, H, W)
    where H and W are typically 8-16 pixels.
    
    Architecture (depth=2, width=32 as baseline):
    - Conv2d(3, width, k=3, s=1, p=1) + ReLU
    - [Conv2d(width, width, k=3, s=1, p=1) + ReLU] × (depth - 1)
    - MaxPool2d(k=2, s=2)  # Reduces H,W by 2
    - Flatten
    - Linear(width * (H//2) * (W//2), features_dim) + ReLU
    
    Parameters
    ----------
    observation_space : gym.Space
        The observation space (should be Box with shape (C, H, W))
    features_dim : int
        Number of output features (default: 256)
    width : int
        Number of channels in conv layers (default: 32)
    depth : int
        Number of conv layers before pooling (default: 2, min: 1, max: 4)
    
    Examples
    --------
    >>> from gymnasium import spaces
    >>> import numpy as np
    >>> obs_space = spaces.Box(low=0, high=1, shape=(3, 12, 12), dtype=np.float32)
    >>> extractor = SnakeScalableCNN(obs_space, features_dim=256, width=64, depth=3)
    >>> x = torch.randn(4, 3, 12, 12)
    >>> features = extractor(x)
    >>> features.shape
    torch.Size([4, 256])
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        width: int = 32,
        depth: int = 2,
    ):
        super().__init__(observation_space, features_dim)
        
        # Validate parameters
        assert len(observation_space.shape) == 3, \
            f"Expected 3D observation space (C, H, W), got {observation_space.shape}"
        assert depth >= 1 and depth <= 4, \
            f"Depth must be between 1 and 4, got {depth}"
        assert width >= 16 and width <= 512, \
            f"Width must be between 16 and 512, got {width}"
        
        n_input_channels = observation_space.shape[0]
        height = observation_space.shape[1]
        width_spatial = observation_space.shape[2]
        
        # CRITICAL: Support frame stacking (C can be 3, 6, 9, 12, etc.)
        if n_input_channels % 3 != 0:
            import warnings
            warnings.warn(
                f"Unexpected channel count {n_input_channels}. "
                f"Snake env produces 3 channels; with frame_stack=N, expect 3*N channels."
            )
        
        # Build convolutional layers
        layers = []
        
        # First layer: input channels -> width
        layers.extend([
            nn.Conv2d(n_input_channels, width, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ])
        
        # Additional layers: width -> width (depth - 1 times)
        for _ in range(depth - 1):
            layers.extend([
                nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ])
        
        # Pooling layer to reduce spatial dimensions
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.cnn = nn.Sequential(*layers)
        
        # Compute shape after convolutions
        # After MaxPool2d with k=2, s=2: H -> H//2, W -> W//2
        pooled_height = height // 2
        pooled_width = width_spatial // 2
        n_flatten = width * pooled_height * pooled_width
        
        # Linear layer to produce final features
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(inplace=True),
        )
        
        # Store config for inspection
        self.width = width
        self.depth = depth
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.
        
        Parameters
        ----------
        observations : torch.Tensor
            Batch of observations with shape (batch_size, C, H, W)
            Values should be float32 in range [0, 1]
        
        Returns
        -------
        torch.Tensor
            Extracted features with shape (batch_size, features_dim)
        """
        # Validate input range (only in training mode, sample check to avoid overhead)
        if self.training and torch.rand(1).item() < 0.01:
            if observations.min() < -0.1 or observations.max() > 1.1:
                import warnings
                warnings.warn(
                    f"Input observations out of expected [0, 1] range: "
                    f"min={observations.min().item():.3f}, max={observations.max().item():.3f}. "
                    f"Check normalize_images setting or env preprocessing."
                )
        
        x = self.cnn(observations)
        x = self.linear(x)
        return x


class SnakeDeepCNN(BaseFeaturesExtractor):
    """
    Deeper CNN with skip connections for larger grids or more complex features.
    
    Architecture with residual-like connections for training stability.
    
    Parameters
    ----------
    observation_space : gym.Space
        The observation space
    features_dim : int
        Number of output features (default: 256)
    width : int
        Base number of channels (default: 64)
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        width: int = 64,
    ):
        super().__init__(observation_space, features_dim)
        
        assert len(observation_space.shape) == 3
        
        n_input_channels = observation_space.shape[0]
        height = observation_space.shape[1]
        width_spatial = observation_space.shape[2]
        
        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_input_channels, width, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Residual-style blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Compute final size
        pooled_height = height // 2
        pooled_width = width_spatial // 2
        n_flatten = width * 2 * pooled_height * pooled_width
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.conv1(observations)
        
        # Add residual connection
        identity = x
        x = self.conv2(x)
        x = x + identity  # Skip connection
        
        x = self.conv3(x)
        x = self.pool(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    """Test scalable CNN architectures."""
    import numpy as np
    from gymnasium import spaces
    
    print("Testing SnakeScalableCNN...")
    
    # Create observation space
    obs_space = spaces.Box(low=0, high=1, shape=(3, 12, 12), dtype=np.float32)
    
    # Test different configurations
    configs = [
        {"width": 32, "depth": 1, "features_dim": 128},
        {"width": 32, "depth": 2, "features_dim": 256},
        {"width": 64, "depth": 2, "features_dim": 256},
        {"width": 64, "depth": 3, "features_dim": 512},
        {"width": 128, "depth": 3, "features_dim": 512},
    ]
    
    for config in configs:
        extractor = SnakeScalableCNN(obs_space, **config)
        
        # Count parameters
        n_params = sum(p.numel() for p in extractor.parameters())
        
        # Test forward pass
        x = torch.randn(4, 3, 12, 12)
        features = extractor(x)
        
        print(f"  Width={config['width']}, Depth={config['depth']}, "
              f"Features={config['features_dim']}: "
              f"{n_params:,} params, output shape {features.shape}")
    
    print("\nTesting SnakeDeepCNN...")
    extractor = SnakeDeepCNN(obs_space, features_dim=256, width=64)
    n_params = sum(p.numel() for p in extractor.parameters())
    x = torch.randn(4, 3, 12, 12)
    features = extractor(x)
    print(f"  Width=64: {n_params:,} params, output shape {features.shape}")