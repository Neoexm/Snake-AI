"""
Production smoke test for multi-GPU training system.
Run this before launching expensive cloud training.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from stable_baselines3.common.env_checker import check_env

from snake_env import SnakeEnv, RewardShaping, FrameStack
from train.models import SnakeTinyCNN, SnakeScalableCNN


def test_environment_checker():
    """Verify environment passes SB3 checks."""
    env = SnakeEnv(grid_size=12)
    env = RewardShaping(env)
    
    # This will raise if environment is invalid
    check_env(env, warn=True)
    
    obs, info = env.reset(seed=42)
    assert obs.shape == (3, 12, 12)
    assert obs.dtype == np.float32
    assert 0.0 <= obs.min() <= obs.max() <= 1.0


def test_observation_space_consistency():
    """Ensure CNN input matches environment output."""
    env = SnakeEnv(grid_size=12)
    env = RewardShaping(env)
    
    obs, _ = env.reset()
    
    # Test with SnakeTinyCNN
    from gymnasium import spaces
    obs_space = spaces.Box(low=0, high=1, shape=(3, 12, 12), dtype=np.float32)
    cnn = SnakeTinyCNN(obs_space, features_dim=256)
    
    # Forward pass
    obs_tensor = torch.from_numpy(obs).unsqueeze(0)  # Add batch dim
    features = cnn(obs_tensor)
    
    assert features.shape == (1, 256)
    assert not torch.isnan(features).any()


def test_deterministic_seeding():
    """Verify same seed produces same trajectory."""
    trajectories = []
    
    for _ in range(2):
        env = SnakeEnv(grid_size=8)
        obs, _ = env.reset(seed=42)
        
        actions = [0, 1, 2, 3, 0, 1, 2, 3]
        trajectory = [obs.copy()]
        
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append(obs.copy())
            if terminated or truncated:
                break
        
        trajectories.append(trajectory)
    
    # Compare trajectories
    assert len(trajectories[0]) == len(trajectories[1])
    for obs1, obs2 in zip(trajectories[0], trajectories[1]):
        np.testing.assert_array_equal(obs1, obs2)


def test_cuda_available():
    """Check CUDA setup."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    assert torch.cuda.device_count() > 0
    print(f"Found {torch.cuda.device_count()} CUDA device(s)")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        assert props.total_memory > 0


def test_amp_compatibility():
    """Verify AMP (mixed precision) works."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    from gymnasium import spaces
    obs_space = spaces.Box(low=0, high=1, shape=(3, 12, 12), dtype=np.float32)
    cnn = SnakeScalableCNN(obs_space, features_dim=256, width=64, depth=2)
    cnn = cnn.cuda()
    
    obs_tensor = torch.randn(4, 3, 12, 12, device='cuda')
    
    # Test with autocast
    with torch.cuda.amp.autocast():
        features = cnn(obs_tensor)
    
    assert features.shape == (4, 256)
    assert not torch.isnan(features).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])