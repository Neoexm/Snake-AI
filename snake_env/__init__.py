"""Snake Gymnasium environment package for IB EE RL experiments."""

from snake_env.snake_env import SnakeEnv
from snake_env.wrappers import FrameStack, RewardShaping

__all__ = ["SnakeEnv", "FrameStack", "RewardShaping"]
__version__ = "1.0.0"