"""Custom neural network models for Snake RL."""

from train.models.snake_tiny_cnn import SnakeTinyCNN
from train.models.snake_scalable_cnn import SnakeScalableCNN, SnakeDeepCNN

__all__ = ["SnakeTinyCNN", "SnakeScalableCNN", "SnakeDeepCNN"]