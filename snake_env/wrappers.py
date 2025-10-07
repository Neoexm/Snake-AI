"""Gymnasium wrappers for Snake environment."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Any, Dict, Tuple


class FrameStack(gym.Wrapper):
    """
    Stack the last N observations along the channel dimension.
    
    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    n_frames : int
        Number of frames to stack.
    """

    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        
        # Update observation space
        low = env.observation_space.low
        high = env.observation_space.high
        shape = env.observation_space.shape
        
        # Stack along channel dimension: (C, H, W) -> (C*n_frames, H, W)
        new_shape = (shape[0] * n_frames, shape[1], shape[2])
        self.observation_space = spaces.Box(
            low=np.repeat(low, n_frames, axis=0),
            high=np.repeat(high, n_frames, axis=0),
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and initialize frame stack."""
        obs, info = self.env.reset(**kwargs)
        # Fill the deque with the initial observation
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return stacked observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Stack frames along the channel dimension."""
        # Stack along axis 0 (channel dimension)
        return np.concatenate(list(self.frames), axis=0)


class RewardShaping(gym.Wrapper):
    """
    Apply reward shaping to encourage specific behaviors.
    
    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    step_penalty : float
        Penalty for each step (default: -0.01).
    death_penalty : float
        Penalty for dying (default: -1.0).
    food_reward : float
        Reward for eating food (default: +1.0).
    distance_reward_scale : float
        Scale factor for distance-based reward shaping (0 = off).
    """

    def __init__(
        self,
        env: gym.Env,
        step_penalty: float = -0.01,
        death_penalty: float = -1.0,
        food_reward: float = 1.0,
        distance_reward_scale: float = 0.0,
    ):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.death_penalty = death_penalty
        self.food_reward = food_reward
        self.distance_reward_scale = distance_reward_scale
        self.prev_distance = None

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        obs, info = self.env.reset(**kwargs)
        self.prev_distance = self._compute_distance()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and apply reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Override base rewards with shaped rewards
        shaped_reward = self.step_penalty
        
        # Check if food was eaten (snake grew)
        if reward > 0:  # Base env gives +1 for food
            shaped_reward = self.food_reward
        elif terminated:  # Death
            shaped_reward = self.death_penalty
        
        # Optional: add distance-based shaping
        if self.distance_reward_scale > 0:
            current_distance = self._compute_distance()
            if self.prev_distance is not None:
                # Reward getting closer to food
                delta = self.prev_distance - current_distance
                shaped_reward += delta * self.distance_reward_scale
            self.prev_distance = current_distance
        
        return obs, shaped_reward, terminated, truncated, info

    def _compute_distance(self) -> float:
        """Compute Manhattan distance from head to food."""
        head = self.env.snake[-1]
        food = self.env.food
        return float(np.abs(head - food).sum())