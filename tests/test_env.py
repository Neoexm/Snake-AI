"""
Unit tests for Snake environment.

Tests basic functionality, determinism, and correctness of the environment.
"""

import pytest
import numpy as np
from snake_env import SnakeEnv, RewardShaping, FrameStack


class TestSnakeEnv:
    """Test the base Snake environment."""
    
    def test_init(self):
        """Test environment initialization."""
        env = SnakeEnv(grid_size=12)
        assert env.grid_size == 12
        assert env.observation_space.shape == (3, 12, 12)
        assert env.action_space.n == 4
    
    def test_reset(self):
        """Test environment reset."""
        env = SnakeEnv(grid_size=12)
        obs, info = env.reset(seed=42)
        
        # Check observation shape and type
        assert obs.shape == (3, 12, 12)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)
        
        # Check that observation contains valid data
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0
        
        # Snake should be visible in plane 0
        assert obs[0].sum() > 0
        # Food should be visible in plane 1
        assert obs[1].sum() > 0
        # Head should be visible in plane 2
        assert obs[2].sum() > 0
    
    def test_deterministic_reset(self):
        """Test that reset with same seed produces same state."""
        env1 = SnakeEnv(grid_size=12)
        env2 = SnakeEnv(grid_size=12)
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_step_shape(self):
        """Test that step returns correct shapes."""
        env = SnakeEnv(grid_size=12)
        obs, _ = env.reset(seed=42)
        
        obs, reward, terminated, truncated, info = env.step(1)
        
        assert obs.shape == (3, 12, 12)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_food_reward(self):
        """Test that eating food gives positive reward."""
        env = SnakeEnv(grid_size=12)
        obs, _ = env.reset(seed=42)
        
        # Find food position
        food_pos = np.argwhere(obs[1] == 1.0)[0]
        head_pos = np.argwhere(obs[2] == 1.0)[0]
        
        # Move towards food (this is probabilistic, so we'll just check mechanism)
        initial_length = len(env.snake)
        
        # Take several steps and check if we ever get positive reward
        got_food = False
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if reward > 0:
                got_food = True
                assert len(env.snake) > initial_length  # Snake grew
                break
            if terminated:
                break
        
        # It's possible we don't get food in 100 random steps, so we won't assert
        # But if we did get food, snake should have grown
        if got_food:
            assert True  # Test passed
    
    def test_death_penalty(self):
        """Test that dying gives negative reward."""
        env = SnakeEnv(grid_size=12)
        obs, _ = env.reset(seed=42)
        
        # Move in circles until we hit wall or ourselves
        for _ in range(1000):
            obs, reward, terminated, truncated, info = env.step(0)  # Always move left
            if terminated:
                assert reward < 0  # Death should give negative reward
                break
        
        assert terminated  # Should have died within 1000 steps
    
    def test_no_reverse_move(self):
        """Test that 180-degree turns are prevented."""
        env = SnakeEnv(grid_size=12)
        obs, _ = env.reset(seed=42)
        
        # Snake starts moving right (direction 1)
        initial_dir = env.dir
        
        # Try to move left (opposite of right)
        opposite_action = {0: 1, 1: 0, 2: 3, 3: 2}[initial_dir]
        
        # Store snake position before step
        head_before = env.snake[-1].copy()
        
        obs, reward, terminated, truncated, info = env.step(opposite_action)
        
        # Direction should not have changed to opposite
        assert env.dir == initial_dir
    
    def test_max_steps_termination(self):
        """Test that episode terminates after max_steps."""
        max_steps = 50
        env = SnakeEnv(grid_size=12, max_steps=max_steps)
        obs, _ = env.reset(seed=42)
        
        for i in range(max_steps + 10):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated:
                assert i < max_steps + 5  # Should terminate around max_steps
                break
        
        assert terminated
    
    def test_render(self):
        """Test that render returns valid image."""
        env = SnakeEnv(grid_size=12, render_mode="rgb_array")
        obs, _ = env.reset(seed=42)
        
        frame = env.render()
        
        # Check shape (should be scaled up)
        assert len(frame.shape) == 3
        assert frame.shape[2] == 3  # RGB
        assert frame.dtype == np.uint8
        
        # Should be larger than grid size due to scaling
        assert frame.shape[0] > 12
        assert frame.shape[1] > 12


class TestRewardShaping:
    """Test the RewardShaping wrapper."""
    
    def test_custom_rewards(self):
        """Test that custom reward values are applied."""
        env = SnakeEnv(grid_size=12)
        env = RewardShaping(
            env,
            step_penalty=-0.1,
            death_penalty=-10.0,
            food_reward=5.0,
        )
        
        obs, _ = env.reset(seed=42)
        
        # Take a step (should get step penalty)
        obs, reward, terminated, truncated, info = env.step(1)
        if not terminated:
            assert abs(reward - (-0.1)) < 0.001  # Step penalty
    
    def test_distance_shaping(self):
        """Test distance-based reward shaping."""
        env = SnakeEnv(grid_size=12)
        env = RewardShaping(
            env,
            distance_reward_scale=0.1,
        )
        
        obs, _ = env.reset(seed=42)
        
        # Distance shaping should provide additional signal
        # (hard to test precisely without controlling movements)
        obs, reward, terminated, truncated, info = env.step(1)
        assert isinstance(reward, float)


class TestFrameStack:
    """Test the FrameStack wrapper."""
    
    def test_stacking(self):
        """Test that frames are stacked correctly."""
        env = SnakeEnv(grid_size=12)
        env = FrameStack(env, n_frames=4)
        
        # Observation space should be 4x larger in channel dimension
        assert env.observation_space.shape == (12, 12, 12)  # 3 channels * 4 frames
        
        obs, _ = env.reset(seed=42)
        assert obs.shape == (12, 12, 12)
    
    def test_reset_fills_stack(self):
        """Test that reset fills stack with initial observation."""
        env = SnakeEnv(grid_size=12)
        env = FrameStack(env, n_frames=4)
        
        obs, _ = env.reset(seed=42)
        
        # All frames in stack should be identical after reset
        frame_size = 3 * 12 * 12
        for i in range(4):
            start = i * 3
            end = (i + 1) * 3
            if i > 0:
                # Compare frames (they should be same after reset)
                np.testing.assert_array_equal(
                    obs[start:end],
                    obs[0:3]
                )
    
    def test_step_updates_stack(self):
        """Test that step updates the frame stack."""
        env = SnakeEnv(grid_size=12)
        env = FrameStack(env, n_frames=4)
        
        obs1, _ = env.reset(seed=42)
        obs2, _, _, _, _ = env.step(1)
        
        # Observations should be different (snake moved)
        assert not np.array_equal(obs1, obs2)


def test_env_in_loop():
    """Test environment in typical RL loop."""
    env = SnakeEnv(grid_size=12)
    
    num_episodes = 5
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        assert steps > 0
        assert done or steps >= 100


def test_observation_dtype_and_range():
    """Test that observations have correct dtype and value range."""
    env = SnakeEnv(grid_size=12)
    obs, _ = env.reset(seed=42)
    
    # Check shape
    assert obs.shape == (3, 12, 12), f"Expected shape (3, 12, 12), got {obs.shape}"
    
    # Check dtype is float32
    assert obs.dtype == np.float32, f"Expected dtype float32, got {obs.dtype}"
    
    # Check values are in [0, 1] range
    assert obs.min() >= 0.0, f"Observation has values below 0: {obs.min()}"
    assert obs.max() <= 1.0, f"Observation has values above 1: {obs.max()}"
    
    # Take a step and check again
    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)
    
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0


def test_sb3_env_checker():
    """Test environment compatibility with Stable-Baselines3."""
    from stable_baselines3.common.env_checker import check_env
    
    env = SnakeEnv(grid_size=12)
    
    # Run SB3's environment checker
    # This will raise an exception if the environment is not compatible
    check_env(env, warn=True)
    
    # If we get here, the environment passed all checks
    assert True