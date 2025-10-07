"""
Smoke tests for training pipeline.

Quick tests to ensure training doesn't crash on startup.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from snake_env import SnakeEnv


def test_training_smoke():
    """Test that training runs for a few timesteps without errors."""
    # Create temporary directory for logs
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simple environment
        def make_env():
            return SnakeEnv(grid_size=8)
        
        env = DummyVecEnv([make_env])
        
        # Create model with minimal settings
        model = PPO(
            "CnnPolicy",
            env,
            n_steps=32,
            batch_size=32,
            learning_rate=0.001,
            verbose=0,
            tensorboard_log=tmpdir,
        )
        
        # Train for a tiny number of steps
        model.learn(total_timesteps=100, progress_bar=False)
        
        # Save model
        model_path = Path(tmpdir) / "test_model.zip"
        model.save(str(model_path))
        
        # Load model
        loaded_model = PPO.load(str(model_path))
        
        # Test prediction
        obs = env.reset()
        action, _ = loaded_model.predict(obs)
        
        assert action is not None
        env.close()


def test_evaluation():
    """Test that evaluation runs without errors."""
    def make_env():
        return SnakeEnv(grid_size=8)
    
    env = DummyVecEnv([make_env])
    
    model = PPO(
        "CnnPolicy",
        env,
        n_steps=32,
        batch_size=32,
        verbose=0,
    )
    
    # Quick train
    model.learn(total_timesteps=100, progress_bar=False)
    
    # Evaluate
    from stable_baselines3.common.evaluation import evaluate_policy
    
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=2,
        deterministic=True,
    )
    
    assert isinstance(mean_reward, float)
    assert isinstance(std_reward, float)
    
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])