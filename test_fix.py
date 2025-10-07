"""
Quick smoke test to verify all fixes:
1. Environment construction (no "type function" error)
2. Image normalization (no "NatureCNN with Box(0.0, 1.0)" assertion)
3. Custom CNN (no "Kernel size can't be greater than actual input size" error)

Run this to test that the training pipeline works correctly.
"""

import sys
from pathlib import Path
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from snake_env import SnakeEnv, RewardShaping
from train.models import SnakeTinyCNN

def make_snake_env(seed=0, rank=0):
    """Create a snake environment instance."""
    env = SnakeEnv(grid_size=8, render_mode=None)
    env = RewardShaping(env)
    env.reset(seed=seed + rank)
    return env

def main():
    print("="*60)
    print("SNAKE RL - QUICK VERIFICATION TEST")
    print("="*60)
    
    # Test 1: Single environment
    print("\n1. Testing single environment...")
    env = make_snake_env(seed=42, rank=0)
    
    print(f"   ✓ Environment created: {type(env)}")
    print(f"   ✓ Observation space: {env.observation_space}")
    print(f"   ✓ Action space: {env.action_space}")
    
    # Check observation dtype and range
    obs, _ = env.reset()
    print(f"   ✓ Observation shape: {obs.shape}")
    print(f"   ✓ Observation dtype: {obs.dtype}")
    print(f"   ✓ Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
    assert obs.min() >= 0.0 and obs.max() <= 1.0, "Observations not in [0,1] range"
    
    # Test 2: SB3 env checker
    print("\n2. Running SB3 environment checker...")
    try:
        check_env(env, warn=True)
        print("   ✓ Environment passed SB3 checks")
    except Exception as e:
        print(f"   ✗ Environment failed SB3 checks: {e}")
        return
    
    # Test 3: Vectorized environment
    print("\n3. Testing vectorized environment...")
    env_fns = [lambda i=i: make_snake_env(seed=42, rank=i) for i in range(2)]
    vec_env = DummyVecEnv(env_fns)
    
    print(f"   ✓ Vectorized environment created: {type(vec_env)}")
    print(f"   ✓ Number of environments: {vec_env.num_envs}")
    
    # Test 4: PPO model creation with custom CNN
    print("\n4. Creating PPO model with SnakeTinyCNN...")
    model = PPO(
        "CnnPolicy",
        vec_env,
        n_steps=32,
        batch_size=32,
        learning_rate=0.001,
        policy_kwargs={
            "features_extractor_class": SnakeTinyCNN,
            "features_extractor_kwargs": {"features_dim": 256},
            "normalize_images": False,
        },
        verbose=0,
    )
    
    print("   ✓ PPO model created successfully")
    print(f"   ✓ Policy: {model.policy.__class__.__name__}")
    print(f"   ✓ Features Extractor: {model.policy.features_extractor.__class__.__name__}")
    print(f"   ✓ normalize_images: {model.policy.normalize_images}")
    
    # Test 5: Training
    print("\n5. Training for 500 timesteps...")
    model.learn(total_timesteps=500, progress_bar=True)
    
    print("   ✓ Training completed without errors")
    
    # Test 6: Prediction
    print("\n6. Testing prediction...")
    obs = vec_env.reset()
    action, _ = model.predict(obs)
    print(f"   ✓ Prediction successful, action: {action}")
    
    vec_env.close()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nAll three fixes are working:")
    print("1. ✓ Environment construction (no 'type function' error)")
    print("2. ✓ Image normalization (normalize_images=False)")
    print("3. ✓ Custom CNN (SnakeTinyCNN for small 12x12 grids)")
    print("\nYou can now run full training with:")
    print("python train\\train_ppo.py --config train\\configs\\base.yaml --device cpu --total-timesteps 50000 --n-envs 2")

if __name__ == "__main__":
    main()