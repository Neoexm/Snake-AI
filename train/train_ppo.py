"""
Main training script for Snake PPO agent.

This script provides a comprehensive CLI for training a PPO agent on the Snake
environment with automatic resource scaling, logging, and evaluation.
"""

import os
import sys
import argparse
import yaml
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from snake_env import SnakeEnv, RewardShaping, FrameStack
from train.autoscale import autoscale
from train.models import SnakeTinyCNN


class ThroughputCallback(BaseCallback):
    """
    Callback to log training throughput (FPS) and resource utilization.
    """
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.start_time = None
        
    def _on_training_start(self):
        import time
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            import time
            elapsed = time.time() - self.start_time
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0
            
            # Log to TensorBoard
            self.logger.record("time/fps", fps)
            
            # Try to get GPU utilization
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem_alloc = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    self.logger.record("system/gpu_memory_allocated_gb", gpu_mem_alloc)
                    self.logger.record("system/gpu_memory_reserved_gb", gpu_mem_reserved)
            except:
                pass
            
            # CPU and RAM
            try:
                import psutil
                self.logger.record("system/cpu_percent", psutil.cpu_percent())
                mem = psutil.virtual_memory()
                self.logger.record("system/ram_percent", mem.percent)
                self.logger.record("system/ram_available_gb", mem.available / (1024**3))
            except:
                pass
        
        return True


def make_snake_env(
    grid_size: int = 12,
    max_steps: Optional[int] = None,
    step_penalty: float = -0.01,
    death_penalty: float = -1.0,
    food_reward: float = 1.0,
    distance_reward_scale: float = 0.0,
    frame_stack: int = 1,
    rank: int = 0,
    seed: int = 0,
    render_mode: Optional[str] = None,
):
    """
    Create a Snake environment with optional wrappers.
    
    This function creates and returns the actual environment instance,
    not a factory function. It's designed to be called within a lambda
    for vectorization.
    
    Parameters
    ----------
    grid_size : int
        Size of the game grid.
    max_steps : int, optional
        Maximum steps per episode.
    step_penalty : float
        Penalty per step (reward shaping).
    death_penalty : float
        Penalty for dying.
    food_reward : float
        Reward for eating food.
    distance_reward_scale : float
        Scale for distance-based reward shaping (0 = off).
    frame_stack : int
        Number of frames to stack (1 = off).
    rank : int
        Environment rank (for vectorized envs).
    seed : int
        Random seed.
    render_mode : str, optional
        Render mode ('human', 'rgb_array', or None).
    
    Returns
    -------
    gym.Env
        The configured Snake environment instance.
    """
    env = SnakeEnv(
        grid_size=grid_size,
        max_steps=max_steps,
        render_mode=render_mode,
    )
    
    # Apply reward shaping
    env = RewardShaping(
        env,
        step_penalty=step_penalty,
        death_penalty=death_penalty,
        food_reward=food_reward,
        distance_reward_scale=distance_reward_scale,
    )
    
    # Apply frame stacking if requested
    if frame_stack > 1:
        env = FrameStack(env, n_frames=frame_stack)
    
    # Set seed
    env.reset(seed=seed + rank)
    return env


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on Snake with autoscaling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config and paths
    parser.add_argument(
        "--config",
        type=str,
        default="train/configs/base.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="runs",
        help="Base directory for logs and models",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (default: timestamp)",
    )
    
    # Training parameters
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total timesteps to train (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    # Resource management
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use for training",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of parallel environments (overrides autoscale)",
    )
    parser.add_argument(
        "--max-utilization",
        action="store_true",
        help="Maximize resource utilization (may risk OOM)",
    )
    
    # Evaluation
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluate every N timesteps",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation",
    )
    
    # Checkpointing
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50000,
        help="Save checkpoint every N timesteps",
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.total_timesteps is not None:
        config['training']['total_timesteps'] = args.total_timesteps
    
    # Autoscale resources
    prefer_device = args.device if args.device != "auto" else None
    resource_config = autoscale(
        max_utilization=args.max_utilization,
        prefer_device=prefer_device,
        override_n_envs=args.n_envs,
    )
    resource_config.print_summary()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create run directory
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.logdir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config and autoscale info
    config['autoscale'] = {
        'device': resource_config.device,
        'n_envs': resource_config.n_envs,
        'n_steps': resource_config.n_steps,
        'batch_size': resource_config.batch_size,
        'use_amp': resource_config.use_amp,
    }
    config['seed'] = args.seed
    save_config(config, str(run_dir / "config.yaml"))
    
    print(f"\n📁 Run directory: {run_dir}")
    print(f"📝 Config saved to: {run_dir / 'config.yaml'}\n")
    
    # Create vectorized environment
    env_kwargs = {
        'grid_size': config['environment']['grid_size'],
        'max_steps': config['environment'].get('max_steps'),
        'step_penalty': config['environment']['step_penalty'],
        'death_penalty': config['environment']['death_penalty'],
        'food_reward': config['environment']['food_reward'],
        'distance_reward_scale': config['environment'].get('distance_reward_scale', 0.0),
        'frame_stack': config['environment'].get('frame_stack', 1),
        'seed': args.seed,
        'render_mode': None,  # Headless for training
    }
    
    # Use SubprocVecEnv on Linux, DummyVecEnv on Windows
    vec_env_cls = SubprocVecEnv if sys.platform != 'win32' else DummyVecEnv
    
    # Create list of environment factories
    # Use lambda with default argument to avoid late binding issue
    env_fns = [
        lambda i=i: make_snake_env(**{**env_kwargs, 'rank': i})
        for i in range(resource_config.n_envs)
    ]
    
    env = vec_env_cls(env_fns)
    env = VecMonitor(env)
    
    # Create eval environment
    eval_env_fns = [
        lambda i=i: make_snake_env(**{**env_kwargs, 'rank': 1000 + i})
        for i in range(min(4, resource_config.n_envs))
    ]
    eval_env = DummyVecEnv(eval_env_fns)
    eval_env = VecMonitor(eval_env)
    
    # PPO hyperparameters (from config with autoscale overrides)
    ppo_kwargs = config['ppo'].copy()
    ppo_kwargs['n_steps'] = resource_config.n_steps
    ppo_kwargs['batch_size'] = resource_config.batch_size
    ppo_kwargs['device'] = resource_config.device
    
    # Extract policy_kwargs from ppo config
    policy_kwargs = ppo_kwargs.pop('policy_kwargs', {})
    
    # Configure custom CNN for small observations (works for 8x8 to 16x16 grids)
    # This replaces SB3's NatureCNN which has kernels too large for 12x12 inputs
    policy_kwargs.setdefault('features_extractor_class', SnakeTinyCNN)
    policy_kwargs.setdefault('features_extractor_kwargs', {'features_dim': 256})
    
    # Ensure normalize_images is False for our pre-normalized float32 observations
    policy_kwargs.setdefault('normalize_images', False)
    
    # Get policy type from config
    policy = config.get('policy', 'CnnPolicy')
    
    # Create PPO model
    print("🤖 Creating PPO model...")
    print(f"Policy: {policy}")
    print(f"Features Extractor: {policy_kwargs.get('features_extractor_class', 'default').__name__}")
    print(f"Features Dim: {policy_kwargs.get('features_extractor_kwargs', {}).get('features_dim', 'default')}")
    print(f"normalize_images: {policy_kwargs.get('normalize_images', False)}")
    
    model = PPO(
        policy=policy,
        env=env,
        policy_kwargs=policy_kwargs,
        **ppo_kwargs,
        tensorboard_log=str(run_dir),
        verbose=1,
    )
    
    # Setup custom logger for CSV output
    logger = configure(str(run_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // resource_config.n_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir / "eval"),
        eval_freq=max(args.eval_freq // resource_config.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    
    # Throughput callback
    throughput_callback = ThroughputCallback(log_freq=1000)
    callbacks.append(throughput_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Train
    print(f"\n🚀 Starting training for {config['training']['total_timesteps']:,} timesteps...\n")
    
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callback_list,
            log_interval=10,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    # Save final model
    final_path = run_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"\n✅ Training complete!")
    print(f"📦 Final model saved to: {final_path}")
    print(f"🏆 Best model saved to: {run_dir / 'best_model.zip'}")
    print(f"\n📊 View training progress:")
    print(f"   tensorboard --logdir {run_dir}")
    
    # Cleanup
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()