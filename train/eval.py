"""
Standalone evaluation script for trained Snake PPO agents.

Runs fixed-seed evaluation episodes and logs metrics without training.
CRITICAL: Does NOT set CUDA_VISIBLE_DEVICES to allow DDP-trained models to load correctly.
"""

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from snake_env import SnakeEnv, RewardShaping, FrameStack


def make_eval_env(config: Dict, seed: int = 0):
    """
    Create evaluation environment from config.
    
    Returns the actual environment instance, not a factory.
    """
    env = SnakeEnv(
        grid_size=config['environment']['grid_size'],
        max_steps=config['environment'].get('max_steps'),
        render_mode=None,
    )
    
    env = RewardShaping(
        env,
        step_penalty=config['environment']['step_penalty'],
        death_penalty=config['environment']['death_penalty'],
        food_reward=config['environment']['food_reward'],
        distance_reward_scale=config['environment'].get('distance_reward_scale', 0.0),
    )
    
    if config['environment'].get('frame_stack', 1) > 1:
        env = FrameStack(env, n_frames=config['environment']['frame_stack'])
    
    env.reset(seed=seed)
    return env


def evaluate_model(
    model_path: str,
    config_path: str,
    n_eval_episodes: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Dict[str, float], List[float], List[int]]:
    """
    Evaluate a trained model.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model (.zip file).
    config_path : str
        Path to the config file used during training.
    n_eval_episodes : int
        Number of episodes to evaluate.
    seed : int
        Random seed for evaluation.
    verbose : bool
        Print progress.
    
    Returns
    -------
    metrics : dict
        Summary statistics (mean, std, min, max, median).
    episode_rewards : list
        List of episode rewards.
    episode_lengths : list
        List of episode lengths.
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model on available GPU (auto-detect, don't hardcode cuda:0)
    if verbose:
        print(f"Loading model from: {model_path}")
    
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set deterministic mode for reproducible evaluation
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(mode=True, warn_only=True)
    
    model = PPO.load(model_path, device=device)
    
    # Create eval environment
    env = DummyVecEnv([lambda: make_eval_env(config, seed=seed)])
    env = VecMonitor(env)
    
    # Evaluate
    if verbose:
        print(f"Evaluating for {n_eval_episodes} episodes (deterministic mode)...")
    
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            done = done[0]
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{n_eval_episodes}: "
                  f"Reward={episode_reward:.2f}, Length={episode_length}")
    
    env.close()
    
    # Compute statistics
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'median_reward': np.median(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'min_length': np.min(episode_lengths),
        'max_length': np.max(episode_lengths),
        'median_length': np.median(episode_lengths),
    }
    
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Episodes: {n_eval_episodes}")
        print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean Length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
        print(f"Reward Range: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
        print(f"Length Range: [{metrics['min_length']}, {metrics['max_length']}]")
        print("="*60 + "\n")
    
    return metrics, episode_rewards, episode_lengths


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Snake PPO agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (auto-detect from model dir if not provided)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results (JSON)",
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Auto-detect config if not provided
    config_path = args.config
    if config_path is None:
        model_dir = Path(args.model).parent
        config_path = model_dir / "config.yaml"
        if not config_path.exists():
            # Try parent directory
            config_path = model_dir.parent / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Could not find config.yaml in {model_dir} or its parent. "
                "Please specify --config explicitly."
            )
    
    # Run evaluation
    metrics, episode_rewards, episode_lengths = evaluate_model(
        model_path=args.model,
        config_path=str(config_path),
        n_eval_episodes=args.n_episodes,
        seed=args.seed,
        verbose=True,
    )
    
    # Save results if requested
    if args.output:
        import json
        output_data = {
            'metrics': metrics,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'model_path': args.model,
            'config_path': str(config_path),
            'n_episodes': args.n_episodes,
            'seed': args.seed,
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"📊 Results saved to: {args.output}")


if __name__ == "__main__":
    main()