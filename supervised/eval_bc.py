import os
import sys
import argparse
import yaml
import json
import csv
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from snake_env import SnakeEnv, RewardShaping, FrameStack
from supervised.models import make_bc_model
from supervised.utils import set_seed
import gymnasium as gym


def make_eval_env(config: dict, seed: int = 0):
    """Create evaluation environment matching PPO setup."""
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
    
    frame_stack = config['environment'].get('frame_stack', 1)
    if frame_stack > 1:
        env = FrameStack(env, n_frames=frame_stack)
    
    env.reset(seed=seed)
    return env


def evaluate_bc_model(
    model_path: str,
    env_config_path: str,
    n_episodes: int = 100,
    seed: int = 42,
    device: str = 'cpu'
):
    """Evaluate BC model with same protocol as PPO."""
    
    checkpoint = torch.load(model_path, map_location=device)
    meta = checkpoint['meta']
    config_bc = checkpoint['config']
    
    with open(env_config_path, 'r') as f:
        env_config = yaml.safe_load(f)
    
    obs_shape = tuple(meta['obs_shape'])
    action_dim = meta['action_space']
    
    obs_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype='float32')
    action_space = gym.spaces.Discrete(action_dim)
    
    model = make_bc_model(obs_space, action_space, config_bc)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    env = make_eval_env(env_config, seed=seed)
    
    episode_rewards = []
    episode_lengths = []
    
    print(f"Evaluating for {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                logits = model(obs_tensor)
                action = logits.argmax(dim=1).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}")
    
    env.close()
    
    return episode_rewards, episode_lengths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained BC model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pt)')
    parser.add_argument('--config', type=str, required=True, help='Path to env config')
    parser.add_argument('--n-episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', default=True, help='Use deterministic policy')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    set_seed(args.seed, deterministic=True)
    
    device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    
    print("="*60)
    print("BEHAVIOR CLONING EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Seed: {args.seed}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    episode_rewards, episode_lengths = evaluate_bc_model(
        model_path=args.model,
        env_config_path=args.config,
        n_episodes=args.n_episodes,
        seed=args.seed,
        device=device
    )
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    min_length = np.min(episode_lengths)
    max_length = np.max(episode_lengths)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Episodes: {args.n_episodes}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Length: {mean_length:.1f} ± {std_length:.1f}")
    print(f"Reward Range: [{min_reward:.2f}, {max_reward:.2f}]")
    print(f"Length Range: [{min_length}, {max_length}]")
    print("="*60 + "\n")
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'total_reward', 'length', 'seed'])
            for i, (reward, length) in enumerate(zip(episode_rewards, episode_lengths)):
                writer.writerow([i, reward, length, args.seed + i])
        
        print(f"Results saved to: {output_path}")
        
        summary = {
            'n_episodes': args.n_episodes,
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'min_reward': float(min_reward),
            'max_reward': float(max_reward),
            'mean_length': float(mean_length),
            'std_length': float(std_length),
            'min_length': int(min_length),
            'max_length': int(max_length),
            'seed': args.seed
        }
        
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {json_path}")


if __name__ == '__main__':
    main()
