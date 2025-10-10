import os
import sys
import argparse
import yaml
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from snake_env import SnakeEnv, RewardShaping, FrameStack
from supervised.utils import set_seed


class ScriptedPolicy:
    """Greedy scripted policy that moves toward food while avoiding collisions."""
    
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
    
    def get_action(self, obs: np.ndarray, env: SnakeEnv) -> int:
        """Get action from scripted policy."""
        head = env.snake[-1]
        food = env.food
        
        delta = food - head
        
        safe_actions = []
        for action in range(4):
            if self._is_safe_move(env, action):
                safe_actions.append(action)
        
        if not safe_actions:
            return env.action_space.sample()
        
        if delta[1] != 0:
            preferred = 1 if delta[1] > 0 else 0
            if preferred in safe_actions:
                return preferred
        
        if delta[0] != 0:
            preferred = 3 if delta[0] > 0 else 2
            if preferred in safe_actions:
                return preferred
        
        return safe_actions[0]
    
    def _is_safe_move(self, env: SnakeEnv, action: int) -> bool:
        """Check if action leads to safe position."""
        from snake_env import DIRS, OPPOSITE
        
        if len(env.snake) > 1 and action == OPPOSITE[env.dir]:
            return False
        
        head = env.snake[-1].copy()
        head += DIRS[action]
        
        y, x = head
        if y < 0 or y >= self.grid_size or x < 0 or x >= self.grid_size:
            return False
        
        body = {tuple(p) for p in env.snake[:-1]}
        if tuple(head) in body:
            return False
        
        return True


class PPOPolicy:
    """Wrapper for trained PPO model."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path, device=device)
    
    def get_action(self, obs: np.ndarray, env: SnakeEnv) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def make_env(config: dict, seed: int = 0):
    """Create environment matching PPO setup."""
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
    return env, frame_stack


def collect_transitions(
    env: SnakeEnv,
    policy,
    num_steps: int,
    shard_size: int,
    save_dir: Path,
    max_episodes: Optional[int] = None
):
    """Collect transitions from policy, saving shards incrementally to disk."""
    
    observations = []
    actions = []
    
    obs, _ = env.reset()
    episode_count = 0
    steps_collected = 0
    shard_count = 0
    start_time = time.time()
    last_print_time = start_time
    
    while steps_collected < num_steps:
        if max_episodes and episode_count >= max_episodes:
            break
        
        action = policy.get_action(obs, env.unwrapped if hasattr(env, 'unwrapped') else env)
        
        observations.append(obs.copy())
        actions.append(action)
        steps_collected += 1
        
        if len(observations) >= shard_size:
            shard_count += 1
            shard_path = save_dir / f"shard_{shard_count:05d}.pt"
            save_shard(observations, actions, shard_path)
            print(f"    Saved {shard_path.name} ({len(observations)} samples)")
            observations = []
            actions = []
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if done:
            obs, _ = env.reset()
            episode_count += 1
        
        current_time = time.time()
        if current_time - last_print_time >= 2.0 or steps_collected >= num_steps:
            elapsed = current_time - start_time
            progress = steps_collected / num_steps
            rate = steps_collected / elapsed if elapsed > 0 else 0
            eta_seconds = (num_steps - steps_collected) / rate if rate > 0 else 0
            
            print(f"  Progress: {steps_collected}/{num_steps} ({progress*100:.1f}%) | "
                  f"Episodes: {episode_count} | "
                  f"Rate: {rate:.0f} steps/s | "
                  f"ETA: {eta_seconds:.0f}s")
            last_print_time = current_time
    
    if observations:
        shard_count += 1
        shard_path = save_dir / f"shard_{shard_count:05d}.pt"
        save_shard(observations, actions, shard_path)
        print(f"    Saved {shard_path.name} ({len(observations)} samples)")
    
    return steps_collected, shard_count


def save_shard(
    observations: List[np.ndarray],
    actions: List[int],
    shard_path: Path
):
    """Save a shard of data to disk."""
    data = {
        'observations': torch.from_numpy(np.stack(observations)),
        'actions': torch.tensor(actions, dtype=torch.long)
    }
    torch.save(data, shard_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect expert demonstrations for behavior cloning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, required=True, help='Path to environment config')
    parser.add_argument('--expert', type=str, choices=['scripted', 'ppo'], required=True)
    parser.add_argument('--ppo-model', type=str, default=None, help='Path to PPO model (if expert=ppo)')
    parser.add_argument('--steps', type=int, required=True, help='Number of steps to collect')
    parser.add_argument('--max-episodes', type=int, default=None, help='Max episodes to collect')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='data/snake_bc', help='Save directory')
    parser.add_argument('--shard-size', type=int, default=100000, help='Samples per shard')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of parallel envs (scripted only)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    set_seed(args.seed, deterministic=True)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("DATASET COLLECTION")
    print("="*60)
    print(f"Expert: {args.expert}")
    print(f"Target steps: {args.steps}")
    print(f"Seed: {args.seed}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"{args.expert}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Save directory: {save_dir}")
    print("="*60 + "\n")
    
    env, frame_stack = make_env(config, seed=args.seed)
    
    if args.expert == 'scripted':
        policy = ScriptedPolicy(config['environment']['grid_size'])
    elif args.expert == 'ppo':
        if not args.ppo_model:
            raise ValueError("--ppo-model required when expert=ppo")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        policy = PPOPolicy(args.ppo_model, device=device)
    
    print("Collecting demonstrations...")
    print(f"Will save shards incrementally every {args.shard_size} samples to save RAM\n")
    start_time = time.time()
    
    total_steps, num_shards = collect_transitions(
        env, policy, args.steps, args.shard_size, save_dir, args.max_episodes
    )
    
    elapsed = time.time() - start_time
    print(f"\nCollected {total_steps} transitions in {elapsed:.1f}s")
    print(f"Rate: {total_steps / elapsed:.1f} steps/sec")
    
    print("\nLoading first shard for metadata...")
    first_shard = torch.load(save_dir / 'shard_00001.pt', map_location='cpu')
    
    meta = {
        'expert_type': args.expert,
        'ppo_model': args.ppo_model,
        'total_steps': total_steps,
        'num_shards': num_shards,
        'shard_size': args.shard_size,
        'obs_shape': list(first_shard['observations'][0].shape),
        'obs_dtype': str(first_shard['observations'].dtype),
        'action_space': 4,
        'env_config': config['environment'],
        'frame_stack': frame_stack,
        'seed': args.seed,
        'timestamp': timestamp,
        'allow_hflip': False
    }
    
    meta_path = save_dir / 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nMetadata saved to {meta_path}")
    
    print("\nValidating & computing dataset statistics...")
    print(f"  Loaded shard 1: obs shape {first_shard['observations'].shape}, actions shape {first_shard['actions'].shape}")
    print(f"  Obs dtype: {first_shard['observations'].dtype}, range [{first_shard['observations'].min():.3f}, {first_shard['observations'].max():.3f}]")
    
    print("\nComputing action class distribution across all shards...")
    action_counts = np.zeros(4, dtype=np.int64)
    obs_min, obs_max = float('inf'), float('-inf')
    obs_sum, obs_sq_sum = 0.0, 0.0
    total_obs_vals = 0
    
    for shard_idx in range(1, num_shards + 1):
        shard_path = save_dir / f"shard_{shard_idx:05d}.pt"
        shard_data = torch.load(shard_path, map_location='cpu')
        
        shard_actions = shard_data['actions'].numpy()
        action_counts += np.bincount(shard_actions, minlength=4)
        
        shard_obs = shard_data['observations']
        obs_min = min(obs_min, float(shard_obs.min()))
        obs_max = max(obs_max, float(shard_obs.max()))
        obs_sum += float(shard_obs.sum())
        obs_sq_sum += float((shard_obs ** 2).sum())
        total_obs_vals += shard_obs.numel()
    
    action_dist = action_counts / action_counts.sum()
    print(f"\nAction class distribution (all {total_steps} samples):")
    for i in range(4):
        print(f"  Action {i}: {action_counts[i]:7d} ({action_dist[i]*100:5.2f}%)")
    
    obs_mean = obs_sum / total_obs_vals
    obs_std = np.sqrt(obs_sq_sum / total_obs_vals - obs_mean ** 2)
    print(f"\nObservation statistics (all samples):")
    print(f"  Min: {obs_min:.4f}, Max: {obs_max:.4f}")
    print(f"  Mean: {obs_mean:.4f}, Std: {obs_std:.4f}")
    
    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)
    print(f"Dataset directory: {save_dir}")
    print(f"Total samples: {total_steps}")
    print(f"Shards: {num_shards}")
    print("="*60)
    
    env.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
