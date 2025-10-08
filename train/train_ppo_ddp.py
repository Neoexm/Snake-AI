"""
Multi-GPU DDP PPO training for Snake environment.

This is a from-scratch implementation with proper DistributedDataParallel support.
Unlike Stable Baselines3, this actually synchronizes gradients across GPUs.
"""

import os
import sys
import time
import argparse
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).parent.parent))

from snake_env import SnakeEnv, RewardShaping, FrameStack
from train.models import SnakeScalableCNN
from train.ddp_utils import (
    setup_distributed,
    cleanup_distributed,
    is_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    setup_nccl_env_for_b200,
    print_distributed_info,
)


@dataclass
class Config:
    env_id: str = "Snake-v0"
    grid_size: int = 12
    total_timesteps: int = 5_000_000
    n_envs: int = 64
    n_steps: int = 1024
    n_epochs: int = 4
    n_minibatches: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_width: int = 128
    policy_depth: int = 3
    checkpoint_freq: int = 100
    log_freq: int = 10
    seed: int = 42


class Agent(nn.Module):
    def __init__(self, obs_shape, action_dim, width=128, depth=3):
        super().__init__()
        self.feature_extractor = SnakeScalableCNN(
            observation_space=gym.spaces.Box(0, 1, obs_shape, dtype=np.float32),
            features_dim=256,
            width=width,
            depth=depth
        )
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return features
    
    def get_value(self, x):
        features = self.forward(x)
        return self.critic(features)
    
    def get_action_and_value(self, x, action=None):
        features = self.forward(x)
        logits = self.actor(features)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(features)


def make_env(rank, seed, grid_size=12):
    def thunk():
        env = SnakeEnv(grid_size=grid_size)
        env = RewardShaping(env)
        env = FrameStack(env, n_frames=1)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + rank)
        return env
    return thunk


def train(config: Config):
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f'cuda:{rank}')
    
    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    
    run_name = f"ppo_ddp_{int(time.time())}"
    run_dir = Path("runs") / run_name
    
    if is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(run_dir))
        print(f"\n{'='*60}")
        print(f"Starting DDP PPO Training")
        print(f"{'='*60}")
        print(f"Total timesteps: {config.total_timesteps:,}")
        print(f"World size: {world_size}")
        print(f"Envs per rank: {config.n_envs}")
        print(f"Global batch size: {config.n_envs * config.n_steps * world_size:,}")
        print(f"{'='*60}\n")
    
    envs = gym.vector.AsyncVectorEnv([
        make_env(rank * config.n_envs + i, config.seed, config.grid_size)
        for i in range(config.n_envs)
    ])
    
    obs_shape = envs.single_observation_space.shape
    action_dim = envs.single_action_space.n
    
    agent = Agent(obs_shape, action_dim, config.policy_width, config.policy_depth).to(device)
    
    if is_distributed():
        agent = DDP(agent, device_ids=[rank])
    
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    
    obs_buf = torch.zeros((config.n_steps, config.n_envs) + obs_shape).to(device)
    actions_buf = torch.zeros((config.n_steps, config.n_envs), dtype=torch.long).to(device)
    logprobs_buf = torch.zeros((config.n_steps, config.n_envs)).to(device)
    rewards_buf = torch.zeros((config.n_steps, config.n_envs)).to(device)
    dones_buf = torch.zeros((config.n_steps, config.n_envs)).to(device)
    values_buf = torch.zeros((config.n_steps, config.n_envs)).to(device)
    
    global_step = 0
    update_count = 0
    start_time = time.time()
    
    next_obs, _ = envs.reset()
    next_obs = torch.FloatTensor(next_obs).to(device)
    next_done = torch.zeros(config.n_envs).to(device)
    
    episode_returns = []
    episode_lengths = []
    
    num_updates = config.total_timesteps // (config.n_envs * config.n_steps * world_size)
    
    for update in range(1, num_updates + 1):
        agent.eval()
        
        for step in range(config.n_steps):
            global_step += config.n_envs * world_size
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.module.get_action_and_value(next_obs) if is_distributed() else agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()
            
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            
            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_obs = torch.FloatTensor(next_obs_np).to(device)
            rewards_buf[step] = torch.FloatTensor(reward).to(device)
            next_done = torch.FloatTensor(np.logical_or(terminations, truncations)).to(device)
            
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_returns.append(info["episode"]["r"])
                        episode_lengths.append(info["episode"]["l"])
        
        with torch.no_grad():
            next_value = agent.module.get_value(next_obs) if is_distributed() else agent.get_value(next_obs)
            next_value = next_value.flatten()
            advantages = torch.zeros_like(rewards_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(config.n_steps)):
                if t == config.n_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + config.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf
        
        agent.train()
        
        b_obs = obs_buf.reshape((-1,) + obs_shape)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        
        b_inds = np.arange(config.n_envs * config.n_steps)
        minibatch_size = (config.n_envs * config.n_steps) // config.n_minibatches
        
        for epoch in range(config.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.n_envs * config.n_steps, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.module.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                ) if is_distributed() else agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()
        
        update_count += 1
        
        if is_main_process() and update % config.log_freq == 0:
            elapsed = time.time() - start_time
            fps = global_step / elapsed
            
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("charts/SPS", fps, global_step)
            
            if len(episode_returns) > 0:
                writer.add_scalar("charts/episodic_return", np.mean(episode_returns[-100:]), global_step)
                writer.add_scalar("charts/episodic_length", np.mean(episode_lengths[-100:]), global_step)
                print(f"[Update {update}/{num_updates}] Step: {global_step:,} | "
                      f"Return: {np.mean(episode_returns[-100:]):.2f} | "
                      f"FPS: {fps:.0f}")
        
        if is_main_process() and update % config.checkpoint_freq == 0:
            checkpoint_path = run_dir / f"checkpoint_{global_step}.pt"
            torch.save({
                'agent': agent.module.state_dict() if is_distributed() else agent.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
                'update': update,
                'config': vars(config),
            }, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    if is_main_process():
        final_path = run_dir / "final_model.pt"
        torch.save(agent.module.state_dict() if is_distributed() else agent.state_dict(), final_path)
        print(f"\n✓ Training complete! Model saved to {final_path}")
        writer.close()
    
    envs.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train/configs/base.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--policy-width", type=int, default=None)
    parser.add_argument("--policy-depth", type=int, default=None)
    args = parser.parse_args()
    
    setup_nccl_env_for_b200()
    
    if is_distributed() or ('RANK' in os.environ and 'WORLD_SIZE' in os.environ):
        rank, world_size, device = setup_distributed('nccl')
        print_distributed_info()
    
    config = Config()
    
    if Path(args.config).exists():
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                for key, value in yaml_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
    
    if args.total_timesteps:
        config.total_timesteps = args.total_timesteps
    if args.n_envs:
        config.n_envs = args.n_envs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.grid_size:
        config.grid_size = args.grid_size
    if args.policy_width:
        config.policy_width = args.policy_width
    if args.policy_depth:
        config.policy_depth = args.policy_depth
    
    try:
        train(config)
    finally:
        if is_distributed():
            cleanup_distributed()


if __name__ == "__main__":
    main()
