# train/eval_ppo_ddp.py
import argparse, json, os, re, time, sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from snake_env import SnakeEnv
from supervised.utils import set_seed
from train.models import SnakeScalableCNN
import gymnasium as gym
import torch.nn as nn

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
    
    def act(self, x):
        features = self.forward(x)
        logits = self.actor(features)
        return logits.argmax(dim=-1)

# ---- Helpers ----------------------------------------------------------------
def latest_checkpoint(folder):
    # pick numerically largest suffix in 'checkpoint_########.pt'
    ckpts = [f for f in os.listdir(folder) if f.startswith("checkpoint_") and f.endswith(".pt")]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint_*.pt in {folder}")
    key = lambda s: int(re.search(r"checkpoint_(\d+)\.pt", s).group(1))
    return os.path.join(folder, sorted(ckpts, key=key)[-1])

def eval_model(ckpt_path, n_episodes, seed, csv_out, device):
    set_seed(seed)
    from snake_env import RewardShaping, FrameStack
    env = SnakeEnv(grid_size=12)
    env = RewardShaping(env)
    env = FrameStack(env, n_frames=1)
    n_actions = env.action_space.n
    device = torch.device(device)

    obs_shape = (3, 12, 12)
    model = Agent(obs_shape=obs_shape, action_dim=n_actions, width=128, depth=3).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("agent") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(sd, strict=True)
    model.eval()

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    returns, lengths = [], []
    
    print(f"Evaluating for {n_episodes} episodes...")
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_ret, ep_len = False, 0.0, 0
        while not done:
            x = torch.from_numpy(obs).unsqueeze(0).float().to(device)
            with torch.no_grad():
                a = model.act(x).item()
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_ret += float(r)
            ep_len += 1
        returns.append(ep_ret)
        lengths.append(ep_len)
        
        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}")

    with open(csv_out, "w") as f:
        f.write("episode,total_reward,length,seed\n")
        for i, (ret, length) in enumerate(zip(returns, lengths)):
            f.write(f"{i},{ret},{length},{seed + i}\n")

    mean_reward = np.mean(returns)
    std_reward = np.std(returns)
    min_reward = np.min(returns)
    max_reward = np.max(returns)
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    min_length = np.min(lengths)
    max_length = np.max(lengths)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Episodes: {n_episodes}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Length: {mean_length:.1f} ± {std_length:.1f}")
    print(f"Reward Range: [{min_reward:.2f}, {max_reward:.2f}]")
    print(f"Length Range: [{min_length}, {max_length}]")
    print("="*60 + "\n")
    
    print(f"Results saved to: {csv_out}")
    
    summary = {
        'n_episodes': n_episodes,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'min_reward': float(min_reward),
        'max_reward': float(max_reward),
        'mean_length': float(mean_length),
        'std_length': float(std_length),
        'min_length': int(min_length),
        'max_length': int(max_length),
        'seed': seed
    }
    
    json_path = Path(csv_out).with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {json_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True, help="Folder with checkpoint_*.pt")
    p.add_argument("--checkpoint", default="", help="Optional specific .pt path; if empty, pick latest")
    p.add_argument("--n-episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="results/ppo_eval.csv")
    args = p.parse_args()

    ckpt = args.checkpoint or latest_checkpoint(args.folder)
    print(f"Evaluating checkpoint: {ckpt}")
    eval_model(ckpt, args.n_episodes, args.seed, args.output, args.device)
