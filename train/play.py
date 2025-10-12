"""
Interactive visualization of a trained Snake PPO agent.

Supports both DDP checkpoints (.pt) and Stable Baselines3 models (.zip).
Supports both local window display (OpenCV/Pygame) and headless mode for saving videos.
"""

import os
import sys
import argparse
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from snake_env import SnakeEnv, RewardShaping, FrameStack
from train.models import SnakeScalableCNN
import gymnasium as gym

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
    
    def get_action_and_value(self, x, action=None):
        features = self.forward(x)
        logits = self.actor(features)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(features)
    
    def act(self, x):
        features = self.forward(x)
        logits = self.actor(features)
        return logits.argmax(dim=-1)

def latest_checkpoint(folder):
    ckpts = [f for f in os.listdir(folder) if f.startswith("checkpoint_") and f.endswith(".pt")]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint_*.pt in {folder}")
    key = lambda s: int(re.search(r"checkpoint_(\d+)\.pt", s).group(1))
    return os.path.join(folder, sorted(ckpts, key=key)[-1])

def load_model(model_path, device='cpu'):
    if model_path.endswith('.pt'):
        obs_shape = (3, 12, 12)
        model = Agent(obs_shape=obs_shape, action_dim=4, width=128, depth=3).to(device)
        ckpt = torch.load(model_path, map_location=device)
        sd = ckpt.get("agent") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model, 'ddp'
    else:
        from stable_baselines3 import PPO
        return PPO.load(model_path, device=device), 'sb3'


def play_local_window(
    model_path: str,
    config_path: Optional[str] = None,
    fps: int = 10,
    deterministic: bool = True,
    n_episodes: Optional[int] = None,
    seed: int = 42,
):
    """
    Play Snake with a local OpenCV window.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model (.pt or .zip file).
    config_path : str, optional
        Path to the config file (not needed for DDP checkpoints).
    fps : int
        Frames per second for display.
    deterministic : bool
        Use deterministic policy.
    n_episodes : int, optional
        Number of episodes to play (None = infinite).
    seed : int
        Random seed.
    """
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python is required for local display.")
        print("Install with: pip install opencv-python")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, model_type = load_model(model_path, device)
    
    env = SnakeEnv(grid_size=12, render_mode="rgb_array")
    env = RewardShaping(env)
    env = FrameStack(env, n_frames=1)
    
    print(f"\n🎮 Playing Snake (Press ESC to quit, SPACE to reset)")
    print(f"Model type: {model_type.upper()}, FPS: {fps}, Deterministic: {deterministic}\n")
    
    episode = 0
    while n_episodes is None or episode < n_episodes:
        obs, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            with torch.no_grad():
                if model_type == 'ddp':
                    x = torch.from_numpy(obs).unsqueeze(0).float().to(device)
                    action = model.act(x).item()
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    action = int(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            frame = env.render()
            frame_with_info = frame.copy()
            text_lines = [
                f"Episode: {episode + 1}",
                f"Reward: {episode_reward:.1f}",
                f"Length: {episode_length}",
                f"Snake: {info.get('length', 0)}",
            ]
            
            y_offset = 30
            for line in text_lines:
                cv2.putText(
                    frame_with_info,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                y_offset += 25
            
            cv2.imshow("Snake Agent", cv2.cvtColor(frame_with_info, cv2.COLOR_RGB2BGR))
            
            key = cv2.waitKey(int(1000 / fps))
            if key == 27:
                print("\n👋 Exiting...")
                cv2.destroyAllWindows()
                return
            elif key == 32:
                print(f"⏭️  Skipping to next episode...")
                break
            
            if done:
                print(f"Episode {episode + 1}: Reward={episode_reward:.1f}, Length={episode_length}, Snake Size={info.get('length', 0)}")
                time.sleep(1.0)
        
        episode += 1
    
    cv2.destroyAllWindows()
    print(f"\n✅ Completed {episode} episodes")


def play_headless_video(
    model_path: str,
    output_path: str,
    fps: int = 10,
    n_episodes: int = 1,
    seed: int = 42,
):
    """
    Save episodes to video file (headless mode).
    """
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python is required for video recording.")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, model_type = load_model(model_path, device)
    
    env = SnakeEnv(grid_size=12, render_mode="rgb_array")
    env = RewardShaping(env)
    env = FrameStack(env, n_frames=1)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    
    print(f"\n🎬 Recording {n_episodes} episodes to {output_path}")
    
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        
        while not done:
            with torch.no_grad():
                if model_type == 'ddp':
                    x = torch.from_numpy(obs).unsqueeze(0).float().to(device)
                    action = model.act(x).item()
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            frame = env.render()
            
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    writer.release()
    print(f"✅ Video saved to: {output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Watch a trained Snake agent play.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (.pt or .zip) or folder with checkpoints",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Folder with checkpoint_*.pt files (uses latest)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (not needed for DDP checkpoints)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (default: True)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help="Number of episodes (None = infinite)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Save video to file (headless mode, deprecated: use --record)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record episodes to videos in --output directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="videos",
        help="Output directory for recorded videos (with --record)",
    )
    
    return parser.parse_args()


def main():
    """Main play function."""
    args = parse_args()
    
    if args.folder:
        model_path = latest_checkpoint(args.folder)
        print(f"Using latest checkpoint: {model_path}")
    elif args.model:
        model_path = args.model
    else:
        raise ValueError("Must specify either --model or --folder")
    
    deterministic = not args.stochastic
    
    if args.video or args.record:
        if args.record:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            n_eps = args.n_episodes or 5
            
            print(f"\n🎬 Recording {n_eps} episodes to {output_dir}")
            for ep in range(n_eps):
                output_path = output_dir / f"episode_{ep+1:03d}.mp4"
                play_headless_video(
                    model_path=model_path,
                    output_path=str(output_path),
                    fps=args.fps,
                    n_episodes=1,
                    seed=args.seed + ep,
                )
            print(f"✅ All videos saved to: {output_dir}")
        else:
            play_headless_video(
                model_path=model_path,
                output_path=args.video,
                fps=args.fps,
                n_episodes=args.n_episodes or 1,
                seed=args.seed,
            )
    else:
        play_local_window(
            model_path=model_path,
            config_path=args.config,
            fps=args.fps,
            deterministic=deterministic,
            n_episodes=args.n_episodes,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()