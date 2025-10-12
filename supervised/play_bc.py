"""
Interactive visualization of a trained Snake BC agent.
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from snake_env import SnakeEnv, RewardShaping, FrameStack
from supervised.models import make_bc_model
from supervised.utils import set_seed
import gymnasium as gym


def play_bc_local_window(
    model_path: str,
    config_path: str,
    fps: int = 10,
    n_episodes: int = None,
    seed: int = 42,
    device: str = 'cpu'
):
    """
    Play Snake with BC model in local OpenCV window.
    """
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python is required for local display.")
        print("Install with: pip install opencv-python")
        sys.exit(1)
    
    # Load BC checkpoint
    print(f"Loading BC model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    meta = checkpoint['meta']
    config_bc = checkpoint['config']
    
    # Load environment config
    with open(config_path, 'r') as f:
        env_config = yaml.safe_load(f)
    
    # Create model
    obs_shape = tuple(meta['obs_shape'])
    action_dim = meta['action_space']
    
    obs_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype='float32')
    action_space = gym.spaces.Discrete(action_dim)
    
    model = make_bc_model(obs_space, action_space, config_bc)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded (val_acc={checkpoint.get('val_acc', 0):.3f})")
    
    # Create environment
    env = SnakeEnv(
        grid_size=env_config['environment']['grid_size'],
        max_steps=env_config['environment'].get('max_steps'),
        render_mode="rgb_array",
    )
    
    env = RewardShaping(
        env,
        step_penalty=env_config['environment']['step_penalty'],
        death_penalty=env_config['environment']['death_penalty'],
        food_reward=env_config['environment']['food_reward'],
        distance_reward_scale=env_config['environment'].get('distance_reward_scale', 0.0),
    )
    
    frame_stack = env_config['environment'].get('frame_stack', 1)
    if frame_stack > 1:
        env = FrameStack(env, n_frames=frame_stack)
    
    print(f"\n🎮 Playing Snake BC Agent (Press ESC to quit, SPACE to reset)")
    print(f"FPS: {fps}\n")
    
    episode = 0
    while n_episodes is None or episode < n_episodes:
        obs, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Get action from BC model
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                logits = model(obs_tensor)
                action = logits.argmax(dim=1).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            # Render
            frame = env.render()
            
            # Add info overlay
            frame_with_info = frame.copy()
            text_lines = [
                f"BC Agent (Epoch {checkpoint.get('epoch', '?')})",
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
            
            # Display
            cv2.imshow("Snake BC Agent", cv2.cvtColor(frame_with_info, cv2.COLOR_RGB2BGR))
            
            # Handle keyboard
            key = cv2.waitKey(int(1000 / fps))
            if key == 27:  # ESC
                print("\n👋 Exiting...")
                cv2.destroyAllWindows()
                return
            elif key == 32:  # SPACE
                print(f"⏭️  Skipping to next episode...")
                break
            
            if done:
                print(f"Episode {episode + 1}: Reward={episode_reward:.1f}, Length={episode_length}, Snake Size={info.get('length', 0)}")
                time.sleep(1.0)
        
        episode += 1
    
    cv2.destroyAllWindows()
    print(f"\n✅ Completed {episode} episodes")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch a trained BC Snake agent play.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained BC model (.pt file, typically best.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to environment config file (train/configs/base.yaml)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second",
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
        "--device",
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help="Device to use",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    print(f"Using device: {device}\n")
    
    play_bc_local_window(
        model_path=args.model,
        config_path=args.config,
        fps=args.fps,
        n_episodes=args.n_episodes,
        seed=args.seed,
        device=device
    )


if __name__ == "__main__":
    main()
