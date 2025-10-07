"""
Interactive visualization of a trained Snake PPO agent.

Supports both local window display (OpenCV/Pygame) and headless mode for saving videos.
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from snake_env import SnakeEnv, RewardShaping, FrameStack
from stable_baselines3 import PPO


def play_local_window(
    model_path: str,
    config_path: str,
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
        Path to the saved model (.zip file).
    config_path : str
        Path to the config file.
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
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    env = SnakeEnv(
        grid_size=config['environment']['grid_size'],
        max_steps=config['environment'].get('max_steps'),
        render_mode="rgb_array",
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
    
    print(f"\n🎮 Playing Snake (Press ESC to quit, SPACE to reset)")
    print(f"FPS: {fps}, Deterministic: {deterministic}\n")
    
    episode = 0
    while n_episodes is None or episode < n_episodes:
        obs, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            # Render
            frame = env.render()
            
            # Add info overlay
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
            
            # Display
            cv2.imshow("Snake Agent", cv2.cvtColor(frame_with_info, cv2.COLOR_RGB2BGR))
            
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
                time.sleep(1.0)  # Pause briefly to show final state
        
        episode += 1
    
    cv2.destroyAllWindows()
    print(f"\n✅ Completed {episode} episodes")


def play_headless_video(
    model_path: str,
    config_path: str,
    output_path: str,
    fps: int = 10,
    n_episodes: int = 1,
    seed: int = 42,
):
    """
    Save episodes to video file (headless mode).
    
    Parameters
    ----------
    model_path : str
        Path to the saved model (.zip file).
    config_path : str
        Path to the config file.
    output_path : str
        Output video file path (.mp4, .avi, etc.).
    fps : int
        Frames per second.
    n_episodes : int
        Number of episodes to record.
    seed : int
        Random seed.
    """
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python is required for video recording.")
        sys.exit(1)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    env = SnakeEnv(
        grid_size=config['environment']['grid_size'],
        max_steps=config['environment'].get('max_steps'),
        render_mode="rgb_array",
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
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    
    print(f"\n🎬 Recording {n_episodes} episodes to {output_path}")
    
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            
            frame = env.render()
            
            # Initialize writer on first frame
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            # Write frame (convert RGB to BGR)
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
        help="Save video to file (headless mode)",
    )
    
    return parser.parse_args()


def main():
    """Main play function."""
    args = parse_args()
    
    # Auto-detect config if not provided
    config_path = args.config
    if config_path is None:
        model_dir = Path(args.model).parent
        config_path = model_dir / "config.yaml"
        if not config_path.exists():
            config_path = model_dir.parent / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Could not find config.yaml. Please specify --config explicitly."
            )
    
    deterministic = not args.stochastic
    
    if args.video:
        # Headless video mode
        play_headless_video(
            model_path=args.model,
            config_path=str(config_path),
            output_path=args.video,
            fps=args.fps,
            n_episodes=args.n_episodes or 1,
            seed=args.seed,
        )
    else:
        # Local window mode
        play_local_window(
            model_path=args.model,
            config_path=str(config_path),
            fps=args.fps,
            deterministic=deterministic,
            n_episodes=args.n_episodes,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()