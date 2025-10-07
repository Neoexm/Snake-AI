"""
Streamlit web dashboard for monitoring Snake RL training and watching live gameplay.

Provides real-time visualization of training metrics, live agent gameplay,
and system resource monitoring.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from snake_env import SnakeEnv, RewardShaping, FrameStack


# Page config
st.set_page_config(
    page_title="Snake RL Dashboard",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_tensorboard_logs(logdir: Path) -> Optional[pd.DataFrame]:
    """Load training logs from TensorBoard CSV files."""
    csv_files = list(logdir.glob("**/progress.csv"))
    
    if not csv_files:
        return None
    
    # Load most recent CSV
    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
    
    try:
        df = pd.read_csv(latest_csv)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None


def get_system_stats() -> Dict[str, Any]:
    """Get current system resource usage."""
    stats = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_available_gb": psutil.virtual_memory().available / (1024**3),
        "ram_total_gb": psutil.virtual_memory().total / (1024**3),
    }
    
    # Try to get GPU stats
    try:
        import torch
        if torch.cuda.is_available():
            stats["gpu_available"] = True
            stats["gpu_name"] = torch.cuda.get_device_name(0)
            stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
            stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
            stats["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            stats["gpu_available"] = False
    except:
        stats["gpu_available"] = False
    
    return stats


def plot_training_curves(df: pd.DataFrame):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Episode reward
    if "rollout/ep_rew_mean" in df.columns:
        axes[0, 0].plot(df["time/total_timesteps"], df["rollout/ep_rew_mean"], linewidth=2)
        axes[0, 0].set_title("Episode Reward Mean")
        axes[0, 0].set_xlabel("Timesteps")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True, alpha=0.3)
    
    # Episode length
    if "rollout/ep_len_mean" in df.columns:
        axes[0, 1].plot(df["time/total_timesteps"], df["rollout/ep_len_mean"], linewidth=2, color='orange')
        axes[0, 1].set_title("Episode Length Mean")
        axes[0, 1].set_xlabel("Timesteps")
        axes[0, 1].set_ylabel("Length")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    if "train/learning_rate" in df.columns:
        axes[1, 0].plot(df["time/total_timesteps"], df["train/learning_rate"], linewidth=2, color='green')
        axes[1, 0].set_title("Learning Rate")
        axes[1, 0].set_xlabel("Timesteps")
        axes[1, 0].set_ylabel("LR")
        axes[1, 0].grid(True, alpha=0.3)
    
    # FPS
    if "time/fps" in df.columns:
        axes[1, 1].plot(df["time/total_timesteps"], df["time/fps"], linewidth=2, color='red')
        axes[1, 1].set_title("Training Throughput")
        axes[1, 1].set_xlabel("Timesteps")
        axes[1, 1].set_ylabel("FPS")
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main dashboard function."""
    st.title("🐍 Snake RL Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Select run directory
    runs_dir = Path("runs")
    if runs_dir.exists():
        run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
        run_names = [d.name for d in run_dirs]
        
        if run_names:
            selected_run = st.sidebar.selectbox(
                "Select Run",
                run_names,
                index=0,
            )
            run_dir = runs_dir / selected_run
        else:
            st.warning("No runs found in 'runs/' directory. Start training first!")
            return
    else:
        st.warning("'runs/' directory not found. Start training first!")
        return
    
    # Tab selection
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Training Metrics", "🎮 Live Agent", "💻 System Resources", "ℹ️ Info"])
    
    # Tab 1: Training Metrics
    with tab1:
        st.header("Training Progress")
        
        # Load logs
        df = load_tensorboard_logs(run_dir)
        
        if df is not None:
            # Show summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if "time/total_timesteps" in df.columns:
                    total_steps = df["time/total_timesteps"].iloc[-1]
                    st.metric("Total Timesteps", f"{int(total_steps):,}")
            
            with col2:
                if "rollout/ep_rew_mean" in df.columns:
                    latest_reward = df["rollout/ep_rew_mean"].iloc[-1]
                    st.metric("Latest Reward", f"{latest_reward:.2f}")
            
            with col3:
                if "rollout/ep_len_mean" in df.columns:
                    latest_length = df["rollout/ep_len_mean"].iloc[-1]
                    st.metric("Latest Episode Length", f"{latest_length:.0f}")
            
            with col4:
                if "time/fps" in df.columns:
                    latest_fps = df["time/fps"].iloc[-1]
                    st.metric("FPS", f"{latest_fps:.0f}")
            
            st.markdown("---")
            
            # Plot training curves
            fig = plot_training_curves(df)
            st.pyplot(fig)
            
            # Show raw data
            with st.expander("📋 Show Raw Data"):
                st.dataframe(df.tail(50))
        else:
            st.info("No training logs found. The model may still be initializing.")
    
    # Tab 2: Live Agent
    with tab2:
        st.header("Watch Agent Play")
        
        # Check for best model
        best_model_path = run_dir / "best_model.zip"
        
        if not best_model_path.exists():
            st.warning("No trained model found yet. Keep training!")
        else:
            from stable_baselines3 import PPO
            import yaml
            
            # Load config
            config_path = run_dir / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                st.error("Config file not found!")
                return
            
            # Load model
            @st.cache_resource
            def load_model(model_path):
                return PPO.load(str(model_path))
            
            model = load_model(best_model_path)
            
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
            
            # Controls
            col1, col2 = st.columns(2)
            with col1:
                fps = st.slider("FPS", min_value=1, max_value=30, value=10)
            with col2:
                if st.button("▶️ Start New Episode"):
                    st.session_state.env_reset = True
            
            # Initialize session state
            if 'obs' not in st.session_state or st.session_state.get('env_reset', False):
                st.session_state.obs, _ = env.reset()
                st.session_state.episode_reward = 0
                st.session_state.episode_length = 0
                st.session_state.done = False
                st.session_state.env_reset = False
            
            # Display area
            frame_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            # Auto-play
            if not st.session_state.done:
                # Take step
                action, _ = model.predict(st.session_state.obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                
                st.session_state.obs = obs
                st.session_state.episode_reward += reward
                st.session_state.episode_length += 1
                st.session_state.done = terminated or truncated
                
                # Render
                frame = env.render()
                frame_placeholder.image(frame, use_container_width=True, caption="Current State")
                
                # Show metrics
                metrics_placeholder.metric(
                    f"Episode: Reward={st.session_state.episode_reward:.1f}, "
                    f"Length={st.session_state.episode_length}, "
                    f"Snake Size={info.get('length', 0)}",
                    ""
                )
                
                # Auto-refresh
                time.sleep(1.0 / fps)
                st.rerun()
            else:
                st.success(f"Episode Complete! Reward: {st.session_state.episode_reward:.1f}, Length: {st.session_state.episode_length}")
                if st.button("🔄 Reset and Play Again"):
                    st.session_state.env_reset = True
                    st.rerun()
    
    # Tab 3: System Resources
    with tab3:
        st.header("System Monitoring")
        
        stats = get_system_stats()
        
        # CPU and RAM
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("CPU")
            st.metric("Usage", f"{stats['cpu_percent']:.1f}%")
            st.progress(stats['cpu_percent'] / 100)
        
        with col2:
            st.subheader("RAM")
            st.metric(
                "Usage",
                f"{stats['ram_percent']:.1f}%",
                f"{stats['ram_available_gb']:.1f} GB available / {stats['ram_total_gb']:.1f} GB total"
            )
            st.progress(stats['ram_percent'] / 100)
        
        # GPU
        if stats['gpu_available']:
            st.markdown("---")
            st.subheader("GPU")
            st.write(f"**Device:** {stats['gpu_name']}")
            
            gpu_usage_percent = (stats['gpu_memory_allocated_gb'] / stats['gpu_memory_total_gb']) * 100
            st.metric(
                "Memory Usage",
                f"{gpu_usage_percent:.1f}%",
                f"{stats['gpu_memory_allocated_gb']:.2f} GB / {stats['gpu_memory_total_gb']:.2f} GB"
            )
            st.progress(gpu_usage_percent / 100)
        else:
            st.info("No GPU detected")
        
        # Auto-refresh button
        if st.button("🔄 Refresh Stats"):
            st.rerun()
    
    # Tab 4: Info
    with tab4:
        st.header("Run Information")
        
        # Config
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            st.subheader("Configuration")
            st.json(config)
        
        # TensorBoard link
        st.markdown("---")
        st.subheader("External Tools")
        st.markdown(f"""
        **TensorBoard:** Launch with:
        ```bash
        tensorboard --logdir {run_dir}
        ```
        Then open: http://localhost:6006
        """)
        
        # Directory info
        st.markdown("---")
        st.subheader("Directory Info")
        st.write(f"**Run Directory:** `{run_dir}`")
        
        if (run_dir / "best_model.zip").exists():
            st.success("✅ Best model available")
        else:
            st.warning("⏳ Best model not yet saved")
        
        if (run_dir / "checkpoints").exists():
            checkpoints = list((run_dir / "checkpoints").glob("*.zip"))
            st.info(f"💾 {len(checkpoints)} checkpoint(s) saved")


if __name__ == "__main__":
    main()