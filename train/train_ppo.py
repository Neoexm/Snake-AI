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
import torch.distributed as dist
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
from train.models import SnakeTinyCNN, SnakeScalableCNN, SnakeDeepCNN
from train.ddp_utils import (
    setup_distributed,
    cleanup_distributed,
    is_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    broadcast_model_parameters,
    all_reduce_dict,
    barrier,
    setup_nccl_env_for_b200,
    print_distributed_info,
)
from tools.hw_monitor import HardwareMonitor


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
            from train.ddp_utils import is_distributed, is_main_process, get_rank, all_reduce_dict
            
            elapsed = time.time() - self.start_time
            local_fps = self.num_timesteps / elapsed if elapsed > 0 else 0
            
            # Aggregate FPS across all ranks
            if is_distributed():
                fps_data = all_reduce_dict({'fps': local_fps})
                global_fps = fps_data['fps']
            else:
                global_fps = local_fps
            
            # Only rank 0 logs to TensorBoard
            if is_main_process():
                self.logger.record("time/global_fps", global_fps)
                self.logger.record("time/local_fps_rank0", local_fps)
            
            # Try to get GPU utilization
            try:
                import torch
                if torch.cuda.is_available():
                    rank = get_rank() if is_distributed() else 0
                    gpu_mem_alloc = torch.cuda.memory_allocated(rank) / (1024**3)
                    gpu_mem_reserved = torch.cuda.memory_reserved(rank) / (1024**3)
                    
                    if is_main_process():
                        self.logger.record(f"system/gpu{rank}_memory_allocated_gb", gpu_mem_alloc)
                        self.logger.record(f"system/gpu{rank}_memory_reserved_gb", gpu_mem_reserved)
                        
                        # Try to get utilization for all GPUs (rank 0 only)
                        try:
                            import nvidia_ml_py3 as nvml
                            nvml.nvmlInit()
                            for i in range(torch.cuda.device_count()):
                                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                                self.logger.record(f"system/gpu{i}_util_percent", util.gpu)
                            nvml.nvmlShutdown()
                        except:
                            pass
            except:
                pass
            
            # CPU and RAM (rank 0 only)
            if is_main_process():
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
    # Set deterministic seeds for this worker
    import random
    worker_seed = seed + rank
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)
    
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
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Suffix to append to run name",
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
    parser.add_argument(
        "--auto-scale",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Enable auto-scaling (default: true)",
    )
    parser.add_argument(
        "--max-n-envs",
        type=int,
        default=None,
        help="Hard cap on number of environments",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Hard cap on batch size",
    )
    
    # Policy/Network configuration
    parser.add_argument(
        "--policy-width",
        type=int,
        default=None,
        help="CNN width (number of channels)",
    )
    parser.add_argument(
        "--policy-depth",
        type=int,
        default=None,
        help="CNN depth (number of conv layers)",
    )
    parser.add_argument(
        "--policy-arch",
        type=str,
        choices=["tiny", "scalable", "deep"],
        default="scalable",
        help="Policy architecture to use",
    )
    
    # Performance optimizations
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "amp"],
        default=None,
        help="Precision mode (fp32 or amp for mixed precision)",
    )
    parser.add_argument(
        "--compile",
        type=str,
        choices=["none", "default"],
        default=None,
        help="Torch compile mode",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Pin memory for faster CPU->GPU transfer",
    )
    parser.add_argument(
        "--env-processes",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Use SubprocVecEnv (true) or DummyVecEnv (false)",
    )
    
    # Image preprocessing
    parser.add_argument(
        "--normalize-images",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Normalize images (our images are already normalized)",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=None,
        help="Number of frames to stack (overrides config)",
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
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (.zip file)",
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Setup NCCL environment variables for B200 GPUs BEFORE initializing distributed
    setup_nccl_env_for_b200()
    
    # Initialize distributed training if running under torchrun
    rank = 0
    world_size = 1
    device = None
    
    if os.environ.get('RANK') is not None:
        # Running under torchrun - initialize distributed
        try:
            rank, world_size, device = setup_distributed(backend='nccl')
            print_distributed_info()
        except Exception as e:
            print(f"Warning: Could not initialize distributed training: {e}")
            print("Falling back to single-GPU mode")
    
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.total_timesteps is not None:
        config['training']['total_timesteps'] = args.total_timesteps
    if args.frame_stack is not None:
        config['environment']['frame_stack'] = args.frame_stack
    
    # Autoscale resources
    auto_scale_enabled = args.auto_scale == "true"
    prefer_device = args.device if args.device != "auto" else None
    
    if auto_scale_enabled:
        resource_config = autoscale(
            max_utilization=args.max_utilization,
            prefer_device=prefer_device,
            override_n_envs=args.n_envs,
            override_policy_width=args.policy_width,
            override_policy_depth=args.policy_depth,
            override_precision=args.precision,
            override_compile=args.compile,
            max_n_envs=args.max_n_envs,
            max_batch_size=args.max_batch_size,
        )
        if is_main_process():
            resource_config.print_summary()
    else:
        # Manual configuration when auto-scale is disabled
        from train.autoscale import detect_cuda, detect_cpu, ResourceConfig
        resource_config = ResourceConfig(
            device=args.device if args.device != "auto" else "cpu",
            n_envs=args.n_envs or 4,
            n_steps=256,
            batch_size=512,
            num_workers=0,
            use_amp=args.precision == "amp" if args.precision else False,
            pin_memory=args.pin_memory,
            num_threads=None,
            gpu_info=detect_cuda(),
            cpu_info=detect_cpu(),
            policy_width=args.policy_width or 32,
            policy_depth=args.policy_depth or 2,
            precision=args.precision or "fp32",
            compile_mode=args.compile or "none",
            env_processes=args.env_processes != "false" if args.env_processes else True,
        )
    
    # Set seeds with rank-dependent offset for diversity
    base_seed = args.seed
    rank_seed = base_seed + rank
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
        # For distributed training, use non-deterministic for better performance
        if is_distributed():
            torch.use_deterministic_algorithms(mode=False)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            # Single GPU can be more deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # Create run directory (only rank 0)
    if is_main_process():
        base_run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.run_suffix:
            run_name = f"{base_run_name}_{args.run_suffix}"
        else:
            run_name = base_run_name
        run_dir = Path(args.logdir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Non-zero ranks: wait for rank 0 to create directory, then use it
        import time
        time.sleep(2)  # Small delay to ensure rank 0 creates directory
        base_run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.run_suffix:
            run_name = f"{base_run_name}_{args.run_suffix}"
        else:
            run_name = base_run_name
        run_dir = Path(args.logdir) / run_name
    
    # Save config and autoscale info (rank 0 only)
    if is_main_process():
        config['autoscale'] = {
            'enabled': auto_scale_enabled,
            'device': resource_config.device,
            'n_envs': resource_config.n_envs,
            'n_steps': resource_config.n_steps,
            'batch_size': resource_config.batch_size,
            'use_amp': resource_config.use_amp,
            'policy_width': resource_config.policy_width,
            'policy_depth': resource_config.policy_depth,
            'precision': resource_config.precision,
            'compile_mode': resource_config.compile_mode,
        }
        config['distributed'] = {
            'enabled': is_distributed(),
            'world_size': world_size,
            'backend': 'nccl' if is_distributed() else None,
        }
        config['seed'] = base_seed
        config['cli_args'] = vars(args)
        save_config(config, str(run_dir / "config.yaml"))
        
        # Save system info using hardware monitor
        try:
            monitor = HardwareMonitor()
            monitor.save_system_info(run_dir / "system.json")
        except Exception as e:
            print(f"Warning: Could not save system info: {e}")
        
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
    
    # Choose vec env class
    if args.env_processes == "true":
        vec_env_cls = SubprocVecEnv
    elif args.env_processes == "false":
        vec_env_cls = DummyVecEnv
    else:
        # Auto: use subprocess on Linux, dummy on Windows
        vec_env_cls = SubprocVecEnv if resource_config.env_processes else DummyVecEnv
    
    # Create list of environment factories with PROPER closure capture
    # Critical: each rank creates n_envs with unique global IDs for seeding diversity
    def _make_env(local_idx: int, rank_id: int, n_per_rank: int, base_seed: int, kw: dict):
        global_env_id = rank_id * n_per_rank + local_idx
        return make_snake_env(**{**kw, 'rank': global_env_id, 'seed': base_seed})
    
    # FIXED: Use functools.partial to properly capture loop variable
    from functools import partial
    env_fns = [
        partial(_make_env, i, rank, resource_config.n_envs, args.seed, env_kwargs)
        for i in range(resource_config.n_envs)
    ]
    
    if is_main_process():
        global_start = rank * resource_config.n_envs
        global_end = (rank + 1) * resource_config.n_envs - 1
        print(f"[Rank {rank}] Creating {resource_config.n_envs} environments (global IDs {global_start}–{global_end})")
    
    env = vec_env_cls(env_fns)
    env = VecMonitor(env)
    
    # CRITICAL: Validate observation space matches expected CNN input
    expected_channels = 3 * config['environment'].get('frame_stack', 1)
    actual_channels = env.observation_space.shape[0]
    if actual_channels != expected_channels:
        raise ValueError(
            f"Observation space channel mismatch! "
            f"Expected {expected_channels} channels (3 × frame_stack={config['environment'].get('frame_stack', 1)}), "
            f"but got {actual_channels}. Check FrameStack wrapper or CNN input layer."
        )
    
    if is_main_process():
        print(f"✓ Observation space validated: {env.observation_space.shape}")
    
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
    
    # CRITICAL: Validate batch_size divides rollout evenly
    effective_envs = resource_config.n_envs * world_size if is_distributed() else resource_config.n_envs
    rollout_size = effective_envs * resource_config.n_steps
    if rollout_size % ppo_kwargs['batch_size'] != 0:
        if is_main_process():
            print(f"⚠️  WARNING: batch_size {ppo_kwargs['batch_size']} does not divide "
                  f"rollout size {rollout_size} evenly!")
            print(f"   This may cause SB3 to truncate data. Adjusting batch_size...")
        
        # Auto-adjust to largest divisor <= batch_size
        for candidate in range(ppo_kwargs['batch_size'], 0, -1):
            if rollout_size % candidate == 0:
                ppo_kwargs['batch_size'] = candidate
                if is_main_process():
                    print(f"   Adjusted batch_size to {candidate}")
                break
    
    # Extract policy_kwargs from ppo config
    policy_kwargs = ppo_kwargs.pop('policy_kwargs', {})
    
    # Select policy architecture
    if args.policy_arch == "tiny":
        extractor_class = SnakeTinyCNN
        extractor_kwargs = {'features_dim': 256}
    elif args.policy_arch == "deep":
        extractor_class = SnakeDeepCNN
        extractor_kwargs = {
            'features_dim': 256,
            'width': resource_config.policy_width,
        }
    else:  # scalable (default)
        extractor_class = SnakeScalableCNN
        extractor_kwargs = {
            'features_dim': 256,
            'width': resource_config.policy_width,
            'depth': resource_config.policy_depth,
        }
    
    policy_kwargs['features_extractor_class'] = extractor_class
    policy_kwargs['features_extractor_kwargs'] = extractor_kwargs
    
    # Normalize images setting
    normalize_images = args.normalize_images == "true" if args.normalize_images else False
    policy_kwargs['normalize_images'] = normalize_images
    
    # Get policy type from config
    policy = config.get('policy', 'CnnPolicy')
    
    # Create PPO model
    if is_main_process():
        print("🤖 Creating PPO model...")
        print(f"Policy: {policy}")
    
    if args.resume_from:
        if is_main_process():
            print(f"📂 Resuming from checkpoint: {args.resume_from}")
        
        # Load on CPU first to avoid GPU memory issues
        model = PPO.load(
            args.resume_from,
            env=env,
            device='cpu',
            tensorboard_log=str(run_dir),
        )
        
        # CRITICAL: Broadcast parameters from rank 0 BEFORE moving to GPU
        if is_distributed():
            broadcast_model_parameters(model.policy, src=0)
            barrier()
            if is_main_process():
                print(f"✓ [Rank {rank}] Checkpoint loaded, weights synchronized")
        
        # Move to correct device AFTER sync
        model.policy.to(resource_config.device)
        if hasattr(model, 'policy_old'):
            model.policy_old.to(resource_config.device)
        
        if is_main_process():
            print(f"✓ Loaded checkpoint at {model.num_timesteps:,} timesteps")
    else:
        if is_main_process():
            print(f"Architecture: {args.policy_arch}")
            print(f"Features Extractor: {policy_kwargs['features_extractor_class'].__name__}")
            print(f"Extractor Config: {policy_kwargs['features_extractor_kwargs']}")
            print(f"normalize_images: {policy_kwargs['normalize_images']}")
            print(f"Precision: {resource_config.precision}")
            print(f"Compile: {resource_config.compile_mode}")
        
        model = PPO(
            policy=policy,
            env=env,
            policy_kwargs=policy_kwargs,
            **ppo_kwargs,
            tensorboard_log=str(run_dir),
            verbose=1 if is_main_process() else 0,
        )
        
        # NOTE: We do NOT wrap policy in DDP because Stable Baselines3
        # expects direct access to policy methods that DDP doesn't expose.
        # SB3 already handles parallelization via vectorized environments.
    
    # Apply torch compile if requested
    if resource_config.compile_mode == "default":
        try:
            if hasattr(torch, 'compile'):
                if is_main_process():
                    print("Compiling model with torch.compile...")
                model.policy = torch.compile(model.policy)
        except Exception as e:
            if is_main_process():
                print(f"Warning: Could not compile model: {e}")
    
    # Setup custom logger for CSV output
    # Only rank 0 should log to TensorBoard to avoid conflicts
    if is_main_process():
        logger = configure(str(run_dir), ["stdout", "csv", "tensorboard"])
    else:
        # Non-zero ranks: only stdout, no TensorBoard
        logger = configure(str(run_dir), ["stdout"])
    model.set_logger(logger)
    
    # Create callbacks (only rank 0 saves checkpoints)
    callbacks = []
    
    # Checkpoint callback (rank 0 only)
    if is_main_process():
        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.save_freq // resource_config.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="model",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback (rank 0 only)
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
    
    # Start hardware monitoring (rank 0 only)
    hw_monitor = None
    if is_main_process():
        try:
            hw_monitor = HardwareMonitor(
                poll_interval=1.0,
                log_dir=run_dir,
                tensorboard_logger=model.logger,
            )
            hw_monitor.start()
            print("✓ Hardware monitoring started")
        except Exception as e:
            print(f"Warning: Could not start hardware monitoring: {e}")
    
    # Train (adjust timesteps for DDP: each rank trains for total_timesteps)
    # In true DDP, all ranks train the SAME number of steps with synchronized gradients
    # The effective batch is world_size × n_envs × n_steps per update
    if is_main_process():
        total_envs = resource_config.n_envs * world_size if is_distributed() else resource_config.n_envs
        print(f"\n🚀 Starting training for {config['training']['total_timesteps']:,} timesteps per rank...")
        print(f"Total environments across all ranks: {total_envs}")
        print(f"Environments per rank: {resource_config.n_envs}")
        print(f"Effective batch size per update: {total_envs * resource_config.n_steps}\n")
    
    training_timesteps = config['training']['total_timesteps']
    if args.resume_from:
        # FIXED: In DDP, all ranks must train for same number of steps
        # Subtract existing timesteps consistently across all ranks
        remaining = training_timesteps - model.num_timesteps
        if is_distributed():
            import torch.distributed as dist
            remaining_tensor = torch.tensor([remaining], device=resource_config.device)
            dist.broadcast(remaining_tensor, src=0)
            training_timesteps = remaining_tensor.item()
        else:
            training_timesteps = remaining
        
        if is_main_process() and training_timesteps <= 0:
            print(f"⚠️  Already trained for {model.num_timesteps:,} timesteps, skipping training")
            training_timesteps = 0
    
    try:
        if training_timesteps > 0:
            model.learn(
                total_timesteps=training_timesteps,
                callback=callback_list,
                log_interval=10,
                progress_bar=True if is_main_process() else False,
            )
        elif is_main_process():
            print("⚠️  No training needed (already at target timesteps)")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    finally:
        # Stop hardware monitoring
        if hw_monitor:
            hw_monitor.stop()
            
            # Print average stats
            avg_stats = hw_monitor.get_average_stats(last_n_seconds=120)
            if avg_stats:
                print("\n" + "="*60)
                print("AVERAGE HARDWARE UTILIZATION (last 2 minutes)")
                print("="*60)
                print(f"CPU: {avg_stats.cpu_percent:.1f}%")
                print(f"RAM: {avg_stats.ram_percent:.1f}%")
                for gpu in avg_stats.gpus:
                    print(f"GPU {gpu.device_id}: {gpu.utilization_percent:.1f}% util, "
                          f"{gpu.memory_percent:.1f}% mem")
                print("="*60)
    
    # Save final model (rank 0 only, unwrap DDP first, atomic write)
    if is_main_process():
        final_path = run_dir / "final_model.zip"
        temp_path = run_dir / f"final_model.tmp.{os.getpid()}.zip"
        
        # Unwrap DDP before saving
        # Atomic save: write to temp with PID, then rename
        try:
            model.save(str(temp_path))
            if temp_path.exists():
                if final_path.exists():
                    final_path.unlink()
                temp_path.rename(final_path)
                print(f"✓ Final model saved atomically to {final_path}")
        except Exception as e:
            print(f"Warning: Failed to save final model: {e}")
            if temp_path.exists():
                temp_path.unlink()
        
        # Save effective config (after auto-tuning)
        effective_config = config.copy()
        effective_config['effective_autoscale'] = {
            'device': resource_config.device,
            'n_envs': resource_config.n_envs,
            'n_steps': resource_config.n_steps,
            'batch_size': resource_config.batch_size,
            'policy_width': resource_config.policy_width,
            'policy_depth': resource_config.policy_depth,
            'precision': resource_config.precision,
        }
        save_config(effective_config, str(run_dir / "effective_config.yaml"))
        
        print(f"\n✅ Training complete!")
        print(f"📦 Final model saved to: {final_path}")
        print(f"🏆 Best model saved to: {run_dir / 'best_model.zip'}")
        print(f"📝 Effective config saved to: {run_dir / 'effective_config.yaml'}")
        print(f"\n📊 View training progress:")
        print(f"   tensorboard --logdir {run_dir}")
    
    # Cleanup
    env.close()
    if eval_env is not None:
        eval_env.close()
    
    # Cleanup distributed
    if is_distributed():
        cleanup_distributed()


if __name__ == "__main__":
    main()