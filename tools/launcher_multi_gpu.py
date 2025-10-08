"""
Multi-GPU launcher for Snake RL training.

Spawns one independent training process per GPU with CUDA_VISIBLE_DEVICES isolation.
Each process gets unique seed, run suffix, and log directory.
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Try to import torch for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available. Cannot detect GPUs automatically.")

# Try to import pynvml for detailed GPU info
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. Install with: pip install nvidia-ml-py3")


def detect_gpus() -> List[int]:
    """
    Detect available GPUs.
    
    Returns
    -------
    List[int]
        List of GPU device IDs
    """
    if not TORCH_AVAILABLE:
        print("ERROR: torch not available. Cannot detect GPUs.")
        return []
    
    if not torch.cuda.is_available():
        print("No CUDA GPUs detected.")
        return []
    
    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} CUDA GPU(s)")
    
    # Get GPU names if NVML available
    if NVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_gb = mem_info.total / (1024**3)
                print(f"  GPU {i}: {name} ({mem_gb:.1f} GB)")
            pynvml.nvmlShutdown()
        except Exception as e:
            print(f"Warning: Could not get GPU details: {e}")
    
    return list(range(gpu_count))


def save_system_info(run_dir: Path, gpu_ids: List[int]):
    """
    Save system information to run directory.
    
    Parameters
    ----------
    run_dir : Path
        Run directory path
    gpu_ids : List[int]
        List of GPU IDs being used
    """
    import psutil
    
    info = {
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "num_gpus": len(gpu_ids),
        "gpu_ids": gpu_ids,
        "cpu": {
            "count": psutil.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
        },
        "ram": {
            "total_gb": psutil.virtual_memory().total / (1024**3),
        },
        "gpus": [],
    }
    
    if NVML_AVAILABLE and gpu_ids:
        try:
            pynvml.nvmlInit()
            for gpu_id in gpu_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                name = pynvml.nvmlDeviceGetName(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                try:
                    uuid = pynvml.nvmlDeviceGetUUID(handle)
                except:
                    uuid = f"gpu-{gpu_id}"
                
                info["gpus"].append({
                    "id": gpu_id,
                    "name": name,
                    "memory_gb": mem_info.total / (1024**3),
                    "uuid": uuid,
                })
            pynvml.nvmlShutdown()
        except Exception as e:
            print(f"Warning: Could not get detailed GPU info: {e}")
    
    system_file = run_dir / "system.json"
    with open(system_file, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"System info saved to: {system_file}")


def launch_training_process(
    gpu_id: int,
    base_seed: int,
    run_name: str,
    logdir: Path,
    config: str,
    total_timesteps: int,
    auto_scale: bool,
    extra_args: List[str],
) -> subprocess.Popen:
    """
    Launch a single training process on a specific GPU.
    
    Parameters
    ----------
    gpu_id : int
        GPU device ID
    base_seed : int
        Base random seed
    run_name : str
        Base run name
    logdir : Path
        Base log directory
    config : str
        Path to config file
    total_timesteps : int
        Total timesteps to train
    auto_scale : bool
        Enable auto-scaling
    extra_args : List[str]
        Additional arguments to pass to training script
    
    Returns
    -------
    subprocess.Popen
        Running process
    """
    # Unique seed for this GPU
    seed = base_seed + gpu_id
    
    # Unique run name (includes GPU ID and seed to prevent log collisions)
    gpu_run_name = f"{run_name}_gpu{gpu_id}_seed{seed}"
    
    # Log file for this GPU
    log_file = logdir / f"run_gpu{gpu_id}.log"
    
    # Build command
    cmd = [
        sys.executable,
        "train/train_ppo.py",
        "--config", config,
        "--device", "cuda",
        "--seed", str(seed),
        "--run-name", gpu_run_name,
        "--logdir", str(logdir),
        "--total-timesteps", str(total_timesteps),
    ]
    
    if auto_scale:
        cmd.append("--auto-scale")
        cmd.append("true")
    
    # Add extra args
    cmd.extend(extra_args)
    
    # Environment variables - isolate to this GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Open log file
    log_f = open(log_file, "w")
    
    print(f"\n{'='*60}")
    print(f"Launching training on GPU {gpu_id}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Seed: {seed}")
    print(f"Run name: {gpu_run_name}")
    print(f"Log file: {log_file}")
    print(f"CUDA_VISIBLE_DEVICES: {gpu_id}")
    
    # Launch process
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=Path.cwd(),
    )
    
    return proc


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch multi-GPU training for Snake RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # GPU selection
    parser.add_argument(
        "--gpus",
        type=str,
        default="all",
        help="Comma-separated GPU IDs (e.g., '0,1,2') or 'all'",
    )
    
    # Training parameters
    parser.add_argument(
        "--config",
        type=str,
        default="train/configs/base.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500000,
        help="Total timesteps per GPU",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (each GPU gets seed + gpu_id)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Base run name (default: timestamp)",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="runs",
        help="Base directory for logs",
    )
    
    # Auto-scaling
    parser.add_argument(
        "--auto-scale",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Enable auto-scaling",
    )
    
    # Monitoring
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for all processes to complete",
    )
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=60,
        help="Seconds between status checks (if --wait)",
    )
    
    return parser.parse_known_args()


def main():
    """Main launcher function."""
    args, extra_args = parse_args()
    
    # Detect GPUs
    available_gpus = detect_gpus()
    if not available_gpus:
        print("ERROR: No GPUs detected. Cannot launch multi-GPU training.")
        sys.exit(1)
    
    # Select GPUs to use
    if args.gpus == "all":
        gpu_ids = available_gpus
    else:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
        # Validate GPU IDs
        for gpu_id in gpu_ids:
            if gpu_id not in available_gpus:
                print(f"ERROR: GPU {gpu_id} not available. Available: {available_gpus}")
                sys.exit(1)
    
    print(f"\nUsing GPUs: {gpu_ids}")
    
    # Create base run directory
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    
    # Save system info
    save_system_info(logdir, gpu_ids)
    
    # Launch processes
    processes = []
    auto_scale = args.auto_scale == "true"
    
    print(f"\n{'='*60}")
    print(f"LAUNCHING {len(gpu_ids)} TRAINING PROCESSES")
    print(f"{'='*60}")
    print(f"Base seed: {args.seed}")
    print(f"Total timesteps per GPU: {args.total_timesteps:,}")
    print(f"Auto-scale: {auto_scale}")
    print(f"Config: {args.config}")
    
    for gpu_id in gpu_ids:
        proc = launch_training_process(
            gpu_id=gpu_id,
            base_seed=args.seed,
            run_name=run_name,
            logdir=logdir,
            config=args.config,
            total_timesteps=args.total_timesteps,
            auto_scale=auto_scale,
            extra_args=extra_args,
        )
        processes.append((gpu_id, proc))
        time.sleep(2)  # Stagger launches slightly
    
    print(f"\n{'='*60}")
    print(f"All {len(processes)} processes launched!")
    print(f"{'='*60}")
    print(f"\nLog files:")
    for gpu_id, _ in processes:
        print(f"  GPU {gpu_id}: {logdir / f'run_gpu{gpu_id}.log'}")
    
    if args.wait:
        print(f"\nWaiting for processes to complete...")
        print(f"(Checking every {args.monitor_interval}s, press Ctrl+C to stop waiting)\n")
        
        try:
            while True:
                # Check status
                running = []
                completed = []
                for gpu_id, proc in processes:
                    if proc.poll() is None:
                        running.append(gpu_id)
                    else:
                        completed.append((gpu_id, proc.returncode))
                
                # Print status
                print(f"[{datetime.now().strftime('%H:%M:%S')}] " +
                      f"Running: {len(running)}/{len(processes)} " +
                      f"(GPUs: {running})")
                
                # Check if all done
                if not running:
                    print("\nAll processes completed!")
                    break
                
                # Wait before next check
                time.sleep(args.monitor_interval)
        
        except KeyboardInterrupt:
            print("\n\nStopping wait (processes continue in background)...")
        
        # Final status
        print("\n" + "="*60)
        print("FINAL STATUS")
        print("="*60)
        for gpu_id, proc in processes:
            status = proc.poll()
            if status is None:
                print(f"GPU {gpu_id}: Still running (PID {proc.pid})")
            elif status == 0:
                print(f"GPU {gpu_id}: Completed successfully")
            else:
                print(f"GPU {gpu_id}: Failed (exit code {status})")
    
    else:
        print("\nProcesses running in background.")
        print("To monitor, check log files or use:")
        print(f"  watch -n 5 'tail -n 20 {logdir}/run_gpu*.log'")
        print("\nTo kill all processes:")
        for gpu_id, proc in processes:
            print(f"  kill {proc.pid}  # GPU {gpu_id}")


if __name__ == "__main__":
    main()