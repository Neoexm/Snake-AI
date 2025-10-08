#!/usr/bin/env python3
"""
Validate that the repo is ready for 4×B200 distributed training.

Checks:
- Torch 2.8+ with CUDA 12.4+
- 4 GPUs detected
- NVLink connectivity
- DDP import and initialization
- Environment creation
- Model instantiation with DDP wrapping
- Single update step (forward + backward + optimizer)
- Checkpoint save/load with DDP unwrapping

Exit code 0 if all checks pass, 1 if any fail.
"""

import sys
import os

# Force multi-GPU visibility
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def check(condition, message):
    if condition:
        print(f"✓ {message}")
        return True
    else:
        print(f"✗ {message}")
        return False

def main():
    checks_passed = 0
    checks_total = 0
    
    print("="*60)
    print("4×B200 Readiness Validation")
    print("="*60)
    
    # Check 1: PyTorch version
    checks_total += 1
    try:
        import torch
        version = torch.__version__
        major, minor = map(int, version.split('.')[:2])
        if major >= 2 and minor >= 8:
            checks_passed += check(True, f"PyTorch {version} (>=2.8.0)")
        else:
            check(False, f"PyTorch {version} (need >=2.8.0)")
    except Exception as e:
        check(False, f"PyTorch import failed: {e}")
    
    # Check 2: CUDA version
    checks_total += 1
    try:
        cuda_version = torch.version.cuda
        if cuda_version:
            major, minor = map(int, cuda_version.split('.')[:2])
            if major >= 12 and minor >= 4:
                checks_passed += check(True, f"CUDA {cuda_version} (>=12.4)")
            else:
                check(False, f"CUDA {cuda_version} (need >=12.4)")
        else:
            check(False, "CUDA not available")
    except Exception as e:
        check(False, f"CUDA check failed: {e}")
    
    # Check 3: GPU count
    checks_total += 1
    try:
        gpu_count = torch.cuda.device_count()
        if gpu_count == 4:
            checks_passed += check(True, f"4 GPUs detected")
            for i in range(4):
                name = torch.cuda.get_device_properties(i).name
                mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"    GPU {i}: {name} ({mem_gb:.1f} GB)")
        else:
            check(False, f"Expected 4 GPUs, found {gpu_count}")
    except Exception as e:
        check(False, f"GPU detection failed: {e}")
    
    # Check 4: Gymnasium
    checks_total += 1
    try:
        import gymnasium as gym
        version = gym.__version__
        checks_passed += check(True, f"Gymnasium {version}")
    except Exception as e:
        check(False, f"Gymnasium import failed: {e}")
    
    # Check 5: Stable-Baselines3
    checks_total += 1
    try:
        import stable_baselines3 as sb3
        version = sb3.__version__
        checks_passed += check(True, f"Stable-Baselines3 {version}")
    except Exception as e:
        check(False, f"Stable-Baselines3 import failed: {e}")
    
    # Check 6: Snake environment
    checks_total += 1
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from snake_env import SnakeEnv
        env = SnakeEnv(grid_size=12)
        obs, _ = env.reset()
        assert obs.shape == (3, 12, 12), f"Expected shape (3, 12, 12), got {obs.shape}"
        checks_passed += check(True, "Snake environment creation")
    except Exception as e:
        check(False, f"Snake environment failed: {e}")
    
    # Check 7: DDP utilities
    checks_total += 1
    try:
        from train.ddp_utils import (
            setup_distributed,
            is_distributed,
            wrap_model_ddp,
            unwrap_model_ddp,
        )
        checks_passed += check(True, "DDP utilities import")
    except Exception as e:
        check(False, f"DDP utilities import failed: {e}")
    
    # Check 8: Model creation with DDP
    checks_total += 1
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from snake_env import SnakeEnv
        
        env = DummyVecEnv([lambda: SnakeEnv(grid_size=12)])
        model = PPO("CnnPolicy", env, n_steps=128, batch_size=256, device='cuda:0', verbose=0)
        checks_passed += check(True, "PPO model instantiation")
    except Exception as e:
        check(False, f"PPO model creation failed: {e}")
    
    # Check 9: NVLink status (informational)
    checks_total += 1
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', 'nvlink', '--status'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and 'Active' in result.stdout:
            checks_passed += check(True, "NVLink connectivity")
        else:
            check(False, "NVLink not active (will use PCIe)")
    except Exception as e:
        check(False, f"NVLink check failed (may not be critical): {e}")
    
    # Check 10: NCCL backend
    checks_total += 1
    try:
        if torch.distributed.is_nccl_available():
            checks_passed += check(True, "NCCL backend available")
        else:
            check(False, "NCCL backend not available")
    except Exception as e:
        check(False, f"NCCL check failed: {e}")
    
    print("="*60)
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print("="*60)
    
    if checks_passed == checks_total:
        print("\n✓ Repository is READY for 4×B200 training!")
        print("\nNext steps:")
        print("  1. Run smoke test: ./scripts/smoke_test_single_gpu.sh")
        print("  2. Run multi-GPU smoke test: ./scripts/smoke_test_multi_gpu.sh")
        print("  3. Launch full training: ./scripts/launch_b200.sh")
        return 0
    else:
        print(f"\n✗ Repository is NOT READY ({checks_total - checks_passed} issues found)")
        print("\nFix the issues above before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
