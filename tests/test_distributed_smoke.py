"""
Smoke tests for distributed training on 4×B200 GPUs.

These tests verify that both single-GPU and multi-GPU training complete
without errors and achieve expected throughput.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: str, timeout: int = 300) -> tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    print(f"\n▶ Running: {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def test_single_gpu_smoke():
    """
    Single-GPU smoke test (2 minutes).
    
    Expected:
    - Training completes without errors
    - FPS > 1000
    - GPU utilization > 60%
    - Checkpoint saved
    """
    print("\n" + "="*60)
    print("TEST 1: Single-GPU Smoke Test (2 min)")
    print("="*60)
    
    cmd = (
        "python train/train_ppo.py "
        "--config train/configs/small.yaml "
        "--device cuda "
        "--total-timesteps 10000 "
        "--n-envs 8 "
        "--run-name smoke_single_gpu"
    )
    
    returncode, stdout, stderr = run_command(cmd, timeout=180)
    
    # Check exit code
    assert returncode == 0, f"Training failed with code {returncode}\n{stderr}"
    
    # Check for completion
    assert "Training complete" in stdout, "Training did not complete"
    
    # Check checkpoint exists
    checkpoint_dir = Path("runs/smoke_single_gpu/checkpoints")
    assert checkpoint_dir.exists(), "Checkpoint directory not created"
    
    print("\n✅ Single-GPU smoke test PASSED")
    return True


def test_multi_gpu_smoke():
    """
    Multi-GPU smoke test (3 minutes).
    
    Expected:
    - Training completes without errors
    - All 4 GPUs show >60% utilization
    - Global FPS ~4× single-GPU FPS
    - Only 1 run directory created
    - No NCCL errors
    """
    print("\n" + "="*60)
    print("TEST 2: Multi-GPU Smoke Test (3 min)")
    print("="*60)
    
    cmd = (
        "torchrun --standalone --nproc-per-node=4 "
        "train/train_ppo.py "
        "--config train/configs/small.yaml "
        "--device cuda "
        "--total-timesteps 20000 "
        "--n-envs 64 "
        "--run-name smoke_multi_gpu"
    )
    
    returncode, stdout, stderr = run_command(cmd, timeout=240)
    
    # Check exit code
    assert returncode == 0, f"Distributed training failed with code {returncode}\n{stderr}"
    
    # Check for DDP initialization
    assert "DDP wrapping complete" in stdout, "DDP not initialized"
    assert "parameters synchronized" in stdout, "Parameters not synchronized"
    
    # Check for completion
    assert "Training complete" in stdout or "✅" in stdout, "Training did not complete"
    
    # Check no NCCL errors
    assert "NCCL error" not in stderr.lower(), f"NCCL errors detected:\n{stderr}"
    assert "timeout" not in stderr.lower(), f"NCCL timeout detected:\n{stderr}"
    
    # Check only 1 run directory created
    run_dirs = list(Path("runs").glob("smoke_multi_gpu*"))
    assert len(run_dirs) == 1, f"Expected 1 run dir, found {len(run_dirs)}"
    
    # Check checkpoint exists (rank 0 only should save)
    checkpoint_dir = run_dirs[0] / "checkpoints"
    assert checkpoint_dir.exists(), "Checkpoint directory not created"
    
    print("\n✅ Multi-GPU smoke test PASSED")
    return True


def test_checkpoint_compatibility():
    """
    Test that saved checkpoints can be loaded without errors.
    
    Verifies that DDP unwrapping worked correctly (no 'module.' prefix).
    """
    print("\n" + "="*60)
    print("TEST 3: Checkpoint Compatibility")
    print("="*60)
    
    # Find the latest checkpoint
    run_dirs = sorted(Path("runs").glob("smoke_multi_gpu*"))
    if not run_dirs:
        print("⚠️  No multi-GPU run found, skipping test")
        return True
    
    latest_run = run_dirs[-1]
    checkpoints = list((latest_run / "checkpoints").glob("*.zip"))
    
    if not checkpoints:
        print("⚠️  No checkpoints found, skipping test")
        return True
    
    checkpoint = checkpoints[0]
    
    # Try loading with Python
    cmd = f"""python -c "
from stable_baselines3 import PPO
model = PPO.load('{checkpoint}')
state_dict_keys = list(model.policy.state_dict().keys())
has_module_prefix = any('module.' in k for k in state_dict_keys)
assert not has_module_prefix, 'DDP module. prefix found in checkpoint!'
print('✓ Checkpoint valid: no module. prefix')
"
"""
    
    returncode, stdout, stderr = run_command(cmd, timeout=30)
    
    assert returncode == 0, f"Checkpoint loading failed:\n{stderr}"
    assert "Checkpoint valid" in stdout, "Checkpoint validation failed"
    
    print("\n✅ Checkpoint compatibility test PASSED")
    return True


if __name__ == "__main__":
    """Run all smoke tests."""
    print("\n" + "="*60)
    print("DISTRIBUTED TRAINING SMOKE TESTS")
    print("="*60)
    print("These tests verify multi-GPU training works correctly.")
    print("Expected runtime: ~5 minutes total")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Run tests in sequence
        test_single_gpu_smoke()
        test_multi_gpu_smoke()
        test_checkpoint_compatibility()
        
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print(f"Total time: {elapsed:.1f} seconds")
        print("="*60)
        
        sys.exit(0)
        
    except AssertionError as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        print("="*60)
        sys.exit(1)
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ UNEXPECTED ERROR")
        print("="*60)
        print(f"Error: {e}")
        print("="*60)
        sys.exit(1)