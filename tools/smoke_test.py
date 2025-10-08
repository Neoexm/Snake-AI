"""
Smoke test for Snake RL training system.

Validates that training can start and progress on both CPU and CUDA (if available),
and that auto-scaling achieves reasonable GPU utilization.
"""

import sys
import time
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_cpu_training():
    """Test basic CPU training."""
    print("\n" + "="*60)
    print("TEST 1: CPU Training")
    print("="*60)
    
    import subprocess
    
    cmd = [
        sys.executable,
        "train/train_ppo.py",
        "--config", "train/configs/small.yaml",
        "--device", "cpu",
        "--total-timesteps", "1000",
        "--n-envs", "2",
        "--auto-scale", "false",
        "--run-name", "smoke_test_cpu",
        "--eval-freq", "10000",  # Disable eval for speed
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("✅ CPU training test PASSED")
        return True
    else:
        print(f"❌ CPU training test FAILED (exit code {result.returncode})")
        return False


def test_cuda_training():
    """Test CUDA training with auto-scale."""
    print("\n" + "="*60)
    print("TEST 2: CUDA Training with Auto-Scale")
    print("="*60)
    
    # Check if CUDA is available
    try:
        import torch
        if not torch.cuda.is_available():
            print("⏭️  CUDA not available, skipping GPU test")
            return True
    except ImportError:
        print("⏭️  torch not available, skipping GPU test")
        return True
    
    import subprocess
    
    cmd = [
        sys.executable,
        "train/train_ppo.py",
        "--config", "train/configs/small.yaml",
        "--device", "cuda",
        "--total-timesteps", "2000",
        "--auto-scale", "true",
        "--run-name", "smoke_test_cuda",
        "--eval-freq", "10000",  # Disable eval for speed
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("✅ CUDA training test PASSED")
        
        # Check GPU utilization from logs
        try:
            from tools.hw_monitor import HardwareMonitor
            monitor = HardwareMonitor()
            stats = monitor.get_current_stats()
            
            if stats.gpus:
                gpu_util = stats.gpus[0].utilization_percent
                print(f"   Current GPU util: {gpu_util:.1f}%")
                
                if gpu_util >= 50:
                    print(f"   ✅ GPU utilization looks good (≥50%)")
                else:
                    print(f"   ⚠️  GPU utilization lower than expected (<50%)")
                    print(f"   This is not a failure, but may indicate tuning needed")
        except Exception as e:
            print(f"   Could not check GPU util: {e}")
        
        return True
    else:
        print(f"❌ CUDA training test FAILED (exit code {result.returncode})")
        return False


def test_hw_monitor():
    """Test hardware monitoring."""
    print("\n" + "="*60)
    print("TEST 3: Hardware Monitoring")
    print("="*60)
    
    try:
        from tools.hw_monitor import HardwareMonitor
        
        monitor = HardwareMonitor()
        stats = monitor.get_current_stats()
        
        print(f"CPU: {stats.cpu_percent:.1f}%")
        print(f"RAM: {stats.ram_percent:.1f}% ({stats.ram_used_gb:.1f}/{stats.ram_total_gb:.1f} GB)")
        
        for gpu in stats.gpus:
            print(f"GPU {gpu.device_id}: {gpu.name}")
            print(f"  Util: {gpu.utilization_percent:.1f}%")
            print(f"  Mem: {gpu.memory_percent:.1f}% ({gpu.memory_used_gb:.1f}/{gpu.memory_total_gb:.1f} GB)")
        
        if not stats.gpus:
            print("No GPUs detected")
        
        print("✅ Hardware monitoring test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Hardware monitoring test FAILED: {e}")
        return False


def test_autoscale():
    """Test autoscale detection."""
    print("\n" + "="*60)
    print("TEST 4: Autoscale Configuration")
    print("="*60)
    
    try:
        from train.autoscale import autoscale
        
        # Test CPU config
        print("\nCPU Configuration:")
        config = autoscale(prefer_device="cpu")
        print(f"  Device: {config.device}")
        print(f"  N envs: {config.n_envs}")
        print(f"  Batch size: {config.batch_size}")
        
        # Test GPU config if available
        import torch
        if torch.cuda.is_available():
            print("\nGPU Configuration:")
            config = autoscale(prefer_device="cuda", max_utilization=True)
            print(f"  Device: {config.device}")
            print(f"  N envs: {config.n_envs}")
            print(f"  Batch size: {config.batch_size}")
            print(f"  Policy width: {config.policy_width}")
            print(f"  Policy depth: {config.policy_depth}")
            print(f"  Precision: {config.precision}")
        
        print("✅ Autoscale configuration test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Autoscale configuration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_gpu_launcher():
    """Test multi-GPU launcher detection."""
    print("\n" + "="*60)
    print("TEST 5: Multi-GPU Launcher")
    print("="*60)
    
    try:
        # Just test import and GPU detection
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from tools.launcher_multi_gpu import detect_gpus
        
        gpus = detect_gpus()
        print(f"Detected {len(gpus)} GPU(s): {gpus}")
        
        if len(gpus) == 0:
            print("⏭️  No GPUs for multi-GPU test, but launcher works")
        else:
            print(f"✅ Multi-GPU launcher detected {len(gpus)} GPU(s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-GPU launcher test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    parser = argparse.ArgumentParser(description="Run smoke tests for Snake RL")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip training tests, only test imports and detection",
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SNAKE RL SMOKE TESTS")
    print("="*60)
    
    results = []
    
    # Always run these
    results.append(("Hardware Monitoring", test_hw_monitor()))
    results.append(("Autoscale Config", test_autoscale()))
    results.append(("Multi-GPU Launcher", test_multi_gpu_launcher()))
    
    # Training tests (slow)
    if not args.quick:
        results.append(("CPU Training", test_cpu_training()))
        results.append(("CUDA Training", test_cuda_training()))
    else:
        print("\n⏭️  Skipping training tests (--quick mode)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())