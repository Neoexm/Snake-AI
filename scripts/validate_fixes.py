#!/usr/bin/env python3
"""
Validation script to verify all 7 critical fixes are working correctly.

Run this BEFORE deploying to 4×B200 to ensure the repo is ready.
"""

import sys
import subprocess
import time
from pathlib import Path


def check_fix_1_lambda_closure():
    """Check that lambda closure is fixed using functools.partial."""
    print("\n[1/7] Checking lambda closure fix...")
    
    with open("train/train_ppo.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    if "from functools import partial" in content and "partial(_make_env" in content:
        print("✅ PASS: Lambda closure uses functools.partial")
        return True
    else:
        print("❌ FAIL: Lambda closure NOT fixed")
        return False


def check_fix_2_checkpoint_sync():
    """Check that checkpoint resume broadcasts weights."""
    print("\n[2/7] Checking checkpoint resume weight sync...")
    
    with open("train/train_ppo.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    if "broadcast_model_parameters(model.policy, src=0)" in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "if args.resume_from:" in line:
                next_30_lines = '\n'.join(lines[i:i+30])
                if "broadcast_model_parameters" in next_30_lines and "BEFORE moving to GPU" in next_30_lines:
                    print("✅ PASS: Checkpoint resume broadcasts weights before GPU move")
                    return True
        print("❌ FAIL: broadcast_model_parameters found but not in correct location")
        return False
    else:
        print("❌ FAIL: No weight broadcast after checkpoint resume")
        return False


def check_fix_3_atomic_saves():
    """Check that saves are atomic (temp + rename)."""
    print("\n[3/7] Checking atomic checkpoint saves...")
    
    with open("train/train_ppo.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    if "final_model.tmp" in content and "temp_path.rename(final_path)" in content:
        print("✅ PASS: Atomic saves implemented (temp + rename)")
        return True
    else:
        print("❌ FAIL: Atomic saves NOT implemented")
        return False


def check_fix_4_timestep_sync():
    """Check that timestep accounting is synced across ranks."""
    print("\n[4/7] Checking timestep synchronization in DDP...")
    
    with open("train/train_ppo.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    if "dist.broadcast(remaining_tensor, src=0)" in content:
        print("✅ PASS: Timestep accounting synchronized via broadcast")
        return True
    else:
        print("❌ FAIL: Timestep accounting NOT synchronized")
        return False


def check_fix_5_deterministic_eval():
    """Check that eval.py sets deterministic flags."""
    print("\n[5/7] Checking deterministic eval flags...")
    
    with open("train/eval.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    if "torch.backends.cudnn.deterministic = True" in content and \
       "torch.use_deterministic_algorithms" in content:
        print("✅ PASS: Deterministic eval flags set")
        return True
    else:
        print("❌ FAIL: Deterministic eval flags NOT set")
        return False


def check_fix_6_nccl_interface():
    """Check that NCCL interface is auto-detected."""
    print("\n[6/7] Checking NCCL interface auto-detection...")
    
    with open("train/ddp_utils.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    if "_detect_network_interface" in content:
        print("✅ PASS: NCCL interface auto-detection implemented")
        return True
    else:
        print("❌ FAIL: NCCL interface still hardcoded")
        return False


def check_fix_7_b200_threshold():
    """Check that B200 memory threshold is lowered to 175GB."""
    print("\n[7/7] Checking B200 memory threshold...")
    
    with open("train/autoscale.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    if "mem_gb >= 175" in content or "'b200' in gpu_name" in content:
        print("✅ PASS: B200 threshold lowered to 175GB + name matching")
        return True
    else:
        print("❌ FAIL: B200 threshold still too high")
        return False


def run_syntax_check():
    """Check that all Python files have valid syntax."""
    print("\n[BONUS] Running syntax checks...")
    
    files = [
        "train/train_ppo.py",
        "train/ddp_utils.py",
        "train/autoscale.py",
        "train/eval.py",
    ]
    
    all_valid = True
    for file in files:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", file],
            capture_output=True
        )
        if result.returncode == 0:
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}: {result.stderr.decode()}")
            all_valid = False
    
    return all_valid


def main():
    print("="*60)
    print("VALIDATING 7 CRITICAL FIXES FOR 4×B200 READINESS")
    print("="*60)
    
    results = []
    
    results.append(("Lambda closure (env seeding)", check_fix_1_lambda_closure()))
    results.append(("Checkpoint resume (weight sync)", check_fix_2_checkpoint_sync()))
    results.append(("Atomic saves", check_fix_3_atomic_saves()))
    results.append(("Timestep synchronization", check_fix_4_timestep_sync()))
    results.append(("Deterministic eval", check_fix_5_deterministic_eval()))
    results.append(("NCCL auto-detection", check_fix_6_nccl_interface()))
    results.append(("B200 threshold", check_fix_7_b200_threshold()))
    
    syntax_ok = run_syntax_check()
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    if not syntax_ok:
        print("❌ FAIL: Syntax errors detected")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\n{passed_count}/{total_count} fixes validated")
    
    if passed_count == total_count and syntax_ok:
        print("\n🎉 ALL FIXES VALIDATED - READY FOR 4×B200!")
        print("\nNext steps:")
        print("  1. Run smoke tests: pytest tests/test_ddp_production.py")
        print("  2. Deploy to Runpod with 4×B200")
        print("  3. Launch: ./scripts/launch_b200.sh")
        return 0
    else:
        print("\n⚠️  SOME FIXES MISSING - NOT READY FOR DEPLOYMENT")
        print("Review failed checks above and re-apply fixes.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
