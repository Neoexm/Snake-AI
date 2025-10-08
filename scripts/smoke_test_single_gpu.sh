#!/bin/bash
# Smoke test for single-GPU training (validates env/model work, ~30 sec)

set -e

echo "=========================================="
echo "Single-GPU Smoke Test (1000 steps)"
echo "=========================================="

echo "Running 1000-step smoke test..."
python train/train_ppo.py \
    --config train/configs/base.yaml \
    --total-timesteps 1000 \
    --device cuda \
    --n-envs 8 \
    --run-name smoke_test_single \
    --seed 42

echo ""
echo "✓ Smoke test passed!"
echo "Check runs/smoke_test_single/ for logs"
