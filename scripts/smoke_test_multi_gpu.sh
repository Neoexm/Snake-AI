#!/bin/bash
# Smoke test for multi-GPU training (validates DDP works, ~2 min)

set -e

echo "=========================================="
echo "Multi-GPU Smoke Test (4×GPU, 5000 steps)"
echo "=========================================="

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt 4 ]; then
    echo "ERROR: Need 4 GPUs, found $GPU_COUNT"
    exit 1
fi

echo "Running 5000-step smoke test..."
torchrun \
    --standalone \
    --nproc-per-node=4 \
    train/train_ppo.py \
    --config train/configs/base.yaml \
    --total-timesteps 5000 \
    --device cuda \
    --precision amp \
    --n-envs 16 \
    --run-name smoke_test_multigpu \
    --seed 42

echo ""
echo "✓ Smoke test passed!"
echo "Check runs/smoke_test_multigpu/ for logs"
