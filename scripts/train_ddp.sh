#!/bin/bash
# Launch distributed training on 4×B200 GPUs using torchrun
#
# Usage:
#   bash scripts/train_ddp.sh [additional args to train_ppo.py]
#
# Examples:
#   bash scripts/train_ddp.sh --config train/configs/large.yaml
#   bash scripts/train_ddp.sh --total-timesteps 10000000 --max-utilization

set -e  # Exit on error

# Distributed training configuration
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Number of GPUs (adjust if not using 4)
NPROC_PER_NODE=4

# Default config if not specified
CONFIG=${1:-train/configs/large.yaml}

# Generate run name with timestamp
RUN_NAME="ddp_4xb200_$(date +%Y%m%d_%H%M%S)"

echo "================================================"
echo "Multi-GPU Distributed Training"
echo "================================================"
echo "GPUs: ${NPROC_PER_NODE}"
echo "Config: ${CONFIG}"
echo "Run Name: ${RUN_NAME}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "================================================"
echo ""

# Launch distributed training with torchrun
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=${NPROC_PER_NODE} \
    train/train_ppo.py \
    --config "${CONFIG}" \
    --device cuda \
    --precision amp \
    --run-name "${RUN_NAME}" \
    --seed 42 \
    --auto-scale true \
    --max-utilization \
    "${@:2}"  # Pass remaining args to train_ppo.py

echo ""
echo "================================================"
echo "Training Complete!"
echo "================================================"
echo "View results with:"
echo "  tensorboard --logdir runs/${RUN_NAME} --host 0.0.0.0 --port 6006"
echo "================================================"