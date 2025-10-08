#!/bin/bash
set -e

# Default config
CONFIG=${1:-train/configs/large.yaml}
TOTAL_TIMESTEPS=${2:-10000000}
SEED=${3:-42}

# Generate run name
RUN_NAME="ddp_4xb200_$(date +%Y%m%d_%H%M%S)"

# Create runs directory if it doesn't exist
mkdir -p runs

echo "================================================"
echo "Multi-GPU DDP Training"
echo "================================================"
echo "GPUs: 4"
echo "Config: ${CONFIG}"
echo "Total Timesteps: ${TOTAL_TIMESTEPS}"
echo "Seed: ${SEED}"
echo "Run Name: ${RUN_NAME}"
echo "================================================"

# Launch with torchrun
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    train/train_ppo.py \
    --config "${CONFIG}" \
    --device cuda \
    --total-timesteps ${TOTAL_TIMESTEPS} \
    --seed ${SEED} \
    --run-name "${RUN_NAME}" \
    --precision amp \
    --max-utilization

echo ""
echo "✅ Training complete! View results:"
echo "   tensorboard --logdir runs/${RUN_NAME} --host 0.0.0.0 --port 6006"