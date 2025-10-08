#!/bin/bash
set -e

# Parse arguments
CONFIG=${1:-train/configs/large.yaml}
TOTAL_TIMESTEPS=${2:-100000000}  # 100M default for full training
SEED=${3:-42}

# Generate run name with timestamp
RUN_NAME="ddp_4xb200_$(date +%Y%m%d_%H%M%S)"

# Create runs directory
mkdir -p runs

echo "================================================"
echo "4×B200 Multi-GPU DDP Training"
echo "================================================"
echo "Config: ${CONFIG}"
echo "Total Timesteps: ${TOTAL_TIMESTEPS}"
echo "Seed: ${SEED}"
echo "Run Name: ${RUN_NAME}"
echo "================================================"

# Launch with torchrun
# CRITICAL: Use --max-utilization for B200 saturation
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
    --max-utilization \
    2>&1 | tee "runs/${RUN_NAME}.log"

echo ""
echo "✅ Training complete! View results:"
echo "   tensorboard --logdir runs/${RUN_NAME} --host 0.0.0.0 --port 6006"
echo "   tail -f runs/${RUN_NAME}.log"