#!/bin/bash
# Launch script for 4×NVIDIA B200 multi-GPU training on Runpod/Lambda/etc.
# Usage: ./scripts/launch_b200.sh [config] [timesteps]

set -e

CONFIG="${1:-train/configs/base.yaml}"
TIMESTEPS="${2:-5000000}"
RUN_NAME="b200_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "4×B200 Distributed Training Launch"
echo "=========================================="
echo "Config: $CONFIG"
echo "Timesteps: $TIMESTEPS"
echo "Run Name: $RUN_NAME"
echo "=========================================="

# Verify GPUs
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are NVIDIA drivers installed?"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $GPU_COUNT GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

if [ "$GPU_COUNT" -ne 4 ]; then
    echo "WARNING: Expected 4 GPUs, found $GPU_COUNT. Proceeding anyway..."
fi

# Export NCCL environment variables for B200 (NVLink 5.0 + InfiniBand)
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export MALLOC_ARENA_MAX=1

echo ""
echo "NCCL Environment:"
echo "  NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL"
echo "  NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo ""

# Create log directory
LOG_DIR="runs/$RUN_NAME"
mkdir -p "$LOG_DIR"

# Launch with torchrun (replaces torch.distributed.launch)
echo "Launching torchrun with 4 GPUs..."
torchrun \
    --standalone \
    --nproc-per-node=4 \
    --nnodes=1 \
    train/train_ppo.py \
    --config "$CONFIG" \
    --total-timesteps "$TIMESTEPS" \
    --device cuda \
    --precision amp \
    --max-utilization \
    --run-name "$RUN_NAME" \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/training.log"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Logs: $LOG_DIR/training.log"
echo "Checkpoints: $LOG_DIR/"
echo "TensorBoard: tensorboard --logdir $LOG_DIR"
echo "=========================================="
