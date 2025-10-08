#!/bin/bash
#
# Launch multi-GPU training for Snake RL
#
# Usage:
#   bash scripts/launch_all_gpus.sh [options]
#
# Examples:
#   # Use all GPUs with default settings
#   bash scripts/launch_all_gpus.sh
#
#   # Specify GPUs and timesteps
#   bash scripts/launch_all_gpus.sh --gpus 0,1,2 --total-timesteps 5000000
#
#   # Custom config and run name
#   bash scripts/launch_all_gpus.sh --config train/configs/large.yaml --run-name my_experiment

set -e  # Exit on error

# Default values
GPUS="all"
CONFIG="train/configs/base.yaml"
TOTAL_TIMESTEPS=500000
SEED=42
RUN_NAME=""
LOGDIR="runs"
AUTO_SCALE="true"
WAIT="false"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --total-timesteps)
            TOTAL_TIMESTEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --logdir)
            LOGDIR="$2"
            shift 2
            ;;
        --auto-scale)
            AUTO_SCALE="$2"
            shift 2
            ;;
        --wait)
            WAIT="true"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --gpus GPUS              Comma-separated GPU IDs or 'all' (default: all)"
            echo "  --config PATH            Config file path (default: train/configs/base.yaml)"
            echo "  --total-timesteps N      Total timesteps per GPU (default: 500000)"
            echo "  --seed N                 Base random seed (default: 42)"
            echo "  --run-name NAME          Base run name (default: timestamp)"
            echo "  --logdir DIR             Log directory (default: runs)"
            echo "  --auto-scale true|false  Enable auto-scaling (default: true)"
            echo "  --wait                   Wait for all processes to complete"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --gpus 0,1,2 --total-timesteps 5000000"
            echo "  $0 --config train/configs/large.yaml --wait"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python tools/launcher_multi_gpu.py"
CMD="$CMD --gpus $GPUS"
CMD="$CMD --config $CONFIG"
CMD="$CMD --total-timesteps $TOTAL_TIMESTEPS"
CMD="$CMD --seed $SEED"
CMD="$CMD --logdir $LOGDIR"
CMD="$CMD --auto-scale $AUTO_SCALE"

if [ -n "$RUN_NAME" ]; then
    CMD="$CMD --run-name $RUN_NAME"
fi

if [ "$WAIT" = "true" ]; then
    CMD="$CMD --wait"
fi

# Print command
echo "Launching multi-GPU training..."
echo "Command: $CMD"
echo ""

# Execute
eval $CMD