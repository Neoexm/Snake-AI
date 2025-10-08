# Distributed Training Guide for Snake RL (4×NVIDIA B200)

This guide covers the **true multi-GPU distributed data-parallel training** implementation for the Snake RL project, optimized for 4×NVIDIA B200 GPUs on Runpod Linux.

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tuning](#performance-tuning)
9. [Validation](#validation)

---

## Overview

### What Changed?
The project now supports **TRUE distributed data-parallel (DDP) training** where:
- ✅ **4 GPUs train ONE agent** with synchronized gradients (not 4 separate agents)
- ✅ Each GPU processes a portion of environments (e.g., 64 envs/GPU × 4 GPUs = 256 total)
- ✅ Gradients are synchronized across GPUs during backpropagation
- ✅ Only rank 0 logs to TensorBoard and saves checkpoints
- ✅ B200-optimized NCCL settings for maximum throughput

### Key Features
- **DistributedDataParallel (DDP)**: PPO policy wrapped in `torch.nn.parallel.DistributedDataParallel`
- **Gradient Synchronization**: Automatic all-reduce during `.backward()`
- **Rank-Safe Operations**: Logging, checkpointing, and evaluation only on rank 0
- **B200 Optimization**: NVLink 5.0 and InfiniBand tuning via NCCL environment variables
- **Scalable**: Works with 1-N GPUs seamlessly

---

## Architecture

### Distributed Training Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         torchrun                                │
│                    (launches 4 processes)                       │
└───────┬──────────────┬──────────────┬──────────────┬───────────┘
        │              │              │              │
   ┌────▼────┐    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
   │ Rank 0  │    │ Rank 1  │    │ Rank 2  │    │ Rank 3  │
   │ GPU 0   │    │ GPU 1   │    │ GPU 2   │    │ GPU 3   │
   │ 64 envs │    │ 64 envs │    │ 64 envs │    │ 64 envs │
   └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘
        │              │              │              │
        │    Collect rollouts independently          │
        │              │              │              │
        ▼              ▼              ▼              ▼
   ┌────────────────────────────────────────────────────┐
   │         DDP-Wrapped PPO Policy (synchronized)      │
   │   - Forward pass: independent per rank             │
   │   - Backward pass: gradients all-reduced via NCCL  │
   │   - Optimizer step: synchronized parameters        │
   └────────────────────────────────────────────────────┘
        │
        │  (Only Rank 0)
        ▼
   ┌────────────────────────────────────────────────────┐
   │  - TensorBoard logging                             │
   │  - Checkpoint saving (DDP-unwrapped)               │
   │  - Evaluation                                      │
   └────────────────────────────────────────────────────┘
```

### File Modifications

#### Core Changes
1. **`train/train_ppo.py`**: Added distributed setup, DDP wrapping, rank-safe operations
2. **`train/autoscale.py`**: Updated for B200 GPU defaults (192-256 envs for 80GB+ VRAM)
3. **`train/ddp_utils.py`**: NEW - Helper functions for DDP operations
4. **`train/eval.py`**: Force single-GPU usage (`CUDA_VISIBLE_DEVICES=0`)
5. **`train/play.py`**: Force single-GPU usage (`CUDA_VISIBLE_DEVICES=0`)
6. **`requirements-gpu.txt`**: Updated to PyTorch 2.8+/CUDA 12.4
7. **`scripts/train_ddp.sh`**: NEW - Torchrun launcher script

---

## Installation

### Prerequisites
- **Hardware**: 4×NVIDIA B200 GPUs (or any NVIDIA GPUs with NVLink)
- **OS**: Linux (Runpod Ubuntu 22.04+)
- **Python**: 3.10+
- **CUDA**: 12.4+

### Setup Steps

```bash
# 1. Clone repository
git clone https://github.com/Neoexm/Snake-AI.git
cd Snake-AI

# 2. Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements-gpu.txt

# 4. Verify CUDA and GPUs
nvidia-smi
python -c "import torch; print(f'{torch.cuda.device_count()} GPUs available')"

# 5. Make launch script executable
chmod +x scripts/train_ddp.sh
```

---

## Quick Start

### Single-GPU Smoke Test (2 minutes)
```bash
python train/train_ppo.py \
    --config train/configs/small.yaml \
    --device cuda \
    --total-timesteps 10000 \
    --n-envs 8 \
    --run-name smoke_single_gpu
```

**Expected**: Training completes, FPS >1000, GPU util >60%, checkpoint saved.

### Multi-GPU Smoke Test (3 minutes)
```bash
torchrun --standalone --nproc-per-node=4 \
    train/train_ppo.py \
    --config train/configs/small.yaml \
    --device cuda \
    --total-timesteps 20000 \
    --n-envs 64 \
    --run-name smoke_multi_gpu
```

**Expected**: 
- All 4 GPUs show >60% utilization in `nvidia-smi`
- Global FPS ~4× single-GPU FPS in TensorBoard (`time/global_fps`)
- Only ONE run directory in `runs/` (not 4)
- No NCCL errors in stdout

### Full 4×B200 Training (2-4 hours)
```bash
bash scripts/train_ddp.sh --total-timesteps 10000000
```

---

## Configuration

### Environment Variables (Auto-Set)

The following NCCL variables are automatically configured in [`train/ddp_utils.py`](train/ddp_utils.py:setup_nccl_env_for_b200()):

```bash
# NVLink & InfiniBand
NCCL_IB_DISABLE=0          # Enable InfiniBand
NCCL_P2P_DISABLE=0         # Enable P2P (NVLink)
NCCL_P2P_LEVEL=NVL         # Use NVLink
NCCL_SOCKET_IFNAME=eth0    # Network interface

# Memory optimization
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
MALLOC_ARENA_MAX=1

# Debugging (set to INFO for verbose logs)
NCCL_DEBUG=WARN
```

### Hyperparameters (B200 Defaults)

For GPUs with ≥80GB VRAM ([`train/autoscale.py`](train/autoscale.py)):

| Parameter | Single-GPU | Multi-GPU (4×B200) | Notes |
|-----------|------------|--------------------|-------|
| `n_envs` | 192 | 256 (64/GPU) | Total environments |
| `n_steps` | 512 | 1024 | Rollout buffer size |
| `batch_size` | 4096 | 8192 | Training batch size |
| `policy_width` | 128 | 128 | CNN channels |
| `policy_depth` | 3 | 3 | CNN layers |
| `precision` | amp | amp | Mixed precision |

---

## Usage Examples

### Example 1: Custom Config
```bash
bash scripts/train_ddp.sh \
    --config train/configs/large.yaml \
    --total-timesteps 50000000 \
    --seed 123
```

### Example 2: Override Hyperparameters
```bash
torchrun --standalone --nproc-per-node=4 \
    train/train_ppo.py \
    --config train/configs/base.yaml \
    --n-envs 128 \
    --policy-width 256 \
    --precision amp \
    --max-utilization
```

### Example 3: Resume from Checkpoint
```bash
torchrun --standalone --nproc-per-node=4 \
    train/train_ppo.py \
    --resume-from runs/ddp_4xb200_20250108_143000/checkpoints/model_1000000_steps.zip \
    --total-timesteps 20000000
```

### Example 4: Evaluate Trained Model
```bash
# Evaluation automatically uses single-GPU (cuda:0)
python train/eval.py \
    --model runs/ddp_4xb200_20250108_143000/best_model.zip \
    --n-episodes 100
```

### Example 5: Watch Agent Play
```bash
# Playback automatically uses single-GPU (cuda:0)
python train/play.py \
    --model runs/ddp_4xb200_20250108_143000/best_model.zip \
    --fps 10
```

---

## Troubleshooting

### Issue 1: NCCL Timeout
**Symptom**: Training hangs or shows "NCCL timeout" errors.

**Solution**:
```bash
# Increase timeout (debugging only)
export NCCL_TIMEOUT=600000  # 10 minutes

# Enable verbose logging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Re-run training
bash scripts/train_ddp.sh
```

### Issue 2: GPU Utilization <60%
**Symptom**: GPUs underutilized despite distributed training.

**Solution**:
```bash
# Increase batch size and environments
torchrun --standalone --nproc-per-node=4 \
    train/train_ppo.py \
    --n-envs 256 \
    --batch-size 8192 \
    --n-steps 1024 \
    --max-utilization
```

### Issue 3: OOM (Out of Memory)
**Symptom**: `RuntimeError: CUDA out of memory`.

**Solution**:
```bash
# Reduce batch size or environments
torchrun --standalone --nproc-per-node=4 \
    train/train_ppo.py \
    --n-envs 192 \
    --batch-size 4096 \
    --n-steps 512
```

### Issue 4: "module." Prefix in Checkpoint
**Symptom**: Loaded model has `module.` prefix in state_dict keys.

**Fix**: Checkpoint saving now automatically unwraps DDP before saving. If you encounter this with old checkpoints:

```python
# Load and unwrap manually
from train.ddp_utils import unwrap_model_ddp
model = PPO.load('checkpoint.zip')
model.policy = unwrap_model_ddp(model.policy)
```

### Issue 5: Multiple Run Directories Created
**Symptom**: 4 separate run directories instead of 1.

**Fix**: Ensure you're using `torchrun` (not `python` directly):
```bash
# ❌ Wrong
python train/train_ppo.py ...

# ✅ Correct
torchrun --standalone --nproc-per-node=4 train/train_ppo.py ...
```

---

## Performance Tuning

### Optimal Settings for 4×B200

```bash
torchrun --standalone --nproc-per-node=4 \
    train/train_ppo.py \
    --config train/configs/large.yaml \
    --n-envs 256 \
    --n-steps 1024 \
    --batch-size 8192 \
    --policy-width 128 \
    --policy-depth 3 \
    --precision amp \
    --max-utilization \
    --total-timesteps 100000000
```

### Expected Throughput

| Configuration | FPS (Frames/Sec) | Training Time (10M steps) |
|---------------|------------------|---------------------------|
| 1×B200 (baseline) | ~8,000 | ~20 minutes |
| 4×B200 (DDP) | ~32,000 (4× speedup) | ~5 minutes |

### Monitoring Performance

```bash
# Terminal 1: Training
bash scripts/train_ddp.sh

# Terminal 2: Watch GPUs
watch -n 1 nvidia-smi

# Terminal 3: TensorBoard
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

**Key Metrics**:
- `time/global_fps`: Total throughput across all GPUs
- `system/gpu{0-3}_util_percent`: Per-GPU utilization (target: >80%)
- `system/gpu{0-3}_memory_allocated_gb`: VRAM usage

---

## Validation

### Checkpoint Validation
```python
# Verify checkpoint has NO "module." prefix
from stable_baselines3 import PPO

model = PPO.load('runs/smoke_multi_gpu/best_model.zip')
assert not any('module.' in k for k in model.policy.state_dict().keys()), \
    "DDP module. prefix found—checkpoint not unwrapped!"
print("✓ Checkpoint valid")
```

### Gradient Sync Validation
Run a short training session and verify logs show:
```
✓ DDP wrapping complete, parameters synchronized
```

### Multi-GPU Utilization Validation
```bash
# During training, run in separate terminal:
nvidia-smi dmon -s u

# All 4 GPUs should show >60% utilization:
# gpu   sm   mem   enc   dec
#   0   85    12     0     0
#   1   83    10     0     0
#   2   87    11     0     0
#   3   84    13     0     0
```

---

## Advanced Topics

### Custom NCCL Backend
If using InfiniBand, verify:
```bash
# Check InfiniBand devices
ibstat

# Adjust NCCL_SOCKET_IFNAME if needed
export NCCL_SOCKET_IFNAME=ib0  # For InfiniBand
```

### Mixed Precision Training (AMP)
Already enabled by default with `--precision amp`. To disable:
```bash
torchrun --standalone --nproc-per-node=4 \
    train/train_ppo.py \
    --precision fp32
```

### Dynamic Scaling (Elastic Training)
To support elastic scaling (add/remove GPUs mid-training):
```bash
# Not yet implemented - future work
# Would require torchrun elastic launch
```

---

## References

- **PyTorch DDP**: https://pytorch.org/docs/stable/nn.html#distributeddataparallel
- **NCCL**: https://docs.nvidia.com/deeplearning/nccl/
- **Torchrun**: https://pytorch.org/docs/stable/elastic/run.html
- **B200 Specs**: https://www.nvidia.com/en-us/data-center/b200/

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Enable debug logging: `export NCCL_DEBUG=INFO`
3. Review TensorBoard metrics
4. File an issue on GitHub with full error logs

---

**Last Updated**: 2025-01-15  
**Maintainer**: Snake RL Team