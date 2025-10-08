# TRUE Multi-GPU DDP Implementation - Summary

## Overview

This implementation adds **TRUE distributed data-parallel (DDP) training** to the Snake RL project, enabling ONE PPO agent to train across 4 GPUs with synchronized gradients, rather than 4 independent trainers.

## Key Changes Made

### 1. DDP Policy Wrapping (train/train_ppo.py)

**Lines 648-695**: Added DDP wrapping of PPO policy after model creation:

```python
# CRITICAL: Wrap policy in DDP for true distributed training
if is_distributed():
    model.policy = wrap_model_ddp(
        model.policy,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,
    )
    # Broadcast initial parameters from rank 0 to all ranks
    broadcast_model_parameters(model.policy, src=0)
    if is_main_process():
        print(f"✓ [Rank {rank}] DDP wrapping complete, parameters synchronized")
```

**What this does**:
- Wraps the PPO policy network in PyTorch's DistributedDataParallel
- Enables automatic gradient all-reduce during backward pass
- Synchronizes parameters across all ranks from rank 0
- Only happens when running under torchrun (when RANK env var is set)

### 2. Per-Rank Environment Ranks (train/train_ppo.py)

**Line 569**: Fixed environment rank calculation for unique global IDs:

```python
# In DDP: each rank creates n_envs environments with unique global ranks
env_fns = [
    lambda i=i: make_snake_env(**{**env_kwargs, 'rank': rank * resource_config.n_envs + i})
    for i in range(resource_config.n_envs)
]
```

**What this does**:
- Rank 0: environments 0-63
- Rank 1: environments 64-127
- Rank 2: environments 128-191
- Rank 3: environments 192-255
- Total: 256 unique environments with different seeds

### 3. Rank-Safe Logging (train/train_ppo.py)

**Lines 686-693**: Only rank 0 logs to TensorBoard:

```python
# Setup custom logger for CSV output
# Only rank 0 should log to TensorBoard to avoid conflicts
if is_main_process():
    logger = configure(str(run_dir), ["stdout", "csv", "tensorboard"])
else:
    # Non-zero ranks: only stdout, no TensorBoard
    logger = configure(str(run_dir), ["stdout"])
model.set_logger(logger)
```

**Lines 697-708**: Only rank 0 saves checkpoints and evaluates:

```python
# Checkpoint callback (rank 0 only)
if is_main_process():
    checkpoint_callback = CheckpointCallback(...)
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback (rank 0 only)
    eval_callback = EvalCallback(...)
    callbacks.append(eval_callback)
```

### 4. DDP-Aware Checkpointing (train/train_ppo.py)

**Lines 774-780**: Unwrap DDP before saving:

```python
# Save final model (rank 0 only, unwrap DDP first)
if is_main_process():
    final_path = run_dir / "final_model.zip"
    
    # Unwrap DDP before saving
    if is_distributed():
        model.policy = unwrap_model_ddp(model.policy)
    
    model.save(str(final_path))
```

**What this does**:
- Removes the DDP wrapper before saving to avoid "module." prefix in state dict
- Only rank 0 saves checkpoints (prevents race conditions)
- Saved models can be loaded without modification

### 5. Autoscale for DDP (train/autoscale.py)

**Lines 206-232**: Updated n_envs calculation for per-rank semantics:

```python
# Determine n_envs (PER-RANK in DDP mode, total in single-GPU mode)
# In DDP: Each rank creates n_envs environments
# Total environments = n_envs × world_size
if use_gpu:
    mem_gb = gpu_info['primary_memory_gb']
    if mem_gb >= 80:  # B200 (192GB), H100 (80GB), MI300X
        # DDP mode: 64 envs/GPU × 4 GPUs = 256 total
        # Single-GPU: 64 envs on one GPU
        n_envs = 64 if max_utilization else 48
```

**Key change**: B200 GPUs now default to 64 envs/GPU (was 256 total) for proper DDP scaling.

### 6. Launch Script (scripts/train_ddp.sh)

Updated to match requirements:
- Takes 3 args: config, total_timesteps, seed
- Launches 4 processes with torchrun
- Uses amp precision and max-utilization by default
- Simplified output

## How DDP Training Works

### Training Flow

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

### Gradient Synchronization

1. Each rank computes forward pass on its 64 environments
2. Each rank computes loss and gradients locally
3. DDP automatically all-reduces gradients across all 4 GPUs using NCCL
4. All ranks apply the same optimizer step with synchronized gradients
5. Result: One agent trained on 256 total environments

## Usage

### Quick Start Commands

```bash
# 1. Single-GPU smoke test (2 minutes)
python train/train_ppo.py \
    --config train/configs/small.yaml \
    --device cuda \
    --total-timesteps 10000 \
    --n-envs 8 \
    --run-name smoke_single_gpu

# 2. Multi-GPU smoke test (3 minutes)
torchrun --standalone --nproc-per-node=4 \
    train/train_ppo.py \
    --config train/configs/small.yaml \
    --device cuda \
    --total-timesteps 20000 \
    --run-name smoke_multi_gpu

# 3. Full 4×B200 training
bash scripts/train_ddp.sh train/configs/large.yaml 100000000 42
```

### Expected Behavior

**Single-GPU**:
- Creates 1 run directory
- GPU 0 shows >60% utilization
- FPS ~8,000

**Multi-GPU DDP**:
- Creates 1 run directory (shared by all ranks)
- All 4 GPUs show >60% utilization
- Global FPS ~32,000 (4× speedup)
- TensorBoard shows `time/global_fps` metric
- Only 1 set of checkpoints saved
- No "module." prefix in saved models

## Validation Checklist

- [ ] `nvidia-smi` shows all 4 GPUs with >60% utilization
- [ ] TensorBoard shows `time/global_fps` ~4× single-GPU FPS
- [ ] Only ONE run directory created (not 4)
- [ ] Log contains "✓ [Rank 0] DDP wrapping complete, parameters synchronized"
- [ ] No NCCL errors or timeouts
- [ ] Checkpoint loads without "module." prefix errors
- [ ] Smoke tests pass without crashes

## What Was NOT Changed

- Environment code (snake_env/)
- Policy architectures (train/models/)
- Evaluation/playback scripts (already force single-GPU)
- Existing configs (train/configs/)
- Test files (except distributed smoke test)
- Documentation files (README, etc.)

## Critical Implementation Notes

1. **SB3 + DDP**: Stable-Baselines3 does NOT natively support DDP. Manual wrapping of `model.policy` is required.

2. **Per-Rank Envs**: In DDP, each rank creates `n_envs` environments. Total = `n_envs × world_size`. This is NOT the same as gradient accumulation.

3. **Gradient Sync**: DDP handles all-reduce automatically during `backward()`. No manual sync needed if policy is wrapped correctly.

4. **Module Prefix**: DDP wraps model as `model.module`. Always unwrap with `unwrap_model_ddp()` before saving.

5. **NCCL Tuning**: B200-specific NCCL settings are in `ddp_utils.setup_nccl_env_for_b200()` and called before distributed init.

6. **Throughput**: Expect 3-4× speedup on 4 GPUs (not linear due to sync overhead and NCCL communication).

## Files Modified

1. **train/train_ppo.py**: DDP wrapping, rank-safe logging, per-rank env ranks
2. **train/autoscale.py**: Per-rank n_envs defaults for B200
3. **scripts/train_ddp.sh**: Simplified launch script matching requirements

## Next Steps

1. Run single-GPU smoke test to ensure no regression
2. Run multi-GPU smoke test to verify DDP works
3. Monitor `nvidia-smi` during training to confirm 4-GPU utilization
4. Check TensorBoard for global FPS metric
5. For full training, expect ~2-4 hours for 100M timesteps on 4×B200