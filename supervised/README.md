# Supervised Learning (Behavior Cloning) Baseline

This module implements behavior cloning for the Snake RL environment as a baseline comparison to PPO.

## Quick Start

### 1. Collect Expert Demonstrations

**Scripted policy (fast smoke test):**
```powershell
python supervised/collect_dataset.py --config train/configs/base.yaml --expert scripted --steps 10000 --seed 0 --save-dir data/snake_bc
```

**From trained PPO model:**
```powershell
python supervised/collect_dataset.py --config train/configs/base.yaml --expert ppo --ppo-model "runs/20251009-140258/best_model.zip" --steps 500000 --seed 42 --save-dir data/snake_bc
```

### 2. Train BC Policy

**Quick CPU smoke test (5 minutes):**
```powershell
python supervised/train_bc.py --data "data/snake_bc/scripted_<TIMESTAMP>" --config supervised/configs/bc_base.yaml --device cpu --time-budget-min 5 --save-dir runs_bc/smoke --seed 0
```

**Full GPU training (4 hours):**
```powershell
python supervised/train_bc.py --data "data/snake_bc/ppo_<TIMESTAMP>" --config supervised/configs/bc_base.yaml --device cuda --time-budget-min 240 --save-dir runs_bc/full --seed 42
```

### 3. Evaluate BC Policy

**Same protocol as PPO (200 episodes):**
```powershell
python supervised/eval_bc.py --model "runs_bc/full/best.pt" --config train/configs/base.yaml --n-episodes 200 --seed 42 --output "runs_bc/full/eval.csv"
```

## Architecture

The BC model automatically selects CNN or MLP based on observation space:
- **CNN** for image observations (C, H, W) - 3x3 convolutions with pooling
- **MLP** for vector observations (D,) - 2-3 hidden layers

## Files

- `collect_dataset.py` - Generate expert demonstrations
- `train_bc.py` - Train behavior cloning policy
- `eval_bc.py` - Evaluate with PPO's protocol
- `models.py` - CNN/MLP policy networks
- `dataset.py` - PyTorch dataset with lazy shard loading
- `utils.py` - Seed control, device selection, time budget
- `configs/bc_base.yaml` - Default hyperparameters
- `test_smoke.py` - Unit smoke tests

## Output Structure

Training creates:
```
runs_bc/<RUN_NAME>/
├── config.yaml           # Resolved configuration
├── best.pt              # Best model (by val loss)
├── last.pt              # Final checkpoint
├── metrics.json         # Training metrics
├── classes.json         # Action space info
└── events.out.tfevents.*  # TensorBoard logs
```

Evaluation creates:
```
eval.csv         # Per-episode results (episode, total_reward, length, seed)
eval.json        # Summary statistics (mean ± std)
```

## Notes

- Uses **same environment wrappers** as PPO for fair comparison
- Supports **time budget** (`--time-budget-min`) for controlled experiments
- **Deterministic** seeds for reproducibility
- **Mixed precision** training on CUDA
- **Multi-worker** data loading on Linux (Windows uses 0 workers)
- Does NOT modify PPO code
