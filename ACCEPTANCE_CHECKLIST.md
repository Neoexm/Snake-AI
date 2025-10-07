# Acceptance Checklist - Snake RL Project

This document verifies that all requirements from the project specification have been met.

## Acceptance Criteria

### Core Functionality

- [x] **Gymnasium-compliant Snake environment** with 3-channel observations `(3, H, W)`
- [x] **Discrete action space** (4 directions) with 180° reverse prevention
- [x] **Configurable rewards**: food (+1), death (-1), step penalty (-0.01)
- [x] **Deterministic reset** with seed support
- [x] **Render mode** `rgb_array` with scale-up factor
- [x] **PPO training pipeline** using Stable-Baselines3 CnnPolicy
- [x] **Autoscaling** to maximize GPU/CPU utilization
- [x] **TensorBoard logging** + CSV exports
- [x] **Live visualization** (local window and web dashboard)

### Training Features

- [x] **CLI with comprehensive flags**: `--config`, `--device`, `--n-envs`, `--total-timesteps`, `--seed`, etc.
- [x] **Multiple config files**: base.yaml, small.yaml, large.yaml, ablations/
- [x] **Automatic resource detection**: GPU/CPU, memory, cores
- [x] **Autoscale knobs**: `n_envs`, `batch_size`, `n_steps`, AMP, pin_memory
- [x] **Evaluation callback**: periodic eval with best model saving
- [x] **Checkpoint saving**: every N timesteps + best model
- [x] **Throughput logging**: FPS, GPU/CPU utilization to TensorBoard

### Visualization & Monitoring

- [x] **Local play window** (OpenCV) with FPS control
- [x] **Headless video export** for cloud environments
- [x] **Streamlit dashboard** with live agent streaming
- [x] **TensorBoard integration** for training curves
- [x] **Resource profiler** (`scripts/profile_resources.py`)
- [x] **Plot export tool** (`scripts/export_plots.py`) for publication-ready figures

### Experiments & Reproducibility

- [x] **Base configuration** (12×12 grid, standard PPO)
- [x] **Small configuration** (8×8 grid, quick testing)
- [x] **Large configuration** (16×16 grid, challenging)
- [x] **Ablation studies**:
  - [x] Distance-based reward shaping
  - [x] Frame stacking (4 frames)
  - [x] High entropy coefficient
- [x] **Fixed seeds** for reproducibility
- [x] **Config snapshots** saved with each run
- [x] **Statistical analysis** tools in results notebook

### Code Quality

- [x] **Unit tests** for environment (`tests/test_env.py`)
- [x] **Training smoke tests** (`tests/test_training_smoke.py`)
- [x] **Config tests** (`tests/test_config.py`)
- [x] **Pre-commit hooks** (black, ruff, mypy)
- [x] **Type hints** where practical
- [x] **Docstrings** (numpy style) on all major functions
- [x] **pyproject.toml** configuration for tools
- [x] **Code formatting**: black
- [x] **Linting**: ruff
- [x] **Type checking**: mypy

### Documentation

- [x] **README.md** with quickstart, usage guide, troubleshooting
- [x] **RUNBOOK.md** with exact commands for Windows and Linux
- [x] **ASSUMPTIONS.md** documenting all design decisions
- [x] **EE_METHODS.md** explaining IB EE methodology mapping
- [x] **results.ipynb** for final analysis and plots
- [x] **.gitignore** for clean version control

### Platform Support

- [x] **Windows CPU development** fully supported
- [x] **Linux GPU cloud** fully supported
- [x] **Cross-platform paths** using pathlib
- [x] **Conditional vectorization**: SubprocVecEnv (Linux) / DummyVecEnv (Windows)
- [x] **PyTorch CPU and CUDA** installation via separate requirements files

### Performance

- [x] **Autoscaling saturates resources** without OOM
- [x] **GPU detection** and automatic device selection
- [x] **AMP support** (optional with `--max-utilization`)
- [x] **Parallel environments** scale with available cores/GPU
- [x] **Throughput monitoring**: FPS logged to TensorBoard
- [x] **Resource warnings**: alerts if GPU available but using CPU

## 🧪 Test Execution

### Environment Tests

```bash
pytest tests/test_env.py -v
```

Expected results:
- Environment initialization
- Deterministic reset
- Correct observation shapes
- Reward signals (food, death, step)
- No reverse moves
- Max steps termination
- Render output
- Wrapper functionality

### Training Smoke Test

```bash
pytest tests/test_training_smoke.py -v
```

Expected results:
- Training runs without crash
- Model saves and loads
- Evaluation completes

### Config Tests

```bash
pytest tests/test_config.py -v
```

Expected results:
- YAML configs load successfully
- All config files are valid
- Config overrides work

## Integration Tests

### Windows CPU

```powershell
# 1. Training starts
python train/train_ppo.py --config train/configs/small.yaml --total-timesteps 1000

# 2. TensorBoard shows data
tensorboard --logdir runs

# 3. Play agent
python train/play.py --model runs/<run>/best_model.zip

# 4. Dashboard launches
streamlit run dashboards/app.py
```

### Linux GPU

```bash
# 1. GPU detected
python -c "from train.autoscale import autoscale; autoscale().print_summary()"

# 2. Training with CUDA
python train/train_ppo.py --config train/configs/small.yaml --device cuda --total-timesteps 1000

# 3. FPS increase with more envs
python scripts/profile_resources.py --compact
```

## Deliverables Checklist

### Code

- [x] `snake_env/` package with environment and wrappers
- [x] `train/train_ppo.py` - comprehensive training CLI
- [x] `train/eval.py` - standalone evaluation
- [x] `train/play.py` - visualization tool
- [x] `train/autoscale.py` - resource optimization
- [x] `train/configs/` - experiment configurations
- [x] `dashboards/app.py` - Streamlit dashboard
- [x] `scripts/profile_resources.py` - monitoring tool
- [x] `scripts/export_plots.py` - plot generation
- [x] `tests/` - comprehensive test suite

### Documentation

- [x] `README.md` - main documentation
- [x] `RUNBOOK.md` - exact commands
- [x] `ASSUMPTIONS.md` - design decisions
- [x] `EE_METHODS.md` - IB EE methodology
- [x] `results.ipynb` - analysis notebook
- [x] `requirements.txt` - Windows CPU deps
- [x] `requirements-gpu.txt` - Linux GPU deps
- [x] `pyproject.toml` - tool configuration
- [x] `.pre-commit-config.yaml` - quality hooks
- [x] `.gitignore` - version control

## Key Features Demonstrated

1. **Production Quality**: real autoscaling, monitoring, logging
2. **Scientific Rigor**: Fixed seeds, multiple trials, statistical analysis
3. **User Friendly**: One-command setup, clear error messages, helpful defaults
4. **Well Documented**: Every assumption documented, every command explained
5. **Fully Tested**: Unit tests, integration tests, smoke tests
6. **Cross-Platform**: Works on Windows (dev) and Linux (cloud)
7. **Scalable**: From laptop CPU to cloud GPU with same code
8. **Reproducible**: All experiments can be exactly reproduced

## Success Metrics

- **Code Quality**: Pre-commit hooks pass 
- **Tests**: All tests pass 
- **Documentation**: Complete and clear 
- **Usability**: One-command quickstart 
- **Performance**: Autoscaling works 
- **Reproducibility**: Fixed seeds and configs 
- **IB EE Ready**: Methodology documented 

**Estimated Time to First Results:**
- Windows CPU: ~30 minutes (small config)
- Linux GPU: ~5 minutes (small config), ~2 hours (large config)