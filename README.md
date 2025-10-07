# Snake RL - Reinforcement Learning for IB Extended Essay

A Project where an AI learns to play snake via RL for a school project

### Training

**Basic training:**
```bash
python train/train_ppo.py --config train/configs/base.yaml
```

**Advanced options:**
```bash
python train/train_ppo.py \
  --config train/configs/large.yaml \
  --device cuda \
  --total-timesteps 5000000 \
  --n-envs 32 \
  --max-utilization \
  --seed 42 \
  --run-name my_experiment
```

**CLI arguments:**
- `--config`: Path to YAML config file (required)
- `--device`: `auto`, `cuda`, or `cpu` (default: auto)
- `--total-timesteps`: Override config timesteps
- `--n-envs`: Number of parallel environments (overrides autoscale)
- `--max-utilization`: Push resource usage aggressively
- `--seed`: Random seed for reproducibility
- `--run-name`: Custom name for this run
- `--eval-freq`: Evaluate every N timesteps
- `--save-freq`: Save checkpoint every N timesteps

### Evaluation

**Evaluate a trained model:**
```bash
python train/eval.py \
  --model runs/<run_name>/best_model.zip \
  --n-episodes 100 \
  --output eval_results.json
```

### Visualization

**Local window (OpenCV):**
```bash
python train/play.py --model runs/<run_name>/best_model.zip --fps 10
```

**Save to video:**
```bash
python train/play.py --model runs/<run_name>/best_model.zip --video agent.mp4
```

**Web dashboard:**
```bash
streamlit run dashboards/app.py
# Then open http://localhost:8501
```

**TensorBoard:**
```bash
tensorboard --logdir runs
# Then open http://localhost:6006
```

### Experiments

**Run ablation studies:**
```bash
# Baseline
python train/train_ppo.py --config train/configs/base.yaml --seed 42

# Distance reward shaping
python train/train_ppo.py --config train/configs/ablations/reward_shaping_distance.yaml --seed 42

# Frame stacking
python train/train_ppo.py --config train/configs/ablations/frame_stacking.yaml --seed 42

# High entropy
python train/train_ppo.py --config train/configs/ablations/high_entropy.yaml --seed 42
```

**Generate comparison plots:**
```bash
python scripts/export_plots.py \
  --runs runs/baseline runs/distance_shaping runs/frame_stack \
  --names "Baseline" "Distance Shaping" "Frame Stack" \
  --output plots/ablations
```

### Resource Monitoring

**Monitor live resource usage:**
```bash
python scripts/profile_resources.py --interval 1.0 --compact
```

**Check autoscale recommendations:**
```bash
python -c "from train.autoscale import autoscale; autoscale().print_summary()"
```

## Configuration

All experiments are defined in YAML config files. Example:

```yaml
environment:
  grid_size: 12
  step_penalty: -0.01
  death_penalty: -1.0
  food_reward: 1.0
  distance_reward_scale: 0.0  # Enable with 0.01
  frame_stack: 1  # Stack frames for temporal context

ppo:
  learning_rate: 0.0003
  n_steps: 256
  batch_size: 512
  n_epochs: 4
  gamma: 0.99
  ent_coef: 0.01

training:
  total_timesteps: 500000
  eval_freq: 10000
```

See [`train/configs/`](train/configs/) for examples.

## Testing

**Run all tests:**
```bash
pytest -v
```

**Run specific test file:**
```bash
pytest tests/test_env.py -v
```

**With coverage:**
```bash
pytest --cov=snake_env --cov=train --cov-report=html
```

## Code Quality

**Format code:**
```bash
black .
```

**Lint:**
```bash
ruff check . --fix
```

**Type check:**
```bash
mypy snake_env train
```

**Pre-commit hooks:**
```bash
pre-commit install
pre-commit run --all-files
```

## Reproducibility (IB EE)

All experiments use fixed seeds and deterministic environments for reproducibility:

```bash
# Run three trials with different seeds
for seed in 42 43 44; do
  python train/train_ppo.py \
    --config train/configs/base.yaml \
    --seed $seed \
    --run-name baseline_seed${seed}
done

# Generate plots with error bars
python scripts/export_plots.py \
  --runs runs/baseline_seed42 runs/baseline_seed43 runs/baseline_seed44 \
  --names "Trial 1" "Trial 2" "Trial 3" \
  --output plots/reproducibility
```

See [`EE_METHODS.md`](EE_METHODS.md) for detailed methodology.

## Performance Tips

### Windows CPU
- Use `--n-envs 4` to `8` for best throughput
- Enable MKL with `set MKL_NUM_THREADS=<cores>`
- Monitor with Task Manager

### Linux GPU
- Use `--device cuda --max-utilization`
- Autoscaling will detect GPU and maximize batch size
- Monitor with `nvidia-smi` or `scripts/profile_resources.py`
- For multi-GPU, use `CUDA_VISIBLE_DEVICES=0,1`

### Cloud (e.g., Google Colab, RunPod)
```python
# In notebook
!pip install -r requirements-gpu.txt
!python train/train_ppo.py --config train/configs/large.yaml --device cuda
```

## Troubleshooting

**Problem:** Training is slow on Windows
- **Solution:** Use fewer environments (`--n-envs 4`) or switch to GPU cloud

**Problem:** CUDA out of memory
- **Solution:** Reduce `--n-envs` or disable `--max-utilization`

**Problem:** TensorBoard not showing data
- **Solution:** Ensure training ran for >100 steps, refresh browser

**Problem:** OpenCV window not appearing
- **Solution:** Install `opencv-python`: `pip install opencv-python`

**Problem:** Tests failing
- **Solution:** Ensure dependencies installed: `pip install -r requirements.txt`

## Acknowledgments

- Built with [Gymnasium](https://gymnasium.farama.org/)
- Training via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- Inspired by classic Snake game and OpenAI Gym environments