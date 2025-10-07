# Snake RL - Quick Start Runbook

Exact commands to get the project running on Windows (CPU) and Linux (GPU).

---

## Windows CPU Development

### 1. Setup (First Time Only)

```powershell
# Navigate to project directory
cd "c:\Users\tomda\Documents\IB EE Work"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install pre-commit hooks
pre-commit install
```

### 2. Run Tests

```powershell
# Activate environment (if not already active)
.venv\Scripts\activate

# Run all tests
pytest -v

# Run with coverage
pytest --cov=snake_env --cov=train --cov-report=html
```

### 3. Quick Training Test (Small Config)

```powershell
# Activate environment
.venv\Scripts\activate

# Train for 50k timesteps on small grid
python train/train_ppo.py --config train/configs/small.yaml --total-timesteps 50000 --seed 42

# Output will be in: runs/<timestamp>/
```

### 4. Full Training (Base Config)

```powershell
# Train with base configuration (500k timesteps)
python train/train_ppo.py --config train/configs/base.yaml --seed 42 --run-name baseline

# Monitor with TensorBoard
tensorboard --logdir runs

# Then open: http://localhost:6006
```

### 5. Watch Trained Agent

```powershell
# Replace <run_name> with your actual run directory name
python train/play.py --model runs/<run_name>/best_model.zip --fps 10

# Press ESC to quit, SPACE to reset episode
```

### 6. Evaluate Performance

```powershell
# Evaluate over 100 episodes
python train/eval.py --model runs/<run_name>/best_model.zip --n-episodes 100 --output eval_results.json
```

### 7. Web Dashboard

```powershell
# Launch Streamlit dashboard
streamlit run dashboards/app.py

# Then open: http://localhost:8501
```

### 8. Generate Plots for EE

```powershell
# Export publication-ready plots
python scripts/export_plots.py --runs runs/baseline --output plots

# Or compare multiple runs:
python scripts/export_plots.py --runs runs/run1 runs/run2 runs/run3 --names "Baseline" "Treatment 1" "Treatment 2" --output plots/comparison
```

---

## Linux GPU Cloud

### 1. Setup (First Time Only)

```bash
# Navigate to project directory
cd ~/snake-rl

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install GPU dependencies
pip install -r requirements-gpu.txt

# Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Optional: Install pre-commit hooks
pre-commit install
```

### 2. Test Autoscaling

```bash
# Activate environment
source .venv/bin/activate

# Check autoscale recommendations
python -c "from train.autoscale import autoscale; autoscale().print_summary()"

# Monitor resources in real-time
python scripts/profile_resources.py --compact
```

### 3. Quick Training Test

```bash
# Activate environment
source .venv/bin/activate

# Quick test with GPU
python train/train_ppo.py \
  --config train/configs/small.yaml \
  --total-timesteps 50000 \
  --device cuda \
  --seed 42
```

### 4. Full Training with Max Utilization

```bash
# Train with autoscaling and maximum GPU utilization
python train/train_ppo.py \
  --config train/configs/base.yaml \
  --device cuda \
  --max-utilization \
  --seed 42 \
  --run-name baseline_gpu

# Run in background (detached)
nohup python train/train_ppo.py \
  --config train/configs/large.yaml \
  --device cuda \
  --max-utilization \
  --seed 42 \
  --run-name large_gpu > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### 5. Remote TensorBoard Access

```bash
# On server: Launch TensorBoard on all interfaces
tensorboard --logdir runs --host 0.0.0.0 --port 6006

# Then access from your local machine:
# http://<server-ip>:6006
```

### 6. Remote Streamlit Dashboard

```bash
# Launch dashboard accessible remotely
streamlit run dashboards/app.py --server.address 0.0.0.0 --server.port 7860

# Then access from your local machine:
# http://<server-ip>:7860
```

### 7. Save Video of Agent

```bash
# Create video file (headless mode)
python train/play.py \
  --model runs/<run_name>/best_model.zip \
  --video agent_gameplay.mp4 \
  --fps 10

# Download video to local machine with scp:
# scp user@server:~/snake-rl/agent_gameplay.mp4 .
```

---

## Running IB EE Experiments

### Complete Experiment Pipeline

#### Windows:

```powershell
# 1. Run all ablations
python train/train_ppo.py --config train/configs/base.yaml --seed 42 --run-name baseline
python train/train_ppo.py --config train/configs/ablations/reward_shaping_distance.yaml --seed 42 --run-name distance_shaping
python train/train_ppo.py --config train/configs/ablations/frame_stacking.yaml --seed 42 --run-name frame_stacking
python train/train_ppo.py --config train/configs/ablations/high_entropy.yaml --seed 42 --run-name high_entropy

# 2. Generate comparison plots
python scripts/export_plots.py --runs runs/baseline runs/distance_shaping runs/frame_stacking runs/high_entropy --names "Baseline" "Distance Shaping" "Frame Stacking" "High Entropy" --output plots/ee_comparison

# 3. Open results notebook
jupyter notebook results.ipynb
```

#### Linux:

```bash
# Run experiments in parallel on GPU
python train/train_ppo.py --config train/configs/base.yaml --device cuda --seed 42 --run-name baseline &
python train/train_ppo.py --config train/configs/ablations/reward_shaping_distance.yaml --device cuda --seed 42 --run-name distance_shaping &
python train/train_ppo.py --config train/configs/ablations/frame_stacking.yaml --device cuda --seed 42 --run-name frame_stacking &
python train/train_ppo.py --config train/configs/ablations/high_entropy.yaml --device cuda --seed 42 --run-name high_entropy &

# Wait for all to complete
wait

# Generate plots
python scripts/export_plots.py \
  --runs runs/baseline runs/distance_shaping runs/frame_stacking runs/high_entropy \
  --names "Baseline" "Distance Shaping" "Frame Stacking" "High Entropy" \
  --output plots/ee_comparison
```

### Multiple Seeds for Statistical Validity

```bash
# Run with 3 different seeds for baseline
for seed in 42 43 44; do
  python train/train_ppo.py \
    --config train/configs/base.yaml \
    --device cuda \
    --seed $seed \
    --run-name baseline_seed${seed}
done

# Compare trials
python scripts/export_plots.py \
  --runs runs/baseline_seed42 runs/baseline_seed43 runs/baseline_seed44 \
  --names "Trial 1" "Trial 2" "Trial 3" \
  --output plots/reproducibility
```

---

## Troubleshooting

### Windows

**Virtual environment not activating:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\activate
```

**Tests failing:**
```powershell
pip install --upgrade -r requirements.txt
pytest -v --tb=short
```

**Training very slow:**
```powershell
# Use fewer environments
python train/train_ppo.py --config train/configs/small.yaml --n-envs 4
```

### Linux

**CUDA out of memory:**
```bash
# Reduce number of environments
python train/train_ppo.py --config train/configs/base.yaml --device cuda --n-envs 16

# Or use smaller config
python train/train_ppo.py --config train/configs/small.yaml --device cuda
```

**SSH connection drops during long training:**
```bash
# Use tmux or screen
tmux new -s training
python train/train_ppo.py --config train/configs/large.yaml --device cuda
# Press Ctrl+B then D to detach
# Reconnect with: tmux attach -t training
```

---

## Quick Reference

### Environment Variables

```bash
# Set number of CPU threads (Windows & Linux)
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8

# Select specific GPU (Linux)
export CUDA_VISIBLE_DEVICES=0

# Multiple GPUs (Linux)
export CUDA_VISIBLE_DEVICES=0,1
```

### File Locations

- **Logs & Models:** `runs/<run_name>/`
- **Best Model:** `runs/<run_name>/best_model.zip`
- **Checkpoints:** `runs/<run_name>/checkpoints/`
- **CSV Logs:** `runs/<run_name>/progress.csv`
- **Config Snapshot:** `runs/<run_name>/config.yaml`

### Common Commands

```bash
# List all runs
ls -lh runs/

# Find best models
find runs/ -name "best_model.zip"

# Delete old runs
rm -rf runs/<old_run_name>

# Check disk usage
du -sh runs/
```

---

## Success Checklist

- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip list | grep stable-baselines3`)
- [ ] Tests pass (`pytest -v`)
- [ ] GPU detected (Linux: `nvidia-smi`, check CUDA available)
- [ ] Training starts without errors
- [ ] TensorBoard shows live plots
- [ ] Agent can be watched with `play.py`
- [ ] Plots exported successfully

---

**You're all set! Start with the small config for testing, then scale up to full experiments.**