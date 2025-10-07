# Quick Start Guide - Fixed Version

This guide provides the exact commands to test the fixed Snake RL project.

## Windows CPU - Quick Test

```powershell
# 1. Activate virtual environment (if not already active)
.venv\Scripts\activate

# 2. Quick smoke test (1000 timesteps, ~30 seconds)
python test_fix.py

# 3. Short training run (20k timesteps, ~5 minutes)
python train\train_ppo.py --config train\configs\base.yaml --device cpu --total-timesteps 20000 --n-envs 2

# 4. Check TensorBoard
tensorboard --logdir runs
# Open: http://localhost:6006
```

## What Was Fixed

The original code had a **double-wrapped callable** issue:
- `make_snake_env()` returned a factory function instead of an env instance
- This caused: `ValueError: The environment is of type <class 'function'>`

**Fix applied:**
- `make_snake_env()` now returns the environment directly
- Vectorized env creation uses proper lambda factories with `lambda i=i:` to avoid capture bugs
- See `FIX_EXPLANATION.md` for full technical details

## Verify the Fix

### 1. Run Quick Test
```powershell
python test_fix.py
```

**Expected output:**
```
Testing environment construction fix...
✓ Environment created successfully: <class 'stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv'>
✓ Observation space: Box(0.0, 1.0, (3, 8, 8), float32)
✓ Action space: Discrete(4)
✓ PPO model created successfully

Training for 1000 timesteps...
[Progress bar showing training]

✅ SUCCESS! Training completed without errors.
```

### 2. Run Full Training
```powershell
# Windows CPU (2 parallel envs)
python train\train_ppo.py --config train\configs\base.yaml --device cpu --total-timesteps 20000 --n-envs 2
```

**Expected output:**
```
AUTOSCALE RESOURCE CONFIGURATION
============================================================
Device: cpu
Parallel Environments: 2
Steps per Rollout: 256
Batch Size: 512
...

📁 Run directory: runs\<timestamp>
📝 Config saved to: runs\<timestamp>\config.yaml

🤖 Creating PPO model...

🚀 Starting training for 20,000 timesteps...

[Training progress with no errors]
```

### 3. Verify TensorBoard
```powershell
tensorboard --logdir runs
```

Open http://localhost:6006 and verify you see:
- **Scalars**: `rollout/ep_rew_mean`, `rollout/ep_len_mean`, `time/fps`
- **Events** updating in real-time

## Linux GPU (If Available)

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. Quick test
python test_fix.py

# 4. GPU training (autoscale will use more envs)
python train/train_ppo.py \
  --config train/configs/base.yaml \
  --device cuda \
  --total-timesteps 50000 \
  --max-utilization
```

## Troubleshooting

### "environment is of type function" error
✅ **FIXED** - This error should no longer occur with the corrected code.

### Training very slow on Windows
- Use `--n-envs 2` or `--n-envs 4` (default autoscale should be fine)
- Close other programs to free CPU resources

### Import errors
```powershell
pip install -r requirements.txt --upgrade
```

### Can't find config file
Use correct path separators:
```powershell
# Windows
python train\train_ppo.py --config train\configs\base.yaml

# Linux
python train/train_ppo.py --config train/configs/base.yaml
```

## Next Steps

After verifying the fix works:

1. **Run experiments** for IB EE:
   ```powershell
   python train\train_ppo.py --config train\configs\base.yaml --seed 42 --run-name baseline
   python train\train_ppo.py --config train\configs\ablations\reward_shaping_distance.yaml --seed 42 --run-name distance
   ```

2. **Generate plots**:
   ```powershell
   python scripts\export_plots.py --runs runs\baseline runs\distance --output plots
   ```

3. **Analyze results**:
   ```powershell
   jupyter notebook results.ipynb
   ```

## Files Modified

The fix changed these files:
- ✅ `train/train_ppo.py` - Fixed env construction, removed nested factory
- ✅ `train/eval.py` - Fixed env construction
- ✅ `test_fix.py` - New quick verification script
- ✅ `FIX_EXPLANATION.md` - Technical explanation of the fix

All other files remain unchanged and fully functional.

## Success Criteria

✅ `test_fix.py` completes without errors  
✅ Training starts and timesteps increase  
✅ TensorBoard shows live metrics  
✅ No "environment is of type function" error  
✅ CPU/GPU resources are utilized  

**Status: All fixes applied and tested! 🎉**