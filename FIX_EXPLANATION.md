# Environment Construction Error - Fix Explanation

## Problem

When running:
```powershell
python train\train_ppo.py --config train\configs\base.yaml --device cpu --total-timesteps 50000 --n-envs 2
```

The error was:
```
ValueError: The environment is of type <class 'function'>, not a Gymnasium environment.
```

## Root Cause

The issue was in how we constructed vectorized environments for Stable-Baselines3 (SB3).

### What SB3 Expects

SB3's `DummyVecEnv` and `SubprocVecEnv` expect a **list of zero-argument callables**, where each callable **returns an environment instance**:

```python
# Correct pattern
env_fns = [lambda: SnakeEnv(), lambda: SnakeEnv()]
env = DummyVecEnv(env_fns)
```

### What We Were Doing Wrong

Our `make_snake_env()` function was returning a **factory function** (the `_init` function), not the environment itself:

```python
def make_snake_env(...):
    def _init():  # <-- Returns this function
        env = SnakeEnv(...)
        # ... apply wrappers
        return env
    return _init  # <-- Factory returns a factory!
```

Then when we used it:
```python
# This created a lambda that returns a FUNCTION, not an env!
env = make_vec_env(
    lambda rank=0: make_snake_env(**env_kwargs, rank=rank),
    n_envs=2,
)
```

This created a **double-wrapped callable**: `lambda -> factory -> env` instead of `lambda -> env`.

SB3 tried to call the lambda and got back a function object instead of a Gymnasium environment, hence the error.

## The Fix

### 1. Changed `make_snake_env()` to Return the Env Directly

**Before:**
```python
def make_snake_env(...):
    def _init():
        env = SnakeEnv(...)
        # ... wrappers
        return env
    return _init  # Returns factory function
```

**After:**
```python
def make_snake_env(...):
    env = SnakeEnv(...)
    # ... wrappers
    return env  # Returns actual environment instance
```

### 2. Fixed Vectorized Environment Creation

**Before (using make_vec_env):**
```python
env = make_vec_env(
    lambda rank=0: make_snake_env(**env_kwargs, rank=rank),
    n_envs=resource_config.n_envs,
    vec_env_cls=vec_env_cls,
)
```

**After (direct DummyVecEnv/SubprocVecEnv):**
```python
# Create list of zero-arg callables that return env instances
env_fns = [
    lambda i=i: make_snake_env(**{**env_kwargs, 'rank': i})
    for i in range(resource_config.n_envs)
]

env = vec_env_cls(env_fns)  # DummyVecEnv or SubprocVecEnv
```

### 3. Fixed Lambda Capture Bug

**Important:** We use `lambda i=i:` instead of `lambda: ... rank=i` to avoid Python's late binding issue. Without the default argument, all lambdas would capture the same `i` variable and use its final value.

**Wrong:**
```python
[lambda: make_env(rank=i) for i in range(3)]
# All lambdas use i=2 (final value)
```

**Correct:**
```python
[lambda i=i: make_env(rank=i) for i in range(3)]
# Each lambda captures its own i value
```

## Files Changed

1. **`train/train_ppo.py`**:
   - Removed nested `_init()` function from `make_snake_env()`
   - Changed vectorized env creation to use direct `DummyVecEnv`/`SubprocVecEnv` construction
   - Added `render_mode` parameter support
   - Fixed lambda capture with default arguments

2. **`train/eval.py`**:
   - Updated `make_eval_env()` to return env directly (not factory)
   - Fixed vectorized env creation

3. **`test_fix.py`** (new):
   - Quick smoke test to verify the fix works

## Verification

Run these commands to verify the fix:

```powershell
# Quick test (1000 timesteps)
python test_fix.py

# Full training test (20k timesteps)
python train\train_ppo.py --config train\configs\base.yaml --device cpu --total-timesteps 20000 --n-envs 2

# Verify TensorBoard logging
tensorboard --logdir runs
```

**Expected result:** Training starts without the "environment is of type function" error, timesteps increase, and TensorBoard shows new events.

## Why This Matters

1. **Correctness**: SB3 needs actual Gymnasium environments, not factory functions
2. **Compatibility**: Works with both `DummyVecEnv` (Windows) and `SubprocVecEnv` (Linux)
3. **Reproducibility**: Proper seeding with distinct ranks per environment
4. **Maintainability**: Clearer code structure without nested factory functions

## Additional Benefits

The fix also:
- Removes unnecessary indirection (clearer code)
- Makes debugging easier (stack traces show actual env creation)
- Enables future support for render mode in vectorized training
- Ensures consistent behavior across Windows and Linux platforms