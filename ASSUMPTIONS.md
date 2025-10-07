# Assumptions and Design Decisions

This document lists all assumptions made during development of the Snake RL project.

## Environment Design

1. **Grid Representation**
   - Assumed 3-channel observation (body, food, head) is sufficient
   - Alternative considered: single-channel with different integer values
   - Rationale: Multi-channel allows CNN to learn feature hierarchies naturally

2. **Action Space**
   - Assumed discrete actions (4 directions) without diagonal movement
   - 180° reverse moves are prevented to avoid instant death
   - Rationale: Matches classic Snake game mechanics

3. **Reward Structure**
   - Default: +1 for food, -1 for death, -0.01 per step
   - Assumed sparse rewards are sufficient for learning
   - Distance-based shaping is optional (may help or hinder, tested in ablations)
   - Rationale: Balance between learning signal and reward engineering

4. **Termination Conditions**
   - Death (wall collision or self-collision) terminates episode
   - max_steps also terminates to prevent infinite episodes
   - Default max_steps = 200 * grid_size (allows full board traversal)
   - Rationale: Prevents runaway episodes while allowing success

## Training Pipeline

1. **Vectorization**
   - SubprocVecEnv on Linux for true parallelism
   - DummyVecEnv on Windows (no fork support)
   - Assumed this difference is acceptable for development
   - Rationale: Platform compatibility vs. performance trade-off

2. **Hyperparameters**
   - PPO defaults from Stable-Baselines3 documentation
   - Learning rate: 3e-4 (standard for PPO)
   - Batch size and n_steps autoscaled based on resources
   - Rationale: Proven hyperparameters from literature

3. **Autoscaling**
   - Assumed GPU memory can be estimated from total VRAM
   - Conservative defaults to avoid OOM crashes
   - `--max-utilization` flag for aggressive resource use
   - Rationale: Balance stability and throughput

4. **Evaluation**
   - Deterministic policy for evaluation (no stochasticity)
   - Separate evaluation environment to avoid data leakage
   - Fixed seeds for reproducibility
   - Rationale: Standard RL evaluation practice

## Data and Logging

1. **TensorBoard + CSV**
   - Assumed TensorBoard is sufficient for live monitoring
   - CSV provides backup for custom analysis
   - Rationale: Industry standard tools

2. **Checkpoint Frequency**
   - Save best model based on evaluation reward
   - Periodic checkpoints every 50k timesteps (configurable)
   - Rationale: Balance disk usage and recovery options

3. **Metrics**
   - Episode reward mean/std/min/max
   - Episode length
   - FPS (training throughput)
   - Policy/value loss, entropy
   - Rationale: Standard RL metrics for diagnosing learning

## Code Quality

1. **Python Version**
   - Minimum 3.10 for better type hints and match/case
   - Rationale: Modern features without cutting-edge instability

2. **Testing**
   - Unit tests for environment correctness
   - Smoke tests for training (not full convergence)
   - Assumed 100-1000 timesteps sufficient for smoke tests
   - Rationale: Balance thoroughness and CI runtime

3. **Type Hints**
   - Not enforced strictly (mypy in permissive mode)
   - Stable-Baselines3 has incomplete type stubs
   - Rationale: Gradual typing approach

## IB Extended Essay Specific

1. **Reproducibility**
   - Fixed seeds throughout (environment, model, data sampling)
   - Deterministic CuDNN when possible
   - Rationale: Scientific rigor for academic work

2. **Experiment Design**
   - Minimum 3 seeds per configuration for statistical validity
   - 500k timesteps for quick experiments, 5M for final results
   - Rationale: Balance computation time and result confidence

3. **Ablations**
   - Selected based on common RL practices (reward shaping, frame stacking, entropy)
   - Each ablation changes one variable at a time
   - Rationale: Isolate causal factors

## Platform Compatibility

1. **Windows**
   - Assumed CPU-only development is acceptable
   - Limited parallelism (DummyVecEnv)
   - Rationale: Most students have Windows laptops

2. **Linux**
   - Assumed CUDA support for cloud training
   - Full parallelism with SubprocVecEnv
   - Rationale: Cloud GPU instances are Linux

3. **Cross-Platform Paths**
   - Used Path from pathlib for compatibility
   - Rationale: Python 3 best practice

## Limitations and Known Issues

1. **AMP (Automatic Mixed Precision)**
   - Enabled only with `--max-utilization` flag
   - May cause numerical instability in some cases
   - Rationale: Trade-off between speed and stability

2. **Headless Rendering**
   - Streamlit dashboard may be slow over network
   - Video export works but requires opencv-python
   - Rationale: Best available solution for remote visualization

3. **Large Batch Sizes**
   - May not fit on smaller GPUs (<8GB)
   - Autoscaling provides conservative defaults
   - Rationale: Prioritize stability over max performance

## Future Improvements (Not Implemented)

1. Could add recurrent policies (LSTM) for partial observability
2. Could implement curiosity-driven exploration
3. Could add multi-agent Snake for competition
4. Could implement custom CNN architecture tuning
5. Could add more sophisticated reward shaping

---

**Note:** All assumptions are documented here for transparency. If you disagree with any, they are controlled by config files and can be changed without modifying code.