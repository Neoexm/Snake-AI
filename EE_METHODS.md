# IB Extended Essay Methodology

This document explains how the Snake RL project maps to the IB Extended Essay scientific methodology.

## Research Question

**Example RQ:** "How do different reward shaping strategies affect the sample efficiency and final performance of a PPO agent learning to play Snake?"

**Variables:**
- **Independent Variable:** Reward shaping strategy (sparse vs. distance-based)
- **Dependent Variables:** 
  - Sample efficiency (reward per 100k timesteps)
  - Final performance (mean episode reward after 500k timesteps)
  - Training stability (standard deviation of rewards)
- **Controlled Variables:**
  - Grid size (12×12)
  - PPO hyperparameters (learning rate, batch size, etc.)
  - Random seed (multiple trials)
  - Hardware (same GPU/CPU)

## Experimental Design

### 1. Treatments

We define experimental conditions in YAML config files:

- **Control:** `train/configs/base.yaml` (sparse rewards only)
- **Treatment 1:** `train/configs/ablations/reward_shaping_distance.yaml` (distance-based shaping)
- **Treatment 2:** Additional ablations (frame stacking, entropy, etc.)

### 2. Replication

Each treatment is run with **3 different random seeds** to ensure reproducibility:

```bash
for seed in 42 43 44; do
  python train/train_ppo.py \
    --config train/configs/base.yaml \
    --seed $seed \
    --run-name control_seed${seed}
done
```

### 3. Data Collection

**Automated logging** captures metrics every 1000 timesteps:

- Episode reward (mean, std, min, max)
- Episode length (proxy for snake performance)
- Policy entropy (exploration level)
- Value function loss (learning progress)
- Training FPS (computational efficiency)

All data is saved to:
- **TensorBoard logs** (time-series visualization)
- **CSV files** (statistical analysis)

### 4. Controls

**Fixed throughout experiments:**
- Environment parameters (grid size, max steps)
- PPO hyperparameters (learning rate, gamma, etc.)
- Evaluation protocol (deterministic policy, 100 episodes)
- Hardware (same GPU model and CPU)

**Randomized:**
- Initial network weights (via seed)
- Environment initialization (via seed)
- Stochastic policy during training

## Data Analysis

### Statistical Analysis

Using `scripts/export_plots.py`, we generate:

1. **Learning Curves** (reward vs. timesteps)
   - Shows convergence rate and stability
   - Confidence intervals from multiple seeds

2. **Sample Efficiency** (reward per 100k timesteps)
   - Measures speed of learning
   - Key metric for comparing strategies

3. **Final Performance Comparison**
   - Bar plot with error bars (std dev)
   - Statistical significance via t-test

### Example Analysis Code

```python
import pandas as pd
import scipy.stats as stats

# Load results from multiple runs
control = pd.read_csv("runs/control_seed42/progress.csv")
treatment = pd.read_csv("runs/distance_seed42/progress.csv")

# Compare final 100 episodes
control_final = control.tail(100)["rollout/ep_rew_mean"]
treatment_final = treatment.tail(100)["rollout/ep_rew_mean"]

# T-test for significance
t_stat, p_value = stats.ttest_ind(control_final, treatment_final)
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
```

## Reproducibility Checklist

✅ **Fixed seeds** for all experiments  
✅ **Version pinning** in requirements.txt  
✅ **Config snapshots** saved with each run  
✅ **Deterministic operations** where possible  
✅ **Same hardware** for all trials  
✅ **Raw data preserved** (CSV logs)  
✅ **Code version control** (Git)  

## Validity Considerations

### Internal Validity

**Confounding variables minimized:**
- Same codebase for all treatments
- Same evaluation protocol
- Multiple seeds to account for randomness

**Potential threats:**
- GPU temperature throttling (monitor with `nvidia-smi`)
- Background processes affecting CPU (close unnecessary programs)
- PyTorch non-determinism in some operations (acceptable due to multiple trials)

### External Validity

**Generalizability:**
- Results specific to Snake game
- PPO algorithm (may differ with other RL algorithms)
- Grid sizes tested (8×8, 12×12, 16×16)

**Transferability:**
- Insights may apply to other grid-based games
- Reward shaping principles are general in RL

### Construct Validity

**Are we measuring what we claim?**
- Episode reward = agent success ✓
- Episode length = game duration (not snake size) ⚠️
- Sample efficiency = learning speed ✓

**Metrics logged in `info` dict:**
- Snake length (actual performance metric)
- Food eaten count
- Collision types (wall vs. self)

## Reporting Results

### For the EE Report

Use the generated plots from `scripts/export_plots.py`:

1. **Figure 1:** Learning curves with confidence intervals
2. **Figure 2:** Sample efficiency comparison
3. **Figure 3:** Final performance bar chart
4. **Table 1:** Summary statistics (mean ± std for each treatment)

### Example Table

| Treatment | Final Reward | Sample Efficiency | Convergence Time |
|-----------|--------------|-------------------|------------------|
| Control (sparse) | 5.2 ± 1.1 | 0.52 | 300k steps |
| Distance shaping | 6.8 ± 0.9 | 0.68 | 200k steps |
| Frame stacking | 5.5 ± 1.3 | 0.55 | 320k steps |

### Statistical Reporting

Report p-values and effect sizes:
- "Distance shaping significantly improved final reward (t=3.45, p=0.003, d=1.2)"
- "No significant difference between frame stacking and control (t=0.87, p=0.42)"

## Ethical Considerations

**For IB EE ethics section:**

- **No human subjects** (computational experiment only)
- **No environmental impact** (beyond standard computing)
- **Reproducible science** (all code and data available)
- **Honest reporting** (negative results are valid results)
- **Acknowledgments** (cite Stable-Baselines3, Gymnasium, etc.)

## Limitations

**Acknowledged limitations:**

1. **Computational constraints:** Limited to 5M timesteps per run
2. **Hyperparameter space:** Only tested default PPO settings
3. **Environment simplicity:** Snake is deterministic and fully observable
4. **Algorithm choice:** Only tested PPO (not A2C, DQN, etc.)
5. **Hardware variance:** Results may differ on different GPUs

## Conclusion

This project provides a **rigorous, reproducible framework** for conducting RL experiments that meet IB Extended Essay standards. All code is open-source, all data is logged, and all decisions are documented.

**Key strengths:**
- Automated logging and reproducibility
- Multiple trials with fixed seeds
- Statistical analysis tools included
- Clear separation of treatments

**Next steps for the EE:**
1. Run all experiments (baseline + ablations)
2. Generate plots with `scripts/export_plots.py`
3. Perform statistical tests in `results.ipynb`
4. Write up results with proper citations
5. Discuss limitations and future work