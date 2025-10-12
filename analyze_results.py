# analyze_results.py
import argparse, json, math, pathlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def load_eval_csv(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    
    reward_col = None
    if "total_reward" in cols:
        reward_col = cols["total_reward"]
    elif "reward" in cols:
        reward_col = cols["reward"]
    else:
        raise ValueError(f"{path} must contain 'reward' or 'total_reward' column")
    
    if "length" not in cols:
        raise ValueError(f"{path} must contain 'length' column")
    
    return df.rename(columns={reward_col: "reward", cols["length"]: "length"})

def summarize(name, x):
    return dict(
        name=name,
        n=len(x),
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)),
        min=float(np.min(x)),
        max=float(np.max(x)),
        ci95_lo=float(stats.t.interval(0.95, len(x)-1, loc=np.mean(x), scale=stats.sem(x))[0]),
        ci95_hi=float(stats.t.interval(0.95, len(x)-1, loc=np.mean(x), scale=stats.sem(x))[1]),
    )

def cliffs_delta(a, b):
    # Effect size for stochastic dominance
    A, B = np.asarray(a), np.asarray(b)
    # downsample if extremely long to keep fast
    maxn = 20000
    if len(A) > maxn: A = np.random.default_rng(0).choice(A, size=maxn, replace=False)
    if len(B) > maxn: B = np.random.default_rng(1).choice(B, size=maxn, replace=False)
    gt = sum((ai > bj) for ai in A for bj in B)
    lt = sum((ai < bj) for ai in A for bj in B)
    d = (gt - lt) / (len(A)*len(B))
    return float(d)

def grade_effect(d):
    ad = abs(d)
    if ad < 0.147: return "negligible"
    if ad < 0.33:  return "small"
    if ad < 0.474: return "medium"
    return "large"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", required=True)
    ap.add_argument("--ppo", required=True)
    ap.add_argument("--out", default="results/figs")
    args = ap.parse_args()
    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    bc  = load_eval_csv(args.bc)
    ppo = load_eval_csv(args.ppo)

    # Summaries
    s_bc  = summarize("BC",  bc["reward"].values)
    s_ppo = summarize("PPO", ppo["reward"].values)

    # Welch t-test on reward
    t, p = stats.ttest_ind(bc["reward"].values, ppo["reward"].values, equal_var=False)
    d = cliffs_delta(ppo["reward"].values, bc["reward"].values)

    summary = {
        "reward_summary": [s_bc, s_ppo],
        "t_test_reward": {"t": float(t), "p": float(p)},
        "cliffs_delta_reward": {"d": d, "magnitude": grade_effect(d)},
        "length_summary": [
            summarize("BC", bc["length"].values),
            summarize("PPO", ppo["length"].values),
        ],
    }
    with open(outdir/"summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Table CSV
    pd.DataFrame(summary["reward_summary"]).to_csv(outdir/"reward_summary.csv", index=False)
    pd.DataFrame(summary["length_summary"]).to_csv(outdir/"length_summary.csv", index=False)

    # Plots (no seaborn, single-figure per plot)
    for metric in ["reward", "length"]:
        plt.figure()
        bins = 50
        plt.hist(bc[metric].values, bins=bins, alpha=0.6, label="BC")
        plt.hist(ppo[metric].values, bins=bins, alpha=0.6, label="PPO")
        plt.xlabel(metric.capitalize()); plt.ylabel("Count"); plt.legend()
        plt.title(f"Distribution of {metric}")
        plt.tight_layout(); plt.savefig(outdir/f"{metric}_hist.png"); plt.close()

        # CDF
        plt.figure()
        for name, data in [("BC", bc[metric].values), ("PPO", ppo[metric].values)]:
            xs = np.sort(data)
            ys = np.arange(1, len(xs)+1)/len(xs)
            plt.plot(xs, ys, label=name)
        plt.xlabel(metric.capitalize()); plt.ylabel("CDF"); plt.legend()
        plt.title(f"CDF of {metric}")
        plt.tight_layout(); plt.savefig(outdir/f"{metric}_cdf.png"); plt.close()

    # Print a quick console digest
    print("\n=== SUMMARY (reward) ===")
    for s in summary["reward_summary"]:
        print(f"{s['name']}: mean {s['mean']:.2f} (95% CI {s['ci95_lo']:.2f}–{s['ci95_hi']:.2f}), "
              f"std {s['std']:.2f}, min {s['min']:.2f}, max {s['max']:.2f}, n={s['n']}")
    print(f"Welch t-test: t={t:.3f}, p={p:.3e}")
    print(f"Cliff's delta (PPO vs BC): d={d:.3f} ({grade_effect(d)})")
    print(f"\nFiles written to: {outdir}\n")

if __name__ == "__main__":
    main()
