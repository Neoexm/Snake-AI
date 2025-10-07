"""
Export publication-ready plots from training logs.

Reads TensorBoard CSV logs and creates clean figures for IB EE reports.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def load_progress_csv(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load progress.csv from a run directory."""
    csv_files = list(run_dir.glob("**/progress.csv"))
    
    if not csv_files:
        print(f"Warning: No progress.csv found in {run_dir}")
        return None
    
    # Use most recent if multiple
    csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return None


def plot_learning_curve(
    data: Dict[str, pd.DataFrame],
    output_path: str,
    metric: str = "rollout/ep_rew_mean",
    ylabel: str = "Episode Reward",
    title: str = "Learning Curve",
    smooth_window: int = 10,
):
    """
    Plot learning curves for multiple runs.
    
    Parameters
    ----------
    data : dict
        Dictionary mapping run names to DataFrames.
    output_path : str
        Path to save the plot.
    metric : str
        Column name to plot.
    ylabel : str
        Y-axis label.
    title : str
        Plot title.
    smooth_window : int
        Moving average window size.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for run_name, df in data.items():
        if metric not in df.columns or "time/total_timesteps" not in df.columns:
            print(f"Warning: {metric} not found in {run_name}")
            continue
        
        x = df["time/total_timesteps"].values
        y = df[metric].values
        
        # Smooth with moving average
        if smooth_window > 1:
            y_smooth = pd.Series(y).rolling(window=smooth_window, min_periods=1).mean().values
        else:
            y_smooth = y
        
        # Plot
        ax.plot(x, y_smooth, label=run_name, linewidth=2, alpha=0.9)
        
        # Optionally show raw data as faint line
        if smooth_window > 1:
            ax.plot(x, y, alpha=0.15, linewidth=1, color=ax.lines[-1].get_color())
    
    ax.set_xlabel("Timesteps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_episode_length(
    data: Dict[str, pd.DataFrame],
    output_path: str,
    smooth_window: int = 10,
):
    """Plot episode length over time."""
    plot_learning_curve(
        data=data,
        output_path=output_path,
        metric="rollout/ep_len_mean",
        ylabel="Episode Length",
        title="Episode Length Over Time",
        smooth_window=smooth_window,
    )


def plot_sample_efficiency(
    data: Dict[str, pd.DataFrame],
    output_path: str,
    interval: int = 100000,
):
    """
    Plot sample efficiency: reward per N timesteps.
    
    Parameters
    ----------
    data : dict
        Dictionary mapping run names to DataFrames.
    output_path : str
        Path to save the plot.
    interval : int
        Timestep interval for binning.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for run_name, df in data.items():
        if "rollout/ep_rew_mean" not in df.columns or "time/total_timesteps" not in df.columns:
            continue
        
        # Bin by intervals
        bins = np.arange(0, df["time/total_timesteps"].max() + interval, interval)
        bin_indices = np.digitize(df["time/total_timesteps"], bins)
        
        rewards_per_bin = []
        bin_centers = []
        
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if mask.sum() > 0:
                rewards_per_bin.append(df.loc[mask, "rollout/ep_rew_mean"].mean())
                bin_centers.append(bins[i-1] + interval/2)
        
        ax.plot(bin_centers, rewards_per_bin, marker='o', label=run_name, linewidth=2, markersize=6)
    
    ax.set_xlabel(f"Timesteps (binned by {interval:,})")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Sample Efficiency")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_training_throughput(
    data: Dict[str, pd.DataFrame],
    output_path: str,
):
    """Plot training throughput (FPS) over time."""
    plot_learning_curve(
        data=data,
        output_path=output_path,
        metric="time/fps",
        ylabel="Frames Per Second (FPS)",
        title="Training Throughput",
        smooth_window=5,
    )


def plot_final_comparison(
    data: Dict[str, pd.DataFrame],
    output_path: str,
    n_last: int = 100,
):
    """
    Bar plot comparing final performance of different runs.
    
    Parameters
    ----------
    data : dict
        Dictionary mapping run names to DataFrames.
    output_path : str
        Path to save the plot.
    n_last : int
        Number of last timesteps to average.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    run_names = []
    mean_rewards = []
    std_rewards = []
    mean_lengths = []
    std_lengths = []
    
    for run_name, df in data.items():
        if "rollout/ep_rew_mean" not in df.columns:
            continue
        
        # Take last n_last rows
        last_data = df.tail(n_last)
        
        run_names.append(run_name)
        mean_rewards.append(last_data["rollout/ep_rew_mean"].mean())
        std_rewards.append(last_data["rollout/ep_rew_mean"].std())
        
        if "rollout/ep_len_mean" in df.columns:
            mean_lengths.append(last_data["rollout/ep_len_mean"].mean())
            std_lengths.append(last_data["rollout/ep_len_mean"].std())
    
    # Reward comparison
    x = np.arange(len(run_names))
    ax1.bar(x, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(run_names, rotation=45, ha='right')
    ax1.set_ylabel("Mean Episode Reward")
    ax1.set_title(f"Final Performance (last {n_last} samples)")
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Length comparison
    if mean_lengths:
        ax2.bar(x, mean_lengths, yerr=std_lengths, capsize=5, alpha=0.8, color='orange')
        ax2.set_xticks(x)
        ax2.set_xticklabels(run_names, rotation=45, ha='right')
        ax2.set_ylabel("Mean Episode Length")
        ax2.set_title(f"Final Episode Length (last {n_last} samples)")
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def export_all_plots(
    run_dirs: List[Path],
    output_dir: Path,
    run_names: Optional[List[str]] = None,
):
    """
    Export all standard plots from multiple runs.
    
    Parameters
    ----------
    run_dirs : list of Path
        List of run directories to analyze.
    output_dir : Path
        Directory to save plots.
    run_names : list of str, optional
        Custom names for runs (default: directory names).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    data = {}
    for i, run_dir in enumerate(run_dirs):
        df = load_progress_csv(run_dir)
        if df is not None:
            name = run_names[i] if run_names else run_dir.name
            data[name] = df
    
    if not data:
        print("Error: No valid data found in any run directory")
        return
    
    print(f"\nLoaded {len(data)} runs:")
    for name in data.keys():
        print(f"  - {name}")
    
    print("\nGenerating plots...")
    
    # Learning curve
    plot_learning_curve(
        data=data,
        output_path=str(output_dir / "learning_curve.png"),
    )
    
    # Episode length
    plot_episode_length(
        data=data,
        output_path=str(output_dir / "episode_length.png"),
    )
    
    # Sample efficiency
    plot_sample_efficiency(
        data=data,
        output_path=str(output_dir / "sample_efficiency.png"),
    )
    
    # Training throughput
    plot_training_throughput(
        data=data,
        output_path=str(output_dir / "training_throughput.png"),
    )
    
    # Final comparison
    plot_final_comparison(
        data=data,
        output_path=str(output_dir / "final_comparison.png"),
    )
    
    print(f"\n✅ All plots exported to: {output_dir}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export publication-ready plots from training logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--runs",
        type=str,
        nargs='+',
        required=True,
        help="Paths to run directories",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs='+',
        default=None,
        help="Custom names for runs (default: use directory names)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots",
        help="Output directory for plots",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    run_dirs = [Path(p) for p in args.runs]
    
    # Validate run directories
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            return
    
    if args.names and len(args.names) != len(run_dirs):
        print(f"Error: Number of names ({len(args.names)}) must match number of runs ({len(run_dirs)})")
        return
    
    export_all_plots(
        run_dirs=run_dirs,
        output_dir=Path(args.output),
        run_names=args.names,
    )


if __name__ == "__main__":
    main()