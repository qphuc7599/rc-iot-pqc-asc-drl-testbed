#!/usr/bin/env python3
"""
plot-drl-results.py — Publication-quality DRL plots with multi-seed statistics.

Usage:
  python3 analysis/plot-drl-results.py
"""

import json
import os
import sys
import statistics
import numpy as np

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def print_training_summary(data):
    if not data:
        return
    rewards = data.get('rewards', [])
    if not rewards:
        return

    print("=" * 65)
    print("  DRL Training Summary")
    print("=" * 65)
    print(f"  Episodes:    {len(rewards)}")
    print(f"  Final Reward: {rewards[-1]:.1f}")
    print(f"  Best Reward:  {max(rewards):.1f}")
    print(f"  Avg (last 50): {statistics.mean(rewards[-50:]):.1f}")
    print("=" * 65)


def plot_training(data, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed")
        return

    rewards = data.get('rewards', [])
    if not rewards:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PPO Training — RC-IoT Committee Selection', fontsize=14)

    # Reward curve
    axes[0].plot(rewards, alpha=0.3, color='blue', label='Per-episode')
    # Moving average
    window = 50
    if len(rewards) > window:
        ma = [statistics.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
        axes[0].plot(ma, color='red', linewidth=2, label=f'Moving avg ({window})')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # TPS + Node Survival from training history/logged episode summaries
    history = data.get('history', [])
    tps_vals = data.get('tps', [])
    alive_vals = data.get('alive', [])
    if history:
        x_vals = range(len(history))
        tps_vals = [h.get('tps', 0) for h in history]
        alive_vals = [h.get('alive', 0) for h in history]
        x_label = 'Step'
    else:
        x_vals = range(len(tps_vals))
        x_label = 'Episode'

    if tps_vals and alive_vals:
        ax1 = axes[1]
        ax2 = ax1.twinx()
        ax1.plot(x_vals, tps_vals, color='green', alpha=0.55, linewidth=0.8)
        ax2.plot(x_vals, alive_vals, color='orange', alpha=0.7, linewidth=0.8)

        ax1.set_xlabel(x_label)
        ax1.set_ylabel('TPS', color='green')
        ax2.set_ylabel('Alive Nodes', color='orange')
        axes[1].set_title('TPS & Node Survival')
    else:
        axes[1].axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, 'drl_training.png')
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()


def plot_disaster(data, output_dir):
    """Plot multi-seed disaster results with mean ± std shaded bands."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Detect format: new multi-seed or old single-seed
    if "all_runs" in data:
        all_runs = data["all_runs"]
        config = data.get("config", {})
        disaster_step = config.get("disaster_step", 1000)
        total_steps = config.get("total_steps", 2000)
        num_seeds = len(next(iter(all_runs.values())))
    else:
        # Legacy single-seed format
        all_runs = {k: [v] for k, v in data.items()}
        disaster_step = 300
        total_steps = 1500
        num_seeds = 1

    colors = {
        'ppo': '#2196F3', 'sa': '#4CAF50', 'gba': '#FF9800',
        'ocd': '#9C27B0', 'random': '#FF5722', 'static': '#9E9E9E'
    }
    strategy_order = ['ppo', 'sa', 'gba', 'ocd', 'random', 'static']
    strategies_present = [s for s in strategy_order if s in all_runs]

    # --- Figure 1: TPS + Node Survival (2 subplots) ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(
        f'Disaster Recovery: Six Committee-Selection Strategies\n'
        f'(kill=30%, disaster@step={disaster_step}, {num_seeds} seeds, mean±std)',
        fontsize=13
    )

    for strategy in strategies_present:
        runs = all_runs[strategy]
        # Pad runs to same length
        max_len = max(len(r) for r in runs)

        tps_matrix = np.full((len(runs), max_len), np.nan)
        alive_matrix = np.full((len(runs), max_len), np.nan)

        for ri, run in enumerate(runs):
            for si, step_data in enumerate(run):
                tps_matrix[ri, si] = step_data['tps']
                alive_matrix[ri, si] = step_data['alive']

        steps = np.arange(max_len)
        tps_mean = np.nanmean(tps_matrix, axis=0)
        tps_std = np.nanstd(tps_matrix, axis=0)
        alive_mean = np.nanmean(alive_matrix, axis=0)
        alive_std = np.nanstd(alive_matrix, axis=0)

        # Smooth for readability (window=10)
        def smooth(arr, w=10):
            kernel = np.ones(w) / w
            return np.convolve(arr, kernel, mode='same')

        tps_mean_s = smooth(tps_mean)
        tps_std_s = smooth(tps_std)
        alive_mean_s = smooth(alive_mean)
        alive_std_s = smooth(alive_std)

        c = colors.get(strategy, 'gray')

        # TPS plot
        axes[0].plot(steps, tps_mean_s, color=c, label=strategy.upper(),
                    linewidth=2)
        if num_seeds > 1:
            axes[0].fill_between(steps,
                                np.maximum(0.0, tps_mean_s - tps_std_s),
                                tps_mean_s + tps_std_s,
                                color=c, alpha=0.15)

        # Alive plot
        axes[1].plot(steps, alive_mean_s, color=c, label=strategy.upper(),
                    linewidth=2)
        if num_seeds > 1:
            axes[1].fill_between(steps,
                                np.maximum(0.0, alive_mean_s - alive_std_s),
                                alive_mean_s + alive_std_s,
                                color=c, alpha=0.15)

    for ax in axes:
        ax.axvline(x=disaster_step, color='red', linestyle='--',
                  alpha=0.6, label=f'Disaster (t={disaster_step})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('TPS (mean ± std)')
    axes[0].set_title('Transaction Throughput')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Alive Nodes (mean ± std)')
    axes[1].set_title('Node Survival')

    plt.tight_layout()
    path = os.path.join(output_dir, 'drl_disaster_recovery.png')
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()

    # --- Figure 2: Summary Table as image ---
    if "summary" in data:
        summary = data["summary"]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        ax.set_title(
            f'Statistical Comparison ({num_seeds} seeds, kill={config.get("kill_ratio",0.3)*100:.0f}%)',
            fontsize=13, pad=20
        )

        col_labels = ['Strategy', 'Pre-TPS', 'Post-TPS', 'Rcv90 (steps)',
                      'Final Alive', 'Imm. Drop']
        table_data = []
        for strat in strategies_present:
            s = summary[strat]
            rcv_key = 'recovery90' if 'recovery90' in s else 'recovery'
            drop_key = 'imm_drop' if 'imm_drop' in s else None
            row = [
                strat.upper(),
                f"{np.mean(s['pre_tps']):.1f} ± {np.std(s['pre_tps']):.1f}",
                f"{np.mean(s['post_tps']):.1f} ± {np.std(s['post_tps']):.1f}",
                f"{np.mean(s[rcv_key]):.0f} ± {np.std(s[rcv_key]):.0f}",
                f"{np.mean(s['alive']):.1f} ± {np.std(s['alive']):.1f}",
                f"{np.mean(s[drop_key]):.1f} ± {np.std(s[drop_key]):.1f}" if drop_key else "N/A",
            ]
            table_data.append(row)

        table = ax.table(cellText=table_data, colLabels=col_labels,
                        loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # Color header
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#E3F2FD')
            table[0, j].set_text_props(fontweight='bold')

        # Color first column
        for i in range(len(table_data)):
            c = colors.get(strategies_present[i], 'gray')
            table[i+1, 0].set_facecolor(c + '30')  # 30 = alpha hex
            table[i+1, 0].set_text_props(fontweight='bold')

        plt.tight_layout()
        path = os.path.join(output_dir, 'drl_disaster_table.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()


def main():
    results_dir = "results"

    # Training results. Prefer the robust-profile run used for Experiment 3;
    # fall back to the legacy default path for older artifacts.
    training = load_json("drl/models_robust/training_log.json")
    if not training:
        training = load_json("drl/models/training_log.json")
    if training:
        print_training_summary(training)
        plot_training(training, results_dir)

    # Disaster results
    disaster = load_json(os.path.join(results_dir, "disaster_results.json"))
    if disaster:
        plot_disaster(disaster, results_dir)

    if not training and not disaster:
        print("[ERROR] No results found.")
        print("  Train:    python3 drl/ppo_agent.py --mode train --sim")
        print("  Disaster: python3 drl/ppo_agent.py --mode disaster --model drl/models/ppo_best.pt")

if __name__ == '__main__':
    main()
