#!/usr/bin/env python3
"""
Plot disaster test results — 6 baselines (PPO, SA, GBA, OCD, Random, Static)
Reads disaster_results.json, creates 4 publication-quality figures.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Config ──
RESULTS_FILE = Path(__file__).parent.parent / "results" / "disaster_results.json"
OUT_DIR = Path(__file__).parent.parent / "results"

COLORS = {
    "ppo":    "#2563eb",  # blue
    "sa":     "#7c3aed",  # purple
    "gba":    "#059669",  # emerald
    "ocd":    "#d97706",  # amber
    "random": "#6b7280",  # gray
    "static": "#dc2626",  # red
}
LABELS = {
    "ppo": "PPO (Ours)", "sa": "SA", "gba": "GBA",
    "ocd": "OCD", "random": "Random", "static": "Static",
}
STRATEGIES = ["ppo", "sa", "gba", "ocd", "random", "static"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "Arial"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def load_data():
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    return data


def fig1_tps_timeline(data):
    """TPS trajectory over time — all 6 baselines, mean ± std across seeds."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    all_runs = data["all_runs"]
    disaster_step = data["config"]["disaster_step"]
    
    for strat in STRATEGIES:
        runs = all_runs[strat]
        # Pad to same length
        max_len = max(len(r) for r in runs)
        tps_matrix = np.full((len(runs), max_len), np.nan)
        for i, run in enumerate(runs):
            for j, h in enumerate(run):
                tps_matrix[i, j] = h["tps"]
        
        mean = np.nanmean(tps_matrix, axis=0)
        std = np.nanstd(tps_matrix, axis=0)
        steps = np.arange(max_len)
        
        # Smooth for readability
        window = 15
        if len(mean) > window:
            kernel = np.ones(window) / window
            mean_s = np.convolve(mean, kernel, mode='same')
            std_s = np.convolve(std, kernel, mode='same')
        else:
            mean_s, std_s = mean, std
        
        ax.plot(steps, mean_s, color=COLORS[strat], label=LABELS[strat],
                linewidth=2.5 if strat == "ppo" else 1.8,
                zorder=10 if strat == "ppo" else 5)
        lower = np.maximum(0.0, mean_s - std_s)
        ax.fill_between(steps, lower, mean_s + std_s,
                        alpha=0.12, color=COLORS[strat])
    
    # Disaster marker
    ax.axvline(disaster_step, color="#ef4444", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(disaster_step + 15, ax.get_ylim()[1] * 0.95,
            f"Disaster\n(t={disaster_step})",
            color="#ef4444", fontsize=10, fontweight="bold", va="top")
    
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("TPS", fontsize=12)
    ax.set_title("Figure 7: TPS Trajectory Under Disaster — 6 Baselines (mean ± σ, 10 seeds)")
    ax.legend(loc="lower left", framealpha=0.9, ncol=3, fontsize=10)
    ax.set_xlim(0, max_len)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    out = OUT_DIR / "disaster_6baselines_timeline.png"
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)


def fig2_bar_comparison(data):
    """Bar chart: Pre-TPS, Post-TPS, AUC for all 6 baselines."""
    summary = data["summary"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    metrics = [
        ("pre_tps", "Pre-Disaster TPS", "TPS"),
        ("post_tps", "Post-Disaster TPS (avg)", "TPS"),
        ("auc", "AUC Ratio (higher = better)", "AUC"),
    ]
    
    for ax, (key, title, ylabel) in zip(axes, metrics):
        means = [np.mean(summary[s][key]) for s in STRATEGIES]
        stds = [np.std(summary[s][key]) for s in STRATEGIES]
        colors = [COLORS[s] for s in STRATEGIES]
        labels = [LABELS[s] for s in STRATEGIES]
        
        bars = ax.bar(labels, means, yerr=stds, color=colors, 
                      edgecolor="white", linewidth=1.2, capsize=4,
                      error_kw={"linewidth": 1.5})
        
        # Value labels
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 5,
                    f"{m:.0f}" if key != "auc" else f"{m:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis='x', rotation=30)
    
    fig.suptitle("Figure 8: Disaster Recovery — 6-Baseline Comparison (10 seeds)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.subplots_adjust(top=0.78, bottom=0.28, wspace=0.35)
    out = OUT_DIR / "disaster_6baselines_bars.png"
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)


def fig3_recovery_detail(data):
    """Zoomed view: post-disaster first 500 steps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    all_runs = data["all_runs"]
    disaster_step = data["config"]["disaster_step"]
    
    # Left: TPS zoom post-disaster
    for strat in STRATEGIES:
        runs = all_runs[strat]
        max_len = max(len(r) for r in runs)
        tps_matrix = np.full((len(runs), max_len), np.nan)
        for i, run in enumerate(runs):
            for j, h in enumerate(run):
                tps_matrix[i, j] = h["tps"]
        
        # Post-disaster window
        post_start = disaster_step
        post_end = min(disaster_step + 500, max_len)
        post_tps = tps_matrix[:, post_start:post_end]
        mean = np.nanmean(post_tps, axis=0)
        std = np.nanstd(post_tps, axis=0)
        steps = np.arange(len(mean))
        
        window = 10
        if len(mean) > window:
            kernel = np.ones(window) / window
            mean_s = np.convolve(mean, kernel, mode='same')
        else:
            mean_s = mean
        
        ax1.plot(steps, mean_s, color=COLORS[strat], label=LABELS[strat],
                linewidth=2.5 if strat == "ppo" else 1.5)
    
    ax1.set_xlabel("Steps After Disaster")
    ax1.set_ylabel("TPS")
    ax1.set_title("Post-Disaster TPS Recovery (500 steps)")
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Right: Alive nodes
    for strat in STRATEGIES:
        runs = all_runs[strat]
        max_len = max(len(r) for r in runs)
        alive_matrix = np.full((len(runs), max_len), np.nan)
        for i, run in enumerate(runs):
            for j, h in enumerate(run):
                alive_matrix[i, j] = h["alive"]
        
        post_start = disaster_step
        post_end = min(disaster_step + 500, max_len)
        post_alive = alive_matrix[:, post_start:post_end]
        mean = np.nanmean(post_alive, axis=0)
        steps = np.arange(len(mean))
        
        ax2.plot(steps, mean, color=COLORS[strat], label=LABELS[strat],
                linewidth=2.5 if strat == "ppo" else 1.5)
    
    ax2.set_xlabel("Steps After Disaster")
    ax2.set_ylabel("Alive Nodes")
    ax2.set_title("Node Survival Post-Disaster")
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle("Figure 9: Post-Disaster Recovery Detail", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "disaster_6baselines_recovery.png"
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)


def fig4_summary_table(data):
    """Publication-quality summary table as image."""
    summary = data["summary"]
    
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    
    headers = ["Strategy", "Pre-TPS", "Post-TPS", "TPS@t+100", "TPS@t+500", "AUC", "Alive"]
    
    rows = []
    for strat in STRATEGIES:
        s = summary[strat]
        rows.append([
            LABELS[strat],
            f"{np.mean(s['pre_tps']):.1f} ± {np.std(s['pre_tps']):.1f}",
            f"{np.mean(s['post_tps']):.1f} ± {np.std(s['post_tps']):.1f}",
            f"{np.mean(s['tps_100']):.1f} ± {np.std(s['tps_100']):.1f}",
            f"{np.mean(s['tps_500']):.1f} ± {np.std(s['tps_500']):.1f}",
            f"{np.mean(s['auc']):.3f} ± {np.std(s['auc']):.3f}",
            f"{np.mean(s['alive']):.1f} ± {np.std(s['alive']):.1f}",
        ])
    
    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#2563eb']*len(headers))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    
    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_facecolor('#1e3a5f')
    
    # Style rows
    for i, strat in enumerate(STRATEGIES):
        color = COLORS[strat]
        for j in range(len(headers)):
            cell = table[i+1, j]
            if j == 0:
                cell.set_text_props(fontweight='bold', color=color)
            # Highlight best AUC
            if j == 5:  # AUC column
                auc_val = np.mean(summary[strat]['auc'])
                if auc_val >= 0.99:
                    cell.set_facecolor('#dcfce7')  # light green
                elif auc_val < 0.60:
                    cell.set_facecolor('#fee2e2')  # light red
    
    ax.set_title("Table 8: Disaster Recovery — Statistical Summary (10 seeds, kill=30%, byz=20%)",
                 fontsize=13, fontweight="bold", pad=20)
    
    fig.tight_layout()
    out = OUT_DIR / "disaster_6baselines_table.png"
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Loading disaster results...")
    data = load_data()
    
    config = data["config"]
    print(f"  Kill: {config['kill_ratio']*100:.0f}%, Disaster step: {config['disaster_step']}")
    print(f"  Seeds: {config['seeds']}")
    print(f"  Strategies: {list(data['all_runs'].keys())}")
    
    fig1_tps_timeline(data)
    fig2_bar_comparison(data)
    fig3_recovery_detail(data)
    fig4_summary_table(data)
    
    print("\nAll 4 figures saved to results/")
