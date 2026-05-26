#!/usr/bin/env python3
"""Create a paper-quality DRL fault-sensitivity figure.

The generic heatmap produced by plot-drl-sensitivity.py is useful for quick
inspection, but it is visually too heavy for the manuscript. This script
creates a cleaner two-panel figure:

  1. PPO post-disaster throughput across the fault grid.
  2. PPO/GBA post-disaster throughput advantage on the same grid.

Input:
  results/drl_gumbel_pooled/drl_sensitivity.json

Output:
  <input-dir>/drl_sensitivity_paper.{png,pdf}
"""

import argparse
import json
import os


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows", [])
    if not rows:
        raise SystemExit(f"[ERROR] No rows found in {path}")
    return rows


def make_grid(rows, strategy, metric):
    selected = [r for r in rows if r.get("strategy") == strategy]
    kill_vals = sorted({float(r["kill_ratio"]) for r in selected})
    byz_vals = sorted({float(r["byz_ratio"]) for r in selected})
    lookup = {
        (float(r["kill_ratio"]), float(r["byz_ratio"])): float(r[metric])
        for r in selected
    }
    grid = [[lookup[(k, b)] for b in byz_vals] for k in kill_vals]
    return kill_vals, byz_vals, grid


def draw(path, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise SystemExit(f"[ERROR] matplotlib is required: {exc}") from exc

    rows = load_rows(path)
    kill_vals, byz_vals, ppo_post = make_grid(rows, "ppo", "post_tps_mean")
    _, _, ppo_ret = make_grid(rows, "ppo", "retention_mean")
    _, _, gba_post = make_grid(rows, "gba", "post_tps_mean")

    ppo_post = np.array(ppo_post)
    ppo_ret = np.array(ppo_ret) * 100.0
    gba_post = np.array(gba_post)
    ratio = ppo_post / np.maximum(gba_post, 1e-9)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    throughput_cmap = LinearSegmentedColormap.from_list(
        "clean_throughput",
        ["#f6f7f2", "#d8ead1", "#9fd3a7", "#4f9f74", "#17664b"],
    )
    advantage_cmap = LinearSegmentedColormap.from_list(
        "clean_advantage",
        ["#f8f5ef", "#ead7b2", "#d99b64", "#be5a48", "#7a2e3a"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), constrained_layout=True)

    panels = [
        (
            axes[0],
            ppo_post,
            throughput_cmap,
            "Post-disaster TPS",
            lambda v, i, j: f"{v:.0f}\n{ppo_ret[i, j]:.1f}%",
            0,
        ),
        (
            axes[1],
            ratio,
            advantage_cmap,
            "PPO / GBA post-TPS",
            lambda v, i, j: f"{v:.2f}x",
            1,
        ),
    ]

    for ax, arr, cmap, title, label_fn, panel_idx in panels:
        im = ax.imshow(arr, cmap=cmap, aspect="auto")
        ax.set_title(title, pad=6)
        ax.set_xticks(range(len(byz_vals)))
        ax.set_xticklabels([f"{b * 100:.0f}%" for b in byz_vals])
        ax.set_yticks(range(len(kill_vals)))
        ax.set_yticklabels([f"{k * 100:.0f}%" for k in kill_vals])
        ax.set_xlabel("Active Byzantine nodes")
        if panel_idx == 0:
            ax.set_ylabel("Physically killed nodes")
        else:
            ax.set_ylabel("")

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(byz_vals), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(kill_vals), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(length=0)

        threshold = arr.min() + 0.58 * (arr.max() - arr.min())
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                color = "white" if arr[i, j] >= threshold else "#1b1b1b"
                ax.text(
                    j,
                    i,
                    label_fn(arr[i, j], i, j),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8.2,
                    linespacing=0.95,
                )

        # Highlight the main paper operating point: 30% kill, 20% Byzantine.
        main_i = kill_vals.index(0.30)
        main_j = byz_vals.index(0.20)
        ax.add_patch(
            Rectangle(
                (main_j - 0.5, main_i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#1f2937",
                linewidth=1.8,
            )
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=7.5, length=0)

    fig.text(
        0.02,
        -0.025,
        "Highlighted cell: main evaluation setting (30% physical kill, 20% active Byzantine).",
        fontsize=8,
        color="#333333",
    )

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(output_dir, f"drl_sensitivity_paper.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.path.join("results", "drl_gumbel_pooled", "drl_sensitivity.json"),
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    draw(args.input, output_dir)


if __name__ == "__main__":
    main()
