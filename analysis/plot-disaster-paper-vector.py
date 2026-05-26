#!/usr/bin/env python3
"""
Create the paper-ready vector PDF for Experiment 3.

This script intentionally keeps the figure as a single full-width panel:
the table in paper.tex already carries the exact numerical summary, while this
figure should show the trajectory-level story without visual clutter.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_CANDIDATES = [
    RESULTS_DIR / "drl_pooled" / "disaster_results.json",
    RESULTS_DIR / "disaster_results.json",
]
POOLED_RESULTS_DIR = RESULTS_DIR / "drl_pooled"
PAPER_FIG_DIR = ROOT / "file_project" / "figures"

STRATEGIES = ["ppo", "sa", "gba", "ocd", "random", "static"]
LABELS = {
    "ppo": "PPO (ours)",
    "sa": "SA",
    "gba": "GBA",
    "ocd": "OCD",
    "random": "Random",
    "static": "Static",
}
COLORS = {
    "ppo": "#2563eb",
    "sa": "#7c3aed",
    "gba": "#059669",
    "ocd": "#d97706",
    "random": "#6b7280",
    "static": "#dc2626",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def load_data(path: Path | None = None) -> dict:
    if path is not None:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["_source_path"] = path.as_posix()
        return data
    for path in RESULTS_CANDIDATES:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            data["_source_path"] = path.as_posix()
            return data
    searched = ", ".join(path.as_posix() for path in RESULTS_CANDIDATES)
    raise FileNotFoundError(f"No disaster_results.json found. Searched: {searched}")


def runs_to_matrix(runs: list[list[dict]], key: str) -> np.ndarray:
    max_len = max(len(run) for run in runs)
    matrix = np.full((len(runs), max_len), np.nan, dtype=float)
    for i, run in enumerate(runs):
        for j, item in enumerate(run):
            matrix[i, j] = item[key]
    return matrix


def smooth_edge(values: np.ndarray, window: int = 15) -> np.ndarray:
    """Centered moving average without zero-padding edge artifacts."""
    if window <= 1 or values.size < window:
        return values
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def plot_paper_timeline(
    data: dict,
    output_dir: Path = POOLED_RESULTS_DIR,
    paper_copy: bool = True,
    title_suffix: str = "",
) -> list[Path]:
    all_runs = data["all_runs"]
    disaster_step = int(data["config"]["disaster_step"])
    seeds = data["config"].get("seeds", [])

    fig, ax = plt.subplots(figsize=(7.2, 3.05))

    active_strategies = [s for s in STRATEGIES if s in all_runs]
    for strat in active_strategies:
        tps = runs_to_matrix(all_runs[strat], "tps")
        mean = np.nanmean(tps, axis=0)
        std = np.nanstd(tps, axis=0)
        steps = np.arange(mean.size)

        mean_s = smooth_edge(mean, window=15)
        std_s = smooth_edge(std, window=15)
        lower = np.maximum(0.0, mean_s - std_s)
        upper = mean_s + std_s

        is_ppo = strat == "ppo"
        ax.plot(
            steps,
            mean_s,
            color=COLORS[strat],
            label=LABELS[strat],
            linewidth=2.3 if is_ppo else 1.35,
            alpha=0.98 if is_ppo else 0.92,
            zorder=8 if is_ppo else 4,
        )
        ax.fill_between(
            steps,
            lower,
            upper,
            color=COLORS[strat],
            alpha=0.13 if is_ppo else 0.07,
            linewidth=0,
            zorder=2 if is_ppo else 1,
        )

    # Single-committee bandwidth ceiling used by the DRL environment.
    ax.axhline(982, color="#64748b", linestyle=(0, (2, 2)), linewidth=0.8, alpha=0.75)
    ax.text(1590, 1000, "single-channel ceiling", color="#475569", fontsize=7.5)

    ax.axvline(disaster_step, color="#ef4444", linestyle="--", linewidth=1.1, alpha=0.9)
    ax.text(
        disaster_step + 18,
        865,
        "30% node kill\n(t=1000)",
        color="#dc2626",
        fontsize=7.8,
        fontweight="bold",
        va="top",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 1.2,
        },
    )

    ax.set_xlim(0, 1999)
    ax.set_ylim(0, 1080)
    ax.set_xlabel("Control step")
    ax.set_ylabel("Throughput (TPS)")
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_yticks([0, 200, 400, 600, 800, 1000])
    ax.grid(True, which="major", color="#94a3b8", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=min(6, len(active_strategies)),
        frameon=False,
        handlelength=1.8,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    for line in legend.get_lines():
        line.set_linewidth(2.2)

    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.17, top=0.77)

    outputs = [
        output_dir / "disaster_6baselines_timeline_paper.pdf",
        output_dir / "disaster_6baselines_timeline_paper.png",
    ]
    if paper_copy:
        outputs.extend(
            [
                RESULTS_DIR / "disaster_6baselines_timeline_paper.pdf",
                RESULTS_DIR / "disaster_6baselines_timeline_paper.png",
                PAPER_FIG_DIR / "disaster_recovery_timeline.pdf",
                PAPER_FIG_DIR / "disaster_recovery_timeline.png",
            ]
        )
    for out in outputs:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".png":
            fig.savefig(out, dpi=300)
        else:
            fig.savefig(
                out,
                metadata={
                    "Title": "DRL disaster recovery trajectory",
                    "Subject": f"Mean +/- std across {len(seeds)} seeds",
                },
            )

    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disaster-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=POOLED_RESULTS_DIR)
    parser.add_argument("--title-suffix", default="")
    parser.add_argument(
        "--no-paper-copy",
        action="store_true",
        help="do not update results/disaster_6baselines_timeline_paper.* or file_project/figures/disaster_recovery_timeline.*",
    )
    args = parser.parse_args()

    data = load_data(args.disaster_json)
    outputs = plot_paper_timeline(
        data,
        output_dir=args.output_dir,
        paper_copy=not args.no_paper_copy,
        title_suffix=args.title_suffix,
    )
    source = data.get("_source_path", "unknown source")
    print(f"Generated paper-ready disaster trajectory from {source}:")
    for out in outputs:
        print(f"  {out}")


if __name__ == "__main__":
    main()
