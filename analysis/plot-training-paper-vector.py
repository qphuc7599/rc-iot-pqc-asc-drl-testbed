#!/usr/bin/env python3
"""
Create a paper-ready vector PDF for PPO training convergence.

This version preserves the original wide two-panel diagnostic layout preferred
for the paper, but generates it directly from training_log.json as vector PDF
for a two-column `figure*` in the Elsevier layout.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOG_CANDIDATES = [
    ROOT / "drl" / "models_robust" / "training_log.json",
    ROOT / "drl" / "models" / "training_log.json",
]
RESULTS_DIR = ROOT / "results"
PAPER_FIG_DIR = ROOT / "file_project" / "figures"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def load_training_log() -> tuple[dict, Path]:
    for path in LOG_CANDIDATES:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f), path
    raise FileNotFoundError("No training_log.json found in drl/models_robust or drl/models")


def moving_average(values: np.ndarray, window: int = 50) -> np.ndarray:
    if values.size == 0:
        return values
    out = np.empty_like(values, dtype=float)
    csum = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    for i in range(values.size):
        start = max(0, i - window + 1)
        out[i] = (csum[i + 1] - csum[start]) / (i - start + 1)
    return out


def plot_training(log: dict, source_path: Path) -> list[Path]:
    rewards = np.asarray(log.get("rewards", []), dtype=float)
    tps = np.asarray(log.get("tps", []), dtype=float)
    alive = np.asarray(log.get("alive", []), dtype=float)
    if rewards.size == 0:
        raise ValueError(f"{source_path} does not contain a non-empty rewards list")

    x = np.arange(1, rewards.size + 1)
    reward_ma = np.asarray(log.get("avg50_rewards", []), dtype=float)
    if reward_ma.size != rewards.size:
        reward_ma = moving_average(rewards, 50)
    fig, (ax_reward, ax_tps) = plt.subplots(1, 2, figsize=(7.2, 2.75))
    fig.suptitle("PPO Training -- RC-IoT Committee Selection", fontsize=10.0, y=1.02)

    ax_reward.plot(x, rewards, color="#a5b4fc", alpha=0.26, linewidth=0.55, label="Per-update")
    ax_reward.plot(x, reward_ma, color="#dc2626", linewidth=1.45, label="Moving avg (50)")
    ax_reward.axhline(0, color="#64748b", linewidth=0.6, linestyle=(0, (2, 2)), alpha=0.8)
    ax_reward.set_title("Training Reward")
    ax_reward.set_xlabel("PPO update")
    ax_reward.set_ylabel("Reward")
    ax_reward.set_xlim(1, rewards.size)
    ax_reward.set_xticks([1, 500, 1000, 1500, 2000, 2500, 3000])
    ax_reward.set_ylim(-24500, 7800)
    ax_reward.grid(True, color="#94a3b8", alpha=0.23, linewidth=0.55)
    ax_reward.legend(loc="upper left", frameon=True, framealpha=0.82, handlelength=1.7)

    if tps.size:
        ax_tps.plot(x, tps, color="#16a34a", alpha=0.55, linewidth=0.72)
        ax_tps.set_title("TPS & Node Survival")
        ax_tps.set_xlabel("PPO update")
        ax_tps.set_ylabel("TPS")
        ax_tps.set_xlim(1, rewards.size)
        ax_tps.set_xticks([1, 500, 1000, 1500, 2000, 2500, 3000])
        ax_tps.set_ylim(-50, 1030)
        ax_tps.tick_params(axis="y", colors="#15803d")
        ax_tps.yaxis.label.set_color("#15803d")

    if alive.size:
        ax_alive = ax_tps.twinx()
        ax_alive.plot(x, alive, color="#f59e0b", alpha=0.70, linewidth=0.72)
        ax_alive.set_ylabel("Alive nodes")
        ax_alive.set_ylim(48, 81.5)
        ax_alive.tick_params(axis="y", colors="#d97706")
        ax_alive.yaxis.label.set_color("#d97706")

    ax_tps.grid(True, color="#94a3b8", alpha=0.23, linewidth=0.55)
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.16, top=0.80, wspace=0.22)

    outputs = [
        RESULTS_DIR / "drl_training_paper.pdf",
        RESULTS_DIR / "drl_training_paper.png",
        PAPER_FIG_DIR / "training_convergence.pdf",
        PAPER_FIG_DIR / "training_convergence.png",
    ]
    for out in outputs:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".png":
            fig.savefig(out, dpi=300)
        else:
            fig.savefig(
                out,
                metadata={
                    "Title": "PPO training convergence",
                    "Subject": f"Generated from {source_path.as_posix()}",
                },
            )
    plt.close(fig)
    return outputs


def main() -> None:
    log, source_path = load_training_log()
    outputs = plot_training(log, source_path)
    print(f"Generated paper-ready training convergence from {source_path}:")
    for out in outputs:
        print(f"  {out}")


if __name__ == "__main__":
    main()
