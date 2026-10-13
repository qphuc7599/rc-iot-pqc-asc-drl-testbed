"""Render the validated security-speed v5 held-out disaster panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "results/paper_recovery_security_speed_v5/heldout_disaster_seed_657111/evaluation.json"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent
COLORS = {
    "PPO deterministic": "#135fbd",
    "PPO stochastic": "#48a2dc",
    "CMAB": "#168a8a",
    "GBA": "#2a9d68",
    "SA": "#7c4dcc",
    "FMG": "#c57f17",
    "OCD-inspired": "#d24d8b",
    "Risk-Prior": "#8a54d6",
    "Random": "#707781",
    "Static-with-valid-refill": "#c93d3d",
}
DISPLAY = {
    "PPO deterministic": "PPO deterministic",
    "PPO stochastic": "PPO stochastic audit",
    "OCD-inspired": "OCD",
    "Static-with-valid-refill": "Static",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "axes.titleweight": "semibold",
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d9dde3",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _smooth(values: list[float], window: int = 25) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    half = window // 2
    padded = np.pad(array, (half, half), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "qrelay-security-speed-v5-heldout-disaster-evaluation-1":
        raise ValueError(f"unexpected schema: {payload.get('schema')!r}")
    validation = json.loads((path.parent / "validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise ValueError("refusing to plot an unvalidated evaluation")
    return payload


def _draw(payload: dict, output: Path) -> None:
    panel = payload["panel"]
    timelines = {
        "PPO deterministic": panel["deterministic_primary"]["mean_tps_timeline"],
        "PPO stochastic": panel["stochastic_audit"]["mean_tps_timeline"],
        **panel["baselines"]["mean_tps_timeline"],
    }
    aggregates = {
        "PPO deterministic": panel["deterministic_primary"]["aggregate"],
        "PPO stochastic": panel["stochastic_audit"]["aggregate"],
        **panel["baselines"]["aggregates"],
    }
    guards = {
        "PPO deterministic": panel["deterministic_primary"]["security_guard"],
        "PPO stochastic": panel["stochastic_audit"]["security_guard"],
        **panel["baselines"]["security_guards"],
    }
    if any(len(values) != 2_000 for values in timelines.values()):
        raise ValueError("timeline length drift")

    _style()
    fig = plt.figure(figsize=(7.2, 5.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.48, 1.0), width_ratios=(1.08, 0.92))
    ax_time = fig.add_subplot(grid[0, :])
    ax_late = fig.add_subplot(grid[1, 0])
    ax_guard = fig.add_subplot(grid[1, 1])

    x = np.arange(2_000)
    ax_time.axvspan(1_000, 1_199, color="#f6c7c7", alpha=0.30, lw=0)
    ax_time.axvspan(1_200, 1_499, color="#f7dfad", alpha=0.28, lw=0)
    ax_time.axvspan(1_500, 1_999, color="#cfe2f5", alpha=0.22, lw=0)
    order = [
        "Static-with-valid-refill",
        "Random",
        "Risk-Prior",
        "OCD-inspired",
        "FMG",
        "SA",
        "GBA",
        "CMAB",
        "PPO stochastic",
        "PPO deterministic",
    ]
    for name in order:
        ppo = name.startswith("PPO")
        strong = name in {"CMAB", "GBA", "SA", "FMG"}
        ax_time.plot(
            x,
            _smooth(timelines[name]),
            color=COLORS[name],
            lw=2.25 if name == "PPO deterministic" else (1.55 if ppo else (1.15 if strong else 0.8)),
            alpha=1.0 if ppo else (0.88 if strong else 0.55),
            ls="--" if name == "PPO stochastic" or (not ppo and not strong) else "-",
            label=DISPLAY.get(name, name),
            zorder=6 if name == "PPO deterministic" else (5 if ppo else 3),
        )
    ax_time.axvline(1_000, color="#9b1c31", ls=(0, (4, 3)), lw=1.35, zorder=7)
    top = max(float(np.max(_smooth(values))) for values in timelines.values())
    ax_time.text(
        1_010,
        top * 0.98,
        "30% permanent node loss + cascading stress",
        color="#8c1830",
        fontsize=7.3,
        weight="semibold",
        va="top",
    )
    ax_time.text(1_100, top * 0.035, "impact", color="#7d5151", ha="center", fontsize=7)
    ax_time.text(1_350, top * 0.035, "damaged", color="#816f45", ha="center", fontsize=7)
    ax_time.text(1_750, top * 0.035, "late recovery", color="#45657f", ha="center", fontsize=7)
    ax_time.set_xlim(0, 1_999)
    ax_time.set_ylim(0, top * 1.05)
    ax_time.set_xticks([0, 500, 1_000, 1_200, 1_500, 1_999])
    ax_time.set_xlabel("Control step")
    ax_time.set_ylabel("Service throughput (TPS)")
    ax_time.set_title("(a) Held-out disaster timeline (25-step moving mean)", loc="left", pad=29)
    ax_time.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        frameon=False,
        handlelength=2.2,
        columnspacing=0.9,
    )

    names = sorted(aggregates, key=lambda name: aggregates[name]["mean_tps_late_1500_1999"])
    late = np.asarray([aggregates[name]["mean_tps_late_1500_1999"] for name in names])
    y = np.arange(len(names))
    ax_late.hlines(y, 0, late, color="#d7dce2", lw=1.1, zorder=1)
    ax_late.scatter(
        late,
        y,
        c=[COLORS[name] for name in names],
        s=[44 if name == "PPO deterministic" else (34 if name == "PPO stochastic" else 23) for name in names],
        marker="o",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    for yi, value in zip(y, late, strict=True):
        ax_late.text(value + max(late) * 0.012, yi, f"{value:.1f}", va="center", fontsize=6.7)
    ax_late.set_yticks(y, [DISPLAY.get(name, name) for name in names])
    ax_late.set_xlim(0, max(late) * 1.23)
    ax_late.set_xlabel("Mean TPS, steps 1500--1999")
    ax_late.set_title("(b) Late-window service", loc="left")
    ax_late.grid(axis="y", visible=False)

    guard_names = list(aggregates)
    failure = np.asarray([100 * aggregates[name]["failure_rate_post_1000_1999"] for name in guard_names])
    exposure = np.asarray([guards[name]["global_max_actual_security_exposure"] for name in guard_names])
    ax_guard.axhline(6, color="#9b1c31", ls=(0, (4, 3)), lw=1.1)
    ax_guard.text(
        1.0,
        6.13,
        r"strict guard $f=6$",
        color="#8c1830",
        ha="left",
        fontsize=6.8,
    )
    for idx, name in enumerate(guard_names):
        ax_guard.scatter(
            failure[idx],
            exposure[idx],
            s=76 if name == "PPO deterministic" else (58 if name == "PPO stochastic" else 27),
            marker="*" if name.startswith("PPO") else "o",
            color=COLORS[name],
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
        ax_guard.annotate(
            DISPLAY.get(name, name),
            (failure[idx], exposure[idx]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.9,
        )
    xmax = max(50.0, float(failure.max()) * 1.12)
    ymin = min(5.5, float(exposure.min()) - 0.5)
    ymax = max(7.0, float(exposure.max()) + 0.8)
    ax_guard.set_xlim(0, xmax)
    ax_guard.set_ylim(ymin, ymax)
    ax_guard.set_xlabel("Post-event zero-service steps (%)")
    ax_guard.set_ylabel("Global max compromised members")
    ax_guard.set_title("(c) Reliability--security guard", loc="left")

    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "security_speed_v5_disaster_timeline.pdf", bbox_inches="tight")
    fig.savefig(output / "security_speed_v5_disaster_timeline.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = _load(args.input.resolve())
    _draw(payload, args.output.resolve())
    print(args.output.resolve() / "security_speed_v5_disaster_timeline.png")


if __name__ == "__main__":
    main()
