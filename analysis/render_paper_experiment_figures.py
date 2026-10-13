#!/usr/bin/env python3
"""Render the three result figures used by the current Q-Relay manuscript.

The renderer reads only the validated, frozen JSON artifacts used by
``paper_forcus_form/paper_revison.tex``.  It deliberately uses high-contrast
hazard encoding only where the data define a boundary (loss of quorum or
security exposure above f); saturated colour is not used to invent severity.

Outputs are vector PDFs for the manuscript and high-resolution PNG previews.
The accompanying manifest binds every output to its source artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch, PathPatch, RegularPolygon
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
import numpy as np
import scienceplots  # noqa: F401  # registers the SciencePlots style sheets
from adjustText import adjust_text


ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURES = (
    ROOT
    / "file_project"
    / "paper_file"
    / "_2026__Elsevier_IoT_HoangPhuc"
    / "paper_forcus_form"
    / "figures"
)
PREVIEW_DIR = ROOT / "results" / "paper_figure_render_high_impact_v2"

THRESHOLD_RUN = ROOT / "results" / "paper_recovery_threshold_v3" / "registered_run.json"
THRESHOLD_VALIDATION = ROOT / "results" / "paper_recovery_threshold_v3" / "validation.json"
SLA_SUMMARY = ROOT / "results" / "stage_bc" / "sla_sweep" / "sla_sweep_summary.json"
DRL_DIR = ROOT / "results" / "paper_recovery_security_speed_v5" / "heldout_disaster_seed_657111"
DRL_EVALUATION = DRL_DIR / "evaluation.json"
DRL_VALIDATION = DRL_DIR / "validation.json"


# Color-blind-safe, print-conscious palette.  Crimson is reserved for a real
# fail/guard boundary; royal blue is reserved for the deployed PPO policy.
INK = "#17202A"
MUTED = "#647181"
GRID = "#D6DEE8"
PPO = "#0057B8"
PPO_LIGHT = "#DCEBFF"
COMPARATOR = "#E07A16"
TEAL = "#007F73"
PURPLE = "#6941C6"
HAZARD = "#B4232D"
HAZARD_LIGHT = "#FBE3E5"
SAFE = "#087A55"
SAFE_LIGHT = "#E1F3EB"
NEUTRAL = "#7A8491"
NEUTRAL_LIGHT = "#E8ECF1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _style() -> None:
    # SciencePlots supplies the publication/IEEE geometry and high-visibility
    # colour defaults.  The explicit overrides below keep the manuscript's
    # established typography and reserve crimson for measured hazards only.
    plt.style.use(["science", "ieee", "no-latex", "high-vis"])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.3,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.4,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.72,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.fontsize": 6.4,
            "legend.frameon": False,
            "grid.color": GRID,
            "grid.linewidth": 0.48,
            "grid.alpha": 0.88,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.constrained_layout.w_pad": 2.0 / 72.0,
            "figure.constrained_layout.h_pad": 2.0 / 72.0,
            "figure.constrained_layout.wspace": 0.045,
            "figure.constrained_layout.hspace": 0.035,
        }
    )


def _panel_title(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(f"{label} {title}", loc="left", pad=5.0, color=INK)


def _quiet_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.65, length=2.8)


def _halo(artist: Any, color: str, linewidth: float = 3.0, alpha: float = 0.25) -> None:
    """Add a restrained print-safe glow without rasterising the artist."""

    artist.set_path_effects(
        [pe.Stroke(linewidth=linewidth, foreground=color, alpha=alpha), pe.Normal()]
    )


def _outline_text(artist: Any, linewidth: float = 1.4) -> None:
    artist.set_path_effects([pe.Stroke(linewidth=linewidth, foreground="white"), pe.Normal()])


def _warning_icon(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    size: float = 0.046,
    transform: Any | None = None,
) -> None:
    """Draw a compact vector warning icon in axes or data coordinates."""

    transform = ax.transAxes if transform is None else transform
    triangle = RegularPolygon(
        (x, y),
        numVertices=3,
        radius=size,
        orientation=np.pi / 2.0,
        transform=transform,
        facecolor=HAZARD,
        edgecolor="white",
        linewidth=0.55,
        zorder=20,
        clip_on=False,
    )
    _halo(triangle, HAZARD_LIGHT, linewidth=3.5, alpha=0.55)
    ax.add_patch(triangle)
    ax.text(
        x,
        y - size * 0.08,
        "!",
        transform=transform,
        ha="center",
        va="center",
        color="white",
        fontsize=5.0,
        weight="bold",
        zorder=21,
        clip_on=False,
    )


def _shield_icon(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    size: float = 0.045,
    transform: Any | None = None,
    color: str = SAFE,
) -> None:
    """Draw a small vector shield with a check mark."""

    transform = ax.transAxes if transform is None else transform
    shield_path = MplPath(
        [
            (0.0, 1.0),
            (0.82, 0.70),
            (0.68, -0.22),
            (0.0, -1.0),
            (-0.68, -0.22),
            (-0.82, 0.70),
            (0.0, 1.0),
        ],
        [
            MplPath.MOVETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.CLOSEPOLY,
        ],
    )
    icon_transform = Affine2D().scale(size).translate(x, y) + transform
    shield = PathPatch(
        shield_path,
        transform=icon_transform,
        facecolor=color,
        edgecolor="white",
        linewidth=0.55,
        zorder=20,
        clip_on=False,
    )
    _halo(shield, SAFE_LIGHT, linewidth=3.5, alpha=0.65)
    ax.add_patch(shield)
    check = PathPatch(
        MplPath(
            [(-0.36, 0.02), (-0.08, -0.28), (0.46, 0.34)],
            [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO],
        ),
        transform=icon_transform,
        facecolor="none",
        edgecolor="white",
        linewidth=0.75,
        capstyle="round",
        joinstyle="round",
        zorder=21,
        clip_on=False,
    )
    ax.add_patch(check)


def _recovery_icon(ax: plt.Axes, x: float, y: float, *, size: float = 0.055) -> None:
    """Draw a minimal circular recovery arrow in axes coordinates."""

    arrow = FancyArrowPatch(
        (x - size * 0.55, y - size * 0.05),
        (x + size * 0.42, y + size * 0.20),
        transform=ax.transAxes,
        connectionstyle="arc3,rad=-0.72",
        arrowstyle="-|>",
        mutation_scale=6.5,
        linewidth=1.35,
        color=PPO,
        zorder=20,
        clip_on=False,
    )
    _halo(arrow, PPO_LIGHT, linewidth=3.8, alpha=0.9)
    ax.add_patch(arrow)


def _save(
    fig: plt.Figure,
    stem: str,
    paper_output: Path,
    preview_output: Path,
) -> dict[str, Any]:
    paper_output.mkdir(parents=True, exist_ok=True)
    preview_output.mkdir(parents=True, exist_ok=True)
    pdf = paper_output / f"{stem}.pdf"
    png = preview_output / f"{stem}.png"
    svg = preview_output / f"{stem}.svg"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(png, dpi=450, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return {
        "pdf": str(pdf.relative_to(ROOT)) if pdf.is_relative_to(ROOT) else str(pdf),
        "pdf_sha256": _sha256(pdf),
        "png": str(png.relative_to(ROOT)) if png.is_relative_to(ROOT) else str(png),
        "png_sha256": _sha256(png),
        "svg": str(svg.relative_to(ROOT)) if svg.is_relative_to(ROOT) else str(svg),
        "svg_sha256": _sha256(svg),
    }


def _threshold_figure(run: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    configs = run["configurations"]
    labels = [f"{item['t']}-of-{item['n']}" for item in configs]
    p50 = np.asarray([item["signing_ms"]["p50"] for item in configs], dtype=float)
    p95 = np.asarray([item["signing_ms"]["p95"] for item in configs], dtype=float)
    p99 = np.asarray([item["signing_ms"]["p99"] for item in configs], dtype=float)
    k_values = np.asarray([item["k_parallel_repetitions"] for item in configs], dtype=float)
    payload = np.asarray(
        [
            item["implementation_defined_outbound_bytes_per_attempt"]["party_to_coordinator_total"]
            / (1024**2)
            for item in configs
        ],
        dtype=float,
    )
    dropout = run["dropout_campaign"]
    dropped = np.asarray([item["dropped"] for item in dropout], dtype=int)
    completions = np.asarray([item["completions"] for item in dropout], dtype=float)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 1.92),
        gridspec_kw={"width_ratios": (1.22, 1.08, 0.96)},
        layout="constrained",
    )

    # (a) Full measured latency distribution summary on a log axis.
    ax = axes[0]
    _quiet_axes(ax)
    y = np.arange(len(labels))[::-1]
    colors = [TEAL, PPO, COMPARATOR, HAZARD]
    ax.axvspan(20.0, 145.0, color=HAZARD_LIGHT, alpha=0.45, zorder=0)
    reference_line = ax.axvline(20.0, color=HAZARD, lw=1.05, ls=(0, (4, 2.5)), zorder=1)
    _halo(reference_line, HAZARD_LIGHT, linewidth=3.2, alpha=0.75)
    for idx, yi in enumerate(y):
        ax.hlines(yi, p50[idx], p99[idx], color=colors[idx], lw=2.6, zorder=3)
        ax.scatter(p50[idx], yi, s=30, color=colors[idx], edgecolor="white", lw=0.55, zorder=5)
        ax.scatter(p95[idx], yi, s=29, marker="|", color=INK, lw=1.0, zorder=6)
        ax.scatter(p99[idx], yi, s=29, marker="|", color=colors[idx], lw=1.2, zorder=6)
        annotation = ax.annotate(
            f"{p50[idx]:.1f}",
            (p50[idx], yi),
            xytext=(-5, 5) if idx == len(y) - 1 else (5, 5),
            textcoords="offset points",
            ha="right" if idx == len(y) - 1 else "left",
            va="bottom",
            color=colors[idx],
            fontsize=6.4,
            weight="bold",
        )
        _outline_text(annotation, linewidth=1.6)
    _warning_icon(ax, 0.66, 0.92, size=0.038)
    ax.text(0.72, 0.965, "published ref.\n<20 ms", transform=ax.transAxes, color=HAZARD, fontsize=5.8, va="top", weight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 145)
    ax.set_ylim(-0.45, 3.68)
    ax.set_xticks([1, 5, 20, 100])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks(y, labels)
    ax.set_xlabel("End-to-end signing latency (ms, log)")
    ax.grid(axis="x", which="major")
    _panel_title(ax, "(a)", "Latency escalation")

    # (b) The SDK-selected work factor and derived payload, both on log scales.
    ax = axes[1]
    _quiet_axes(ax)
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        k_values,
        width=0.58,
        color=PPO_LIGHT,
        edgecolor=PPO,
        linewidth=0.9,
        zorder=3,
        label="parallel repetitions $K$",
    )
    ax.set_yscale("log")
    ax.set_ylim(1, 320)
    ax.set_ylabel("Parallel repetitions $K$ (log)", color=PPO)
    ax.tick_params(axis="y", colors=PPO)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_xlabel("Executed configuration")
    ax.grid(axis="y", which="major")
    for bar, value in zip(bars, k_values, strict=True):
        value_label = ax.annotate(
            f"{value:.0f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(-5, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=PPO,
            fontsize=6.2,
            weight="bold",
            zorder=8,
        )
        _outline_text(value_label, linewidth=1.55)
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(COMPARATOR)
    ax2.set_yscale("log")
    ax2.set_ylim(0.018, 5.2)
    payload_line = ax2.plot(
        x,
        payload,
        color=COMPARATOR,
        lw=1.6,
        marker="D",
        ms=4.2,
        mec="white",
        mew=0.45,
        zorder=5,
        label="payload/attempt",
    )[0]
    _halo(payload_line, "#FFE2BD", linewidth=3.3, alpha=0.62)
    ax2.set_ylabel("Payload (MiB/attempt, log)", color=COMPARATOR)
    ax2.tick_params(axis="y", colors=COMPARATOR, width=0.65, length=2.8)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
    _warning_icon(ax, 0.06, 0.91, size=0.035)
    ax.text(0.12, 0.965, f"{k_values[-1] / k_values[0]:.0f}× work", transform=ax.transAxes, ha="left", va="top", color=PPO, fontsize=6.2, weight="bold")
    ax2.text(0.97, 0.965, f"{payload[-1] / payload[0]:.0f}× traffic", transform=ax2.transAxes, ha="right", va="top", color=COMPARATOR, fontsize=6.2, weight="bold")
    _panel_title(ax, "(b)", "Non-linear work growth")

    # (c) Exact completion/fail-close transition.
    ax = axes[2]
    _quiet_axes(ax)
    ax.axvspan(-0.5, 2.5, color=SAFE_LIGHT, alpha=0.64, zorder=0)
    ax.axvspan(2.5, 3.5, color=HAZARD_LIGHT, alpha=0.82, hatch="////", zorder=0)
    bar_colors = [SAFE, SAFE, SAFE, HAZARD]
    bars = ax.bar(
        dropped,
        completions,
        width=0.62,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    boundary = ax.axvline(2.5, color=HAZARD, lw=1.1, ls=(0, (4, 2.5)), zorder=4)
    _halo(boundary, HAZARD_LIGHT, linewidth=3.2, alpha=0.78)
    for bar, value in zip(bars, completions, strict=True):
        xmid = bar.get_x() + bar.get_width() / 2
        if value > 0:
            ax.text(xmid, value - 7, f"{value:.0f}", ha="center", va="top", color="white", fontsize=6.3, weight="bold")
        else:
            ax.scatter(xmid, 3.2, marker="X", s=38, color=HAZARD, edgecolor="white", lw=0.45, zorder=5)
            ax.text(xmid, 10.0, "0", ha="center", color=HAZARD, fontsize=6.4, weight="bold")
    _shield_icon(ax, 0.12, 0.91, size=0.035)
    ax.text(0.20, 0.96, "LIVE ≥$Q$", transform=ax.transAxes, ha="left", va="top", color=SAFE, fontsize=6.0, weight="bold")
    _warning_icon(ax, 0.72, 0.91, size=0.035)
    ax.text(0.80, 0.96, "CLOSED <$Q$", transform=ax.transAxes, ha="left", va="top", color=HAZARD, fontsize=5.8, weight="bold")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 126)
    ax.set_xticks(dropped, [f"d={value}" for value in dropped])
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Simultaneous party dropouts")
    ax.set_ylabel("Completed signatures")
    ax.grid(axis="y")
    _panel_title(ax, "(c)", "Exact fail-close boundary")

    metrics = {
        "configurations": labels,
        "p50_ms": p50.tolist(),
        "p95_ms": p95.tolist(),
        "p99_ms": p99.tolist(),
        "parallel_repetitions": k_values.astype(int).tolist(),
        "payload_mib": payload.tolist(),
        "dropout_completions": completions.astype(int).tolist(),
    }
    return fig, metrics


def _sla_figure(payload: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    rows = sorted(
        (row for row in payload["summary"] if np.isclose(row["load_fraction"], 0.9)),
        key=lambda row: row["batch_size"],
    )
    _require([row["batch_size"] for row in rows] == [10, 25, 50, 100], "SLA 90% batch grid drift")
    b = np.asarray([row["batch_size"] for row in rows], dtype=int)
    tps = np.asarray([row["delivered_tps_mean"] for row in rows], dtype=float)
    tps_ci = np.asarray([row["delivered_tps_ci95_half"] for row in rows], dtype=float)
    p95 = np.asarray([row["latency_p95_ms_mean"] for row in rows], dtype=float)
    p95_ci = np.asarray([row["latency_p95_ms_ci95_half"] for row in rows], dtype=float)
    backlog = np.asarray([row["queue_backlog_p95_records_mean"] for row in rows], dtype=float)
    archive = np.asarray([row["archive_bytes_per_record_mean"] for row in rows], dtype=float)
    exposure = np.asarray([row["checkpoint_exposure_mean_ms_mean"] for row in rows], dtype=float)

    fig, (ax, ax_cost) = plt.subplots(
        1,
        2,
        figsize=(7.16, 1.92),
        gridspec_kw={"width_ratios": (1.18, 1.0)},
        layout="constrained",
    )

    # (a) Pareto-like frontier with uncertainty and backlog encoded by marker area.
    _quiet_axes(ax)
    ax.plot(p95, tps, color=PPO, lw=1.55, zorder=2)
    marker_colors = [TEAL, PPO, COMPARATOR, HAZARD]
    sizes = 30.0 + 0.58 * backlog
    frontier_labels: list[Any] = []
    label_x_offsets = [1.0, 1.1, 1.0, -1.0]
    label_y_offsets = [260.0, 250.0, 330.0, -380.0]
    for idx in range(len(rows)):
        ax.errorbar(
            p95[idx],
            tps[idx],
            xerr=p95_ci[idx],
            yerr=tps_ci[idx],
            fmt="none",
            ecolor=marker_colors[idx],
            elinewidth=0.85,
            capsize=2.0,
            zorder=3,
        )
        ax.scatter(
            p95[idx],
            tps[idx],
            s=sizes[idx],
            color=marker_colors[idx],
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        x_seed = p95[idx] + label_x_offsets[idx]
        y_seed = tps[idx] + label_y_offsets[idx]
        label = ax.text(
            x_seed,
            y_seed,
            f"B={b[idx]} | q95={backlog[idx]:.0f}",
            ha="left" if idx < 3 else "right",
            va="center",
            color=marker_colors[idx],
            fontsize=5.9,
            weight="bold" if b[idx] == 50 else "normal",
            zorder=7,
        )
        _outline_text(label, linewidth=1.65)
        frontier_labels.append(label)
    idx50 = int(np.where(b == 50)[0][0])
    ax.scatter(
        p95[idx50],
        tps[idx50],
        s=sizes[idx50] + 95,
        facecolors="none",
        edgecolors=COMPARATOR,
        linewidth=1.25,
        zorder=5,
    )
    ax.axvline(p95[idx50], color=COMPARATOR, lw=0.8, ls=(0, (3, 2)), alpha=0.8)
    _shield_icon(ax, 0.04, 0.90, size=0.034, color=COMPARATOR)
    ax.text(0.10, 0.96, "registered B=50", transform=ax.transAxes, ha="left", va="top", color=COMPARATOR, fontsize=5.8, weight="bold")
    ax.set_xlim(91, 128)
    ax.set_ylim(700, 4630)
    ax.set_xlabel("p95 end-to-end latency (ms)")
    ax.set_ylabel("Delivered throughput (record equiv./s)")
    ax.grid()
    _panel_title(ax, "(a)", "Throughput–latency–backlog frontier")
    adjust_text(
        frontier_labels,
        x=p95,
        y=tps,
        ax=ax,
        expand=(1.04, 1.16),
        force_text=(0.18, 0.25),
        force_static=(0.10, 0.18),
        ensure_inside_axes=True,
        min_arrow_len=4.0,
        time_lim=0.5,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.45},
    )

    # (b) Storage savings against longer checkpoint exposure.  Keep only the
    # endpoint exposure markers because those are the cells reported in the paper.
    _quiet_axes(ax_cost)
    selected = np.asarray([0, 2, 3])
    x = np.arange(3)
    archive_sel = archive[selected]
    bars = ax_cost.bar(
        x,
        archive_sel,
        width=0.58,
        color=[NEUTRAL_LIGHT, PPO_LIGHT, NEUTRAL_LIGHT],
        edgecolor=[NEUTRAL, PPO, NEUTRAL],
        linewidth=0.9,
        zorder=3,
        label="archive overhead",
    )
    ax_cost.axvspan(0.62, 1.38, color=PPO_LIGHT, alpha=0.32, zorder=0)
    for bar, value in zip(bars, archive_sel, strict=True):
        ax_cost.text(
            bar.get_x() + bar.get_width() / 2,
            value + 14,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=INK,
            weight="bold" if np.isclose(value, archive[idx50]) else "normal",
        )
    ax_cost.set_ylim(0, 680)
    ax_cost.set_ylabel("Archive overhead (B/record)")
    ax_cost.set_xticks(x, ["10", "50", "100"])
    ax_cost.set_xlabel("Batch size $B$")
    ax_cost.grid(axis="y")
    ax_exp = ax_cost.twinx()
    ax_exp.spines["top"].set_visible(False)
    ax_exp.spines["right"].set_color(HAZARD)
    endpoint_x = np.asarray([0, 2])
    endpoint_exp = exposure[[0, 3]]
    exposure_line = ax_exp.plot(
        endpoint_x,
        endpoint_exp,
        color=HAZARD,
        lw=1.55,
        ls=(0, (2, 2)),
        marker="D",
        ms=4.5,
        mec="white",
        mew=0.45,
        zorder=5,
    )[0]
    _halo(exposure_line, HAZARD_LIGHT, linewidth=3.5, alpha=0.85)
    ax_exp.set_ylim(0, 16.0)
    ax_exp.set_ylabel("Checkpoint exposure (ms)", color=HAZARD)
    ax_exp.tick_params(axis="y", colors=HAZARD, width=0.65, length=2.8)
    _recovery_icon(ax_exp, 0.05, 0.90, size=0.050)
    ax_exp.text(0.11, 0.965, f"−{100 * (1 - archive[-1] / archive[0]):.1f}% storage", transform=ax_exp.transAxes, ha="left", va="top", color=PPO, fontsize=6.2, weight="bold")
    _warning_icon(ax_exp, 0.66, 0.91, size=0.034)
    ax_exp.text(0.72, 0.965, f"{exposure[-1] / exposure[0]:.1f}× exposure", transform=ax_exp.transAxes, ha="left", va="top", color=HAZARD, fontsize=6.1, weight="bold")
    _panel_title(ax_cost, "(b)", "Storage gain versus exposure cost")

    metrics = {
        "batch_sizes": b.tolist(),
        "delivered_tps": tps.tolist(),
        "p95_latency_ms": p95.tolist(),
        "p95_backlog_records": backlog.tolist(),
        "archive_bytes_per_record": archive.tolist(),
        "checkpoint_exposure_ms": exposure.tolist(),
    }
    return fig, metrics


def _drl_figure(payload: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    panel = payload["panel"]
    ppo_agg = panel["deterministic_primary"]["aggregate"]
    ppo_guard = panel["deterministic_primary"]["security_guard"]
    baseline_agg = panel["baselines"]["aggregates"]
    baseline_guard = panel["baselines"]["security_guards"]
    comparison = panel["comparisons"]["deterministic_primary"]["CMAB"]
    cmab = baseline_agg["CMAB"]
    cmab_guard = baseline_guard["CMAB"]

    ppo_late = float(ppo_agg["mean_tps_late_1500_1999"])
    cmab_late = float(cmab["mean_tps_late_1500_1999"])
    effect = comparison["late_tps_difference"]
    ppo_failure = 100.0 * float(ppo_agg["failure_rate_post_1000_1999"])
    cmab_failure = 100.0 * float(cmab["failure_rate_post_1000_1999"])

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 1.92),
        gridspec_kw={"width_ratios": (0.95, 1.0, 1.22)},
        layout="constrained",
    )

    # (a) Late-window service with the paired gain made explicit.
    ax = axes[0]
    _quiet_axes(ax)
    y = np.asarray([1, 0])
    values = np.asarray([ppo_late, cmab_late])
    bars = ax.barh(
        y,
        values,
        height=0.50,
        color=[PPO, COMPARATOR],
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(value - 7, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", ha="right", va="center", color="white", fontsize=7.0, weight="bold")
    _halo(bars.patches[0], PPO_LIGHT, linewidth=4.2, alpha=0.95)
    _recovery_icon(ax, 0.56, 0.89, size=0.055)
    ax.text(0.64, 0.965, f"+{effect['mean']:.1f} TPS | +{100 * (ppo_late / cmab_late - 1):.1f}%", transform=ax.transAxes, ha="left", va="top", color=PPO, fontsize=6.2, weight="bold")
    ax.set_xlim(0, 355)
    ax.set_ylim(-0.55, 1.82)
    ax.set_yticks(y, ["PPO", "CMAB"])
    ax.set_xlabel("Late-window service (TPS)")
    ax.grid(axis="x")
    _panel_title(ax, "(a)", "Recovery gain")

    # (b) Paired effect and confidence interval against a visible zero boundary.
    ax = axes[1]
    _quiet_axes(ax)
    ax.axvspan(-14, 0, color=HAZARD_LIGHT, alpha=0.75, hatch="////", zorder=0)
    ax.axvspan(0, 96, color=PPO_LIGHT, alpha=0.45, zorder=0)
    ax.axvline(0, color=HAZARD, lw=1.0, ls=(0, (4, 2.5)), zorder=2)
    lo = float(effect["ci95_lower"])
    hi = float(effect["ci95_upper"])
    mean = float(effect["mean"])
    ax.hlines(0.5, lo, hi, color=PPO, lw=3.0, zorder=4)
    ax.scatter([lo, hi], [0.5, 0.5], marker="|", s=65, color=PPO, lw=1.2, zorder=5)
    ax.scatter(mean, 0.5, marker="D", s=44, color=PPO, edgecolor="white", lw=0.55, zorder=6)
    ax.text(mean, 0.74, f"+{mean:.1f}", ha="center", color=PPO, fontsize=7.0, weight="bold")
    ax.text(mean, 0.27, f"95% CI {lo:.1f}–{hi:.1f}", ha="center", color=INK, fontsize=6.3)
    _warning_icon(ax, 0.08, 0.90, size=0.034)
    ax.text(0.14, 0.965, "NO GAIN", transform=ax.transAxes, ha="left", va="top", color=HAZARD, fontsize=5.8, weight="bold")
    _shield_icon(ax, 0.51, 0.90, size=0.034, color=PPO)
    ax.text(0.58, 0.965, "CI ABOVE ZERO", transform=ax.transAxes, ha="left", va="top", color=PPO, fontsize=5.8, weight="bold")
    ax.set_xlim(-14, 96)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.set_xlabel("Paired PPO − CMAB difference (TPS)")
    ax.grid(axis="x")
    _panel_title(ax, "(b)", "Paired effect (n=20)")

    # (c) Show all frozen baselines as context while reserving emphasis for CMAB.
    ax = axes[2]
    _quiet_axes(ax)
    fault_budget = float(ppo_guard["fault_budget"])
    ax.axvspan(0, fault_budget, color=SAFE_LIGHT, alpha=0.38, zorder=0)
    ax.axvspan(fault_budget, 10.8, color=HAZARD_LIGHT, alpha=0.86, hatch="////", zorder=0)
    guard_line = ax.axvline(fault_budget, color=HAZARD, lw=1.15, ls=(0, (4, 2.5)), zorder=2)
    _halo(guard_line, HAZARD_LIGHT, linewidth=3.2, alpha=0.85)
    for name, agg in baseline_agg.items():
        if name == "CMAB":
            continue
        exposure = float(baseline_guard[name]["global_max_actual_security_exposure"])
        failure = 100.0 * float(agg["failure_rate_post_1000_1999"])
        crossed = exposure > fault_budget
        ax.scatter(
            exposure,
            failure,
            s=30 if crossed else 24,
            marker="^" if crossed else "o",
            facecolors=HAZARD if crossed else "white",
            edgecolors="white" if crossed else NEUTRAL,
            linewidth=0.65 if crossed else 0.8,
            zorder=4 if crossed else 3,
        )
    # Label only the comparators that cross the registered guard; automatic
    # placement is applied after axes geometry is final.
    risk_label_offsets = {
        "SA": (5, 5, "left"),
        "Random": (-5, 6, "right"),
        "Static-with-valid-refill": (5, -11, "left"),
    }
    for name in ("SA", "Random", "Static-with-valid-refill"):
        exposure = float(baseline_guard[name]["global_max_actual_security_exposure"])
        failure = 100.0 * float(baseline_agg[name]["failure_rate_post_1000_1999"])
        label = "Static" if name == "Static-with-valid-refill" else name
        dx, dy, ha = risk_label_offsets[name]
        text_artist = ax.annotate(label, (exposure, failure), xytext=(dx, dy), textcoords="offset points", ha=ha, color=HAZARD, fontsize=5.7, weight="bold", zorder=8)
        _outline_text(text_artist, linewidth=1.55)
    ax.scatter(
        cmab_guard["global_max_actual_security_exposure"],
        cmab_failure,
        marker="s",
        s=43,
        color=COMPARATOR,
        edgecolor="white",
        lw=0.55,
        zorder=5,
    )
    ax.scatter(
        ppo_guard["global_max_actual_security_exposure"],
        ppo_failure,
        marker="*",
        s=92,
        color=PPO,
        edgecolor="white",
        lw=0.55,
        zorder=6,
    )
    cmab_text = ax.annotate("CMAB", (cmab_guard["global_max_actual_security_exposure"], cmab_failure), xytext=(4, 5), textcoords="offset points", color=COMPARATOR, fontsize=6.0, weight="bold")
    ppo_text = ax.annotate("PPO", (ppo_guard["global_max_actual_security_exposure"], ppo_failure), xytext=(5, -10), textcoords="offset points", color=PPO, fontsize=6.2, weight="bold")
    _outline_text(cmab_text, linewidth=1.5)
    _outline_text(ppo_text, linewidth=1.5)
    improvement_arrow = FancyArrowPatch(
        (float(cmab_guard["global_max_actual_security_exposure"]), cmab_failure),
        (float(ppo_guard["global_max_actual_security_exposure"]), ppo_failure),
        arrowstyle="-|>",
        mutation_scale=7.5,
        color=PPO,
        linewidth=1.25,
        connectionstyle="arc3,rad=-0.08",
        zorder=5,
    )
    _halo(improvement_arrow, PPO_LIGHT, linewidth=4.0, alpha=0.95)
    ax.add_patch(improvement_arrow)
    summary = ax.text(
        0.05,
        0.78,
        f"−{cmab_failure - ppo_failure:.1f} pp outage\n−{cmab_guard['global_max_actual_security_exposure'] - ppo_guard['global_max_actual_security_exposure']:.0f} exposure",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=PPO,
        fontsize=5.8,
        weight="bold",
        zorder=8,
    )
    _outline_text(summary, linewidth=1.6)
    _shield_icon(ax, 0.05, 0.92, size=0.033)
    ax.text(0.11, 0.97, "SAFE + RELIABLE", transform=ax.transAxes, ha="left", va="top", color=SAFE, fontsize=5.7, weight="bold")
    _warning_icon(ax, 0.57, 0.92, size=0.033)
    ax.text(0.63, 0.97, "GUARD > 6", transform=ax.transAxes, ha="left", va="top", color=HAZARD, fontsize=5.7, weight="bold")
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 54)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel(r"Max. actual exposure $|\mathcal{B}\cap C_t|$")
    ax.set_ylabel("Zero-service steps (%)")
    ax.grid()
    _panel_title(ax, "(c)", "Reliability–security map")

    metrics = {
        "ppo_late_tps": ppo_late,
        "cmab_late_tps": cmab_late,
        "paired_difference_tps": mean,
        "paired_ci95": [lo, hi],
        "ppo_zero_service_percent": ppo_failure,
        "cmab_zero_service_percent": cmab_failure,
        "ppo_max_exposure": ppo_guard["global_max_actual_security_exposure"],
        "cmab_max_exposure": cmab_guard["global_max_actual_security_exposure"],
        "fault_budget": ppo_guard["fault_budget"],
        "context_baselines": sorted(baseline_agg),
    }
    return fig, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-output", type=Path, default=PAPER_FIGURES)
    parser.add_argument("--preview-output", type=Path, default=PREVIEW_DIR)
    args = parser.parse_args()
    paper_output = args.paper_output.resolve()
    preview_output = args.preview_output.resolve()

    threshold_validation = _read_json(THRESHOLD_VALIDATION)
    threshold_run = _read_json(THRESHOLD_RUN)
    _require(threshold_validation.get("status") == "PASS", "threshold validation is not PASS")
    _require(
        threshold_run.get("schema") == "qrelay-real-threshold-backend-receipt-1"
        and threshold_run.get("status") == "PASS",
        "threshold receipt is not the frozen PASS artifact",
    )

    sla = _read_json(SLA_SUMMARY)
    _require(sla.get("schema") == "qrelay-stage-b-unsaturated-sla-sweep-1", "SLA schema drift")
    _require(sla.get("design", {}).get("replicates") == 8, "SLA replicate count drift")

    drl_validation = _read_json(DRL_VALIDATION)
    drl = _read_json(DRL_EVALUATION)
    _require(drl_validation.get("status") == "PASS", "DRL validation is not PASS")
    _require(
        drl.get("schema") == "qrelay-security-speed-v5-heldout-disaster-evaluation-1",
        "DRL evaluation schema drift",
    )
    _require(len(drl["panel"]["seeds"]) == 20, "DRL held-out seed count drift")

    _style()
    outputs: dict[str, Any] = {}
    figures = (
        ("threshold_backend_v3", _threshold_figure(threshold_run)),
        ("stage_bc_sla_pareto", _sla_figure(sla)),
        ("stage_drl_security_speed_v5_heldout", _drl_figure(drl)),
    )
    for stem, (fig, metrics) in figures:
        outputs[stem] = {
            **_save(fig, stem, paper_output, preview_output),
            "metrics": metrics,
        }

    manifest = {
        "schema": "qrelay-paper-result-figure-render-2",
        "style": "collision-aware IEEE high-vis hazard encoding with restrained vector icons and path effects; no data substitution",
        "sources": {
            str(THRESHOLD_RUN.relative_to(ROOT)): _sha256(THRESHOLD_RUN),
            str(THRESHOLD_VALIDATION.relative_to(ROOT)): _sha256(THRESHOLD_VALIDATION),
            str(SLA_SUMMARY.relative_to(ROOT)): _sha256(SLA_SUMMARY),
            str(DRL_EVALUATION.relative_to(ROOT)): _sha256(DRL_EVALUATION),
            str(DRL_VALIDATION.relative_to(ROOT)): _sha256(DRL_VALIDATION),
        },
        "drl_evidence_role": {
            "paper_evidence_eligible": drl_validation.get("paper_evidence_eligible"),
            "single_training_seed_diagnostic": drl_validation.get("single_training_seed_diagnostic"),
        },
        "outputs": outputs,
    }
    preview_output.mkdir(parents=True, exist_ok=True)
    manifest_path = preview_output / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: rendered {len(outputs)} source-bound figures")
    print(f"manifest={manifest_path}")
    for stem, record in outputs.items():
        print(f"{stem}={record['pdf']} sha256={record['pdf_sha256']}")


if __name__ == "__main__":
    main()
