#!/usr/bin/env python3
"""Validate and plot the affinity-controlled n=100 threshold campaign."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "paper_recovery_threshold_v3"
ENGINE = ROOT / "experiments" / "paper_recovery_threshold_v1"
RESULTS = ROOT / "results" / "paper_recovery_threshold_v3"
PAPER = ROOT / "file_project" / "paper_file" / "_2026__Elsevier_IoT_HoangPhuc"
FIGURES = PAPER / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise RuntimeError(message)
    checks.append(message)


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def check_distribution(
    samples: list[float], summary: dict[str, Any], label: str, checks: list[str]
) -> None:
    expected = {
        "count": len(samples),
        "mean": sum(samples) / len(samples),
        "p50": quantile(samples, 0.50),
        "p95": quantile(samples, 0.95),
        "p99": quantile(samples, 0.99),
        "min": min(samples),
        "max": max(samples),
    }
    require(summary["count"] == expected["count"], f"{label} count recomputes", checks)
    for key in ("mean", "p50", "p95", "p99", "min", "max"):
        value = summary[key]
        require(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= 0
            and math.isclose(value, expected[key], rel_tol=1e-12, abs_tol=1e-9),
            f"{label} {key} recomputes",
            checks,
        )


def validate() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    registration = read_json(EXPERIMENT / "protocol_registration.json")
    protocol = read_json(EXPERIMENT / "protocol.json")
    run = read_json(RESULTS / "registered_run.json")
    checks: list[str] = []

    require(
        registration["schema"] == "qrelay-real-threshold-backend-registration-3",
        "registration schema matches",
        checks,
    )
    require(
        protocol["schema"] == "qrelay-real-threshold-backend-protocol-1",
        "protocol schema matches",
        checks,
    )
    require(
        run["schema"] == "qrelay-real-threshold-backend-receipt-1",
        "receipt schema matches",
        checks,
    )
    require(run["status"] == "PASS", "campaign status is PASS", checks)
    require(
        registration["campaign_id"] == protocol["campaign_id"] == run["campaign_id"],
        "campaign identifiers agree",
        checks,
    )

    current_hashes = {
        "protocol_sha256": sha256(EXPERIMENT / "protocol.json"),
        "source_sha256": sha256(ENGINE / "src" / "main.rs"),
        "cargo_lock_sha256": sha256(ENGINE / "Cargo.lock"),
        "cargo_toml_sha256": sha256(ENGINE / "Cargo.toml"),
        "executable_sha256": sha256(
            ENGINE / "target" / "release" / "qrelay-threshold-backend-bench.exe"
        ),
        "receipt_sha256": sha256(RESULTS / "registered_run.json"),
    }
    for key, value in current_hashes.items():
        if key == "receipt_sha256":
            continue
        require(registration[key] == value, f"registered {key} matches", checks)
    for key in ("protocol_sha256", "source_sha256", "cargo_lock_sha256", "executable_sha256"):
        require(run["provenance"][key] == current_hashes[key], f"receipt {key} matches", checks)

    require(protocol["samples_per_config"] == 100, "100 signatures per configuration registered", checks)
    require(protocol["dropout_samples_per_cell"] == 100, "100 trials per dropout cell registered", checks)
    require(
        registration["execution_control"]["processor_affinity_mask_hex"] == "0x1"
        and registration["execution_control"]["priority_class"] == "High",
        "host scheduling control is registered",
        checks,
    )
    require(run["backend"]["version"] == "0.3.6", "backend version is locked", checks)
    require(
        run["backend"]["published_party_envelope"] == "2 <= T <= N <= 6",
        "execution envelope remains N<=6",
        checks,
    )

    expected_configs = [(2, 3, 3, 3), (3, 4, 7, 6), (4, 6, 74, 20), (5, 6, 100, 15)]
    observed = [
        (item["t"], item["n"], item["k_parallel_repetitions"], item["rss_subsets"])
        for item in run["configurations"]
    ]
    require(observed == expected_configs, "configuration, K, and RSS grid matches", checks)
    for config in run["configurations"]:
        label = f"{config['t']}-of-{config['n']}"
        samples = config["samples"]
        require(len(samples) == 100, f"{label} has 100 samples", checks)
        require(config["all_signatures_verified"], f"{label} signatures all verify", checks)
        require(all(item["verified"] for item in samples), f"{label} per-sample verification passes", checks)
        require(
            all(1 <= item["attempts"] <= protocol["max_single_attempts"] for item in samples),
            f"{label} attempts stay within the registered bound",
            checks,
        )
        check_distribution([item["sign_ms"] for item in samples], config["signing_ms"], f"{label} sign_ms", checks)
        check_distribution([item["verify_us"] for item in samples], config["verification_us"], f"{label} verify_us", checks)
        check_distribution(
            [float(item["attempts"]) for item in samples],
            config["protocol_attempts"],
            f"{label} attempts",
            checks,
        )
        require(config["signature_bytes"] == 2420, f"{label} signature size matches ML-DSA-44", checks)
        require(config["public_key_bytes"] == 1312, f"{label} public-key size matches ML-DSA-44", checks)
        require(config["restart_same_public_key"], f"{label} restart preserves public key", checks)
        require(config["restart_verified_prior_signature"], f"{label} restart verifies prior signature", checks)
        require(config["rotation_changed_public_key"], f"{label} rotation changes public key", checks)
        require(config["rotation_rejected_prior_signature"], f"{label} rotation rejects prior signature", checks)

    dropout = run["dropout_campaign"]
    require([item["dropped"] for item in dropout] == [0, 1, 2, 3], "dropout grid is d=0..3", checks)
    for cell in dropout:
        dropped = cell["dropped"]
        require(cell["trials"] == 100, f"dropout d={dropped} has 100 trials", checks)
        if cell["survivors"] >= cell["t"]:
            require(cell["completions"] == 100, f"dropout d={dropped} completes 100/100", checks)
        else:
            require(cell["completions"] == 0, f"dropout d={dropped} completes 0/100", checks)
            require(cell["expected_fail_closed"], f"dropout d={dropped} is fail-closed", checks)
            require(cell["rejection_class"] == "InvalidParameters", f"dropout d={dropped} rejects explicitly", checks)
    require(run["negative_tests"]["all_passed"], "all negative tests pass", checks)
    require(
        run["negative_tests"]["unsupported_14_of_21_rejected"],
        "unsupported 14-of-21 is rejected rather than simulated",
        checks,
    )
    return registration, protocol, run, checks


def write_validation(
    registration: dict[str, Any], protocol: dict[str, Any], run: dict[str, Any], checks: list[str]
) -> None:
    payload = {
        "schema": "qrelay-real-threshold-backend-validation-3",
        "status": "PASS",
        "campaign_id": run["campaign_id"],
        "registered_at": registration["registered_at"],
        "receipt_sha256": sha256(RESULTS / "registered_run.json"),
        "samples_per_config": protocol["samples_per_config"],
        "dropout_samples_per_cell": protocol["dropout_samples_per_cell"],
        "execution_control": registration["execution_control"],
        "scope": {
            "executed": "Mithril-style threshold ML-DSA-44, sequential in-process SDK, 2 <= T <= N <= 6",
            "not_executed": ["parallel parties", "network transport", "hardware", "14-of-21 backend", "PPO retraining"],
        },
        "checks": checks,
        "configurations": run["configurations"],
        "dropout_campaign": run["dropout_campaign"],
        "negative_tests": run["negative_tests"],
    }
    with (RESULTS / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_macros(run: dict[str, Any]) -> None:
    by_config = {(item["t"], item["n"]): item for item in run["configurations"]}
    qcase = by_config[(4, 6)]
    five_six = by_config[(5, 6)]
    d2 = next(item for item in run["dropout_campaign"] if item["dropped"] == 2)
    d3 = next(item for item in run["dropout_campaign"] if item["dropped"] == 3)
    lines = [
        "% Auto-generated by analysis/validate_and_plot_threshold_backend_v3.py; do not edit.",
        r"\newcommand{\ThresholdBackendName}{threshold-ml-dsa}",
        r"\newcommand{\ThresholdBackendVersion}{0.3.6}",
        r"\newcommand{\ThresholdBackendSamples}{100}",
        r"\newcommand{\ThresholdDropoutSamples}{100}",
        rf"\newcommand{{\ThresholdFourSixMedian}}{{{qcase['signing_ms']['p50']:.1f}}}",
        rf"\newcommand{{\ThresholdFourSixNinetyFive}}{{{qcase['signing_ms']['p95']:.1f}}}",
        rf"\newcommand{{\ThresholdFourSixNinetyNine}}{{{qcase['signing_ms']['p99']:.1f}}}",
        rf"\newcommand{{\ThresholdFourSixVerifyMedian}}{{{qcase['verification_us']['p50']:.1f}}}",
        rf"\newcommand{{\ThresholdFourSixOutboundMiB}}{{{qcase['implementation_defined_outbound_bytes_per_attempt']['party_to_coordinator_total'] / (1024**2):.2f}}}",
        rf"\newcommand{{\ThresholdFiveSixMedian}}{{{five_six['signing_ms']['p50']:.1f}}}",
        rf"\newcommand{{\ThresholdFiveSixOutboundMiB}}{{{five_six['implementation_defined_outbound_bytes_per_attempt']['party_to_coordinator_total'] / (1024**2):.2f}}}",
        rf"\newcommand{{\ThresholdDropTwoSuccess}}{{{100*d2['completion_rate']:.0f}\%}}",
        rf"\newcommand{{\ThresholdDropThreeSuccess}}{{{100*d3['completion_rate']:.0f}\%}}",
        r"\newcommand{\ThresholdKSequence}{3, 7, 74, and 100}",
        r"\newcommand{\MithrilReportedLocalBound}{20}",
        rf"\newcommand{{\ThresholdReceiptShort}}{{{sha256(RESULTS / 'registered_run.json')[:12]}}}",
    ]
    with (PAPER / "threshold_backend_generated.tex").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def annotate_bars(ax: plt.Axes, bars: Any, values: list[float], fmt: str = "{:.0f}") -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.08 if bar.get_height() > 0 else 3,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=6.6,
        )


def plot(run: dict[str, Any]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.6,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    ax_latency, ax_scale, ax_dropout = axes
    configs = run["configurations"]
    labels = [f"{item['t']}-of-{item['n']}" for item in configs]
    x = np.arange(len(configs))

    width = 0.22
    styles = [
        ("p50", "#2F6690", ""),
        ("p95", "#6FA8DC", "//"),
        ("p99", "#B4C7E7", "xx"),
    ]
    for offset, (metric, color, hatch) in zip((-width, 0.0, width), styles):
        values = [item["signing_ms"][metric] for item in configs]
        ax_latency.bar(
            x + offset,
            values,
            width=width,
            label=metric.upper(),
            color=color,
            edgecolor="#333333",
            linewidth=0.35,
            hatch=hatch,
        )
    ax_latency.axhline(20, color="#9C2F2F", linestyle="--", linewidth=1.0)
    ax_latency.text(
        0.03,
        21.5,
        "Authors: <20 ms*",
        color="#8B1E1E",
        fontsize=6.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.5},
    )
    ax_latency.set_yscale("log")
    ax_latency.set_ylim(0.8, 180)
    ax_latency.set_xticks(x, labels, rotation=18)
    ax_latency.set_ylabel("Signing latency (ms, log)")
    ax_latency.set_title("(a) 100 verified signatures/config")
    ax_latency.grid(axis="y", which="both", color="#D9D9D9", linewidth=0.5, alpha=0.85)
    ax_latency.legend(frameon=False, ncol=3, loc="upper left")
    ax_latency.text(
        0.0,
        -0.34,
        "*Separate Go/M3, parallel-party, one-attempt benchmark",
        transform=ax_latency.transAxes,
        fontsize=5.9,
        color="#555555",
    )

    k_values = [item["k_parallel_repetitions"] for item in configs]
    rss_values = [item["rss_subsets"] for item in configs]
    bars_k = ax_scale.bar(
        x - 0.18,
        k_values,
        width=0.36,
        color="#7A5195",
        edgecolor="#333333",
        linewidth=0.35,
        label="Parallel repetitions K",
    )
    bars_rss = ax_scale.bar(
        x + 0.18,
        rss_values,
        width=0.36,
        color="#4E9F7A",
        edgecolor="#333333",
        linewidth=0.35,
        hatch="//",
        label="RSS subsets",
    )
    ax_scale.set_yscale("log")
    ax_scale.set_ylim(1, 210)
    ax_scale.set_xticks(x, labels, rotation=18)
    ax_scale.set_ylabel("Configured count (log)")
    ax_scale.set_title("(b) Backend work-factor indicators")
    ax_scale.grid(axis="y", which="both", color="#D9D9D9", linewidth=0.5, alpha=0.85)
    ax_scale.legend(frameon=False, loc="upper left")
    annotate_bars(ax_scale, bars_k, k_values)
    annotate_bars(ax_scale, bars_rss, rss_values)

    dropout = run["dropout_campaign"]
    dropped = np.array([item["dropped"] for item in dropout])
    completion = [100 * item["completion_rate"] for item in dropout]
    colors = ["#4E9F7A" if value == 100 else "#C94C4C" for value in completion]
    bars = ax_dropout.bar(
        dropped,
        completion,
        width=0.62,
        color=colors,
        edgecolor="#333333",
        linewidth=0.4,
    )
    ax_dropout.axvline(2.5, color="#9C2F2F", linestyle="--", linewidth=1.0)
    ax_dropout.set_ylim(0, 116)
    ax_dropout.set_xticks(dropped, [f"d={value}\n({6-value} live)" for value in dropped])
    ax_dropout.set_ylabel("Certificate completion (%)")
    ax_dropout.set_title("(c) Exact 4-of-6 fail-close boundary")
    ax_dropout.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.85)
    annotate_bars(ax_dropout, bars, completion)
    ax_dropout.text(2.55, 108, "<Q", color="#8B1E1E", fontsize=7.0, va="top")

    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"threshold_backend_v3.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    registration, protocol, run, checks = validate()
    write_validation(registration, protocol, run, checks)
    write_macros(run)
    plot(run)
    print(f"PASS: {len(checks)} checks")
    print(f"receipt_sha256={sha256(RESULTS / 'registered_run.json')}")
    print(f"validation={RESULTS / 'validation.json'}")
    print(f"figure={FIGURES / 'threshold_backend_v3.pdf'}")


if __name__ == "__main__":
    main()
