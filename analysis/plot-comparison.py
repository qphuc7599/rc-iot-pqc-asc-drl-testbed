#!/usr/bin/env python3
"""Plot baseline comparison results.

Expected directories:
  results/comparison_A_pbft_ecdsa/gateway_summary.json
  results/comparison_B_bls_aggregate/gateway_summary.json
  results/comparison_C_offchain_statechannel/gateway_summary.json
  results/comparison_E_offchain_ecdsa/gateway_summary.json
  results/comparison_D_pbft_batched_ecdsa/gateway_summary.json
  results/comparison_F_simplex_batched_ecdsa/gateway_summary.json
  results/comparison_G_bullshark_dag_ecdsa/gateway_summary.json
  results/comparison_H_hydra_ecdsa/gateway_summary.json
"""

import argparse
import json
import os
import sys


PROTOCOL_CONFIG = {
    "A_pbft_ecdsa": {
        "label": "PBFT + ECDSA\nbatch=1",
        "color": "#d95f02",
        "sig": "ECDSA-P256 (72B)",
        "quantum": "No",
        "consensus": "PBFT O(N^2)",
        "model": "On-chain",
    },
    "B_bls_aggregate": {
        "label": "BLS Aggregate\nbatch=1",
        "color": "#7570b3",
        "sig": "BLS12-381 (96B)",
        "quantum": "No",
        "consensus": "Aggregate only",
        "model": "On-chain",
    },
    "D_pbft_batched_ecdsa": {
        "label": "PBFT + ECDSA\nbatch=50",
        "color": "#e6ab02",
        "sig": "ECDSA-P256 (72B)",
        "quantum": "No",
        "consensus": "PBFT O(N^2)",
        "model": "Batched on-chain",
    },
    "E_offchain_ecdsa": {
        "label": "State Channel\nECDSA",
        "color": "#377eb8",
        "sig": "ECDSA-P256 (72B)",
        "quantum": "No",
        "consensus": "Off-chain",
        "model": "ASC batch=50",
    },
    "F_simplex_batched_ecdsa": {
        "label": "Simplex\nBFT batch=50",
        "color": "#a6761d",
        "sig": "ECDSA-P256 (72B)",
        "quantum": "No",
        "consensus": "Simplex",
        "model": "Batched L1 protocol",
    },
    "G_bullshark_dag_ecdsa": {
        "label": "Bullshark\nDAG-BFT",
        "color": "#666666",
        "sig": "ECDSA-P256 (72B)",
        "quantum": "No",
        "consensus": "Bullshark DAG",
        "model": "DAG-BFT protocol",
    },
    "H_hydra_ecdsa": {
        "label": "Hydra Head\nState Channel",
        "color": "#984ea3",
        "sig": "ECDSA-P256 (72B)",
        "quantum": "No",
        "consensus": "Off-chain snapshots",
        "model": "Layer-2 protocol",
    },
    "C_offchain_statechannel": {
        "label": "State Channel\nML-DSA",
        "color": "#1b9e77",
        "sig": "ML-DSA-44 (2420B)",
        "quantum": "Yes",
        "consensus": "Off-chain",
        "model": "ASC batch=50",
    },
}


def load_comparison_data(results_dir):
    data = {}
    for name in sorted(os.listdir(results_dir)):
        if not name.startswith("comparison_"):
            continue
        path = os.path.join(results_dir, name, "gateway_summary.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            data[name.replace("comparison_", "")] = json.load(f)
    return data


def plot_comparison(data, results_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not installed")
        return

    keys = [k for k in PROTOCOL_CONFIG if k in data]
    if len(keys) < 2:
        print(f"[WARN] Need at least 2 experiments, found {len(keys)}")
        return

    labels = [PROTOCOL_CONFIG[k]["label"] for k in keys]
    colors = [PROTOCOL_CONFIG[k]["color"] for k in keys]
    tps_values = [data[k]["avg_tps"] for k in keys]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        gridspec_kw={"height_ratios": [2.4, 1]},
    )
    fig.suptitle(
        "Baseline Comparison: Throughput (TPS)\n"
        "100 IoT nodes, 802.11g WiFi, same offered load",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0]
    x = np.arange(len(keys))
    bars = ax.bar(x, tps_values, color=colors, width=0.58, edgecolor="black", linewidth=0.8)
    for bar, tps, color in zip(bars, tps_values, colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(tps_values) * 0.02,
            f"{tps:,.0f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=color,
        )

    if "A_pbft_ecdsa" in data and "C_offchain_statechannel" in data:
        per_tx_gain = data["C_offchain_statechannel"]["avg_tps"] / max(
            data["A_pbft_ecdsa"]["avg_tps"], 1e-9
        )
        ax.text(
            0.02,
            0.95,
            f"ASC vs per-tx PBFT: {per_tx_gain:.1f}x",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color="#1b9e77",
            va="top",
        )
    if "D_pbft_batched_ecdsa" in data and "C_offchain_statechannel" in data:
        batch_gain = data["C_offchain_statechannel"]["avg_tps"] / max(
            data["D_pbft_batched_ecdsa"]["avg_tps"], 1e-9
        )
        ax.text(
            0.02,
            0.88,
            f"ASC vs batched PBFT: {batch_gain:.1f}x",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color="#1b9e77",
            va="top",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Transactions per second")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(tps_values) * 1.25)

    ax2 = axes[1]
    ax2.axis("off")
    table_data = []
    for k in keys:
        cfg = PROTOCOL_CONFIG[k]
        d = data[k]
        updates = d.get("total_updates", d["total_tx"])
        tx_per_pkt = d["total_tx"] / updates if updates > 0 else 0.0
        table_data.append(
            [
                cfg["label"].replace("\n", " "),
                f"{d['avg_tps']:,.0f}",
                f"{tx_per_pkt:.1f}",
                cfg["sig"],
                cfg["consensus"],
                cfg["quantum"],
            ]
        )

    col_labels = ["Protocol", "TPS", "TX/pkt", "Signature", "Consensus", "PQ"]
    table = ax2.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.28, 0.10, 0.10, 0.22, 0.18, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.9)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#333333")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i, k in enumerate(keys):
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(PROTOCOL_CONFIG[k]["color"] + "20")

    plt.tight_layout()
    path = os.path.join(results_dir, "baseline_comparison.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    print("\n" + "=" * 72)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 72)
    for k in keys:
        cfg = PROTOCOL_CONFIG[k]
        d = data[k]
        updates = d.get("total_updates", d["total_tx"])
        tx_per_pkt = d["total_tx"] / updates if updates > 0 else 0.0
        print(f"{cfg['label'].replace(chr(10), ' '):32s} TPS={d['avg_tps']:8.1f} TX/pkt={tx_per_pkt:5.1f}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    data = load_comparison_data(args.results_dir)
    if not data:
        print("[ERROR] No comparison data found. Run run-baseline-comparison.sh first.")
        sys.exit(1)
    plot_comparison(data, args.results_dir)


if __name__ == "__main__":
    main()
