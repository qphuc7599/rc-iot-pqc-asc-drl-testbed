#!/usr/bin/env python3
"""Plot and summarize batch-size sensitivity results.

Expected input:
  results/batch_sweep/summary.csv

The runner records two latency families:
  - settlement_*: measured from state-update packet timestamp to gateway commit.
  - direct_e2e_*: transaction-level latency from tx-generator v2 first/last
    logical-TX timestamps to gateway commit.
  - estimated_e2e_*: fallback latency for older artifacts that lack v2
    transaction timestamps.
"""

import argparse
import csv
import os
from collections import defaultdict


PROTOCOL_LABELS = {
    "asc": "ASC + ML-DSA",
    "pbft_batched": "Batched PBFT + ECDSA",
}

PROTOCOL_COLORS = {
    "asc": "#1b9e77",
    "pbft_batched": "#e6ab02",
}


def to_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def to_int(value, default=0):
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def load_summary(results_dir):
    path = os.path.join(results_dir, "summary.csv")
    rows = []
    if not os.path.exists(path):
        return rows

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "protocol": row.get("protocol", ""),
                    "batch_size": to_int(row.get("batch_size")),
                    "avg_tps": to_float(row.get("avg_tps")),
                    "tx_per_update": to_float(row.get("tx_per_update")),
                    "settlement_p50_ms": to_float(row.get("settlement_p50_ms")),
                    "settlement_p95_ms": to_float(row.get("settlement_p95_ms")),
                    "direct_e2e_p50_ms": to_float(row.get("direct_e2e_p50_ms")),
                    "direct_e2e_p95_ms": to_float(row.get("direct_e2e_p95_ms")),
                    "estimated_wait_p50_ms": to_float(row.get("estimated_batch_wait_p50_ms")),
                    "estimated_wait_p95_ms": to_float(row.get("estimated_batch_wait_p95_ms")),
                    "estimated_e2e_p50_ms": to_float(row.get("estimated_e2e_p50_ms")),
                    "estimated_e2e_p95_ms": to_float(row.get("estimated_e2e_p95_ms")),
                    "latency_source": row.get("latency_source", "") or "estimated",
                    "total_tx": to_int(row.get("total_tx")),
                    "result_dir": row.get("result_dir", ""),
                }
            )
    rows.sort(key=lambda r: (r["protocol"], r["batch_size"]))
    return rows


def latest_by_protocol_batch(rows):
    latest = {}
    for row in rows:
        latest[(row["protocol"], row["batch_size"])] = row
    return [latest[k] for k in sorted(latest, key=lambda x: (x[0], x[1]))]


def print_table(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["protocol"]].append(row)

    print("\n" + "=" * 96)
    print("BATCH-SIZE SENSITIVITY")
    print("=" * 96)
    print(
        f"{'Protocol':24s} {'B':>5s} {'TPS':>10s} {'TX/update':>10s} "
        f"{'settle p95 ms':>15s} {'e2e p95 ms':>13s} {'src':>3s} {'wait p95 ms':>13s}"
    )
    print("-" * 96)
    for protocol in sorted(grouped):
        for row in grouped[protocol]:
            e2e_p95 = row["direct_e2e_p95_ms"] or row["estimated_e2e_p95_ms"]
            source = "meas" if row.get("latency_source") == "measured" else "est"
            print(
                f"{PROTOCOL_LABELS.get(protocol, protocol):24s} "
                f"{row['batch_size']:5d} "
                f"{row['avg_tps']:10.1f} "
                f"{row['tx_per_update']:10.1f} "
                f"{row['settlement_p95_ms']:15.1f} "
                f"{e2e_p95:13.1f} {source:>2s} "
                f"{row['estimated_wait_p95_ms']:13.1f}"
            )
    print("=" * 96)

    asc_rows = grouped.get("asc", [])
    if asc_rows:
        best_tps = max(asc_rows, key=lambda r: r["avg_tps"])
        best_latency = min(asc_rows, key=lambda r: r["direct_e2e_p95_ms"] or r["estimated_e2e_p95_ms"])
        best_latency_p95 = best_latency["direct_e2e_p95_ms"] or best_latency["estimated_e2e_p95_ms"]
        b50 = next((r for r in asc_rows if r["batch_size"] == 50), None)
        print(
            f"ASC max TPS at B={best_tps['batch_size']} "
            f"({best_tps['avg_tps']:.1f} TPS); lowest p95 latency at "
            f"B={best_latency['batch_size']} ({best_latency_p95:.1f} ms)."
        )
        if b50:
            b50_p95 = b50["direct_e2e_p95_ms"] or b50["estimated_e2e_p95_ms"]
            print(
                f"B=50 point: {b50['avg_tps']:.1f} TPS, "
                f"p95 latency {b50_p95:.1f} ms ({b50['latency_source']})."
            )


def write_latex_table(rows, results_dir):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["batch_size"]][row["protocol"]] = row

    batches = sorted(grouped)
    path = os.path.join(results_dir, "batch_sweep_table.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by analysis/plot-batch-sweep.py\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\caption{Batch-size sensitivity at $N=100$ and 100 tx/s/node offered load. Direct transaction-level p95 is used when tx-generator v2 timestamps are present; otherwise the table falls back to analytical batch-fill latency plus measured gateway settlement latency.}\n")
        f.write("\\label{tab:batch_sensitivity}\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{@{}rrrrr@{}}\n")
        f.write("\\toprule\n")
        f.write("$B$ & ASC TPS & ASC p95 (ms) & PBFT-batched TPS & PBFT p95 (ms) \\\\\n")
        f.write("\\midrule\n")
        for batch in batches:
            asc = grouped[batch].get("asc", {})
            pbft = grouped[batch].get("pbft_batched", {})
            asc_p95 = to_float(asc.get("direct_e2e_p95_ms")) or to_float(asc.get("estimated_e2e_p95_ms"))
            pbft_p95 = to_float(pbft.get("direct_e2e_p95_ms")) or to_float(pbft.get("estimated_e2e_p95_ms"))
            f.write(
                f"{batch} & "
                f"{to_float(asc.get('avg_tps')):,.0f} & "
                f"{asc_p95:,.0f} & "
                f"{to_float(pbft.get('avg_tps')):,.0f} & "
                f"{pbft_p95:,.0f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Saved: {path}")


def plot(rows, results_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping plots.")
        return

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["protocol"]].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.suptitle(
        "Batch-Size Sensitivity (100 nodes, 802.11g, 100 tx/s/node)",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0]
    for protocol, data in grouped.items():
        data = sorted(data, key=lambda r: r["batch_size"])
        ax.plot(
            [r["batch_size"] for r in data],
            [r["avg_tps"] for r in data],
            marker="o",
            linewidth=2.2,
            label=PROTOCOL_LABELS.get(protocol, protocol),
            color=PROTOCOL_COLORS.get(protocol),
        )
    ax.axvline(50, color="#444444", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_xlabel("Logical batch size B (TX/state update)")
    ax.set_ylabel("Accepted TPS")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for protocol, data in grouped.items():
        data = sorted(data, key=lambda r: r["batch_size"])
        ax.plot(
            [r["batch_size"] for r in data],
            [r["direct_e2e_p95_ms"] or r["estimated_e2e_p95_ms"] for r in data],
            marker="s",
            linewidth=2.2,
            label=PROTOCOL_LABELS.get(protocol, protocol),
            color=PROTOCOL_COLORS.get(protocol),
        )
    ax.axvline(50, color="#444444", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_xlabel("Logical batch size B (TX/state update)")
    ax.set_ylabel("End-to-end p95 latency (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(results_dir, f"batch_sweep.{ext}")
        plt.savefig(out, dpi=180, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join("results", "batch_sweep"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    rows = latest_by_protocol_batch(load_summary(args.results_dir))
    if not rows:
        print(f"[ERROR] No batch sweep data found in {args.results_dir}")
        raise SystemExit(1)

    print_table(rows)
    write_latex_table(rows, args.results_dir)
    if not args.no_plot:
        plot(rows, args.results_dir)


if __name__ == "__main__":
    main()
