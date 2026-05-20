#!/usr/bin/env python3
"""
plot-tps.py — State Channel TPS Analysis

Fix theo nhận xét chuyên gia:
- Tách off-chain signing throughput vs end-to-end network TPS
- Fix histogram (clip P99.5)
- Box plots by device type (phải khác nhau nếu CPU scale đúng)
- Scalability curve: bỏ Target TPS, thêm burst TPS
"""

import csv
import json
import os
import sys
import argparse
import statistics
import glob
from collections import defaultdict

def load_gateway_summary(results_dir):
    path = os.path.join(results_dir, 'gateway_summary.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def load_tx_logs(results_dir, num_nodes=100):
    all_txs = []
    for i in range(1, num_nodes + 1):
        path = os.path.join(results_dir, f'tx_node_{i}.csv')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row['node_id'] = int(row['node_id'])
                    # Support both old and new CSV format
                    if 'emulated_sign_us' in row:
                        row['sign_time_us'] = float(row['emulated_sign_us'])
                        row['raw_sign_us'] = float(row.get('sign_time_us', row['emulated_sign_us']))
                    else:
                        row['sign_time_us'] = float(row['sign_time_us'])
                        row['raw_sign_us'] = row['sign_time_us']
                    row['pkt_size'] = int(row['pkt_size'])
                    row['device_type'] = row.get('device_type', get_device_type(row['node_id']))
                    all_txs.append(row)
                except (ValueError, KeyError):
                    pass
    return all_txs

def get_device_type(node_id):
    DEVICE_ORDER = ["ESP32","ESP32-S3","STM32L4-M4","STM32F4-M4",
                    "STM32H7-M7","nRF52840","RP2040"]
    return DEVICE_ORDER[(int(node_id) - 1) % len(DEVICE_ORDER)]

def print_summary(summary, all_txs):
    print("=" * 70)
    print("  State Channel — Dual Metric Analysis")
    print("=" * 70)

    if all_txs:
        sign_times = [t['sign_time_us'] for t in all_txs]
        p99 = sorted(sign_times)[int(len(sign_times) * 0.99)]
        clean = [t for t in sign_times if t <= p99]

        by_type = defaultdict(list)
        for t in all_txs:
            by_type[t['device_type']].append(t['sign_time_us'])

        print(f"\n  [METRIC 1] Off-chain ML-DSA Signing Throughput:")
        print(f"    Total observations: {len(sign_times)}")
        print(f"    Mean sign time:  {statistics.mean(clean):.1f} µs")
        print(f"    Median:          {statistics.median(clean):.1f} µs")
        print(f"    P99:             {p99:.1f} µs")

        for dtype in sorted(by_type.keys()):
            vals = by_type[dtype]
            per_node_tps = 1e6 / statistics.mean(vals) if statistics.mean(vals) > 0 else 0
            print(f"    {dtype:15s}: median={statistics.median(vals):8.1f}µs "
                  f"(max TPS/node: {per_node_tps:6.0f})")

    if summary:
        print(f"\n  [METRIC 2] End-to-End Network TPS (via NS-3 WiFi):")
        print(f"    Duration:    {summary.get('duration_s', 0):.1f}s")
        print(f"    Total TX:    {summary.get('total_tx', 0)}")
        print(f"    Avg TPS:     {summary.get('avg_tps', 0):.1f}")
        print(f"    Batches:     {summary.get('total_batches', 0)}")

        # Compute burst TPS (excluding zero periods)
        if 'batches' in summary and summary['batches']:
            batches = summary['batches']
            if batches[0].get('timestamp'):
                t0 = batches[0]['timestamp']
                times = [(b['timestamp'] - t0) for b in batches]
                max_time = int(times[-1]) + 1
                batch_size = batches[0].get('batch_size', summary.get('batch_size', 50))

                active_seconds = 0
                for sec in range(max_time):
                    count = sum(1 for t in times if sec <= t < sec + 1)
                    if count > 0:
                        active_seconds += 1

                burst_tps = summary.get('total_tx', 0) / max(active_seconds, 1)
                zero_seconds = max_time - active_seconds
                print(f"    Burst TPS:   {burst_tps:.1f} (excluding {zero_seconds}s dead periods)")
                print(f"    Active time: {active_seconds}s / {max_time}s ({100*active_seconds/max(max_time,1):.0f}%)")

    print("=" * 70)

def plot_results(summary, all_txs, results_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not installed")
        return

    # --- Fig 1: End-to-End Network TPS ---
    if summary and 'batches' in summary:
        batches = summary['batches']
        if batches and batches[0].get('timestamp'):
            fig, axes = plt.subplots(3, 1, figsize=(12, 11))
            fig.suptitle('Off-chain State Channel — End-to-End TPS via NS-3 802.11g\n'
                         f'(100 nodes, batch_size={summary.get("batch_size",50)})',
                         fontsize=14)

            t0 = batches[0]['timestamp']
            times = [(b['timestamp'] - t0) for b in batches]
            # Each batch contains multiple state channel updates,
            # each update represents tx_count off-chain TXs.
            # Use 'verified' field = actual off-chain TXs verified in each batch
            cumulative_tx = []
            running_total = 0
            for b in batches:
                running_total += b.get('verified', b.get('batch_size', 50))
                cumulative_tx.append(running_total)

            # Panel 1: Cumulative
            axes[0].plot(times, cumulative_tx, 'b-', linewidth=2)
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Cumulative Transactions')
            axes[0].set_title('Cumulative Transaction Throughput')
            axes[0].grid(True, alpha=0.3)

            # Count ACTUAL verified TX per second, not packets!
            max_time = int(times[-1]) + 1
            inst_tps = []
            for sec in range(max_time):
                # Find batches that fell in this second
                sec_txs = sum(
                    b.get('verified', b.get('batch_size', 50))
                    for b, t in zip(batches, times)
                    if sec <= t < sec + 1
                )
                inst_tps.append(sec_txs)

            avg_tps = summary.get('avg_tps', np.mean(inst_tps))
            zero_count = sum(1 for t in inst_tps if t == 0)
            active_tps = [t for t in inst_tps if t > 0]
            burst_tps = np.mean(active_tps) if active_tps else 0

            colors = ['#FF5722' if t == 0 else '#2196F3' for t in inst_tps]
            axes[1].bar(range(max_time), inst_tps, width=0.9, color=colors, alpha=0.7)
            axes[1].axhline(y=avg_tps, color='red', linestyle='--',
                           linewidth=2, label=f'Avg: {avg_tps:.0f} TPS')
            axes[1].axhline(y=burst_tps, color='green', linestyle='-.',
                           linewidth=2, label=f'Burst: {burst_tps:.0f} TPS')
            if zero_count > 0:
                axes[1].annotate(f'{zero_count}s zero-TPS\n(WiFi contention)',
                               xy=(max_time * 0.7, burst_tps * 0.1), fontsize=9,
                               color='red', fontweight='bold')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('TPS')
            axes[1].set_title('Instantaneous TPS (1-second windows)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Panel 3: Batch processing
            batch_ids = [b['batch_id'] for b in batches]
            agg_times = [b.get('aggregate_time_us', 0) for b in batches]
            verify_times = [b.get('verify_time_us', 0) for b in batches]

            axes[2].bar(batch_ids, verify_times, label='Verify (µs)',
                       alpha=0.7, color='#2196F3')
            axes[2].bar(batch_ids, agg_times, bottom=verify_times,
                       label='Aggregate (µs)', alpha=0.7, color='#FF9800')
            axes[2].set_xlabel('Batch ID')
            axes[2].set_ylabel('Time (µs)')
            axes[2].set_title('Per-Batch Processing Time')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            path = os.path.join(results_dir, 'tps_throughput.png')
            plt.savefig(path, dpi=150)
            print(f"  Saved: {path}")
            plt.close()

    # --- Fig 2: Sign Performance (clipped + by device type) ---
    if all_txs:
        sign_times = [t['sign_time_us'] for t in all_txs]
        p995 = sorted(sign_times)[int(len(sign_times) * 0.995)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle('ML-DSA Sign Performance — Heterogeneous IoT Devices', fontsize=14)

        # Histogram (clipped)
        clipped = [t for t in sign_times if t <= p995]
        axes[0].hist(clipped, bins=80, color='#4CAF50', alpha=0.7,
                    edgecolor='black', linewidth=0.3)
        mean_val = statistics.mean(clipped)
        med_val = statistics.median(clipped)
        axes[0].axvline(mean_val, color='red', linestyle='--',
                       label=f'Mean: {mean_val:.0f}µs')
        axes[0].axvline(med_val, color='blue', linestyle=':',
                       label=f'Median: {med_val:.0f}µs')
        axes[0].set_xlabel('Sign Time (µs)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Distribution (n={len(clipped)}, clipped P99.5={p995:.0f}µs)')
        axes[0].legend()

        # Box plots by device type
        DEVICE_ORDER = ["STM32H7-M7", "ESP32-S3", "ESP32", "STM32F4-M4",
                        "RP2040", "STM32L4-M4", "nRF52840"]

        by_type = defaultdict(list)
        for t in all_txs:
            by_type[t['device_type']].append(t['sign_time_us'])

        types_present = [t for t in DEVICE_ORDER if t in by_type]
        box_data = [by_type[t] for t in types_present]

        bp = axes[1].boxplot(box_data, labels=[t.replace('-', '\n') for t in types_present],
                            showfliers=False, patch_artist=True)
        colors = ['#4CAF50','#03A9F4','#2196F3','#8BC34A','#FF9800','#FF5722','#F44336']
        for patch, c in zip(bp['boxes'], colors[:len(types_present)]):
            patch.set_facecolor(c + '60')
        axes[1].set_xlabel('Device Type (sorted by speed)')
        axes[1].set_ylabel('Sign Time (µs)')
        axes[1].set_title('Sign Time by Device Type (no outliers)')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        path = os.path.join(results_dir, 'tps_sign_distribution.png')
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")
        plt.close()

    # --- Fig 3: Scalability curve ---
    plot_scalability_curve(results_dir)

def plot_scalability_curve(results_dir):
    """Plot TPS scalability — WITHOUT misleading Target line."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    rate_dirs = glob.glob(os.path.join(results_dir, 'tps_rate_*'))
    if len(rate_dirs) < 2:
        return

    rates = []
    avg_tps_list = []
    burst_tps_list = []

    for rd in sorted(rate_dirs):
        rate = int(os.path.basename(rd).replace('tps_rate_', ''))
        summary_path = os.path.join(rd, 'gateway_summary.json')
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            s = json.load(f)

        rates.append(rate)
        avg_tps_list.append(s.get('avg_tps', 0))

        # Compute burst TPS using actual verified TX, not packets
        if 'batches' in s and s['batches'] and s['batches'][0].get('timestamp'):
            batches = s['batches']
            t0 = batches[0]['timestamp']
            times = [(b['timestamp'] - t0) for b in batches]
            max_time = int(times[-1]) + 1

            # Count active seconds and their TX
            active = 0
            for sec in range(max_time):
                if sum(1 for t in times if sec <= t < sec + 1) > 0:
                    active += 1
            burst_tps_list.append(s.get('total_tx', 0) / max(active, 1))
        else:
            burst_tps_list.append(avg_tps_list[-1])

    if len(rates) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rates, avg_tps_list, 'bo-', linewidth=2, markersize=10,
            label='Avg TPS (incl. dead periods)')
    ax.plot(rates, burst_tps_list, 'gs--', linewidth=2, markersize=8,
            label='Burst TPS (active periods only)')

    for i, (r, a, b) in enumerate(zip(rates, avg_tps_list, burst_tps_list)):
        ax.annotate(f'{a:.0f}', (r, a), textcoords="offset points",
                   xytext=(0, -18), ha='center', fontsize=10, fontweight='bold',
                   color='blue')
        ax.annotate(f'{b:.0f}', (r, b), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=10, fontweight='bold',
                   color='green')

    ax.set_xlabel('TX Rate per Node (tx/s)', fontsize=12)
    ax.set_ylabel('Measured TPS', fontsize=12)
    ax.set_title('State Channel Scalability — 802.11g WiFi (100 nodes)\n'
                 'Network layer is the bottleneck, not ML-DSA computation',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Annotation: WiFi contention explanation
    ax.annotate('802.11g CSMA/CA:\ncollision domain saturation\n→ throughput degrades',
               xy=(rates[-1], avg_tps_list[-1]),
               xytext=(rates[-1]*0.3, avg_tps_list[0]*0.7),
               arrowprops=dict(arrowstyle='->', color='red'),
               fontsize=9, color='red', style='italic')

    plt.tight_layout()
    path = os.path.join(results_dir, 'tps_scalability.png')
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results/')
    args = parser.parse_args()

    summary = load_gateway_summary(args.results_dir)
    all_txs = load_tx_logs(args.results_dir)

    if not summary and not all_txs:
        print("[ERROR] No results found in", args.results_dir)
        sys.exit(1)

    print_summary(summary, all_txs)
    plot_results(summary, all_txs, args.results_dir)

if __name__ == '__main__':
    main()
