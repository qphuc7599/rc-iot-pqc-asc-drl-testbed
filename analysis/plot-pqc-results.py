#!/usr/bin/env python3
"""
plot-pqc-results.py — Publication-quality PQC benchmark visualization
                      Groups results by device type (7 heterogeneous IoT chips)

Output: results/pqc_*.png (4 figures)
"""

import csv, os, sys, json, statistics, argparse

def load_csv(filepath):
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    row[k] = float(v) if '.' in str(v) else int(v)
                except (ValueError, TypeError):
                    pass
            data.append(row)
    return data

def load_node_map(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def get_device_type(node_id, node_map):
    return node_map.get(str(node_id), {}).get("device_type", "Unknown")

def print_summary(data, node_map):
    algo = data[0].get('algorithm', 'Unknown')
    n = len(data)

    # Group by device type
    by_type = {}
    for d in data:
        nid = d['node_id']
        dtype = get_device_type(nid, node_map)
        by_type.setdefault(dtype, []).append(d)

    print("=" * 80)
    print(f"  ML-DSA Benchmark — {algo} | {n} IoT Containers (Heterogeneous)")
    print("=" * 80)
    print(f"  Key sizes: PK={data[0].get('pk_bytes','?')}B, "
          f"SK={data[0].get('sk_bytes','?')}B, Sig={data[0].get('sig_bytes','?')}B")
    print("-" * 80)
    print(f"  {'Device Type':<15} {'N':>3} {'KeyGen(µs)':>12} {'Sign(µs)':>12} "
          f"{'Verify(µs)':>12} {'Energy(mJ)':>12}")
    print("-" * 80)

    for dtype in ["ESP32", "ESP32-S3", "STM32L4-M4", "STM32F4-M4",
                   "STM32H7-M7", "nRF52840", "RP2040"]:
        nodes = by_type.get(dtype, [])
        if not nodes:
            continue
        kg = [d['keygen_avg_us'] for d in nodes]
        sn = [d['sign_avg_us'] for d in nodes]
        vf = [d['verify_avg_us'] for d in nodes]
        e_total = [(d.get('energy_keygen_esp32_mj',0) +
                    d.get('energy_sign_esp32_mj',0) +
                    d.get('energy_verify_esp32_mj',0)) for d in nodes]
        print(f"  {dtype:<15} {len(nodes):3d} {statistics.mean(kg):12.1f} "
              f"{statistics.mean(sn):12.1f} {statistics.mean(vf):12.1f} "
              f"{statistics.mean(e_total):12.4f}")

    # Overall
    all_kg = [d['keygen_avg_us'] for d in data]
    all_sn = [d['sign_avg_us'] for d in data]
    all_vf = [d['verify_avg_us'] for d in data]
    print("-" * 80)
    print(f"  {'OVERALL':<15} {n:3d} {statistics.mean(all_kg):12.1f} "
          f"{statistics.mean(all_sn):12.1f} {statistics.mean(all_vf):12.1f}")
    print("=" * 80)

def plot_results(data, node_map, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n[WARN] matplotlib not installed. pip install matplotlib")
        return

    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.3,
    })

    # Group data by device type
    DEVICE_ORDER = ["nRF52840", "STM32L4-M4", "RP2040", "ESP32", "ESP32-S3",
                    "STM32F4-M4", "STM32H7-M7"]
    COLORS = {
        "ESP32": "#4CAF50", "ESP32-S3": "#8BC34A", "STM32L4-M4": "#2196F3",
        "STM32F4-M4": "#FF9800", "STM32H7-M7": "#F44336",
        "nRF52840": "#9C27B0", "RP2040": "#00BCD4",
    }
    CPU_LABELS = {
        "ESP32": "0.050", "ESP32-S3": "0.060", "STM32L4-M4": "0.030",
        "STM32F4-M4": "0.080", "STM32H7-M7": "0.150",
        "nRF52840": "0.020", "RP2040": "0.040",
    }

    by_type = {}
    for d in data:
        dtype = get_device_type(d['node_id'], node_map)
        by_type.setdefault(dtype, []).append(d)

    types_present = [t for t in DEVICE_ORDER if t in by_type]

    # ============================================================
    # Figure 1: Grouped Bar Chart — Avg latency per device type
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(types_present))
    width = 0.25

    kg_avgs, sn_avgs, vf_avgs = [], [], []
    kg_errs, sn_errs, vf_errs = [], [], []
    for t in types_present:
        nodes = by_type[t]
        kg = [d['keygen_avg_us'] for d in nodes]
        sn = [d['sign_avg_us'] for d in nodes]
        vf = [d['verify_avg_us'] for d in nodes]
        kg_avgs.append(statistics.mean(kg))
        sn_avgs.append(statistics.mean(sn))
        vf_avgs.append(statistics.mean(vf))
        kg_errs.append(statistics.stdev(kg) if len(kg) > 1 else 0)
        sn_errs.append(statistics.stdev(sn) if len(sn) > 1 else 0)
        vf_errs.append(statistics.stdev(vf) if len(vf) > 1 else 0)

    bars1 = ax.bar(x - width, kg_avgs, width, yerr=kg_errs, label='KeyGen',
                   color='#2196F3', alpha=0.85, capsize=3, edgecolor='white')
    bars2 = ax.bar(x, sn_avgs, width, yerr=sn_errs, label='Sign',
                   color='#FF9800', alpha=0.85, capsize=3, edgecolor='white')
    bars3 = ax.bar(x + width, vf_avgs, width, yerr=vf_errs, label='Verify',
                   color='#4CAF50', alpha=0.85, capsize=3, edgecolor='white')

    xlabels = [f"{t}\n(CPU={CPU_LABELS.get(t,'')})" for t in types_present]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel('Latency (µs)')
    ax.set_title('ML-DSA-44 Latency by Device Type — 100 IoT Containers')
    ax.legend(loc='upper left')

    plt.tight_layout()
    path = os.path.join(output_dir, 'pqc_latency_by_device.png')
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()

    # ============================================================
    # Figure 2: Box Plot — Sign latency distribution per device
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('ML-DSA-44 Latency Distribution by Device Type (Outliers clipped P99.5)', fontsize=14)

    for idx, (op_name, op_key) in enumerate([
        ('KeyGen', 'keygen_avg_us'), ('Sign', 'sign_avg_us'), ('Verify', 'verify_avg_us')
    ]):
        ax = axes[idx]
        box_data = []
        box_labels = []
        box_colors = []
        for t in types_present:
            vals = [d[op_key] for d in by_type[t]]
            box_data.append(vals)
            box_labels.append(t)
            box_colors.append(COLORS.get(t, 'gray'))

        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                        showfliers=False)  # Bug fix: hide extreme outliers
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Latency (µs)')
        ax.set_title(op_name)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    path = os.path.join(output_dir, 'pqc_boxplot_by_device.png')
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()

    # ============================================================
    # Figure 3: Energy consumption by device type
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(types_present))

    POWER_MW = {
        "nRF52840": 23.0,
        "STM32L4-M4": 30.0,
        "RP2040": 45.0,
        "STM32F4-M4": 50.0,
        "STM32H7-M7": 90.0,
        "ESP32": 160.0,
        "ESP32-S3": 170.0,
    }

    e_kg_avg, e_sn_avg, e_vf_avg = [], [], []
    for t in types_present:
        nodes = by_type[t]
        power = POWER_MW.get(t, 160.0)
        # Energy (mJ) = Time (us) * Power (mW) / 1,000,000
        e_kg_avg.append(statistics.mean([(d['keygen_avg_us'] * power) / 1e6 for d in nodes]))
        e_sn_avg.append(statistics.mean([(d['sign_avg_us'] * power) / 1e6 for d in nodes]))
        e_vf_avg.append(statistics.mean([(d['verify_avg_us'] * power) / 1e6 for d in nodes]))

    ax.bar(x - width, e_kg_avg, width, label='KeyGen', color='#2196F3', alpha=0.85)
    ax.bar(x, e_sn_avg, width, label='Sign', color='#FF9800', alpha=0.85)
    ax.bar(x + width, e_vf_avg, width, label='Verify', color='#4CAF50', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel('Energy (mJ)')
    ax.set_title('ML-DSA-44 Heterogeneous Energy Consumption (Accurate HW Models)')
    ax.legend()

    # Add total energy labels
    for i, t in enumerate(types_present):
        total = e_kg_avg[i] + e_sn_avg[i] + e_vf_avg[i]
        ax.text(i, max(e_kg_avg[i], e_sn_avg[i], e_vf_avg[i]) * 1.1,
                f'Σ={total:.3f}', ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, 'pqc_energy_by_device.png')
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()

    # ============================================================
    # Figure 4: Per-node scatter — all 100 nodes colored by type
    # ============================================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('ML-DSA-44 Per-Node Performance — 100 Heterogeneous IoT Containers', fontsize=14)

    ops = [('KeyGen', 'keygen_avg_us'), ('Sign', 'sign_avg_us'), ('Verify', 'verify_avg_us')]
    for ax, (op_name, op_key) in zip(axes, ops):
        for t in types_present:
            nodes = by_type[t]
            nids = [d['node_id'] for d in nodes]
            vals = [d[op_key] for d in nodes]
            ax.scatter(nids, vals, c=COLORS.get(t, 'gray'), label=t,
                      alpha=0.7, s=20, edgecolors='white', linewidth=0.5)
        ax.set_ylabel(f'{op_name} (µs)')
        if ax == axes[0]:
            ax.legend(loc='upper right', fontsize=8, ncol=4)

    axes[-1].set_xlabel('Node ID')
    plt.tight_layout()
    path = os.path.join(output_dir, 'pqc_per_node_scatter.png')
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()

    print(f"\n  All figures saved to {output_dir}/")


def plot_security_levels(output_dir):
    """Figure 5: ML-DSA Security Level Comparison (44 vs 65 vs 87)"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    # Check for level-specific summary files
    levels = []
    level_data = {}
    for level in [44, 65, 87]:
        path = os.path.join(output_dir, f'summary_{level}.csv')
        if os.path.exists(path):
            data = load_csv(path)
            if data:
                levels.append(level)
                level_data[level] = data

    if len(levels) < 2:
        print("  [SKIP] Security level comparison: need summary_44.csv, summary_65.csv, summary_87.csv")
        return

    # NIST security parameters
    LEVEL_INFO = {
        44: {'nist': 2, 'pk': 1312, 'sig': 2420, 'color': '#4CAF50'},
        65: {'nist': 3, 'pk': 1952, 'sig': 3293, 'color': '#FF9800'},
        87: {'nist': 5, 'pk': 2592, 'sig': 4595, 'color': '#F44336'},
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle('ML-DSA Security Level Comparison — NIST FIPS 204\n'
                 '100 Heterogeneous IoT Nodes', fontsize=14, fontweight='bold')

    x = np.arange(len(levels))
    labels = [f'ML-DSA-{l}\n(NIST L{LEVEL_INFO[l]["nist"]})' for l in levels]
    colors = [LEVEL_INFO[l]['color'] for l in levels]

    # Panel 1: Latency comparison (KeyGen, Sign, Verify)
    ax = axes[0]
    width = 0.25
    kg_avgs = [statistics.mean([d['keygen_avg_us'] for d in level_data[l]]) for l in levels]
    sn_avgs = [statistics.mean([d['sign_avg_us'] for d in level_data[l]]) for l in levels]
    vf_avgs = [statistics.mean([d['verify_avg_us'] for d in level_data[l]]) for l in levels]

    ax.bar(x - width, kg_avgs, width, label='KeyGen', color='#2196F3', alpha=0.85)
    ax.bar(x, sn_avgs, width, label='Sign', color='#FF9800', alpha=0.85)
    ax.bar(x + width, vf_avgs, width, label='Verify', color='#4CAF50', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Latency (µs)')
    ax.set_title('Operation Latency')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Signature + Key sizes
    ax2 = axes[1]
    sig_sizes = [LEVEL_INFO[l]['sig'] for l in levels]
    pk_sizes = [LEVEL_INFO[l]['pk'] for l in levels]

    ax2.bar(x - 0.15, pk_sizes, 0.3, label='Public Key', color='#03A9F4', alpha=0.85)
    ax2.bar(x + 0.15, sig_sizes, 0.3, label='Signature', color='#E91E63', alpha=0.85)

    for i, (pk, sig) in enumerate(zip(pk_sizes, sig_sizes)):
        ax2.text(i - 0.15, pk + 50, f'{pk}B', ha='center', fontsize=9)
        ax2.text(i + 0.15, sig + 50, f'{sig}B', ha='center', fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Size (bytes)')
    ax2.set_title('Key & Signature Sizes')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Cost ratio (normalized to ML-DSA-44)
    ax3 = axes[2]
    base_kg = kg_avgs[0]
    base_sn = sn_avgs[0]
    base_vf = vf_avgs[0]

    norm_kg = [v / base_kg for v in kg_avgs]
    norm_sn = [v / base_sn for v in sn_avgs]
    norm_vf = [v / base_vf for v in vf_avgs]
    norm_sig = [s / sig_sizes[0] for s in sig_sizes]

    ax3.plot(x, norm_kg, 'o-', label='KeyGen', color='#2196F3', linewidth=2, markersize=8)
    ax3.plot(x, norm_sn, 's-', label='Sign', color='#FF9800', linewidth=2, markersize=8)
    ax3.plot(x, norm_vf, '^-', label='Verify', color='#4CAF50', linewidth=2, markersize=8)
    ax3.plot(x, norm_sig, 'D--', label='Sig Size', color='#E91E63', linewidth=2, markersize=8)
    ax3.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel('Relative Cost (ML-DSA-44 = 1.0)')
    ax3.set_title('Security vs Performance Tradeoff')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'pqc_security_levels.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot PQC benchmark results')
    parser.add_argument('--csv', default='results/summary.csv')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] File not found: {args.csv}")
        sys.exit(1)

    data = load_csv(args.csv)
    node_map = load_node_map('results/node_map.json')

    print_summary(data, node_map)
    plot_results(data, node_map, os.path.dirname(args.csv))
    plot_security_levels(os.path.dirname(args.csv))

if __name__ == '__main__':
    main()
