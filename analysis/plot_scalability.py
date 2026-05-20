"""
Publication-Quality Scalability Figures — RQ5
=============================================
NS-3 Testbed Results: ASC vs PBFT vs BLS at N = 10, 25, 50, 75, 100
Target: Q1 Journal (IEEE/ACM format)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Load real NS-3 results ──
SCALE_DIR = 'results/scalability'
N_vals = [10, 25, 50, 75, 100]

# Actual NS-3 testbed data
ASC  = [787, 1980, 3834, 4020, 4408]
PBFT = [20,  17,   15,   16,   18]
BLS  = [29,  25,   20,   20,   23]
PBFT_BATCHED = [0, 0, 0, 0, 0]
ASC_ECDSA = [0, 0, 0, 0, 0]

# Try loading from summary.json
if os.path.exists(f'{SCALE_DIR}/summary.json'):
    with open(f'{SCALE_DIR}/summary.json') as f:
        data = json.load(f)
    summary_asc = data.get('asc', data.get('asc_tps', []))
    if all(v > 0 for v in summary_asc):
        ASC  = summary_asc
        PBFT = data.get('pbft', data.get('pbft_tps', PBFT))
        BLS  = data.get('bls', BLS)
        PBFT_BATCHED = data.get('pbft_batched', PBFT_BATCHED)
        ASC_ECDSA = data.get('asc_ecdsa', ASC_ECDSA)
        print(f"Loaded NS-3 results from {SCALE_DIR}/summary.json")

# ── Publication Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.4,
    'lines.linewidth': 2.2,
    'lines.markersize': 8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
})

COLORS = {
    'ASC':  '#c0392b',   # Deep red
    'PBFT': '#2980b9',   # Steel blue
    'BLS':  '#27ae60',   # Forest green
    'PBFT50': '#8e44ad',
    'ASC_ECDSA': '#d68910',
}
MARKERS = {'ASC': 'o', 'PBFT': 's', 'BLS': '^', 'PBFT50': 'D', 'ASC_ECDSA': 'P'}
LABELS  = {
    'ASC':  'ASC (Off-chain SC + ML-DSA)',
    'PBFT': 'PBFT (On-chain + ECDSA)',
    'BLS':  'BLS (On-chain Aggregate)',
    'PBFT50': 'PBFT Batched (ECDSA, batch=50)',
    'ASC_ECDSA': 'ASC No-PQC (ECDSA, batch=50)',
}

os.makedirs(SCALE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  FIGURE 1: Main Comparison — 2-panel (Linear + Log)
# ═══════════════════════════════════════════════════════════
def plot_five_protocols():
    series = [
        ('PBFT', PBFT),
        ('BLS', BLS),
        ('ASC', ASC),
        ('PBFT50', PBFT_BATCHED),
        ('ASC_ECDSA', ASC_ECDSA),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle('Scalability Across Five Protocol Configurations',
                 fontsize=14, fontweight='bold', y=0.98)

    for ax, yscale, panel in [(ax1, 'linear', '(a) Linear Scale'),
                               (ax2, 'log', '(b) Logarithmic Scale')]:
        for proto, vals in series:
            if vals and any(v > 0 for v in vals):
                ax.plot(N_vals, vals,
                        marker=MARKERS[proto], color=COLORS[proto],
                        linewidth=2.3, markersize=8, markeredgewidth=1.3,
                        markeredgecolor='white', label=LABELS[proto],
                        zorder=3)

        ax.set_xlabel('Number of Nodes ($N$)')
        ax.set_ylabel('Peak Throughput (TPS)')
        ax.set_title(panel, fontsize=12, pad=8)
        ax.set_yscale(yscale)
        ax.set_xticks(N_vals)
        ax.grid(True, alpha=0.4, linestyle='--', which='both')
        ax.legend(loc='best', framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = f'{SCALE_DIR}/fig_scalability_5protocols.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


def plot_main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Fig. 8: Throughput Scalability — TPS vs Number of IoT Nodes',
                 fontsize=14, fontweight='bold', y=0.98)

    for ax, yscale, panel in [(ax1, 'linear', '(a) Linear Scale'),
                               (ax2, 'log', '(b) Logarithmic Scale')]:
        for proto, vals in [('ASC', ASC), ('PBFT', PBFT), ('BLS', BLS)]:
            ax.plot(N_vals, vals,
                    marker=MARKERS[proto], color=COLORS[proto],
                    linewidth=2.5, markersize=9, markeredgewidth=1.5,
                    markeredgecolor='white', label=LABELS[proto],
                    zorder=3)

        ax.set_xlabel('Number of Nodes ($N$)', fontsize=12)
        ax.set_ylabel('Peak Throughput (TPS)', fontsize=12)
        ax.set_title(panel, fontsize=12, pad=8)
        ax.set_yscale(yscale)
        ax.set_xticks(N_vals)
        ax.grid(True, alpha=0.4, linestyle='--', which='both')
        ax.legend(loc='upper left' if yscale == 'linear' else 'center left',
                  framealpha=0.9, edgecolor='#cccccc')

        # Annotate key values
        if yscale == 'linear':
            ax.annotate(f'{int(round(ASC[-1])):,}', xy=(100, ASC[-1]),
                       xytext=(85, ASC[-1] + 300),
                       fontsize=9, fontweight='bold', color=COLORS['ASC'],
                       arrowprops=dict(arrowstyle='->', color=COLORS['ASC'], lw=1.2))
            ax.annotate(f'{int(round(PBFT[-1]))}', xy=(100, PBFT[-1]),
                       xytext=(85, PBFT[-1] + 500),
                       fontsize=9, fontweight='bold', color=COLORS['PBFT'],
                       arrowprops=dict(arrowstyle='->', color=COLORS['PBFT'], lw=1.2))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = f'{SCALE_DIR}/fig8_scalability_main.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


# ═══════════════════════════════════════════════════════════
#  FIGURE 2: ASC/Baseline Ratio (Speedup)
# ═══════════════════════════════════════════════════════════
def plot_speedup():
    fig, ax = plt.subplots(figsize=(8, 5))

    ratio_pbft = [a / max(p, 1) for a, p in zip(ASC, PBFT)]
    ratio_bls  = [a / max(b, 1) for a, b in zip(ASC, BLS)]

    ax.bar([n - 1.5 for n in N_vals], ratio_pbft, width=3, color=COLORS['PBFT'],
           alpha=0.85, label='ASC / PBFT', edgecolor='white', linewidth=0.8)
    ax.bar([n + 1.5 for n in N_vals], ratio_bls, width=3, color=COLORS['BLS'],
           alpha=0.85, label='ASC / BLS', edgecolor='white', linewidth=0.8)

    # Value labels on bars
    for i, n in enumerate(N_vals):
        ax.text(n - 1.5, ratio_pbft[i] + 5, f'{ratio_pbft[i]:.0f}×',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=COLORS['PBFT'])
        ax.text(n + 1.5, ratio_bls[i] + 5, f'{ratio_bls[i]:.0f}×',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=COLORS['BLS'])

    ax.set_xlabel('Number of Nodes ($N$)', fontsize=12)
    ax.set_ylabel('Speedup Ratio (ASC / Baseline)', fontsize=12)
    ax.set_title('Fig. 9: ASC Speedup Over On-chain Baselines',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(N_vals)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(max(ratio_pbft), max(ratio_bls)) * 1.15)

    plt.tight_layout()
    out = f'{SCALE_DIR}/fig9_speedup_ratio.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


# ═══════════════════════════════════════════════════════════
#  FIGURE 3: Stacked Area — Protocol Capacity Breakdown
# ═══════════════════════════════════════════════════════════
def plot_area():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.fill_between(N_vals, 0, PBFT, alpha=0.3, color=COLORS['PBFT'],
                    label=LABELS['PBFT'])
    ax.fill_between(N_vals, PBFT, [p + b for p, b in zip(PBFT, BLS)],
                    alpha=0.3, color=COLORS['BLS'], label=LABELS['BLS'])
    ax.plot(N_vals, ASC, '-o', color=COLORS['ASC'], linewidth=3,
            markersize=10, markeredgecolor='white', markeredgewidth=2,
            label=LABELS['ASC'], zorder=5)

    # Shade the gap = ASC advantage
    ax.fill_between(N_vals, [p + b for p, b in zip(PBFT, BLS)], ASC,
                    alpha=0.12, color=COLORS['ASC'], hatch='///')
    ax.annotate('Off-chain\nAdvantage', xy=(60, 2500), fontsize=10,
               fontstyle='italic', color=COLORS['ASC'], ha='center')

    ax.set_xlabel('Number of Nodes ($N$)', fontsize=12)
    ax.set_ylabel('Throughput (TPS)', fontsize=12)
    ax.set_title('Fig. 10: Protocol Capacity — Off-chain vs On-chain',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xticks(N_vals)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    out = f'{SCALE_DIR}/fig10_capacity_area.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


# ═══════════════════════════════════════════════════════════
#  FIGURE 4: Summary Table (publication-ready)
# ═══════════════════════════════════════════════════════════
def plot_table():
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis('off')

    columns = ['$N$', 'ASC\n(TPS)', 'PBFT\n(TPS)', 'BLS\n(TPS)',
               'ASC/PBFT', 'ASC/BLS', 'ASC\nGrowth']
    cell_data = []
    for i, n in enumerate(N_vals):
        a = int(round(ASC[i]))
        p = int(round(PBFT[i]))
        b = int(round(BLS[i]))
        r_pbft = a / max(p, 1)
        r_bls  = a / max(b, 1)
        growth = a / int(round(ASC[0])) if ASC[0] > 0 else 0
        cell_data.append([
            str(n),
            f'{a:,}',
            f'{p:,}',
            f'{b:,}',
            f'{r_pbft:.1f}×',
            f'{r_bls:.1f}×',
            f'{growth:.1f}×',
        ])

    table = ax.table(cellText=cell_data, colLabels=columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    # Header style
    for col in range(len(columns)):
        table[0, col].set_facecolor('#1a1a2e')
        table[0, col].set_text_props(color='white', fontweight='bold', fontsize=10)

    # Highlight ASC column
    for row in range(1, len(cell_data) + 1):
        table[row, 1].set_facecolor('#fce4e4')
        table[row, 1].set_text_props(fontweight='bold')
        # Alternating rows
        if row % 2 == 0:
            for col in range(len(columns)):
                if col != 1:
                    table[row, col].set_facecolor('#f5f5f5')

    ax.set_title('Table V: Scalability Results — NS-3 WiFi 802.11g Testbed (100 tx/s/node)',
                 fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    out = f'{SCALE_DIR}/table5_scalability.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


# ═══════════════════════════════════════════════════════════
#  FIGURE 5: Combined 4-panel (for paper single figure)
# ═══════════════════════════════════════════════════════════
def plot_combined():
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel (a): Linear
    ax1 = fig.add_subplot(gs[0, 0])
    for proto, vals in [('ASC', ASC), ('PBFT', PBFT), ('BLS', BLS)]:
        ax1.plot(N_vals, vals, marker=MARKERS[proto], color=COLORS[proto],
                 linewidth=2.5, markersize=9, markeredgecolor='white',
                 markeredgewidth=1.5, label=LABELS[proto])
    ax1.set_xlabel('Number of Nodes ($N$)')
    ax1.set_ylabel('Peak TPS')
    ax1.set_title('(a) TPS vs $N$ — Linear Scale', fontsize=12)
    ax1.set_xticks(N_vals)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Panel (b): Log
    ax2 = fig.add_subplot(gs[0, 1])
    for proto, vals in [('ASC', ASC), ('PBFT', PBFT), ('BLS', BLS)]:
        ax2.plot(N_vals, vals, marker=MARKERS[proto], color=COLORS[proto],
                 linewidth=2.5, markersize=9, markeredgecolor='white',
                 markeredgewidth=1.5, label=LABELS[proto])
    ax2.set_xlabel('Number of Nodes ($N$)')
    ax2.set_ylabel('Peak TPS (log)')
    ax2.set_title('(b) TPS vs $N$ — Log Scale', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_xticks(N_vals)
    ax2.legend(fontsize=9, loc='center left')
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')

    # Panel (c): Speedup bars
    ax3 = fig.add_subplot(gs[1, 0])
    ratio_pbft = [a / max(p, 1) for a, p in zip(ASC, PBFT)]
    ratio_bls  = [a / max(b, 1) for a, b in zip(ASC, BLS)]
    x = np.arange(len(N_vals))
    w = 0.35
    bars1 = ax3.bar(x - w/2, ratio_pbft, w, color=COLORS['PBFT'],
                    alpha=0.85, label='ASC / PBFT', edgecolor='white')
    bars2 = ax3.bar(x + w/2, ratio_bls, w, color=COLORS['BLS'],
                    alpha=0.85, label='ASC / BLS', edgecolor='white')
    for bar, val in zip(bars1, ratio_pbft):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val:.0f}×', ha='center', fontsize=8, fontweight='bold',
                color=COLORS['PBFT'])
    for bar, val in zip(bars2, ratio_bls):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val:.0f}×', ha='center', fontsize=8, fontweight='bold',
                color=COLORS['BLS'])
    ax3.set_xlabel('Number of Nodes ($N$)')
    ax3.set_ylabel('Speedup (×)')
    ax3.set_title('(c) ASC Speedup Over Baselines', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(N_vals)
    ax3.legend(fontsize=9)
    ax3.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Panel (d): Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    columns = ['N', 'ASC', 'PBFT', 'BLS', 'Ratio']
    cell_data = []
    for i, n in enumerate(N_vals):
        r = int(round(ASC[i])) // max(int(round(PBFT[i])), 1)
        cell_data.append([str(n), f'{int(round(ASC[i])):,}', f'{int(round(PBFT[i]))}',
                          f'{int(round(BLS[i]))}', f'{r}×'])
    table = ax4.table(cellText=cell_data, colLabels=columns,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.2)
    for col in range(len(columns)):
        table[0, col].set_facecolor('#1a1a2e')
        table[0, col].set_text_props(color='white', fontweight='bold')
    for row in range(1, len(cell_data) + 1):
        table[row, 1].set_facecolor('#fce4e4')
        table[row, 1].set_text_props(fontweight='bold')
    ax4.set_title('(d) Numerical Summary', fontsize=12, pad=15)

    fig.suptitle('Scalability Analysis — NS-3 WiFi 802.11g Testbed\n'
                 '100 tx/s/node, ML-DSA-44, 3 Protocol Architectures',
                 fontsize=15, fontweight='bold', y=1.02)

    out = f'{SCALE_DIR}/fig_scalability_combined.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


# ═══════════════════════════════════════════════════════════
#  RUN ALL
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("  SCALABILITY FIGURES — Publication Quality (Q1 Journal)")
    print("=" * 60)
    print(f"\n  NS-3 Data:")
    print(f"  {'N':>5} | {'A PBFT':>8} | {'B BLS':>8} | {'C ASC':>8} | {'D PBFT50':>9} | {'E ASC-E':>8}")
    print(f"  {'-'*64}")
    for i, n in enumerate(N_vals):
        print(
            f"  {n:5d} | {PBFT[i]:8.0f} | {BLS[i]:8.0f} | {ASC[i]:8.0f} | "
            f"{PBFT_BATCHED[i]:9.0f} | {ASC_ECDSA[i]:8.0f}"
        )
    print()

    plot_five_protocols()
    plot_main()
    plot_speedup()
    plot_area()
    plot_table()
    plot_combined()

    print(f"\n{'='*60}")
    print(f"  All figures saved to: {SCALE_DIR}/")
    print(f"  Formats: PNG (300 DPI) + PDF (vector)")
    print(f"{'='*60}")
