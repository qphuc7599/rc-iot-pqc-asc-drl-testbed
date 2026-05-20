#!/usr/bin/env python3
"""
plot-pqc-vs-classical.py — PQC vs Classical Crypto Comparison
Uses published benchmark data from:
  - pqm4 project (Kannwischer et al., CHES 2019)
  - Becker et al., TCHES 2022 (ML-DSA on Cortex-M4)
  - NIST PQC Round 3 Report (2022)
  - wolfSSL benchmark data for ECDSA/Ed25519

Output: results/pqc_vs_classical.png
"""

import os, sys, statistics

def main():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[ERROR] pip install matplotlib numpy")
        sys.exit(1)

    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.3,
    })
    # ================================================================
    # Load experimental data if available, else use published values
    # ================================================================
    import csv

    def load_csv_data(filepath):
        """Load summary CSV → list of dicts"""
        if not os.path.exists(filepath):
            return None
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
        return data if data else None

    def avg_field(data, field):
        vals = [d[field] for d in data if field in d]
        return statistics.mean(vals) if vals else 0

    # Try loading experimental data
    exp_ecdsa = load_csv_data('results/summary_ecdsa_p256.csv')
    exp_ed25519 = load_csv_data('results/summary_ed25519.csv')
    exp_mldsa = load_csv_data('results/summary.csv')

    use_experimental = (exp_ecdsa and exp_ed25519 and exp_mldsa)

    if use_experimental:
        data_source = 'Experimental (100 IoT containers, same testbed)'
        print("[INFO] Using EXPERIMENTAL data from testbed")

        SCHEMES = {
            'ECDSA-P256': {
                'label': 'ECDSA\n(P-256)',
                'keygen_us': avg_field(exp_ecdsa, 'keygen_avg_us'),
                'sign_us': avg_field(exp_ecdsa, 'sign_avg_us'),
                'verify_us': avg_field(exp_ecdsa, 'verify_avg_us'),
                'pk_bytes':  64,
                'sig_bytes': 72,
                'quantum_safe': False,
                'color': '#F44336',
            },
            'Ed25519': {
                'label': 'Ed25519\n(Curve25519)',
                'keygen_us': avg_field(exp_ed25519, 'keygen_avg_us'),
                'sign_us': avg_field(exp_ed25519, 'sign_avg_us'),
                'verify_us': avg_field(exp_ed25519, 'verify_avg_us'),
                'pk_bytes':  32,
                'sig_bytes': 64,
                'quantum_safe': False,
                'color': '#FF9800',
            },
            'ML-DSA-44': {
                'label': 'ML-DSA-44\n(FIPS 204)',
                'keygen_us': avg_field(exp_mldsa, 'keygen_avg_us'),
                'sign_us': avg_field(exp_mldsa, 'sign_avg_us'),
                'verify_us': avg_field(exp_mldsa, 'verify_avg_us'),
                'pk_bytes':  1312,
                'sig_bytes': 2420,
                'quantum_safe': True,
                'color': '#4CAF50',
            },
        }
    else:
        data_source = 'Published: pqm4 [1], Becker TCHES 2022 [2], wolfSSL [3]'
        print("[INFO] Using ANALYTICAL data (no experimental CSV found)")
        print("[INFO] Run: sudo ./pqc/run-benchmark-classical.sh")
        REF_FREQ_MHZ = 168

        SCHEMES = {
            'ECDSA-P256': {
                'label': 'ECDSA\n(P-256)',
                'keygen_us': 2_500_000 / REF_FREQ_MHZ,
                'sign_us':   3_100_000 / REF_FREQ_MHZ,
                'verify_us': 6_200_000 / REF_FREQ_MHZ,
                'pk_bytes':  64, 'sig_bytes': 64,
                'quantum_safe': False, 'color': '#F44336',
            },
            'Ed25519': {
                'label': 'Ed25519\n(Curve25519)',
                'keygen_us': 1_300_000 / REF_FREQ_MHZ,
                'sign_us':   1_500_000 / REF_FREQ_MHZ,
                'verify_us': 3_100_000 / REF_FREQ_MHZ,
                'pk_bytes':  32, 'sig_bytes': 64,
                'quantum_safe': False, 'color': '#FF9800',
            },
            'ML-DSA-44': {
                'label': 'ML-DSA-44\n(FIPS 204)',
                'keygen_us': 1_400_000 / REF_FREQ_MHZ,
                'sign_us':   4_000_000 / REF_FREQ_MHZ,
                'verify_us': 1_400_000 / REF_FREQ_MHZ,
                'pk_bytes': 1312, 'sig_bytes': 2420,
                'quantum_safe': True, 'color': '#4CAF50',
            },
        }

    scheme_names = list(SCHEMES.keys())

    # ================================================================
    # Figure: 3-panel comparison
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Post-Quantum vs Classical Cryptography — IoT Performance\n'
                 f'{data_source}',
                 fontsize=14, fontweight='bold')

    # ────────────────────────────────────────────────
    # Panel 1: Latency on STM32F4 (reference platform)
    # ────────────────────────────────────────────────
    ax = axes[0, 0]
    x = np.arange(len(scheme_names))
    width = 0.25
    colors_scheme = [SCHEMES[s]['color'] for s in scheme_names]

    kg = [SCHEMES[s]['keygen_us'] for s in scheme_names]
    sn = [SCHEMES[s]['sign_us'] for s in scheme_names]
    vf = [SCHEMES[s]['verify_us'] for s in scheme_names]

    ax.bar(x - width, kg, width, label='KeyGen', color='#2196F3', alpha=0.85)
    ax.bar(x, sn, width, label='Sign', color='#FF9800', alpha=0.85)
    ax.bar(x + width, vf, width, label='Verify', color='#4CAF50', alpha=0.85)

    # Value labels
    for i in range(len(scheme_names)):
        ax.text(i, max(kg[i], sn[i], vf[i]) * 1.05,
                f'{max(sn[i], vf[i])/1000:.1f}ms',
                ha='center', fontsize=9, fontweight='bold')

    labels = [SCHEMES[s]['label'] for s in scheme_names]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Latency (µs)')
    ax.set_title('Operation Latency (avg µs)')
    ax.legend(fontsize=9)

    # Highlight: ML-DSA verify is fastest!
    vf_improvement = (1 - vf[2] / vf[0]) * 100
    ax.annotate(f'Verify {vf_improvement:.0f}%\nfaster!', xy=(2 + width, vf[2]),
               xytext=(2.5, max(vf) * 0.8),
               arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
               fontsize=10, fontweight='bold', color='green')

    # ────────────────────────────────────────────────
    # Panel 2: Signature + Key sizes
    # ────────────────────────────────────────────────
    ax2 = axes[0, 1]
    pk_sizes = [SCHEMES[s]['pk_bytes'] for s in scheme_names]
    sig_sizes = [SCHEMES[s]['sig_bytes'] for s in scheme_names]

    bars1 = ax2.bar(x - 0.15, pk_sizes, 0.3, label='Public Key', color='#03A9F4', alpha=0.85)
    bars2 = ax2.bar(x + 0.15, sig_sizes, 0.3, label='Signature', color='#E91E63', alpha=0.85)

    for i, (pk, sig) in enumerate(zip(pk_sizes, sig_sizes)):
        ax2.text(i - 0.15, pk + max(sig_sizes)*0.03, f'{pk}B', ha='center', fontsize=9)
        ax2.text(i + 0.15, sig + max(sig_sizes)*0.03, f'{sig}B', ha='center', fontsize=9)

    # ML-DSA overhead annotation
    ax2.annotate(f'38x larger sig\n(tradeoff for\nquantum safety)',
                xy=(2 + 0.15, sig_sizes[2]),
                xytext=(1.3, sig_sizes[2] * 0.7),
                arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.5),
                fontsize=9, color='#E91E63')

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('Size (bytes)')
    ax2.set_title('Key & Signature Sizes')
    ax2.legend(fontsize=9)

    # ────────────────────────────────────────────────
    # Panel 3: Normalized comparison (radar-like bar)
    # ────────────────────────────────────────────────
    ax3 = axes[1, 0]
    metrics = ['Sign', 'Verify', 'KeyGen']
    base_vals = [sn[0], vf[0], kg[0]]  # ECDSA as baseline

    x_met = np.arange(len(metrics))
    for idx, s in enumerate(scheme_names):
        vals = [
            SCHEMES[s]['sign_us'] / base_vals[0],
            SCHEMES[s]['verify_us'] / base_vals[1],
            SCHEMES[s]['keygen_us'] / base_vals[2],
        ]
        offset = (idx - 1) * width
        ax3.bar(x_met + offset, vals, width,
               label=s, color=SCHEMES[s]['color'], alpha=0.85)

    ax3.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xticks(x_met)
    ax3.set_xticklabels(metrics)
    ax3.set_ylabel('Relative to ECDSA (=1.0)')
    ax3.set_title('Normalized Performance (lower = better)')
    ax3.legend(fontsize=9)

    # ────────────────────────────────────────────────
    # Panel 4: Summary table
    # ────────────────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.axis('off')

    POWER_MW = {'nRF52840': 23, 'ESP32': 160, 'STM32H7-M7': 90}
    table_data = []
    for s in scheme_names:
        sc = SCHEMES[s]
        e_sign = sc['sign_us'] * 1e-6 * 90  # mJ on STM32H7
        qr = 'Yes' if sc['quantum_safe'] else 'No'
        table_data.append([
            s,
            f"{sc['sign_us']:.0f}",
            f"{sc['verify_us']:.0f}",
            f"{sc['sig_bytes']}",
            f"{e_sign:.4f}",
            qr,
        ])

    col_labels = ['Scheme', 'Sign(µs)', 'Verify(µs)', 'Sig(B)', 'E_sign(mJ)', 'QR']
    table = ax4.table(cellText=table_data,
                      colLabels=col_labels,
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.22, 0.14, 0.14, 0.12, 0.18, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.2)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#333333')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i, s in enumerate(scheme_names):
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor(SCHEMES[s]['color'] + '20')

    ax4.set_title('Comparison Summary', fontsize=12, pad=10)

    # ────────────────────────────────────────────────
    # Summary table below
    # ────────────────────────────────────────────────
    plt.tight_layout()

    path = os.path.join('results', 'pqc_vs_classical.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    # Print summary table
    print("\n" + "=" * 80)
    print(f"  {'Scheme':15s} {'Sign(µs)':>12} {'Verify(µs)':>12} "
          f"{'Sig(B)':>8} {'QR':>4}")
    print("-" * 80)
    for s in scheme_names:
        sc = SCHEMES[s]
        qr = 'Yes' if sc['quantum_safe'] else 'No'
        print(f"  {s:15s} {sc['sign_us']:12.0f} {sc['verify_us']:12.0f} "
              f"{sc['sig_bytes']:8d} {qr:>4}")

    print("-" * 80)
    mldsa_vf = SCHEMES['ML-DSA-44']['verify_us']
    ecdsa_vf = SCHEMES['ECDSA-P256']['verify_us']
    ed_vf = SCHEMES['Ed25519']['verify_us']
    print(f"\n  ML-DSA Verify vs ECDSA: {(1 - mldsa_vf/ecdsa_vf)*100:.0f}% faster")
    print(f"  ML-DSA Verify vs Ed25519: {(1 - mldsa_vf/ed_vf)*100:.0f}% faster")
    print("=" * 80)


if __name__ == '__main__':
    main()
