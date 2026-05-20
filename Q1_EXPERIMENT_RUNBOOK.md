# Q1 Experiment Runbook

This repo follows the advisor's original setup:

- IoT edge: Docker containers with cgroup CPU/RAM limits.
- Network/gateway: NS-3 realtime 802.11g tap-bridge simulation.
- PQC: liboqs ML-DSA in the generator plus calibrated device delays.
- ASC: gateway-side off-chain batching and signature verification.
- DRL: PyTorch/Gym environment calibrated from PQC + network measurements.

Canonical ASC batch size: `50 tx/update`. The offered load remains
`100 tx/s/node`; do not interpret that rate as the batch size.

Run these commands from WSL:

```bash
cd /mnt/d/DoAnChuyenNganh/rc-iot-testbed
```

## 0. Prepare Binaries

Moderate runtime. Needed after changing `pqc/tx-generator.c`.

```bash
bash pqc/build.sh
python3 generate.py
```

## 1. N=100 NS-3 Baseline, Required For Q1 Claims

Heavy. This produces the clean A/B/C/D/E comparison under NS-3:

- A: PBFT + ECDSA, batch=1
- B: BLS-sized aggregate control, batch=1
- C: ASC + ML-DSA, batch=50
- D: PBFT + ECDSA, batch=50 control
- E: ASC + ECDSA, batch=50 no-PQC ablation

```bash
sudo bash run-q1-ns3-baseline.sh 60
```

The wrapper now fails fast if the gateway cannot run Python, if a summary file
is stale/missing, or if the generated summary does not match the expected
batch/signature/PBFT configuration.

Network note: the NS-3 WiFi tap-bridge uses `10.1.1.0/24`; the gateway is
assigned `10.1.1.100`, and IoT nodes are assigned `10.1.1.101` onward.

Isolation note: the wrapper restarts Docker + NS-3 for each A/B/C/D/E protocol
mode. This is slower, but prevents high-rate traffic from one protocol from
leaking through NS-3/tap queues into the next protocol measurement.

Outputs:

```text
results/comparison_A_pbft_ecdsa/gateway_summary.json
results/comparison_B_bls_aggregate/gateway_summary.json
results/comparison_C_offchain_statechannel/gateway_summary.json
results/comparison_D_pbft_batched_ecdsa/gateway_summary.json
results/comparison_E_offchain_ecdsa/gateway_summary.json
results/baseline_comparison.png
```

Use these for:

- Table throughput comparison.
- Table ablation No-PQC / No-ASC / PBFT-batched control.
- Text claims: ASC vs per-tx PBFT and ASC vs batched PBFT.

## 2. Full NS-3 Scalability Curve

Very heavy. This runs N=10,25,50,75,100 for all five A/B/C/D/E protocol
configurations: PBFT+ECDSA, BLS aggregate, ASC+ML-DSA, batched PBFT+ECDSA,
and ASC+ECDSA.
The offered load is fixed at `100 tx/s/node`; at N=100 this is 10,000 logical
transactions per second, high enough for throughput to be network/protocol
limited rather than source limited.
Each protocol is run in a fresh Docker + NS-3 session. This avoids high-rate
batch=1 traffic leaving packets/backlog in the tap/NS-3 queues and corrupting
the following ASC measurement.

```bash
sudo bash run-scalability-full.sh 60
python3 analysis/plot_scalability.py
```

Outputs:

```text
results/scalability/summary.json
file_project/figures/scalability.pdf
```

Use this for:

- Scalability figure.
- Packet-level NS-3 evidence for MAC-layer congestion.

## 3. DRL Disaster Evaluation

No NS-3 required. Run only if the paper needs regenerated DRL statistics after
code changes.

```bash
python3 drl/ppo_agent.py --mode disaster
python3 analysis/plot-drl-results.py
```

Use this for:

- Disaster-recovery table.
- AUC retention and post-disaster TPS.

## 4. Light Plot Regeneration

Run after new JSON results exist:

```bash
python3 analysis/plot-comparison.py
python3 analysis/plot_ablation.py
python3 analysis/plot_scalability.py
```

## Decision Rule

For a Q1 submission, rerun at least Step 1. If time allows, rerun Step 2.
Step 3 is optional because DRL is a calibrated simulation, not an NS-3 runtime
measurement.
