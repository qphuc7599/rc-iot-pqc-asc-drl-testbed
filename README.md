# Q-Relay virtualised emulation artifact

## R1 evidence for IoT-73941-2026

The minor-revision extension adds **retrained component ablations and inference
sensitivity analysis**, with explicit training budgets, paired disaster scenarios,
checkpoint-level results and uncertainty estimates. The frozen research artifact
is available as [GitHub release v1.1.0](https://github.com/qphuc7599/rc-iot-pqc-asc-drl-testbed/releases/tag/v1.1.0)
and [Zenodo version DOI 10.5281/zenodo.23335168](https://doi.org/10.5281/zenodo.23335168).
These identifiers refer to the R1 source, checkpoints and evidence; the legacy
development commands further below do not replace that versioned protocol.

## Ablation and sensitivity analysis

### Retrained component ablation

Four arms have **three independent training checkpoints each**: A0 retains the
complete controller; A1 zeros lag-difference history while retaining the network
parameterisation; A2 removes hard public admission while retaining physical
validity; A3 removes the security critic and fixes its multiplier to zero. Every
checkpoint uses the same **3,000 updates and 9 million transitions**, with
11 physical workers and 33 logical environments, and only the fixed final
checkpoint is evaluated. Training seed lists were fixed before their runs.

All 12 checkpoints use the same 20 disaster scenarios (657201-657220), giving
240 checkpoint-scenario outcomes. These are three training replicates per arm,
not 60 training replicates. The primary endpoint is late TPS at steps 1500-1999;
post-event zero service, actual global maximum exposure and guard violations are
reported separately. This separates service recovery from availability and risk.

| Arm | Late TPS mean +/- training-seed SD | Zero service (%) | Maximum exposure | Difference from A0 and hierarchical 95% interval | Conditional Holm p |
|---|---:|---:|---:|---|---:|
| A0 full | 303.26 +/- 16.30 | 13.56 | 3 | Reference | -- |
| A1 no lag-difference history | 309.65 +/- 13.27 | 11.81 | 2 | +6.38 [-15.11, 28.23] | 0.149 |
| A2 no hard admission | 305.79 +/- 4.73 | 16.40 | 3 | +2.52 [-13.26, 22.17] | 0.885 |
| A3 no security critic | 304.81 +/- 20.87 | 14.12 | 3 | +1.54 [-24.46, 28.65] | 0.885 |

All full-controller checkpoints exceed the fixed CMAB comparator's mean late TPS
on this panel, and all 12 checkpoints have zero recorded exposure-guard
violations. The incremental component contrasts remain unresolved: their
hierarchical intervals include zero and none is Holm-significant. This is useful
evidence about the retained architecture, not proof of equivalence or component
irrelevance. A1 retains temporal summaries, A2 soft risk scoring, and A3 admission
and risk-aware actor scoring. The already-observed panel makes this an
exploratory analysis rather than unseen-deployment qualification.

The registered analysis uses 10,000 hierarchical bootstrap resamples, training
checkpoints resampled independently within arms, and scenario IDs paired across
arms. Exact sign flips enumerate all 2^20 scenario assignments; Holm adjustment
is limited to the three primary late-TPS contrasts. Individual checkpoints and
all outcomes are retained without selecting successful seeds.

### Frozen inference sensitivity and decoder analysis

Sensitivity tests hold the original A0 checkpoint fixed and compare PPO with
CMAB on 20 paired scenarios per setting: **360 new 2,000-step inference episodes
plus 40 reused default episodes**, with no retraining. The grid varies public
admission floors (0.35/0.45/0.55), telemetry lag (10/20/40 steps), node-loss
fractions (10/20/30/40%) and compromise fractions (10/20/30%). The admission floor
changes PPO's mask; CMAB's selector remains unchanged.

At the default, PPO attains 312.73 late TPS versus CMAB's 252.91, with 11.65%
versus 23.74% post-event zero service. Admission 0.35 yields 325.01 versus
252.91 TPS, and lag 40 yields 269.36 versus 149.47 TPS. These tests characterise
configuration sensitivity: at admission 0.55, PPO's mean difference is -37.47
TPS (conditional interval [-71.52, -2.43]); at 40% node loss it is -11.15 TPS
(interval [-32.23, 12.03]) with 41.04%/39.36% zero service for PPO/CMAB.
PPO has no recorded post-event exposure violation across the grid. CMAB records
four violating steps at kill 40 and eleven at compromise 30. Guard compliance
and reliable service are distinct outcomes. Intervals condition on one fixed
checkpoint and are descriptive, not a new primary significance family.

The original checkpoint's **60 previously frozen stochastic action streams** are
also summarised without new simulation. Deterministic decoding has 25.31 more
mean late TPS than stochastic decoding on the paired scenarios (conditional
interval [11.93, 39.14]), providing evidence for the specified deployment decoder
within this setting. Action streams are not independently trained policies.

### Inspecting the evidence and reproducing the protocol

| Evidence | Versioned files |
|---|---|
| Every ablation outcome and arm summary | [ablation_results.csv](https://github.com/qphuc7599/rc-iot-pqc-asc-drl-testbed/blob/v1.1.0/revision_R1_results/03_ablation/final_campaign_export_20261012/ablation_results.csv), [ablation_summary.csv](https://github.com/qphuc7599/rc-iot-pqc-asc-drl-testbed/blob/v1.1.0/revision_R1_results/03_ablation/final_campaign_export_20261012/ablation_summary.csv) |
| All 12 checkpoints and selected switch patches | [checkpoint_manifest.csv](https://github.com/qphuc7599/rc-iot-pqc-asc-drl-testbed/blob/v1.1.0/revision_R1_results/03_ablation/final_campaign_export_20261012/checkpoint_manifest.csv) and [configuration exports](https://github.com/qphuc7599/rc-iot-pqc-asc-drl-testbed/tree/v1.1.0/revision_R1_results/03_ablation/final_campaign_export_20261012) |
| Sensitivity settings and outcomes | [plan.json](https://github.com/qphuc7599/rc-iot-pqc-asc-drl-testbed/blob/v1.1.0/revision_R1_results/04_optional/inference_sensitivity_20261012/plan.json), [sensitivity_inference.csv](https://github.com/qphuc7599/rc-iot-pqc-asc-drl-testbed/blob/v1.1.0/revision_R1_results/04_optional/inference_sensitivity_20261012/sensitivity_inference.csv) |
| Frozen-stream summary | [stochastic_summary.json](https://github.com/qphuc7599/rc-iot-pqc-asc-drl-testbed/blob/v1.1.0/revision_R1_results/04_optional/ledger_export_20261012/stochastic_summary.json) |
| Nine-selector diagnostic | [heldout_all_policies.csv](https://github.com/qphuc7599/rc-iot-pqc-asc-drl-testbed/blob/v1.1.0/revision_R1_results/02_verification/completion_20261012/heldout_all_policies.csv) |
| Artifact requirements and single-scenario reproduction | [Frozen v1.1.0 README](https://github.com/qphuc7599/rc-iot-pqc-asc-drl-testbed/blob/v1.1.0/README.md) |

Receipts and validation files bind source, configuration, checkpoint and paired
exogenous inputs; raw TPS is not forced to zero by an exposure failure. Source is
MIT and group-owned data/checkpoints are CC BY 4.0, subject to the versioned
per-file licence map. A4/A5 and physical-board SRAM/stack were not performed;
node-count scaling in R1 remains analytical. These scope boundaries define what
can be independently checked from the released evidence.

## Legacy development workflow

The following layout and commands document the original development branch.
Use the versioned R1 links above for the published controller and revision data.

This repository contains virtualised emulation for a relay-chain IoT
architecture with:

- Post-Quantum Threshold Signature (PQC-TS) experiments based on ML-DSA.
- Asynchronous State Channel (ASC) throughput experiments over Docker + NS-3.
- PPO-based DRL committee selection and disaster-recovery experiments.

The repository intentionally excludes the paper draft, review notes, generated
figures, old logs, full `results/`, local virtual environments, full `ns-3-dev/`,
and local `liboqs/` build trees. Rebuild external dependencies locally.

## Repository Layout

```text
analysis/                  Plotting and result-table generation scripts
drl/                       IoT environment and PPO committee-selection agent
gateway/                   UDP gateway and state-channel aggregator
pqc/                       ML-DSA/ECDSA benchmark and traffic-generator sources
ns3/scratch/               Custom NS-3 scratch simulation source
generate.py                Generates the 100-node Docker Compose topology
run-ns3-baseline.sh        Full N=100 A/B/C/D/E/F/G/H baseline wrapper
run-scalability-full.sh    Full N in {10,25,50,75,100} scalability wrapper
run-batch-sweep-full.sh    Full batch-size sensitivity wrapper
run-batch-size-sweep.sh    Batch sweep when Docker + NS-3 are already running
run-drl-experiments.sh     PPO/DRL training, disaster, and sensitivity wrapper
```

## Requirements

Recommended environment: WSL2 Ubuntu or native Linux.

- Docker and Docker Compose.
- Python 3.10+.
- NS-3 built locally under `ns-3-dev/`.
- Root/sudo access for tap devices, bridges, and network namespaces.
- Internet access during `pqc/build.sh`, because the Docker build clones liboqs.

Install Python packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Build PQC Binaries

Run from the repository root:

```bash
bash pqc/build.sh
python3 generate.py
```

This builds the static PQC binaries and regenerates `docker-compose.yml` for the
100-node heterogeneous IoT topology.

## Prepare NS-3

Place or clone NS-3 in `ns-3-dev/`, then copy the tracked scratch source:

```bash
cp ns3/scratch/iot-storm-network.cc ns-3-dev/scratch/
cd ns-3-dev
./ns3 configure --enable-examples --enable-tests
./ns3 build
cd ..
```

The wrappers expect this binary:

```text
ns-3-dev/build/scratch/ns3-dev-iot-storm-network-optimized
```

## Run Main Experiments

### 1. N=100 Baseline Comparison

Runs the paper protocol modes in isolated Docker + NS-3 sessions:

- A: PBFT + ECDSA, batch=1.
- B: BLS-sized aggregate control, batch=1.
- C: ASC + ML-DSA, batch=50.
- D: Batched PBFT + ECDSA, batch=50.
- E: ASC + ECDSA, batch=50.
- F: Simplex-style ECDSA BFT protocol-emulation baseline.
- G: Bullshark-style DAG-BFT ECDSA protocol-emulation baseline.
- H: Hydra-like ECDSA state-channel upper-bound control.

```bash
sudo bash run-ns3-baseline.sh 60
python3 analysis/plot-comparison.py
python3 analysis/plot_ablation.py
```

### 2. Scalability Sweep

Runs N in `{10,25,50,75,100}` for all protocol modes:

```bash
sudo bash run-scalability-full.sh 60
python3 analysis/plot_scalability.py
```

### 3. Batch-Size Sensitivity

Run all paper sweep points:

```bash
sudo bash run-batch-sweep-full.sh 60 all all
python3 analysis/plot-batch-sweep.py --results-dir results/batch_sweep
```

Use a custom comma-separated batch list when needed:

```bash
sudo bash run-batch-sweep-full.sh 60 asc 25,50,100
```

### 4. DRL Training and Disaster Evaluation

The historical development path uses model-level invalid-action masking, a pooled critic,
and the masked Gumbel-Top-k actor. The wrapper auto-selects a Python
environment with `numpy`, `torch`, and `gymnasium`.

Smoke test:

```bash
bash run-drl-experiments.sh smoke
```

Train the robust PPO policy:

```bash
bash run-drl-experiments.sh train-gumbel 3000 3000 11
```

Evaluate disaster recovery:

```bash
bash run-drl-experiments.sh disaster-gumbel
```

Run the DRL fault-magnitude sensitivity grid:

```bash
bash run-drl-experiments.sh sensitivity-gumbel
python3 analysis/plot-drl-sensitivity-paper.py \
  --input results/drl_gumbel_pooled/drl_sensitivity.json \
  --output-dir results/drl_gumbel_pooled
```

## Notes

- The canonical ASC logical batch size is 50 transactions per update.
- Offered load is normally 100 transactions/s/node; do not confuse this with
  the batch size.
- The Docker containers emulate topology and resource constraints. Device-side
  cryptographic latencies are calibrated from published microcontroller
  benchmark data and injected into the traffic generator/environment.
- `results/` is ignored by Git. Archive final result artifacts separately when
  submitting a paper artifact bundle.
