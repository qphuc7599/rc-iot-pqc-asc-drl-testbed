# RC-IoT Post-Quantum Testbed

This repository contains the experimental testbed for a relay-chain IoT
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

The paper DRL path uses model-level invalid-action masking, a pooled critic,
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
