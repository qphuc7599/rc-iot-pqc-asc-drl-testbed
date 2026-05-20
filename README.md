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
run-q1-ns3-baseline.sh     Full N=100 A/B/C/D/E baseline wrapper
run-scalability-full.sh    Full N in {10,25,50,75,100} scalability wrapper
run-q1-batch-sweep.sh      Batch-size sensitivity wrapper
run-batch-size-sweep.sh    Batch sweep when Docker + NS-3 are already running
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

Runs five protocol modes in isolated Docker + NS-3 sessions:

- A: PBFT + ECDSA, batch=1.
- B: BLS-sized aggregate control, batch=1.
- C: ASC + ML-DSA, batch=50.
- D: Batched PBFT + ECDSA, batch=50.
- E: ASC + ECDSA, batch=50.

```bash
sudo bash run-q1-ns3-baseline.sh 60
python3 analysis/plot-comparison.py
python3 analysis/plot_ablation.py
```

### 2. Scalability Sweep

Runs N in `{10,25,50,75,100}` for all five protocol modes:

```bash
sudo bash run-scalability-full.sh 60
python3 analysis/plot_scalability.py
```

### 3. Batch-Size Sensitivity

Run all paper sweep points:

```bash
sudo bash run-q1-batch-sweep.sh 60 all
python3 analysis/plot-batch-sweep.py --results-dir results/batch_sweep
```

Use a custom comma-separated batch list when needed:

```bash
sudo bash run-q1-batch-sweep.sh 60 asc 25,50,100
```

### 4. DRL Training and Disaster Evaluation

Train the robust PPO policy:

```bash
python3 drl/ppo_agent.py --mode train --sim --episodes 3000 --steps 3000 \
  --seed 2026 --workers 11 --save-dir drl/models_robust --train-profile robust
```

Evaluate disaster recovery:

```bash
python3 drl/ppo_agent.py --mode disaster --sim \
  --model drl/models_robust/ppo_best.pt --seed 42
python3 analysis/plot-drl-results.py
```

Run the DRL fault-magnitude sensitivity grid:

```bash
python3 drl/ppo_agent.py --mode sensitivity --sim \
  --model drl/models_robust/ppo_best.pt --output results/drl_sensitivity.json
python3 analysis/plot-drl-sensitivity.py --input results/drl_sensitivity.json
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
