#!/usr/bin/env bash
#
# Baseline comparison for the RC-IoT testbed.
#
# Modes:
#   A) Per-transaction PBFT + ECDSA-sized packets
#   B) Aggregate-signature control with BLS-sized packets and no PBFT
#   C) Off-chain state channel + real ML-DSA-44
#   D) Batched PBFT + ECDSA-sized packets at the same 50 TX/packet granularity
#   E) Off-chain state channel + ECDSA-sized packets for the "No PQC" ablation
#   F) Simplex batched BFT protocol emulation with ECDSA-sized packets
#   G) Bullshark DAG-BFT protocol emulation with ECDSA-sized packets
#   H) Hydra Head ECDSA state-channel protocol emulation
#
# The A/B/D modes emulate classical signature size and signing delay in
# tx-generator so the network packets actually match the label. Mode C uses
# liboqs ML-DSA-44 for the proposed post-quantum path.
#
# Usage: ./run-baseline-comparison.sh [duration_seconds]

set -euo pipefail

DURATION=${1:-60}
REQUESTED_MODE=${2:-all}
NUM_NODES=100
PORT=9000
RESULTS_DIR="results"
GW_IP="10.1.1.100"

MLDSA_SIG_BYTES=2420
MLDSA_VERIFY_US=796

ECDSA_SIG_BYTES=72
ECDSA_VERIFY_US=69
ECDSA_SIGN_US=27

BLS_SIG_BYTES=96
BLS_VERIFY_US=0
BLS_SIGN_US=0

# Calibrated controls used by the paper.
# A: 1 / 0.055 ~= 18 TPS when gateway batch size is 1.
# D: 50 / 0.0463 ~= 1080 TPS when tx-generator batches 50 TX/update.
ASC_BATCH_SIZE=50
PBFT_PER_TX_DELAY=0.055
PBFT_BATCH50_DELAY=0.0463

# Validator-side protocol emulation for SOTA baselines.  F/G use the active
# 100-node validator set; H uses a Hydra-head committee by default.
BFT_VALIDATORS=${BFT_VALIDATORS:-100}
HYDRA_HEAD_SIZE=${HYDRA_HEAD_SIZE:-21}
PROTOCOL_RTT_MS=${PROTOCOL_RTT_MS:-4.0}
PROTOCOL_BANDWIDTH_MBPS=${PROTOCOL_BANDWIDTH_MBPS:-1000.0}
BULLSHARK_ROUND_MS=${BULLSHARK_ROUND_MS:-100.0}

echo "================================================================"
echo "  BASELINE COMPARISON"
echo "  Duration: ${DURATION}s, Nodes: ${NUM_NODES}, Offered load: 100 tx/s/node"
echo "  ASC logical batch size: ${ASC_BATCH_SIZE} tx/update"
echo "================================================================"

if [ "$REQUESTED_MODE" = "all" ]; then
    for old_dir in "$RESULTS_DIR"/comparison_*; do
        [ -d "$old_dir" ] || continue
        rm -f "$old_dir/gateway_summary.json" "$old_dir/gateway_batches.csv" "$old_dir/gateway_log.txt"
    done
else
    rm -f "$RESULTS_DIR/comparison_${REQUESTED_MODE}/gateway_summary.json" \
          "$RESULTS_DIR/comparison_${REQUESTED_MODE}/gateway_batches.csv" \
          "$RESULTS_DIR/comparison_${REQUESTED_MODE}/gateway_log.txt"
fi

should_run() {
    local mode=$1
    [ "$REQUESTED_MODE" = "all" ] || [ "$REQUESTED_MODE" = "$mode" ]
}

cooldown_if_all() {
    if [ "$REQUESTED_MODE" = "all" ]; then
        echo "[COOLDOWN] 15s..."
        sleep 15
    fi
}

find_container() {
    local service=$1
    docker ps --format '{{.Names}}' | grep -E "(^|[-_])${service}[-_][0-9]+$" | head -1
}

require_container() {
    local service=$1
    local container
    container=$(find_container "$service")
    if [ -z "$container" ]; then
        echo "[ERROR] Container for service '$service' is not running." >&2
        echo "        Start Docker + NS-3 first, or run run-ns3-baseline.sh." >&2
        exit 1
    fi
    printf '%s\n' "$container"
}

GW_CONT=$(require_container gateway_node)
MISSING_NODES=0
for i in $(seq 1 "$NUM_NODES"); do
    if [ -z "$(find_container "iot_node_${i}")" ]; then
        echo "[ERROR] Missing container for iot_node_${i}" >&2
        MISSING_NODES=$((MISSING_NODES + 1))
    fi
done
if [ "$MISSING_NODES" -gt 0 ]; then
    echo "[ERROR] Baseline requires all ${NUM_NODES} IoT containers; missing ${MISSING_NODES}." >&2
    echo "        Start the full NS-3/Docker wrapper: sudo bash run-ns3-baseline.sh ${DURATION}" >&2
    exit 1
fi

ensure_gateway_python() {
    if docker exec "$GW_CONT" python3 --version >/dev/null 2>&1; then
        return
    fi
    echo "[SETUP] Installing python3 in gateway container..."
    docker exec "$GW_CONT" bash -c "apt-get update -qq && apt-get install -y -qq python3" >/dev/null
    docker exec "$GW_CONT" python3 --version
}

ensure_gateway_python

run_experiment() {
    local exp_name=$1
    local exp_port=$2
    local rate=$3
    local tx_batch=$4
    local gw_batch=$5
    local pbft_delay=$6
    local expected_sig_len=$7
    local verify_delay_us=$8
    local emulated_sign_us=$9
    local mode_label=${10}
    local protocol_mode=${11:-auto}
    local protocol_n=${12:-21}
    local exp_dir="${RESULTS_DIR}/comparison_${exp_name}"
    local summary_file="${RESULTS_DIR}/gateway_summary_${exp_name}.json"
    local batches_file="${RESULTS_DIR}/gateway_batches_${exp_name}.csv"
    local log_file="${RESULTS_DIR}/gateway_log_${exp_name}.txt"

    echo ""
    echo "------------------------------------------------------------"
    echo "  [${exp_name}] ${mode_label}"
    echo "  port=${exp_port}, rate=${rate}, tx_batch=${tx_batch}, gateway_batch=${gw_batch}, pbft=${pbft_delay}s"
    echo "  protocol=${protocol_mode}, protocol_n=${protocol_n}, rtt=${PROTOCOL_RTT_MS}ms, bw=${PROTOCOL_BANDWIDTH_MBPS}Mbps"
    echo "  expected_sig=${expected_sig_len}B, verify_delay=${verify_delay_us}us, emulated_sign=${emulated_sign_us}us"
    echo "------------------------------------------------------------"

    mkdir -p "$exp_dir"
    rm -f "$summary_file" "$batches_file" "$log_file" "$exp_dir/gateway_summary.json"

    docker exec "$GW_CONT" pkill -f sig-aggregator 2>/dev/null || true
    for i in $(seq 1 "$NUM_NODES"); do
        local node_cont
        node_cont=$(find_container "iot_node_${i}")
        [ -z "$node_cont" ] && continue
        docker exec "$node_cont" pkill -f tx-generator 2>/dev/null || true
    done
    sleep 3

    docker exec -d "$GW_CONT" bash -c "
        python3 /opt/gateway/sig-aggregator.py \
            --port $exp_port \
            --batch-size $gw_batch \
            --duration $((DURATION + 15)) \
            --pbft-delay $pbft_delay \
            --verify-delay-us $verify_delay_us \
            --expected-sig-len $expected_sig_len \
            --protocol-mode $protocol_mode \
            --protocol-n $protocol_n \
            --protocol-rtt-ms $PROTOCOL_RTT_MS \
            --protocol-bandwidth-mbps $PROTOCOL_BANDWIDTH_MBPS \
            --bullshark-round-ms $BULLSHARK_ROUND_MS \
            --output /opt/results/gateway_summary_${exp_name}.json \
            > /opt/results/gateway_batches_${exp_name}.csv \
            2> /opt/results/gateway_log_${exp_name}.txt
    "
    sleep 3
    if ! docker exec "$GW_CONT" pgrep -f "sig-aggregator.py" >/dev/null 2>&1; then
        echo "[ERROR] Gateway aggregator failed to start for ${exp_name}."
        echo "---- gateway log ----"
        cat "$log_file" 2>/dev/null || true
        exit 1
    fi

    local extra_args=""
    if [ "$emulated_sign_us" -ge 0 ]; then
        extra_args="--emulated-sig-bytes $expected_sig_len --emulated-sign-us $emulated_sign_us"
    fi

    for i in $(seq 1 "$NUM_NODES"); do
        local container
        container=$(find_container "iot_node_${i}")
        [ -z "$container" ] && {
            echo "  [WARN] Missing iot_node_${i}; skipping"
            continue
        }
        docker exec -d "$container" bash -c "
            /opt/pqc/tx-generator \
                --node-id $i \
                --gateway $GW_IP \
                --port $exp_port \
                --algo ML-DSA-44 \
                --rate $rate \
                --batch $tx_batch \
                --duration $DURATION \
                $extra_args \
                > /opt/results/tx_node_${exp_name}_${i}.csv \
                2> /opt/results/tx_node_${exp_name}_${i}.log
        "
        if (( i % 10 == 0 )); then
            sleep 0.5
        fi
    done

    echo "  Waiting ${DURATION}s + 15s buffer..."
    sleep $((DURATION + 15))

    if [ ! -f "$summary_file" ]; then
        echo "[ERROR] Missing summary for ${exp_name}; gateway probably failed."
        echo "---- gateway log ----"
        cat "$log_file" 2>/dev/null || true
        exit 1
    fi

    python3 - <<PY
import json, math, sys
from pathlib import Path
p = Path("${summary_file}")
d = json.loads(p.read_text())
errors = []
if int(d.get("batch_size", -1)) != int("${gw_batch}"):
    errors.append(f"batch_size={d.get('batch_size')} expected ${gw_batch}")
if int(d.get("expected_sig_len", -1)) != int("${expected_sig_len}"):
    errors.append(f"expected_sig_len={d.get('expected_sig_len')} expected ${expected_sig_len}")
if int(d.get("port", -1)) != int("${exp_port}"):
    errors.append(f"port={d.get('port')} expected ${exp_port}")
if abs(float(d.get("pbft_delay_s", -999)) - float("${pbft_delay}")) > 1e-6:
    errors.append(f"pbft_delay_s={d.get('pbft_delay_s')} expected ${pbft_delay}")
if abs(float(d.get("verify_delay_us", -999)) - float("${verify_delay_us}")) > 1e-6:
    errors.append(f"verify_delay_us={d.get('verify_delay_us')} expected ${verify_delay_us}")
if d.get("protocol_mode") != "${protocol_mode}":
    errors.append(f"protocol_mode={d.get('protocol_mode')} expected ${protocol_mode}")
if int(d.get("protocol", {}).get("participants", -1)) != int("${protocol_n}"):
    errors.append(f"protocol participants={d.get('protocol', {}).get('participants')} expected ${protocol_n}")
if int(d.get("total_updates", 0)) <= 0:
    errors.append("total_updates is zero")
accepted_updates = max(int(d.get("accepted_updates", d.get("total_updates", 0))), 1)
tx_per_update = float(d.get("total_tx", 0)) / accepted_updates
if abs(tx_per_update - float("${tx_batch}")) > max(2.0, 0.10 * float("${tx_batch}")):
    errors.append(f"TX/update={tx_per_update:.2f} expected around ${tx_batch}")
if errors:
    print("[ERROR] Invalid summary for ${exp_name}:")
    for e in errors:
        print("  -", e)
    print("  observed_sig_lens:", d.get("observed_sig_lens", {}))
    print("  failed_sig_lens:", d.get("failed_sig_lens", {}))
    sys.exit(1)
PY

    cp "$summary_file" "$exp_dir/gateway_summary.json"
    cp "$batches_file" "$exp_dir/gateway_batches.csv" 2>/dev/null || true
    cp "$log_file" "$exp_dir/gateway_log.txt" 2>/dev/null || true

    python3 - <<PY
import json
from pathlib import Path
p = Path("${exp_dir}") / "gateway_summary.json"
d = json.loads(p.read_text())
print()
print("  RESULT ${exp_name}:")
print(f"    accepted TPS: {d['avg_tps']:.1f}")
print(f"    accepted TX:  {d['total_tx']}")
print(f"    received TX:  {d.get('received_tx', 'N/A')}")
print(f"    updates:      {d.get('total_updates', 'N/A')}")
print()
PY
}

echo ""
echo "[A] Per-transaction PBFT + ECDSA-sized packets"
if should_run "A_pbft_ecdsa"; then
    run_experiment \
        "A_pbft_ecdsa" \
        9001 100 1 1 "$PBFT_PER_TX_DELAY" \
        "$ECDSA_SIG_BYTES" "$ECDSA_VERIFY_US" "$ECDSA_SIGN_US" \
        "Per-tx PBFT baseline" \
        "legacy_pbft" "$BFT_VALIDATORS"
    cooldown_if_all
fi

echo ""
echo "[B] Aggregate-signature control, no PBFT"
if should_run "B_bls_aggregate"; then
    run_experiment \
        "B_bls_aggregate" \
        9002 100 1 1 0.0 \
        "$BLS_SIG_BYTES" "$BLS_VERIFY_US" "$BLS_SIGN_US" \
        "BLS-sized aggregate-signature control" \
        "none" "$BFT_VALIDATORS"
    cooldown_if_all
fi

echo ""
echo "[C] Off-chain state channel + real ML-DSA-44"
if should_run "C_offchain_statechannel"; then
    run_experiment \
        "C_offchain_statechannel" \
        9003 100 "$ASC_BATCH_SIZE" "$ASC_BATCH_SIZE" 0.0 \
        "$MLDSA_SIG_BYTES" "$MLDSA_VERIFY_US" -1 \
        "ASC + real ML-DSA-44" \
        "none" "$HYDRA_HEAD_SIZE"
    cooldown_if_all
fi

echo ""
echo "[E] Off-chain state channel + ECDSA-sized packets"
if should_run "E_offchain_ecdsa"; then
    run_experiment \
        "E_offchain_ecdsa" \
        9004 100 "$ASC_BATCH_SIZE" "$ASC_BATCH_SIZE" 0.0 \
        "$ECDSA_SIG_BYTES" "$ECDSA_VERIFY_US" "$ECDSA_SIGN_US" \
        "ASC no-PQC ablation" \
        "none" "$HYDRA_HEAD_SIZE"
    cooldown_if_all
fi

echo ""
echo "[D] Batched PBFT + ECDSA-sized packets"
if should_run "D_pbft_batched_ecdsa"; then
    run_experiment \
        "D_pbft_batched_ecdsa" \
        9005 100 "$ASC_BATCH_SIZE" 1 "$PBFT_BATCH50_DELAY" \
        "$ECDSA_SIG_BYTES" "$ECDSA_VERIFY_US" "$ECDSA_SIGN_US" \
        "Batched PBFT analytical-control run" \
        "legacy_pbft" "$BFT_VALIDATORS"
fi

echo ""
echo "[F] Simplex batched BFT protocol + ECDSA-sized packets"
if should_run "F_simplex_batched_ecdsa"; then
    run_experiment \
        "F_simplex_batched_ecdsa" \
        9006 100 "$ASC_BATCH_SIZE" 1 0.0 \
        "$ECDSA_SIG_BYTES" "$ECDSA_VERIFY_US" "$ECDSA_SIGN_US" \
        "Simplex full-protocol batched BFT" \
        "simplex" "$BFT_VALIDATORS"
    cooldown_if_all
fi

echo ""
echo "[G] Bullshark DAG-BFT protocol ordering + ECDSA-sized packets"
if should_run "G_bullshark_dag_ecdsa"; then
    run_experiment \
        "G_bullshark_dag_ecdsa" \
        9007 100 "$ASC_BATCH_SIZE" 1 0.0 \
        "$ECDSA_SIG_BYTES" "$ECDSA_VERIFY_US" "$ECDSA_SIGN_US" \
        "Bullshark full-protocol DAG-BFT ordering" \
        "bullshark" "$BFT_VALIDATORS"
    cooldown_if_all
fi

echo ""
echo "[H] Hydra Head ECDSA state-channel protocol"
if should_run "H_hydra_ecdsa"; then
    run_experiment \
        "H_hydra_ecdsa" \
        9008 100 "$ASC_BATCH_SIZE" "$ASC_BATCH_SIZE" 0.0 \
        "$ECDSA_SIG_BYTES" "$ECDSA_VERIFY_US" "$ECDSA_SIGN_US" \
        "Hydra Head full-protocol state-channel baseline" \
        "hydra" "$HYDRA_HEAD_SIZE"
fi

echo ""
echo "================================================================"
echo "  COMPARISON SUMMARY"
echo "================================================================"

for dir in "$RESULTS_DIR"/comparison_*; do
    name=$(basename "$dir" | sed 's/comparison_//')
    if [ -f "$dir/gateway_summary.json" ]; then
        python3 - <<PY
import json
d = json.load(open("$dir/gateway_summary.json"))
tps = d["avg_tps"]
tx = d["total_tx"]
rx = d.get("received_tx", tx)
updates = d.get("total_updates", tx)
tx_per_pkt = tx / updates if updates else 0.0
print(f"  {'${name}':35s} | TPS: {tps:8.1f} | accepted TX: {tx:8d} | received TX: {rx:8d} | TX/pkt: {tx_per_pkt:5.1f}")
PY
    fi
done

echo ""
echo "Plot: python3 analysis/plot-comparison.py"
