#!/usr/bin/env bash
#
# Batch-size sensitivity sweep for the RC-IoT paper.
#
# This runner expects the Docker containers and NS-3 tap/bridge session to
# already be running, just like run-baseline-comparison.sh. For publication
# numbers, run it through run-q1-batch-sweep.sh so every (B, protocol) pair gets
# a fresh Docker + NS-3 session.
#
# Usage:
#   bash run-batch-size-sweep.sh [duration_seconds] [batch|all] [asc|pbft_batched|all]
#
# Examples:
#   bash run-batch-size-sweep.sh 60 all asc
#   bash run-batch-size-sweep.sh 60 50 all

set -euo pipefail

DURATION=${1:-60}
REQUESTED_BATCH=${2:-all}
REQUESTED_PROTOCOL=${3:-asc}

NUM_NODES=100
RATE=100
PORT_ASC=9201
PORT_PBFT_BATCHED=9202
RESULTS_DIR="results"
SWEEP_DIR="${RESULTS_DIR}/batch_sweep"
GW_IP="10.1.1.100"

MLDSA_SIG_BYTES=2420
MLDSA_VERIFY_US=796

ECDSA_SIG_BYTES=72
ECDSA_VERIFY_US=69
ECDSA_SIGN_US=27

# Calibrated packet-level batched-PBFT delay used by the paper's batch=50
# control. In this sweep we keep the commit-delay model fixed and vary only the
# logical transaction batch carried by one commit/update.
PBFT_BATCH_DELAY=0.0463

if [ "$REQUESTED_BATCH" = "all" ]; then
    # B=50 is already covered by the paper's main baseline artifact. The
    # default sensitivity sweep adds surrounding points only; pass "50"
    # explicitly as the second argument if a same-run anchor is needed.
    BATCHES=(1 10 25 100)
else
    BATCHES=("$REQUESTED_BATCH")
fi

mkdir -p "$SWEEP_DIR"
SUMMARY_FILE="${SWEEP_DIR}/summary.csv"
SUMMARY_HEADER="protocol,batch_size,rate_per_node,target_tps,avg_tps,total_tx,total_updates,tx_per_update,total_batches,gateway_batch_size,pbft_delay_s,verify_delay_us,expected_sig_len,rx_latency_p50_ms,rx_latency_p95_ms,gateway_queue_p50_ms,gateway_queue_p95_ms,settlement_p50_ms,settlement_p95_ms,direct_e2e_p50_ms,direct_e2e_p95_ms,estimated_batch_wait_p50_ms,estimated_batch_wait_p95_ms,estimated_e2e_p50_ms,estimated_e2e_p95_ms,latency_source,result_dir"
if [ -f "$SUMMARY_FILE" ] && ! head -n 1 "$SUMMARY_FILE" | grep -q "direct_e2e_p95_ms"; then
    mv "$SUMMARY_FILE" "${SUMMARY_FILE}.pre-direct-latency.bak"
fi
if [ ! -f "$SUMMARY_FILE" ] || [ "$REQUESTED_BATCH" = "all" ]; then
    echo "$SUMMARY_HEADER" > "$SUMMARY_FILE"
fi

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
        echo "        Start Docker + NS-3 first, or use: sudo bash run-q1-batch-sweep.sh ${DURATION}" >&2
        exit 1
    fi
    printf '%s\n' "$container"
}

should_run_protocol() {
    local protocol=$1
    [ "$REQUESTED_PROTOCOL" = "all" ] || [ "$REQUESTED_PROTOCOL" = "$protocol" ]
}

json_get() {
    local path=$1
    local expr=$2
    python3 - "$path" "$expr" <<'PY'
import json
import sys

path, expr = sys.argv[1:3]
with open(path, encoding="utf-8") as f:
    d = json.load(f)

cur = d
for part in expr.split("."):
    if part == "":
        continue
    if isinstance(cur, dict):
        cur = cur.get(part)
    else:
        cur = None
        break

if cur is None:
    print("")
elif isinstance(cur, float):
    print(f"{cur:.6f}")
else:
    print(cur)
PY
}

append_summary_row() {
    local protocol=$1
    local batch=$2
    local result_dir=$3
    local summary_json="${result_dir}/gateway_summary.json"
    python3 - "$protocol" "$batch" "$RATE" "$NUM_NODES" "$summary_json" "$result_dir" <<'PY' >> "$SUMMARY_FILE"
import json
import math
import sys

protocol, batch, rate, n_nodes, summary_path, result_dir = sys.argv[1:7]
batch = int(batch)
rate = float(rate)
n_nodes = int(n_nodes)

with open(summary_path, encoding="utf-8") as f:
    d = json.load(f)

def nested(name, stat):
    obj = d.get(name) or {}
    val = obj.get(stat)
    return "" if val is None else float(val)

def fmt(val):
    if val == "" or val is None:
        return ""
    if isinstance(val, float):
        if not math.isfinite(val):
            return ""
        return f"{val:.3f}"
    return str(val)

total_tx = int(d.get("total_tx", 0))
updates = int(d.get("accepted_updates", d.get("total_updates", 0)) or 0)
tx_per_update = total_tx / updates if updates else 0.0

rx_p50 = nested("rx_latency_ms", "p50")
rx_p95 = nested("rx_latency_ms", "p95")
queue_p50 = nested("gateway_queue_ms", "p50")
queue_p95 = nested("gateway_queue_ms", "p95")
settle_p50 = nested("settlement_latency_ms", "p50")
settle_p95 = nested("settlement_latency_ms", "p95")

# Direct transaction-level latency is available when tx-generator v2 packets
# carry first/last logical-TX timestamps. Older artifacts fall back to an
# analytical intra-node batch-fill delay.
direct_p50 = nested("transaction_e2e_latency_ms", "p50")
direct_p95 = nested("transaction_e2e_latency_ms", "p95")
fill_ms = min((batch / rate) * 1000.0, 2000.0)
wait_p50 = 0.5 * fill_ms
wait_p95 = 0.95 * fill_ms
e2e_p50 = (settle_p50 if settle_p50 != "" else 0.0) + wait_p50
e2e_p95 = (settle_p95 if settle_p95 != "" else 0.0) + wait_p95
latency_source = "measured" if direct_p95 != "" else "estimated"

row = [
    protocol,
    batch,
    int(rate),
    int(rate * n_nodes),
    float(d.get("avg_tps", 0.0)),
    total_tx,
    int(d.get("total_updates", 0) or 0),
    tx_per_update,
    int(d.get("total_batches", 0) or 0),
    int(d.get("batch_size", 0) or 0),
    float(d.get("pbft_delay_s", 0.0) or 0.0),
    float(d.get("verify_delay_us", 0.0) or 0.0),
    int(d.get("expected_sig_len", 0) or 0),
    rx_p50,
    rx_p95,
    queue_p50,
    queue_p95,
    settle_p50,
    settle_p95,
    direct_p50,
    direct_p95,
    wait_p50,
    wait_p95,
    e2e_p50,
    e2e_p95,
    latency_source,
    result_dir,
]
print(",".join(fmt(v) for v in row))
PY
}

run_experiment() {
    local protocol=$1
    local batch=$2
    local port=$3
    local tx_batch=$4
    local gw_batch=$5
    local pbft_delay=$6
    local expected_sig_len=$7
    local verify_delay_us=$8
    local emulated_sign_us=$9
    local label=${10}

    local slug="${protocol}_B${batch}"
    local run_dir="${SWEEP_DIR}/${slug}"
    local summary_file="${RESULTS_DIR}/gateway_summary_${slug}.json"
    local batches_file="${RESULTS_DIR}/gateway_batches_${slug}.csv"
    local log_file="${RESULTS_DIR}/gateway_log_${slug}.txt"

    mkdir -p "$run_dir"
    rm -f "$summary_file" "$batches_file" "$log_file" \
          "$run_dir/gateway_summary.json" \
          "$run_dir/gateway_batches.csv" \
          "$run_dir/gateway_log.txt"

    echo ""
    echo "------------------------------------------------------------"
    echo "  [B=${batch} ${protocol}] ${label}"
    echo "  rate=${RATE} tx/s/node, tx_batch=${tx_batch}, gateway_batch=${gw_batch}"
    echo "  pbft=${pbft_delay}s, sig=${expected_sig_len}B, verify=${verify_delay_us}us"
    echo "------------------------------------------------------------"

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
            --port $port \
            --batch-size $gw_batch \
            --duration $((DURATION + 15)) \
            --pbft-delay $pbft_delay \
            --verify-delay-us $verify_delay_us \
            --expected-sig-len $expected_sig_len \
            --output /opt/results/gateway_summary_${slug}.json \
            > /opt/results/gateway_batches_${slug}.csv \
            2> /opt/results/gateway_log_${slug}.txt
    "
    sleep 3

    if ! docker exec "$GW_CONT" pgrep -f "sig-aggregator.py" >/dev/null 2>&1; then
        echo "[ERROR] Gateway aggregator failed to start."
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
        [ -z "$container" ] && continue
        docker exec -d "$container" bash -c "
            /opt/pqc/tx-generator \
                --node-id $i \
                --gateway $GW_IP \
                --port $port \
                --algo ML-DSA-44 \
                --rate $RATE \
                --batch $tx_batch \
                --duration $DURATION \
                $extra_args \
                > /opt/results/tx_node_${slug}_${i}.csv \
                2> /opt/results/tx_node_${slug}_${i}.log
        "
        if (( i % 10 == 0 )); then
            sleep 0.5
        fi
    done

    echo "  Waiting ${DURATION}s + 15s gateway window..."
    sleep $((DURATION + 15))

    if [ ! -f "$summary_file" ]; then
        echo "[ERROR] Missing summary for ${slug}."
        cat "$log_file" 2>/dev/null || true
        exit 1
    fi

    python3 - "$summary_file" "$tx_batch" "$gw_batch" "$expected_sig_len" "$port" "$pbft_delay" "$verify_delay_us" <<'PY'
import json
import math
import sys

path, tx_batch, gw_batch, sig_len, port, pbft, verify_us = sys.argv[1:8]
tx_batch = float(tx_batch)
with open(path, encoding="utf-8") as f:
    d = json.load(f)

errors = []
if int(d.get("batch_size", -1)) != int(gw_batch):
    errors.append(f"gateway batch_size={d.get('batch_size')} expected {gw_batch}")
if int(d.get("expected_sig_len", -1)) != int(sig_len):
    errors.append(f"expected_sig_len={d.get('expected_sig_len')} expected {sig_len}")
if int(d.get("port", -1)) != int(port):
    errors.append(f"port={d.get('port')} expected {port}")
if abs(float(d.get("pbft_delay_s", -999)) - float(pbft)) > 1e-6:
    errors.append(f"pbft_delay_s={d.get('pbft_delay_s')} expected {pbft}")
if abs(float(d.get("verify_delay_us", -999)) - float(verify_us)) > 1e-6:
    errors.append(f"verify_delay_us={d.get('verify_delay_us')} expected {verify_us}")
if int(d.get("total_updates", 0)) <= 0:
    errors.append("total_updates is zero")

accepted_updates = max(int(d.get("accepted_updates", d.get("total_updates", 0))), 1)
tx_per_update = float(d.get("total_tx", 0)) / accepted_updates
if abs(tx_per_update - tx_batch) > max(2.0, 0.10 * tx_batch):
    errors.append(f"TX/update={tx_per_update:.2f} expected around {tx_batch:.0f}")

if errors:
    print("[ERROR] Invalid batch-sweep summary:")
    for e in errors:
        print("  -", e)
    print("  observed_sig_lens:", d.get("observed_sig_lens", {}))
    sys.exit(1)
PY

    cp "$summary_file" "$run_dir/gateway_summary.json"
    cp "$batches_file" "$run_dir/gateway_batches.csv" 2>/dev/null || true
    cp "$log_file" "$run_dir/gateway_log.txt" 2>/dev/null || true
    append_summary_row "$protocol" "$batch" "$run_dir"

    python3 - "$run_dir/gateway_summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    d = json.load(f)
updates = max(d.get("accepted_updates", d.get("total_updates", 0)), 1)
tx_per_update = d.get("total_tx", 0) / updates
lat = d.get("settlement_latency_ms") or {}
print(f"  RESULT: TPS={d.get('avg_tps', 0):.1f}, TX/update={tx_per_update:.1f}, "
      f"settlement p95={lat.get('p95', 0) or 0:.1f} ms")
PY
}

echo "================================================================"
echo "  BATCH-SIZE SENSITIVITY SWEEP"
echo "  Duration: ${DURATION}s | Nodes: ${NUM_NODES} | Rate: ${RATE} tx/s/node"
echo "  Batch sizes: ${BATCHES[*]} | Protocol: ${REQUESTED_PROTOCOL}"
echo "  Results: ${SWEEP_DIR}"
echo "================================================================"

GW_CONT=$(require_container gateway_node)
for i in $(seq 1 "$NUM_NODES"); do
    if [ -z "$(find_container "iot_node_${i}")" ]; then
        echo "[ERROR] Missing container for iot_node_${i}" >&2
        exit 1
    fi
done

if ! docker exec "$GW_CONT" python3 --version >/dev/null 2>&1; then
    echo "[SETUP] Installing python3 in gateway container..."
    docker exec "$GW_CONT" bash -c "apt-get update -qq && apt-get install -y -qq python3" >/dev/null
fi

for batch in "${BATCHES[@]}"; do
    if should_run_protocol "asc"; then
        # Match the existing baseline path at B=50: both the state-update
        # transaction batch and the gateway settlement batch use B.
        run_experiment "asc" "$batch" "$PORT_ASC" "$batch" "$batch" 0.0 \
            "$MLDSA_SIG_BYTES" "$MLDSA_VERIFY_US" -1 \
            "ASC + real ML-DSA-44"
    fi

    if should_run_protocol "pbft_batched"; then
        # Batched PBFT commits each logical tx batch as one gateway block.
        run_experiment "pbft_batched" "$batch" "$PORT_PBFT_BATCHED" "$batch" 1 "$PBFT_BATCH_DELAY" \
            "$ECDSA_SIG_BYTES" "$ECDSA_VERIFY_US" "$ECDSA_SIGN_US" \
            "Batched PBFT + ECDSA-sized packets"
    fi
done

echo ""
echo "================================================================"
echo "  SWEEP SUMMARY"
echo "================================================================"
python3 analysis/plot-batch-sweep.py --results-dir "$SWEEP_DIR" --no-plot
echo ""
echo "Plot/table: python3 analysis/plot-batch-sweep.py --results-dir $SWEEP_DIR"
