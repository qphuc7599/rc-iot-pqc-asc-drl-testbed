#!/bin/bash
#
# run-scalability-full.sh — Full Scalability Test Pipeline (RQ5)
#
# Chạy NS-3 + Docker testbed, đo ASC vs PBFT TPS tại N = 10, 25, 50, 75, 100
#
# CÁCH CHẠY (trong WSL, với sudo):
#   cd /mnt/d/DoAnChuyenNganh/rc-iot-testbed
#   sudo bash run-scalability-full.sh
#
# Pipeline cho MỖI giá trị N:
#   1. docker-compose up gateway + N nodes
#   2. For each protocol, start a fresh NS-3 WiFi simulation
#   3. connect_nodes.sh (bridge tap → docker)
#   4. Run one protocol workload
#   5. Cleanup before the next protocol
#   6. Collect results
#

set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$BASE_DIR/results/scalability"
DURATION=${1:-60}  # seconds per test
RATE=100           # tx/sec/node; high enough to expose network bottlenecks
PROTOCOL_COUNT=5
NS3_SIM_TIME=$((DURATION * 4 + 240))
GW_IP="10.1.1.100"
ASC_PORT=9101
PBFT_PORT=9102
BLS_PORT=9103
PBFT_BATCHED_PORT=9104
ASC_ECDSA_PORT=9105
ALGO="ML-DSA-44"
ASC_BATCH_SIZE=50
MLDSA_SIG_BYTES=2420
MLDSA_VERIFY_US=796
ECDSA_SIG_BYTES=72
ECDSA_VERIFY_US=69
ECDSA_SIGN_US=27
BLS_SIG_BYTES=96
BLS_VERIFY_US=0
BLS_SIGN_US=0
PBFT_PER_TX_DELAY=0.055
PBFT_BATCH50_DELAY=0.0463

NODE_COUNTS=(10 25 50 75 100)

mkdir -p "$RESULTS_DIR"

compose() {
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose "$@"
    else
        docker compose "$@"
    fi
}

find_gateway_container() {
    docker ps --format '{{.Names}}' | grep gateway | head -1
}

ensure_gateway_python() {
    local gw_cont
    gw_cont=$(find_gateway_container)
    if [ -z "$gw_cont" ]; then
        echo "  [ERROR] Gateway container is not running."
        return 1
    fi
    if docker exec "$gw_cont" python3 --version >/dev/null 2>&1; then
        return
    fi
    echo "  Installing python3 in gateway container..."
    docker exec "$gw_cont" bash -c "apt-get update -qq && apt-get install -y -qq python3" >/dev/null
    docker exec "$gw_cont" python3 --version
}

wait_for_summary() {
    local label=$1
    local N=$2
    local summary_file=$3
    local log_file=$4
    local timeout=$((DURATION + 30))

    echo "  [${label}] Running ${DURATION}s + gateway flush window..."
    for elapsed in $(seq 1 "$timeout"); do
        if [ -f "$summary_file" ] && python3 - "$summary_file" >/dev/null 2>&1 <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    json.load(f)
PY
        then
            return 0
        fi
        sleep 1
        if (( elapsed % 10 == 0 )); then
            LAST=$(tail -1 "$log_file" 2>/dev/null || echo "waiting for gateway log...")
            echo "    [${label} N=${N} t=${elapsed}s] $LAST"
        fi
    done

    echo "  [ERROR] Missing ${label} summary for N=${N} after ${timeout}s"
    echo "  Expected: $summary_file"
    echo "  Gateway log tail:"
    tail -80 "$log_file" 2>/dev/null || true
    exit 1
}

echo "================================================================="
echo "  SCALABILITY TEST — Full Pipeline (NS-3 + Docker)"
echo "  N = ${NODE_COUNTS[*]}"
echo "  Duration = ${DURATION}s per test | Rate = ${RATE} tx/s/node"
echo "  NS-3 simTime = ${NS3_SIM_TIME}s per isolated protocol session"
echo "  ASC logical batch size = ${ASC_BATCH_SIZE} tx/update"
echo "  Protocols: A PBFT, B BLS, C ASC+ML-DSA, D PBFT-batched, E ASC+ECDSA"
echo "  Isolation: fresh Docker + NS-3 session for every N/protocol pair"
echo "================================================================="

# ── Helper: cleanup everything ──
cleanup_all() {
    echo "  Cleanup: stopping all containers and NS-3..."
    cd "$BASE_DIR"
    compose down 2>/dev/null || true
    pkill -f "iot-storm-network" 2>/dev/null || true
    # Remove tap/bridge interfaces
    for i in $(seq 0 100); do
        ip link delete "veth${i}" 2>/dev/null || true
        ip link delete "br${i}" 2>/dev/null || true
    done
    sleep 3
}

# ── Helper: start Docker containers for N nodes ──
trap cleanup_all EXIT

start_containers() {
    local N=$1
    echo "  Starting gateway + $N IoT containers..."
    cd "$BASE_DIR"

    # Gateway first
    compose up -d gateway_node 2>&1 | tail -1
    sleep 2

    # IoT nodes in batches of 10
    for start in $(seq 1 10 $N); do
        end=$((start + 9))
        [ $end -gt $N ] && end=$N
        SERVICES=""
        for i in $(seq $start $end); do
            SERVICES="$SERVICES iot_node_${i}"
        done
        compose up -d $SERVICES 2>&1 | tail -1
        sleep 2
    done

    RUNNING=$(docker ps -q | wc -l)
    echo "  Containers running: $RUNNING"
    ensure_gateway_python
}

# ── Helper: start NS-3 and connect ──
start_ns3_and_connect() {
    local N=$1
    echo "  Starting NS-3 WiFi simulation (nIoT=$N, simTime=${NS3_SIM_TIME}s)..."
    cd "$BASE_DIR/ns-3-dev"

    # Start NS-3 in background
    ./build/scratch/ns3-dev-iot-storm-network-optimized \
        --nIoT=$N --simTime=$NS3_SIM_TIME &
    NS3_PID=$!
    echo "  NS-3 PID: $NS3_PID"

    # Wait for tap devices
    echo "  Waiting for NS-3 tap devices..."
    TIMEOUT=60
    ELAPSED=0
    while ! ip link show tap0 &>/dev/null; do
        sleep 1
        ELAPSED=$((ELAPSED + 1))
        [ $ELAPSED -ge $TIMEOUT ] && {
            echo "  [ERROR] NS-3 tap timeout!"
            kill $NS3_PID 2>/dev/null || true
            return 1
        }
    done
    sleep 5

    # Connect containers to NS-3 via bridges
    echo "  Connecting $N nodes to NS-3..."
    cd "$BASE_DIR"

    # Disable bridge iptables
    sysctl -w net.bridge.bridge-nf-call-iptables=0 >/dev/null 2>&1 || true

    SUBNET="10.1.1"

    # Create bridges and veths
    for i in $(seq 0 $N); do
        TAP="tap${i}"
        BR="br${i}"
        VETH="veth${i}"
        VETHC="veth${i}c"

        ip link delete "$VETH" 2>/dev/null || true
        ip link delete "$BR" 2>/dev/null || true

        TAP_MAC=$(cat "/sys/class/net/${TAP}/address" 2>/dev/null || echo "")
        [ -z "$TAP_MAC" ] && continue

        ip link add "$BR" type bridge
        echo 0 > "/sys/class/net/${BR}/bridge/stp_state" 2>/dev/null || true
        ip link set "$BR" type bridge forward_delay 0
        ip link set "$BR" up

        ip link add "$VETH" type veth peer name "$VETHC"
        ip link set "$VETHC" address "$TAP_MAC"
        ip link set "$VETH" up

        DUMMY_MAC=$(printf "02:ff:ff:ff:ff:%02x" "$i")
        ip link set "$TAP" address "$DUMMY_MAC"

        ip link set "$TAP" master "$BR"
        ip link set "$VETH" master "$BR"
        ip link set "$TAP" txqueuelen 1000 2>/dev/null || true
    done

    # Move veth ends into containers
    mkdir -p /var/run/netns

    # Gateway
    PID_GW=$(docker inspect -f '{{.State.Pid}}' rc-iot-testbed_gateway_node_1 2>/dev/null || \
             docker inspect -f '{{.State.Pid}}' rc-iot-testbed-gateway_node-1 2>/dev/null || true)
    if [ -n "$PID_GW" ] && [ "$PID_GW" != "0" ]; then
        ln -sf "/proc/$PID_GW/ns/net" "/var/run/netns/$PID_GW"
        ip link set veth0c netns "$PID_GW"
        ip netns exec "$PID_GW" ip addr add "${GW_IP}/24" dev veth0c 2>/dev/null || true
        ip netns exec "$PID_GW" ip link set veth0c up
    fi

    # IoT Nodes
    for i in $(seq 1 $N); do
        CONTAINER="rc-iot-testbed_iot_node_${i}_1"
        PID=$(docker inspect -f '{{.State.Pid}}' "$CONTAINER" 2>/dev/null || true)
        [ -z "$PID" ] || [ "$PID" = "0" ] && {
            CONTAINER="rc-iot-testbed-iot_node_${i}-1"
            PID=$(docker inspect -f '{{.State.Pid}}' "$CONTAINER" 2>/dev/null || true)
        }
        [ -z "$PID" ] || [ "$PID" = "0" ] && continue

        ln -sf "/proc/$PID/ns/net" "/var/run/netns/$PID"
        ip link set "veth${i}c" netns "$PID"
        IP_LAST=$((100 + i))
        ip netns exec "$PID" ip addr add "${SUBNET}.${IP_LAST}/24" dev "veth${i}c" 2>/dev/null || true
        ip netns exec "$PID" ip link set "veth${i}c" up
        sleep 0.02
    done

    echo "  NS-3 + Docker connected OK"
}

# ── Run ASC test ──
run_protocol_test() {
    local N=$1
    local label=$2
    local slug=$3
    local port=$4
    local gw_batch=$5
    local tx_batch=$6
    local pbft_delay=$7
    local sig_bytes=$8
    local verify_us=$9
    local emulated_sign_us=${10}
    local description=${11}

    local OUT_DIR="$RESULTS_DIR/N_${N}"
    local summary_file="$BASE_DIR/results/gateway_summary_${slug}_N${N}.json"
    local log_file="$BASE_DIR/results/gateway_log_${slug}_N${N}.txt"
    local batches_file="$BASE_DIR/results/gateway_batches_${slug}_N${N}.csv"
    local extra_args=""

    mkdir -p "$OUT_DIR"
    rm -f "$summary_file" "$log_file" "$batches_file" \
          "$OUT_DIR/gateway_summary_${slug}_N${N}.json" \
          "$OUT_DIR/gateway_batches_${slug}_N${N}.csv" \
          "$OUT_DIR/gateway_log_${slug}_N${N}.txt"

    GW_CONT=$(find_gateway_container)
    docker exec "$GW_CONT" pkill -f sig-aggregator 2>/dev/null || true
    for i in $(seq 1 $N); do
        CONT=$(docker ps --format '{{.Names}}' | grep "iot_node_${i}[_-]" | head -1)
        [ -z "$CONT" ] && continue
        docker exec "$CONT" pkill -f tx-generator 2>/dev/null || true
    done
    sleep 3

    echo "  [${label}] ${description}"
    echo "  [${label}] port=${port}, tx_batch=${tx_batch}, gateway_batch=${gw_batch}, pbft=${pbft_delay}s, sig=${sig_bytes}B"

    docker exec -d "$GW_CONT" bash -c "
        python3 /opt/gateway/sig-aggregator.py \
            --port $port \
            --batch-size $gw_batch \
            --duration $((DURATION + 15)) \
            --pbft-delay $pbft_delay \
            --verify-delay-us $verify_us \
            --expected-sig-len $sig_bytes \
            --output /opt/results/gateway_summary_${slug}_N${N}.json \
            > /opt/results/gateway_batches_${slug}_N${N}.csv \
            2> /opt/results/gateway_log_${slug}_N${N}.txt
    "
    sleep 3

    if [ "$emulated_sign_us" -ge 0 ]; then
        extra_args="--emulated-sig-bytes $sig_bytes --emulated-sign-us $emulated_sign_us"
    fi

    echo "  [${label}] Starting $N tx-generators..."
    for i in $(seq 1 $N); do
        CONT=$(docker ps --format '{{.Names}}' | grep "iot_node_${i}[_-]" | head -1)
        [ -z "$CONT" ] && continue
        docker exec -d "$CONT" bash -c "
            /opt/pqc/tx-generator --node-id $i --gateway $GW_IP \
                --port $port --rate $RATE --duration $DURATION \
                --algo $ALGO --batch $tx_batch \
                $extra_args \
                > /dev/null 2>&1
        "
        (( i % 10 == 0 )) && sleep 0.3
    done

    wait_for_summary "$label" "$N" "$summary_file" "$log_file"

    cp "$summary_file" "$OUT_DIR/" 2>/dev/null || true
    cp "$batches_file" "$OUT_DIR/" 2>/dev/null || true
    cp "$log_file" "$OUT_DIR/" 2>/dev/null || true

    if ! python3 - "$OUT_DIR/gateway_summary_${slug}_N${N}.json" "$label" "$N" <<'PY'
import json
import sys

summary_path, label, n_nodes = sys.argv[1:4]
with open(summary_path) as f:
    d = json.load(f)

total_tx = int(d.get("total_tx", 0))
if total_tx <= 0:
    print(f"  [ERROR] {label} N={n_nodes}: summary has total_tx=0; this run is invalid.", file=sys.stderr)
    sys.exit(2)

updates = max(d.get("accepted_updates", d.get("total_updates", 0)), 1)
tx_per_update = total_tx / updates
print(f"  [{label}] N={n_nodes}: TPS = {d['avg_tps']:.0f} | TX = {total_tx} | TX/update = {tx_per_update:.1f}")
PY
    then
        echo "  Gateway log tail:"
        tail -80 "$log_file" 2>/dev/null || true
        exit 1
    fi
}

run_isolated_protocol() {
    local N=$1
    shift
    local label=$1

    echo ""
    echo "  --- Isolated run: N=${N}, protocol ${label} ---"
    cleanup_all
    start_containers "$N"
    start_ns3_and_connect "$N"
    run_protocol_test "$N" "$@"
    cleanup_all
}

run_asc_test() {
    local N=$1
    local OUT_DIR="$RESULTS_DIR/N_${N}"
    local summary_file="$BASE_DIR/results/gateway_summary_asc_N${N}.json"
    local log_file="$BASE_DIR/results/gateway_log_asc_N${N}.txt"
    local batches_file="$BASE_DIR/results/gateway_batches_asc_N${N}.csv"
    mkdir -p "$OUT_DIR"
    rm -f "$summary_file" "$log_file" "$batches_file" "$OUT_DIR/gateway_summary_asc_N${N}.json"

    echo "  [ASC] Starting gateway sig-aggregator..."

    # Find gateway container name
    GW_CONT=$(find_gateway_container)

    docker exec -d "$GW_CONT" bash -c "
        python3 /opt/gateway/sig-aggregator.py \
            --port $ASC_PORT --batch-size $ASC_BATCH_SIZE \
            --duration $((DURATION + 15)) \
            --verify-delay-us $MLDSA_VERIFY_US \
            --expected-sig-len $MLDSA_SIG_BYTES \
            --output /opt/results/gateway_summary_asc_N${N}.json \
            > /opt/results/gateway_batches_asc_N${N}.csv \
            2> /opt/results/gateway_log_asc_N${N}.txt
    "
    sleep 3

    echo "  [ASC] Starting $N tx-generators (batch=${ASC_BATCH_SIZE})..."
    for i in $(seq 1 $N); do
        CONT=$(docker ps --format '{{.Names}}' | grep "iot_node_${i}[_-]" | head -1)
        [ -z "$CONT" ] && continue
        docker exec -d "$CONT" bash -c "
            /opt/pqc/tx-generator --node-id $i --gateway $GW_IP \
                --port $ASC_PORT --rate $RATE --duration $DURATION \
                --algo $ALGO --batch $ASC_BATCH_SIZE \
                > /dev/null 2>&1
        "
        (( i % 10 == 0 )) && sleep 0.3
    done

    wait_for_summary "ASC" "$N" "$summary_file" "$log_file"

    # Collect
    cp "$summary_file" "$OUT_DIR/" 2>/dev/null || true
    cp "$batches_file" "$OUT_DIR/" 2>/dev/null || true
    cp "$log_file" "$OUT_DIR/" 2>/dev/null || true

    if [ -f "$OUT_DIR/gateway_summary_asc_N${N}.json" ]; then
        python3 -c "
import json
with open('$OUT_DIR/gateway_summary_asc_N${N}.json') as f:
    d = json.load(f)
print(f'  [ASC] N={$N}: TPS = {d[\"avg_tps\"]:.0f} | TX = {d[\"total_tx\"]}')
"
    fi
}

# ── Run PBFT test ──
run_pbft_test() {
    local N=$1
    local OUT_DIR="$RESULTS_DIR/N_${N}"
    local summary_file="$BASE_DIR/results/gateway_summary_pbft_N${N}.json"
    local log_file="$BASE_DIR/results/gateway_log_pbft_N${N}.txt"
    local batches_file="$BASE_DIR/results/gateway_batches_pbft_N${N}.csv"
    mkdir -p "$OUT_DIR"
    rm -f "$summary_file" "$log_file" "$batches_file" "$OUT_DIR/gateway_summary_pbft_N${N}.json"

    # Kill old generators
    GW_CONT=$(find_gateway_container)
    docker exec "$GW_CONT" pkill -f sig-aggregator 2>/dev/null || true
    for i in $(seq 1 $N); do
        CONT=$(docker ps --format '{{.Names}}' | grep "iot_node_${i}[_-]" | head -1)
        [ -z "$CONT" ] && continue
        docker exec "$CONT" pkill -f tx-generator 2>/dev/null || true
    done
    sleep 3

    # Per-transaction PBFT control: one accepted update per commit.
    PBFT_DELAY=$PBFT_PER_TX_DELAY
    echo "  [PBFT] block_delay=${PBFT_DELAY}s (per-transaction commit control)"

    docker exec -d "$GW_CONT" bash -c "
        python3 /opt/gateway/sig-aggregator.py \
            --port $PBFT_PORT --batch-size 1 \
            --duration $((DURATION + 15)) \
            --pbft-delay $PBFT_DELAY \
            --verify-delay-us $ECDSA_VERIFY_US \
            --expected-sig-len $ECDSA_SIG_BYTES \
            --output /opt/results/gateway_summary_pbft_N${N}.json \
            > /opt/results/gateway_batches_pbft_N${N}.csv \
            2> /opt/results/gateway_log_pbft_N${N}.txt
    "
    sleep 3

    echo "  [PBFT] Starting $N tx-generators (batch=1, on-chain)..."
    for i in $(seq 1 $N); do
        CONT=$(docker ps --format '{{.Names}}' | grep "iot_node_${i}[_-]" | head -1)
        [ -z "$CONT" ] && continue
        docker exec -d "$CONT" bash -c "
            /opt/pqc/tx-generator --node-id $i --gateway $GW_IP \
                --port $PBFT_PORT --rate $RATE --duration $DURATION \
                --algo $ALGO --batch 1 \
                --emulated-sig-bytes $ECDSA_SIG_BYTES \
                --emulated-sign-us $ECDSA_SIGN_US \
                > /dev/null 2>&1
        "
        (( i % 10 == 0 )) && sleep 0.3
    done

    wait_for_summary "PBFT" "$N" "$summary_file" "$log_file"

    cp "$summary_file" "$OUT_DIR/" 2>/dev/null || true
    cp "$batches_file" "$OUT_DIR/" 2>/dev/null || true
    cp "$log_file" "$OUT_DIR/" 2>/dev/null || true

    if [ -f "$OUT_DIR/gateway_summary_pbft_N${N}.json" ]; then
        python3 -c "
import json
with open('$OUT_DIR/gateway_summary_pbft_N${N}.json') as f:
    d = json.load(f)
print(f'  [PBFT] N={$N}: TPS = {d[\"avg_tps\"]:.0f}')
"
    fi
}

# ══════════════════════════════════════════════════════
# -- Run BLS aggregate-signature control --
run_bls_test() {
    local N=$1
    local OUT_DIR="$RESULTS_DIR/N_${N}"
    local summary_file="$BASE_DIR/results/gateway_summary_bls_N${N}.json"
    local log_file="$BASE_DIR/results/gateway_log_bls_N${N}.txt"
    local batches_file="$BASE_DIR/results/gateway_batches_bls_N${N}.csv"
    mkdir -p "$OUT_DIR"
    rm -f "$summary_file" "$log_file" "$batches_file" "$OUT_DIR/gateway_summary_bls_N${N}.json"

    GW_CONT=$(find_gateway_container)
    docker exec "$GW_CONT" pkill -f sig-aggregator 2>/dev/null || true
    for i in $(seq 1 $N); do
        CONT=$(docker ps --format '{{.Names}}' | grep "iot_node_${i}[_-]" | head -1)
        [ -z "$CONT" ] && continue
        docker exec "$CONT" pkill -f tx-generator 2>/dev/null || true
    done
    sleep 3

    echo "  [BLS] Starting aggregate-signature control (batch=1, no PBFT)..."
    docker exec -d "$GW_CONT" bash -c "
        python3 /opt/gateway/sig-aggregator.py \
            --port $BLS_PORT --batch-size 1 \
            --duration $((DURATION + 15)) \
            --verify-delay-us $BLS_VERIFY_US \
            --expected-sig-len $BLS_SIG_BYTES \
            --output /opt/results/gateway_summary_bls_N${N}.json \
            > /opt/results/gateway_batches_bls_N${N}.csv \
            2> /opt/results/gateway_log_bls_N${N}.txt
    "
    sleep 3

    echo "  [BLS] Starting $N tx-generators (BLS-sized packets)..."
    for i in $(seq 1 $N); do
        CONT=$(docker ps --format '{{.Names}}' | grep "iot_node_${i}[_-]" | head -1)
        [ -z "$CONT" ] && continue
        docker exec -d "$CONT" bash -c "
            /opt/pqc/tx-generator --node-id $i --gateway $GW_IP \
                --port $BLS_PORT --rate $RATE --duration $DURATION \
                --algo $ALGO --batch 1 \
                --emulated-sig-bytes $BLS_SIG_BYTES \
                --emulated-sign-us $BLS_SIGN_US \
                > /dev/null 2>&1
        "
        (( i % 10 == 0 )) && sleep 0.3
    done

    wait_for_summary "BLS" "$N" "$summary_file" "$log_file"

    cp "$summary_file" "$OUT_DIR/" 2>/dev/null || true
    cp "$batches_file" "$OUT_DIR/" 2>/dev/null || true
    cp "$log_file" "$OUT_DIR/" 2>/dev/null || true

    if [ -f "$OUT_DIR/gateway_summary_bls_N${N}.json" ]; then
        python3 -c "
import json
with open('$OUT_DIR/gateway_summary_bls_N${N}.json') as f:
    d = json.load(f)
print(f'  [BLS] N={$N}: TPS = {d[\"avg_tps\"]:.0f}')
"
    fi
}

#  MAIN LOOP
# ══════════════════════════════════════════════════════

for N in "${NODE_COUNTS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  N = $N nodes"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Step 1: Clean stale outputs for this N
    cleanup_all
    rm -rf "$RESULTS_DIR/N_${N}"
    rm -f "$BASE_DIR"/results/gateway_summary_*_N${N}.json \
          "$BASE_DIR"/results/gateway_batches_*_N${N}.csv \
          "$BASE_DIR"/results/gateway_log_*_N${N}.txt

    # A) Per-transaction PBFT + ECDSA-sized packets
    run_isolated_protocol "$N" "A" "pbft" "$PBFT_PORT" 1 1 "$PBFT_PER_TX_DELAY" \
        "$ECDSA_SIG_BYTES" "$ECDSA_VERIFY_US" "$ECDSA_SIGN_US" \
        "Per-transaction PBFT + ECDSA-sized packets"

    # B) Aggregate-signature control, no PBFT
    run_isolated_protocol "$N" "B" "bls" "$BLS_PORT" 1 1 0.0 \
        "$BLS_SIG_BYTES" "$BLS_VERIFY_US" "$BLS_SIGN_US" \
        "BLS-sized aggregate-signature control"

    # C) Off-chain state channel + real ML-DSA-44
    run_isolated_protocol "$N" "C" "asc" "$ASC_PORT" "$ASC_BATCH_SIZE" "$ASC_BATCH_SIZE" 0.0 \
        "$MLDSA_SIG_BYTES" "$MLDSA_VERIFY_US" -1 \
        "ASC + real ML-DSA-44"

    # D) Batched PBFT + ECDSA-sized packets
    run_isolated_protocol "$N" "D" "pbft_batched" "$PBFT_BATCHED_PORT" 1 "$ASC_BATCH_SIZE" "$PBFT_BATCH50_DELAY" \
        "$ECDSA_SIG_BYTES" "$ECDSA_VERIFY_US" "$ECDSA_SIGN_US" \
        "Batched PBFT + ECDSA-sized packets"

    # E) Off-chain state channel + ECDSA-sized packets
    run_isolated_protocol "$N" "E" "asc_ecdsa" "$ASC_ECDSA_PORT" "$ASC_BATCH_SIZE" "$ASC_BATCH_SIZE" 0.0 \
        "$ECDSA_SIG_BYTES" "$ECDSA_VERIFY_US" "$ECDSA_SIGN_US" \
        "ASC no-PQC ablation"

    echo "  ✓ N=$N complete"
done

# ── Final cleanup ──
cleanup_all

# ── Generate summary ──
echo ""
echo "================================================================="
echo "  RESULTS SUMMARY"
echo "================================================================="

python3 <<'PYEOF'
import json, os

scale_dir = os.environ.get("RESULTS_DIR", "results/scalability")
if not os.path.isabs(scale_dir):
    scale_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), scale_dir)

node_counts = [10, 25, 50, 75, 100]
print(f"\n{'N':>5} | {'A PBFT':>9} | {'B BLS':>9} | {'C ASC':>9} | {'D PBFT50':>9} | {'E ASC-ECDSA':>12}")
print("-" * 72)

protocols = ["pbft", "bls", "asc", "pbft_batched", "asc_ecdsa"]
summary = {"node_counts": node_counts, **{p: [] for p in protocols}}
for N in node_counts:
    d = f"results/scalability/N_{N}"
    row = {}
    for proto in protocols:
        val = 0
        try:
            with open(f"{d}/gateway_summary_{proto}_N{N}.json") as f:
                val = json.load(f).get("avg_tps", 0)
        except Exception:
            pass
        row[proto] = val
        summary[proto].append(val)
    print(
        f"{N:5d} | {row['pbft']:9.0f} | {row['bls']:9.0f} | "
        f"{row['asc']:9.0f} | {row['pbft_batched']:9.0f} | {row['asc_ecdsa']:12.0f}"
    )

with open("results/scalability/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved: results/scalability/summary.json")
PYEOF

echo ""
echo "  Tiếp theo: chạy 'python analysis/plot_scalability.py' để vẽ biểu đồ"
echo "================================================================="
