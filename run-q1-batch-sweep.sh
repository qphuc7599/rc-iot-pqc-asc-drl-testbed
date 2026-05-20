#!/usr/bin/env bash
#
# Publication-grade batch-size sensitivity wrapper.
#
# Runs every requested (batch size, protocol) pair in a fresh Docker + NS-3
# session to avoid queue/backlog leakage between measurements.
#
# Usage in WSL:
#   sudo bash run-q1-batch-sweep.sh [duration_seconds] [asc|pbft_batched|all] [default|all|batch_csv]
#
# Examples:
#   sudo bash run-q1-batch-sweep.sh 60 asc
#   sudo bash run-q1-batch-sweep.sh 60 all
#   sudo bash run-q1-batch-sweep.sh 60 all 50
#   sudo bash run-q1-batch-sweep.sh 60 all all

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DURATION="${1:-60}"
REQUESTED_PROTOCOL="${2:-asc}"
REQUESTED_BATCHES="${3:-default}"
NUM_NODES=100
GW_IP="10.1.1.100"
SUBNET="10.1.1"
NS3_BIN="$BASE_DIR/ns-3-dev/build/scratch/ns3-dev-iot-storm-network-optimized"
NS3_PID=""
# B=50 is already measured by the main paper baseline. The default wrapper adds
# surrounding sensitivity points only. Pass "50" for a direct B=50 p95 anchor,
# "all" for 1/10/25/50/100, or a comma-separated list such as "25,50,100".
if [ "$REQUESTED_BATCHES" = "default" ]; then
    BATCHES=(1 10 25 100)
elif [ "$REQUESTED_BATCHES" = "all" ]; then
    BATCHES=(1 10 25 50 100)
else
    IFS=',' read -r -a BATCHES <<< "$REQUESTED_BATCHES"
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "[ERROR] Run with sudo. NS-3 tap/bridge setup needs root."
    exit 1
fi

if [ ! -x "$NS3_BIN" ]; then
    echo "[ERROR] NS-3 binary not found or not executable:"
    echo "        $NS3_BIN"
    echo "        Build ns-3 first, then rerun this script."
    exit 1
fi

if [ ! -x "$BASE_DIR/pqc/bin/tx-generator" ]; then
    echo "[ERROR] pqc/bin/tx-generator is missing."
    echo "        Run: bash pqc/build.sh"
    exit 1
fi

compose() {
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose "$@"
    else
        docker compose "$@"
    fi
}

cleanup_all() {
    echo "[CLEANUP] Stopping containers, NS-3, and bridges..."
    cd "$BASE_DIR"
    compose down 2>/dev/null || true
    if [ -n "${NS3_PID:-}" ]; then
        kill "$NS3_PID" 2>/dev/null || true
    fi
    pkill -f "iot-storm-network" 2>/dev/null || true
    for i in $(seq 0 "$NUM_NODES"); do
        ip link delete "veth${i}" 2>/dev/null || true
        ip link delete "br${i}" 2>/dev/null || true
    done
    NS3_PID=""
}

trap cleanup_all EXIT

find_container() {
    local service=$1
    docker ps --format '{{.Names}}' | grep -E "(^|[-_])${service}[-_][0-9]+$" | head -1
}

start_containers() {
    echo "[1/4] Starting gateway + $NUM_NODES IoT containers..."
    cd "$BASE_DIR"
    compose up -d gateway_node
    sleep 2

    for start in $(seq 1 10 "$NUM_NODES"); do
        local end=$((start + 9))
        [ "$end" -gt "$NUM_NODES" ] && end="$NUM_NODES"
        local services=""
        for i in $(seq "$start" "$end"); do
            services="$services iot_node_${i}"
        done
        compose up -d $services
        sleep 2
        echo "  Started nodes $start-$end"
    done
}

start_ns3() {
    local sim_time=$((DURATION * 4 + 240))
    echo "[2/4] Starting NS-3 realtime WiFi simulation, simTime=${sim_time}s..."
    cd "$BASE_DIR/ns-3-dev"
    "$NS3_BIN" --nIoT="$NUM_NODES" --simTime="$sim_time" &
    NS3_PID=$!

    echo "  Waiting for tap0..."
    local elapsed=0
    while ! ip link show tap0 >/dev/null 2>&1; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ "$elapsed" -ge 60 ]; then
            echo "[ERROR] NS-3 tap timeout."
            exit 1
        fi
    done
    sleep 5
}

connect_ns3_to_docker() {
    echo "[3/4] Connecting NS-3 tap devices to Docker containers..."
    cd "$BASE_DIR"
    sysctl -w net.bridge.bridge-nf-call-iptables=0 >/dev/null 2>&1 || true
    mkdir -p /var/run/netns

    for i in $(seq 0 "$NUM_NODES"); do
        local tap="tap${i}"
        local br="br${i}"
        local veth="veth${i}"
        local vethc="veth${i}c"

        ip link delete "$veth" 2>/dev/null || true
        ip link delete "$br" 2>/dev/null || true

        local tap_mac
        tap_mac=$(cat "/sys/class/net/${tap}/address" 2>/dev/null || true)
        [ -z "$tap_mac" ] && continue

        ip link add "$br" type bridge
        echo 0 > "/sys/class/net/${br}/bridge/stp_state" 2>/dev/null || true
        ip link set "$br" type bridge forward_delay 0
        ip link set "$br" up

        ip link add "$veth" type veth peer name "$vethc"
        ip link set "$vethc" address "$tap_mac"
        ip link set "$veth" up

        local dummy_mac
        dummy_mac=$(printf "02:ff:ff:ff:ff:%02x" "$i")
        ip link set "$tap" address "$dummy_mac"
        ip link set "$tap" master "$br"
        ip link set "$veth" master "$br"
        ip link set "$tap" txqueuelen 1000 2>/dev/null || true
    done

    local gw_cont
    gw_cont=$(find_container gateway_node)
    if [ -z "$gw_cont" ]; then
        echo "[ERROR] Gateway container not found."
        exit 1
    fi
    local pid_gw
    pid_gw=$(docker inspect -f '{{.State.Pid}}' "$gw_cont")
    ln -sf "/proc/$pid_gw/ns/net" "/var/run/netns/$pid_gw"
    ip link set veth0c netns "$pid_gw"
    ip netns exec "$pid_gw" ip addr add "${GW_IP}/24" dev veth0c 2>/dev/null || true
    ip netns exec "$pid_gw" ip link set veth0c up

    for i in $(seq 1 "$NUM_NODES"); do
        local cont
        cont=$(find_container "iot_node_${i}")
        [ -z "$cont" ] && {
            echo "  [WARN] Missing container iot_node_${i}"
            continue
        }
        local pid
        pid=$(docker inspect -f '{{.State.Pid}}' "$cont")
        ln -sf "/proc/$pid/ns/net" "/var/run/netns/$pid"
        ip link set "veth${i}c" netns "$pid"
        local ip_last=$((100 + i))
        ip netns exec "$pid" ip addr add "${SUBNET}.${ip_last}/24" dev "veth${i}c" 2>/dev/null || true
        ip netns exec "$pid" ip link set "veth${i}c" up
        sleep 0.02
    done
}

protocols_to_run() {
    if [ "$REQUESTED_PROTOCOL" = "all" ]; then
        echo "asc pbft_batched"
    else
        echo "$REQUESTED_PROTOCOL"
    fi
}

echo "================================================================"
echo "  Q1 BATCH-SIZE SENSITIVITY WRAPPER"
echo "  Duration per run: ${DURATION}s | Nodes: ${NUM_NODES}"
echo "  Batch sizes: ${BATCHES[*]} | Protocol: ${REQUESTED_PROTOCOL}"
echo "================================================================"

rm -rf "$BASE_DIR/results/batch_sweep"
mkdir -p "$BASE_DIR/results/batch_sweep"

for batch in "${BATCHES[@]}"; do
    for protocol in $(protocols_to_run); do
        echo ""
        echo "================================================================"
        echo "  B=${batch}, protocol=${protocol}"
        echo "================================================================"
        cleanup_all
        start_containers
        start_ns3
        connect_ns3_to_docker
        echo "[4/4] Running batch sweep workload..."
        cd "$BASE_DIR"
        bash "$BASE_DIR/run-batch-size-sweep.sh" "$DURATION" "$batch" "$protocol"
    done
done

cleanup_all
python3 "$BASE_DIR/analysis/plot-batch-sweep.py" --results-dir "$BASE_DIR/results/batch_sweep"

echo "================================================================"
echo "  DONE"
echo "  Results: results/batch_sweep"
echo "================================================================"
