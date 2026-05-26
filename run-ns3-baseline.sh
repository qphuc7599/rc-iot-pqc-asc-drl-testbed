#!/usr/bin/env bash
#
# Publication-grade NS-3 baseline wrapper.
#
# This script runs the N=100 baseline comparison through the original
# Docker + NS-3 setup described by the advisor:
#   1. Start gateway + 100 constrained IoT containers.
#   2. Start the realtime NS-3 802.11g tap-bridge simulation.
#   3. Bridge tap devices into Docker network namespaces.
#   4. Run A/B/C/D/E baseline comparison.
#
# Usage in WSL:
#   cd /mnt/d/DoAnChuyenNganh/rc-iot-testbed
#   sudo bash run-ns3-baseline.sh 60
#

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DURATION="${1:-60}"
NUM_NODES=100
GW_IP="10.1.1.100"
SUBNET="10.1.1"
NS3_BIN="$BASE_DIR/ns-3-dev/build/scratch/ns3-dev-iot-storm-network-optimized"
NS3_PID=""

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
        ip link delete "tap${i}" 2>/dev/null || true
        ip tuntap del dev "tap${i}" mode tap 2>/dev/null || true
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
        end=$((start + 9))
        [ "$end" -gt "$NUM_NODES" ] && end="$NUM_NODES"
        services=""
        for i in $(seq "$start" "$end"); do
            services="$services iot_node_${i}"
        done
        compose up -d $services
        sleep 2
        echo "  Started nodes $start-$end"
    done
}

start_ns3() {
    local sim_time=$((DURATION * 8 + 300))
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
    if ! kill -0 "$NS3_PID" 2>/dev/null; then
        echo "[ERROR] NS-3 exited while creating tap devices."
        echo "        This is usually caused by stale tap devices. Cleanup has been updated;"
        echo "        rerun the wrapper, or manually remove stale tap devices."
        exit 1
    fi
}

connect_ns3_to_docker() {
    echo "[3/4] Connecting NS-3 tap devices to Docker containers..."
    cd "$BASE_DIR"
    sysctl -w net.bridge.bridge-nf-call-iptables=0 >/dev/null 2>&1 || true
    mkdir -p /var/run/netns

    for i in $(seq 0 "$NUM_NODES"); do
        tap="tap${i}"
        br="br${i}"
        veth="veth${i}"
        vethc="veth${i}c"

        ip link delete "$veth" 2>/dev/null || true
        ip link delete "$br" 2>/dev/null || true

        tap_mac=$(cat "/sys/class/net/${tap}/address" 2>/dev/null || true)
        [ -z "$tap_mac" ] && continue

        ip link add "$br" type bridge
        echo 0 > "/sys/class/net/${br}/bridge/stp_state" 2>/dev/null || true
        ip link set "$br" type bridge forward_delay 0
        ip link set "$br" up

        ip link add "$veth" type veth peer name "$vethc"
        ip link set "$vethc" address "$tap_mac"
        ip link set "$veth" up

        dummy_mac=$(printf "02:ff:ff:ff:ff:%02x" "$i")
        ip link set "$tap" address "$dummy_mac"
        ip link set "$tap" master "$br"
        ip link set "$veth" master "$br"
        ip link set "$tap" txqueuelen 1000 2>/dev/null || true
    done

    gw_cont=$(find_container gateway_node)
    if [ -z "$gw_cont" ]; then
        echo "[ERROR] Gateway container not found."
        exit 1
    fi
    pid_gw=$(docker inspect -f '{{.State.Pid}}' "$gw_cont")
    ln -sf "/proc/$pid_gw/ns/net" "/var/run/netns/$pid_gw"
    ip link set veth0c netns "$pid_gw"
    ip netns exec "$pid_gw" ip addr add "${GW_IP}/24" dev veth0c 2>/dev/null || true
    ip netns exec "$pid_gw" ip link set veth0c up

    for i in $(seq 1 "$NUM_NODES"); do
        cont=$(find_container "iot_node_${i}")
        [ -z "$cont" ] && {
            echo "  [WARN] Missing container iot_node_${i}"
            continue
        }
        pid=$(docker inspect -f '{{.State.Pid}}' "$cont")
        ln -sf "/proc/$pid/ns/net" "/var/run/netns/$pid"
        ip link set "veth${i}c" netns "$pid"
        ip_last=$((100 + i))
        ip netns exec "$pid" ip addr add "${SUBNET}.${ip_last}/24" dev "veth${i}c" 2>/dev/null || true
        ip netns exec "$pid" ip link set "veth${i}c" up
        sleep 0.02
    done
}

run_one_baseline() {
    local mode=$1
    echo "[4/4] Running baseline mode ${mode} through a fresh NS-3 session..."
    cd "$BASE_DIR"
    bash "$BASE_DIR/run-baseline-comparison.sh" "$DURATION" "$mode"
}

echo "================================================================"
echo "  NS-3 BASELINE WRAPPER"
echo "  Duration per mode: ${DURATION}s | Nodes: ${NUM_NODES}"
echo "================================================================"

BASELINE_MODES=(
    A_pbft_ecdsa
    B_bls_aggregate
    C_offchain_statechannel
    E_offchain_ecdsa
    D_pbft_batched_ecdsa
    F_simplex_batched_ecdsa
    G_bullshark_dag_ecdsa
    H_hydra_ecdsa
)

for mode in "${BASELINE_MODES[@]}"; do
    echo ""
    echo "================================================================"
    echo "  MODE: ${mode}"
    echo "================================================================"
    cleanup_all
    start_containers
    start_ns3
    connect_ns3_to_docker
    run_one_baseline "$mode"
done

cleanup_all
python3 "$BASE_DIR/analysis/plot-comparison.py"

echo "================================================================"
echo "  DONE"
echo "  Results: results/comparison_* and results/baseline_comparison.png"
echo "================================================================"
