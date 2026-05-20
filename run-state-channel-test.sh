#!/bin/bash
#
# run-state-channel-test.sh — Test kenh trang thai di dong bo
#
# Usage: sudo ./run-state-channel-test.sh [duration_sec] [rate_per_node]
#

set -e

DURATION=${1:-60}
RATE=${2:-10}
NUM_NODES=100
GW_IP="10.1.1.100"
PORT=9000
RESULTS_DIR="$(cd "$(dirname "$0")" && pwd)/results"
ALGO="ML-DSA-44"
ASC_BATCH_SIZE=50

# Tao thu muc rieng cho moi rate (scalability curve)
RUN_DIR="$RESULTS_DIR/tps_rate_${RATE}"
mkdir -p "$RUN_DIR"

echo "========================================================="
echo "[STATE-CHANNEL] Test Kenh Trang Thai"
echo "  Duration: ${DURATION}s | Rate: ${RATE} tx/s/node"
echo "  Nodes: $NUM_NODES | Target TPS: $((RATE * NUM_NODES))"
echo "  Algorithm: $ALGO | Gateway: $GW_IP:$PORT"
echo "  ASC logical batch size: ${ASC_BATCH_SIZE} tx/update"
echo "  Output: $RUN_DIR"
echo "========================================================="

# --- Kiem tra prerequisites ---
if ! docker exec rc-iot-testbed_iot_node_1_1 test -f /opt/pqc/tx-generator 2>/dev/null; then
    echo "[ERROR] /opt/pqc/tx-generator not found!"
    exit 1
fi

PID_GW=$(docker inspect -f '{{.State.Pid}}' rc-iot-testbed_gateway_node_1 2>/dev/null || true)
if [ -z "$PID_GW" ] || [ "$PID_GW" = "0" ]; then
    echo "[ERROR] Gateway container chua chay!"
    exit 1
fi

# --- Buoc 1: Kill old processes ---
echo "[1/4] Cleanup..."
docker exec rc-iot-testbed_gateway_node_1 pkill -f sig-aggregator 2>/dev/null || true
for i in $(seq 1 $NUM_NODES); do
    docker exec "rc-iot-testbed_iot_node_${i}_1" pkill -f tx-generator 2>/dev/null || true
done
sleep 2

# --- Buoc 2: Kiem tra Python ---
echo "[2/4] Kiem tra Gateway..."
if ! docker exec rc-iot-testbed_gateway_node_1 python3 --version 2>/dev/null; then
    echo "   Cai Python3..."
    docker exec rc-iot-testbed_gateway_node_1 bash -c "apt-get update -qq && apt-get install -y -qq python3" 2>&1 | tail -3
fi
echo "   Gateway OK"

# --- Buoc 3: Start Gateway aggregator ---
echo "[3/4] Start Gateway sig-aggregator..."

docker exec -d rc-iot-testbed_gateway_node_1 bash -c "
    python3 /opt/gateway/sig-aggregator.py \
        --port $PORT \
        --batch-size $ASC_BATCH_SIZE \
        --duration $((DURATION + 15)) \
        --output /opt/results/gateway_summary.json \
        > /opt/results/gateway_batches.csv \
        2> /opt/results/gateway_log.txt
"
sleep 3

# --- Buoc 4: Start IoT tx-generators (staggered start) ---
echo "[4/4] Start $NUM_NODES IoT tx-generators (staggered)..."

# Stagger start: 10 nodes moi 0.5 giay de tranh channel saturation
for i in $(seq 1 $NUM_NODES); do
    CONTAINER="rc-iot-testbed_iot_node_${i}_1"
    docker exec -d "$CONTAINER" bash -c "
        /opt/pqc/tx-generator \
            --node-id $i \
            --gateway $GW_IP \
            --port $PORT \
            --rate $RATE \
            --duration $DURATION \
            --algo $ALGO \
            --batch $ASC_BATCH_SIZE \
            > /opt/results/tx_node_${i}.csv \
            2> /opt/results/tx_node_${i}_log.txt
    "
    # Stagger start de giam WiFi contention burst
    (( i % 10 == 0 )) && {
        echo "   Started nodes 1-$i"
        sleep 0.5
    }
done

echo ""
echo "[RUNNING] $NUM_NODES nodes x $RATE tx/s = target TPS $((RATE * NUM_NODES))"
echo "[RUNNING] Doi ${DURATION}s..."

# Progress
for t in $(seq 10 10 $DURATION); do
    sleep 10
    LAST_LINE=$(tail -1 "$RESULTS_DIR/gateway_log.txt" 2>/dev/null || echo "waiting...")
    echo "  [t=${t}s] $LAST_LINE"
done

sleep 5

# --- Copy results ---
echo ""
echo "[COPY] Copy results..."
cp "$RESULTS_DIR/gateway_summary.json" "$RUN_DIR/" 2>/dev/null || true
cp "$RESULTS_DIR/gateway_batches.csv" "$RUN_DIR/" 2>/dev/null || true
cp "$RESULTS_DIR/gateway_log.txt" "$RUN_DIR/" 2>/dev/null || true
for i in $(seq 1 $NUM_NODES); do
    cp "$RESULTS_DIR/tx_node_${i}.csv" "$RUN_DIR/" 2>/dev/null || true
done

echo ""
echo "========================================================="
echo "[DONE] Rate=$RATE → $RUN_DIR"
if [ -f "$RUN_DIR/gateway_summary.json" ]; then
    cat "$RUN_DIR/gateway_summary.json" | head -10
fi
echo "========================================================="
