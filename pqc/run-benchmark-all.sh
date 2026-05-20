#!/bin/bash
#
# run-benchmark-all.sh — Chay ML-DSA benchmark tren tat ca IoT containers
#
# Su dung: sudo ./pqc/run-benchmark-all.sh [iterations] [algo]
#
# Output:  results/node_*.json
#          results/summary.csv
#

set -e

ITERATIONS=${1:-100}
ALGO=${2:-"ML-DSA-44"}
RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"
NUM_NODES=100
PARALLEL=20  # Chay song song 20 nodes 1 luc (tranh overload host)

mkdir -p "$RESULTS_DIR"

echo "========================================================="
echo "[BENCH] ML-DSA Benchmark - $NUM_NODES IoT Containers"
echo "[BENCH] Algorithm: $ALGO | Iterations: $ITERATIONS"
echo "========================================================="

# Kiem tra binary co mount dung khong
FIRST_CONTAINER="rc-iot-testbed_iot_node_1_1"
if ! docker exec "$FIRST_CONTAINER" test -f /opt/pqc/pqc-benchmark 2>/dev/null; then
    echo "[ERROR] Binary /opt/pqc/pqc-benchmark khong tim thay trong container!"
    echo "[ERROR] Hay rebuild containers voi docker-compose up -d"
    exit 1
fi

echo "[BENCH] Bat dau chay benchmark..."
echo ""

# Chay theo batch
BATCH=0
for i in $(seq 1 $NUM_NODES); do
    CONTAINER="rc-iot-testbed_iot_node_${i}_1"
    OUTPUT="$RESULTS_DIR/node_${i}.json"

    docker exec "$CONTAINER" /opt/pqc/pqc-benchmark all \
        --algo "$ALGO" \
        --iterations "$ITERATIONS" \
        --node-id "$i" \
        > "$OUTPUT" 2>/dev/null &

    BATCH=$((BATCH + 1))

    if [ $BATCH -ge $PARALLEL ]; then
        wait
        BATCH=0
        echo "[BENCH] Nodes 1-$i done"
    fi
done
wait
echo "[BENCH] Nodes 1-$NUM_NODES done"

# Tao summary CSV
echo ""
echo "[BENCH] Tao summary CSV..."
SUMMARY="$RESULTS_DIR/summary.csv"
echo "node_id,algorithm,iterations,pk_bytes,sk_bytes,sig_bytes,keygen_avg_us,keygen_stddev_us,sign_avg_us,sign_stddev_us,verify_avg_us,verify_stddev_us,energy_keygen_esp32_mj,energy_sign_esp32_mj,energy_verify_esp32_mj" > "$SUMMARY"

for i in $(seq 1 $NUM_NODES); do
    FILE="$RESULTS_DIR/node_${i}.json"
    [ -f "$FILE" ] || continue

    # Parse JSON (dung sed/grep vi khong co jq trong containers)
    ALGO_NAME=$(grep '"algorithm"' "$FILE" | sed 's/.*: "\(.*\)".*/\1/')
    ITER=$(grep '"iterations"' "$FILE" | sed 's/.*: \([0-9]*\).*/\1/')
    PK=$(grep '"pk_bytes"' "$FILE" | sed 's/.*: \([0-9]*\).*/\1/')
    SK=$(grep '"sk_bytes"' "$FILE" | sed 's/.*: \([0-9]*\).*/\1/')
    SIG=$(grep '"sig_bytes"' "$FILE" | sed 's/.*: \([0-9]*\).*/\1/')

    KG_AVG=$(grep '"keygen_us"' "$FILE" | sed 's/.*"avg": \([0-9.]*\).*/\1/')
    KG_STD=$(grep '"keygen_us"' "$FILE" | sed 's/.*"stddev": \([0-9.]*\).*/\1/')
    SN_AVG=$(grep '"sign_us"' "$FILE" | sed 's/.*"avg": \([0-9.]*\).*/\1/')
    SN_STD=$(grep '"sign_us"' "$FILE" | sed 's/.*"stddev": \([0-9.]*\).*/\1/')
    VF_AVG=$(grep '"verify_us"' "$FILE" | sed 's/.*"avg": \([0-9.]*\).*/\1/')
    VF_STD=$(grep '"verify_us"' "$FILE" | sed 's/.*"stddev": \([0-9.]*\).*/\1/')

    E_KG=$(grep '"keygen_esp32"' "$FILE" | sed 's/.*: \([0-9.]*\).*/\1/')
    E_SN=$(grep '"sign_esp32"' "$FILE" | sed 's/.*: \([0-9.]*\).*/\1/')
    E_VF=$(grep '"verify_esp32"' "$FILE" | sed 's/.*: \([0-9.]*\).*/\1/')

    echo "$i,$ALGO_NAME,$ITER,$PK,$SK,$SIG,$KG_AVG,$KG_STD,$SN_AVG,$SN_STD,$VF_AVG,$VF_STD,$E_KG,$E_SN,$E_VF" >> "$SUMMARY"
done

echo ""
echo "========================================================="
echo "[BENCH] HOAN TAT!"
echo "[BENCH] Results: $RESULTS_DIR/"
echo "[BENCH] Summary: $SUMMARY"
echo "[BENCH] Plot:    python3 analysis/plot-pqc-results.py"
echo "========================================================="
