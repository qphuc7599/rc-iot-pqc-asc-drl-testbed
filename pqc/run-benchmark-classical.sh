#!/bin/bash
#
# run-benchmark-classical.sh — Benchmark ECDSA-P256 & Ed25519 tren 100 containers
# Fair comparison: CUNG testbed voi ML-DSA
#
# Usage: sudo ./pqc/run-benchmark-classical.sh [iterations]
# Output: results/summary_ecdsa.csv, results/summary_ed25519.csv
#

set -e
ITERATIONS=${1:-100}
NUM_NODES=100
PARALLEL=20
RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"

mkdir -p "$RESULTS_DIR"

echo "========================================================="
echo "[BENCH] Classical Crypto Benchmark — Same Testbed"
echo "[BENCH] ECDSA-P256 + Ed25519 | Iterations: $ITERATIONS"
echo "========================================================="

for ALGO in ecdsa-p256 ed25519; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${ALGO}] Running on ${NUM_NODES} nodes..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    BATCH=0
    for i in $(seq 1 $NUM_NODES); do
        CONTAINER="rc-iot-testbed_iot_node_${i}_1"
        OUTPUT="$RESULTS_DIR/${ALGO}_node_${i}.json"

        docker exec "$CONTAINER" /opt/pqc/ecdsa-benchmark all \
            --algo "$ALGO" \
            --iterations "$ITERATIONS" \
            --node-id "$i" \
            > "$OUTPUT" 2>/dev/null &

        BATCH=$((BATCH + 1))
        if [ $BATCH -ge $PARALLEL ]; then
            wait
            BATCH=0
            echo "  [${ALGO}] Nodes 1-$i done"
        fi
    done
    wait
    echo "  [${ALGO}] All $NUM_NODES nodes done"

    # Create summary CSV
    SAFE_NAME=$(echo "$ALGO" | tr '-' '_')
    SUMMARY="$RESULTS_DIR/summary_${SAFE_NAME}.csv"
    echo "node_id,algorithm,iterations,pk_bytes,sig_bytes,keygen_avg_us,keygen_stddev_us,sign_avg_us,sign_stddev_us,verify_avg_us,verify_stddev_us" > "$SUMMARY"

    for i in $(seq 1 $NUM_NODES); do
        FILE="$RESULTS_DIR/${ALGO}_node_${i}.json"
        [ -f "$FILE" ] || continue

        ALGO_NAME=$(grep '"algorithm"' "$FILE" | sed 's/.*: "\(.*\)".*/\1/')
        ITER=$(grep '"iterations"' "$FILE" | sed 's/.*: \([0-9]*\).*/\1/')
        PK=$(grep '"pk_bytes"' "$FILE" | sed 's/.*: \([0-9]*\).*/\1/')
        SIG=$(grep '"sig_bytes"' "$FILE" | sed 's/.*: \([0-9]*\).*/\1/')
        KG_AVG=$(grep '"keygen_us"' "$FILE" | sed 's/.*"avg": \([0-9.]*\).*/\1/')
        KG_STD=$(grep '"keygen_us"' "$FILE" | sed 's/.*"stddev": \([0-9.]*\).*/\1/')
        SN_AVG=$(grep '"sign_us"' "$FILE" | sed 's/.*"avg": \([0-9.]*\).*/\1/')
        SN_STD=$(grep '"sign_us"' "$FILE" | sed 's/.*"stddev": \([0-9.]*\).*/\1/')
        VF_AVG=$(grep '"verify_us"' "$FILE" | sed 's/.*"avg": \([0-9.]*\).*/\1/')
        VF_STD=$(grep '"verify_us"' "$FILE" | sed 's/.*"stddev": \([0-9.]*\).*/\1/')

        echo "$i,$ALGO_NAME,$ITER,$PK,$SIG,$KG_AVG,$KG_STD,$SN_AVG,$SN_STD,$VF_AVG,$VF_STD" >> "$SUMMARY"
    done

    echo "  ✅ ${ALGO} → $SUMMARY"
done

echo ""
echo "========================================================="
echo "  DONE! Plot: python3 analysis/plot-pqc-vs-classical.py"
echo "========================================================="
