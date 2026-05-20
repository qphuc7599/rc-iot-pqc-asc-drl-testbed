#!/bin/bash
#
# run-benchmark-levels.sh — Benchmark ML-DSA 3 security levels
# Output: results/summary_44.csv, summary_65.csv, summary_87.csv
#
set -e
ITERATIONS=${1:-50}
NUM_NODES=100
RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"

echo "========================================================="
echo "[BENCH] ML-DSA Security Level Comparison"
echo "[BENCH] Levels: 44, 65, 87 | Iterations: $ITERATIONS"
echo "========================================================="

for LEVEL in 44 65 87; do
    ALGO="ML-DSA-${LEVEL}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${ALGO}] Running on ${NUM_NODES} nodes..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Run benchmark (20 nodes parallel)
    BATCH=0
    for i in $(seq 1 $NUM_NODES); do
        CONTAINER="rc-iot-testbed_iot_node_${i}_1"
        docker exec "$CONTAINER" /opt/pqc/pqc-benchmark all \
            --algo "$ALGO" \
            --iterations "$ITERATIONS" \
            --node-id "$i" \
            > "$RESULTS_DIR/node_${LEVEL}_${i}.json" 2>/dev/null &
        
        BATCH=$((BATCH + 1))
        if [ $BATCH -ge 20 ]; then
            wait
            BATCH=0
        fi
    done
    wait
    
    # Create summary CSV
    SUMMARY="$RESULTS_DIR/summary_${LEVEL}.csv"
    echo "node_id,algorithm,iterations,pk_bytes,sk_bytes,sig_bytes,keygen_avg_us,sign_avg_us,verify_avg_us" > "$SUMMARY"
    
    for i in $(seq 1 $NUM_NODES); do
        FILE="$RESULTS_DIR/node_${LEVEL}_${i}.json"
        [ -f "$FILE" ] || continue
        
        ALGO_NAME=$(grep '"algorithm"' "$FILE" | sed 's/.*: "\(.*\)".*/\1/')
        PK=$(grep '"pk_bytes"' "$FILE" | sed 's/.*: \([0-9]*\).*/\1/')
        SK=$(grep '"sk_bytes"' "$FILE" | sed 's/.*: \([0-9]*\).*/\1/')
        SIG=$(grep '"sig_bytes"' "$FILE" | sed 's/.*: \([0-9]*\).*/\1/')
        KG_AVG=$(grep '"keygen_us"' "$FILE" | sed 's/.*"avg": \([0-9.]*\).*/\1/')
        SN_AVG=$(grep '"sign_us"' "$FILE" | sed 's/.*"avg": \([0-9.]*\).*/\1/')
        VF_AVG=$(grep '"verify_us"' "$FILE" | sed 's/.*"avg": \([0-9.]*\).*/\1/')
        
        echo "$i,$ALGO_NAME,$ITERATIONS,$PK,$SK,$SIG,$KG_AVG,$SN_AVG,$VF_AVG" >> "$SUMMARY"
    done
    
    echo "  ✅ ${ALGO} done → $SUMMARY"
done

echo ""
echo "========================================================="
echo "  Plot: python3 analysis/plot-pqc-results.py"
echo "========================================================="
