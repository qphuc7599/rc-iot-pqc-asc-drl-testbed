#!/usr/bin/env bash
# ===========================================================
# run-comparison-benchmark.sh
# Chay ca ML-DSA-44 va BLS12-381 tren 100 IoT containers
# So sanh Lattice (PQC) vs Bilinear Pairings (baseline)
# ===========================================================
set -e

ITERATIONS=${1:-50}
RESULTS_DIR="results"
SUMMARY_MLDSA="$RESULTS_DIR/summary_mldsa.csv"
SUMMARY_BLS="$RESULTS_DIR/summary_bls.csv"

mkdir -p "$RESULTS_DIR"

# --- Kiem tra binary ---
SAMPLE=$(docker ps --format '{{.Names}}' | grep iot_node | head -1)
if [ -z "$SAMPLE" ]; then
    echo "[ERROR] Khong tim thay IoT containers. Chay docker-compose up -d truoc."
    exit 1
fi

# Kiem tra pqc-benchmark
if ! docker exec "$SAMPLE" test -f /usr/local/bin/pqc-benchmark 2>/dev/null; then
    echo "[WARN] pqc-benchmark chua co. Copy vao containers..."
    if [ ! -f pqc/pqc-benchmark ]; then
        echo "[BUILD] Build benchmarks..."
        docker build -f pqc/Dockerfile.build -t pqc-builder pqc/
        CONTAINER_ID=$(docker create pqc-builder)
        docker cp "$CONTAINER_ID:/pqc-benchmark" pqc/pqc-benchmark
        docker cp "$CONTAINER_ID:/bilinear-benchmark" pqc/bilinear-benchmark 2>/dev/null || true
        docker rm "$CONTAINER_ID"
    fi
    echo "[COPY] Copy pqc-benchmark vao 100 containers..."
    for c in $(docker ps --format '{{.Names}}' | grep iot_node); do
        docker cp pqc/pqc-benchmark "$c:/usr/local/bin/pqc-benchmark"
    done
fi

# Kiem tra bilinear-benchmark
if ! docker exec "$SAMPLE" test -f /usr/local/bin/bilinear-benchmark 2>/dev/null; then
    if [ ! -f pqc/bilinear-benchmark ]; then
        echo "[BUILD] Build bilinear-benchmark..."
        docker build -f pqc/Dockerfile.build -t pqc-builder pqc/
        CONTAINER_ID=$(docker create pqc-builder)
        docker cp "$CONTAINER_ID:/bilinear-benchmark" pqc/bilinear-benchmark
        docker rm "$CONTAINER_ID"
    fi
    echo "[COPY] Copy bilinear-benchmark vao 100 containers..."
    for c in $(docker ps --format '{{.Names}}' | grep iot_node); do
        docker cp pqc/bilinear-benchmark "$c:/usr/local/bin/bilinear-benchmark"
    done
fi

# --- CSV headers ---
echo "node_id,algorithm,iterations,pk_bytes,sk_bytes,sig_bytes,keygen_avg_us,keygen_stddev_us,sign_avg_us,sign_stddev_us,verify_avg_us,verify_stddev_us" > "$SUMMARY_MLDSA"
echo "node_id,algorithm,iterations,pk_bytes,sk_bytes,sig_bytes,keygen_avg_us,keygen_stddev_us,sign_avg_us,sign_stddev_us,verify_avg_us,verify_stddev_us" > "$SUMMARY_BLS"

echo ""
echo "==========================================================="
echo "  BENCHMARK: ML-DSA-44 vs BLS12-381 trên 100 IoT containers"
echo "  Iterations: $ITERATIONS per operation per node"
echo "==========================================================="

# --- Chay benchmark tren tung node ---
TOTAL=$(docker ps --format '{{.Names}}' | grep iot_node | wc -l)
COUNT=0

for CONTAINER in $(docker ps --format '{{.Names}}' | grep iot_node | sort); do
    COUNT=$((COUNT + 1))
    NODE_ID=$(echo "$CONTAINER" | grep -oP '\d+' | tail -1)

    # --- ML-DSA-44 ---
    JSON_ML=$(docker exec "$CONTAINER" /usr/local/bin/pqc-benchmark all \
        --algo ML-DSA-44 --iterations "$ITERATIONS" --node-id "$NODE_ID" 2>/dev/null) || true

    if [ -n "$JSON_ML" ]; then
        echo "$JSON_ML" > "$RESULTS_DIR/mldsa_node_${NODE_ID}.json"
        # Extract CSV fields
        KG_AVG=$(echo "$JSON_ML" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['keygen_us']['avg'])" 2>/dev/null || echo "0")
        KG_STD=$(echo "$JSON_ML" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['keygen_us']['stddev'])" 2>/dev/null || echo "0")
        SN_AVG=$(echo "$JSON_ML" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['sign_us']['avg'])" 2>/dev/null || echo "0")
        SN_STD=$(echo "$JSON_ML" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['sign_us']['stddev'])" 2>/dev/null || echo "0")
        VF_AVG=$(echo "$JSON_ML" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['verify_us']['avg'])" 2>/dev/null || echo "0")
        VF_STD=$(echo "$JSON_ML" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['verify_us']['stddev'])" 2>/dev/null || echo "0")
        echo "$NODE_ID,ML-DSA-44,$ITERATIONS,1312,2560,2420,$KG_AVG,$KG_STD,$SN_AVG,$SN_STD,$VF_AVG,$VF_STD" >> "$SUMMARY_MLDSA"
    fi

    # --- BLS12-381 ---
    JSON_BLS=$(docker exec "$CONTAINER" /usr/local/bin/bilinear-benchmark \
        --iterations "$ITERATIONS" --node-id "$NODE_ID" 2>/dev/null) || true

    if [ -n "$JSON_BLS" ]; then
        echo "$JSON_BLS" > "$RESULTS_DIR/bls_node_${NODE_ID}.json"
        KG_AVG=$(echo "$JSON_BLS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['keygen_us']['avg'])" 2>/dev/null || echo "0")
        KG_STD=$(echo "$JSON_BLS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['keygen_us']['stddev'])" 2>/dev/null || echo "0")
        SN_AVG=$(echo "$JSON_BLS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['sign_us']['avg'])" 2>/dev/null || echo "0")
        SN_STD=$(echo "$JSON_BLS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['sign_us']['stddev'])" 2>/dev/null || echo "0")
        VF_AVG=$(echo "$JSON_BLS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['verify_us']['avg'])" 2>/dev/null || echo "0")
        VF_STD=$(echo "$JSON_BLS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['verify_us']['stddev'])" 2>/dev/null || echo "0")
        echo "$NODE_ID,BLS12-381,$ITERATIONS,48,32,96,$KG_AVG,$KG_STD,$SN_AVG,$SN_STD,$VF_AVG,$VF_STD" >> "$SUMMARY_BLS"
    fi

    if [ $((COUNT % 10)) -eq 0 ]; then
        echo "  [$COUNT/$TOTAL] Done"
    fi
done

echo ""
echo "==========================================================="
echo "  RESULTS:"
echo "    ML-DSA:  $SUMMARY_MLDSA ($(wc -l < "$SUMMARY_MLDSA") rows)"
echo "    BLS:     $SUMMARY_BLS ($(wc -l < "$SUMMARY_BLS") rows)"
echo ""
echo "  Plot: python3 analysis/plot-comparison.py"
echo "==========================================================="
