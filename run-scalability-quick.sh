#!/bin/bash
#
# run-scalability-quick.sh — Scalability Test RQ5
# So sánh 3 protocol × 5 node counts, rate=100 cố định (fair comparison)
#
# Usage: sudo bash run-scalability-quick.sh [duration]
#

set -e

DURATION=${1:-60}
RATE=100  # Cố định 100 tx/s/node — công bằng cho cả 3 protocol
GW_IP="10.1.1.100"
PORT=9000
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
RESULTS_DIR="$(cd "$(dirname "$0")" && pwd)/results"
SCALE_DIR="$RESULTS_DIR/scalability"

NODE_COUNTS=(10 25 50 75 100)

echo "================================================================="
echo "  SCALABILITY TEST — 3 Protocols × ${#NODE_COUNTS[@]} N values"
echo "  Rate = $RATE tx/s/node (fixed) | Duration = ${DURATION}s"
echo "  ASC logical batch size = ${ASC_BATCH_SIZE} tx/update"
echo "================================================================="

mkdir -p "$SCALE_DIR"

# ── Find gateway container ──
GW_CONT=""
for name in rc-iot-testbed_gateway_node_1 rc-iot-testbed-gateway_node-1; do
    if docker inspect "$name" &>/dev/null; then
        GW_CONT="$name"
        break
    fi
done
[ -z "$GW_CONT" ] && { echo "ERROR: Gateway container not found!"; exit 1; }
echo "Gateway: $GW_CONT"

# ── Find IoT container name pattern ──
find_container() {
    local i=$1
    for pat in "rc-iot-testbed_iot_node_${i}_1" "rc-iot-testbed-iot_node_${i}-1"; do
        if docker inspect "$pat" &>/dev/null; then
            echo "$pat"
            return
        fi
    done
}

# ── Run one protocol test ──
run_protocol() {
    local PROTO=$1    # pbft / bls / asc
    local N=$2
    local BATCH=$3
    local PBFT_DELAY=$4
    local SIG_BYTES=$MLDSA_SIG_BYTES
    local VERIFY_US=$MLDSA_VERIFY_US
    local EXTRA_ARGS=""

    if [ "$PROTO" = "pbft" ]; then
        SIG_BYTES=$ECDSA_SIG_BYTES
        VERIFY_US=$ECDSA_VERIFY_US
        EXTRA_ARGS="--emulated-sig-bytes $ECDSA_SIG_BYTES --emulated-sign-us $ECDSA_SIGN_US"
    elif [ "$PROTO" = "bls" ]; then
        SIG_BYTES=$BLS_SIG_BYTES
        VERIFY_US=$BLS_VERIFY_US
        EXTRA_ARGS="--emulated-sig-bytes $BLS_SIG_BYTES --emulated-sign-us $BLS_SIGN_US"
    fi

    echo "  [${PROTO^^}] N=$N batch=$BATCH delay=${PBFT_DELAY}s"

    # Kill old
    docker exec "$GW_CONT" pkill -f sig-aggregator 2>/dev/null || true
    for i in $(seq 1 100); do
        local C=$(find_container $i)
        [ -n "$C" ] && docker exec "$C" pkill -f tx-generator 2>/dev/null || true
    done
    sleep 3

    # Start Gateway
    docker exec -d "$GW_CONT" bash -c "
        python3 /opt/gateway/sig-aggregator.py \
            --port $PORT --batch-size $BATCH \
            --duration $((DURATION + 15)) \
            --pbft-delay $PBFT_DELAY \
            --verify-delay-us $VERIFY_US \
            --expected-sig-len $SIG_BYTES \
            --output /opt/results/${PROTO}_N${N}.json \
            > /opt/results/${PROTO}_N${N}.csv \
            2> /opt/results/${PROTO}_N${N}_log.txt
    "
    sleep 3

    # Start N tx-generators
    for i in $(seq 1 $N); do
        local C=$(find_container $i)
        [ -z "$C" ] && continue
        docker exec -d "$C" bash -c "
            /opt/pqc/tx-generator --node-id $i --gateway $GW_IP \
                --port $PORT --rate $RATE --duration $DURATION \
                --algo $ALGO --batch $BATCH \
                $EXTRA_ARGS \
                > /dev/null 2>&1
        "
        (( i % 10 == 0 )) && sleep 0.3
    done

    # Chờ đủ DURATION + 20s (đảm bảo gateway viết xong JSON)
    echo "    Running ${DURATION}s + 20s buffer..."
    sleep $((DURATION + 20))

    # Collect
    if [ -f "$RESULTS_DIR/${PROTO}_N${N}.json" ]; then
        cp "$RESULTS_DIR/${PROTO}_N${N}.json" "$SCALE_DIR/"
        python3 -c "
import json
with open('$SCALE_DIR/${PROTO}_N${N}.json') as f:
    d = json.load(f)
print(f'    ✓ ${PROTO^^} N=$N: TPS = {d[\"avg_tps\"]:.0f} | TX = {d[\"total_tx\"]:,}')
"
    else
        echo "    ✗ No result — check: cat $RESULTS_DIR/${PROTO}_N${N}_log.txt"
    fi
}

# ══════════════════════════════════════════════════
#  MAIN: 5 node counts × 3 protocols
# ══════════════════════════════════════════════════

for N in "${NODE_COUNTS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  N = $N nodes | Rate = $RATE tx/s/node"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # A) PBFT: on-chain, batch=1, calibrated consensus delay
    run_protocol "pbft" $N 1 "$PBFT_PER_TX_DELAY"

    # B) BLS: on-chain, batch=1, no PBFT
    run_protocol "bls" $N 1 0.0

    # C) ASC: off-chain, batch=50, instant
    run_protocol "asc" $N "$ASC_BATCH_SIZE" 0.0

    echo "  ✓ N=$N all protocols done"
done

# ══════════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════════
echo ""
echo "================================================================="
echo "  SCALABILITY RESULTS"
echo "================================================================="

python3 -c "
import json, os

scale_dir = '$SCALE_DIR'
node_counts = [10, 25, 50, 75, 100]
protos = ['asc', 'pbft', 'bls']

results = {p: [] for p in protos}
for N in node_counts:
    for p in protos:
        f = f'{scale_dir}/{p}_N{N}.json'
        tps = 0
        if os.path.exists(f):
            with open(f) as fh:
                tps = json.load(fh).get('avg_tps', 0)
        results[p].append(tps)

print(f'  {\"N\":>5} | {\"ASC TPS\":>10} | {\"PBFT TPS\":>10} | {\"BLS TPS\":>10} | {\"ASC/PBFT\":>8}')
print('  ' + '-' * 58)
for i, N in enumerate(node_counts):
    a, p, b = results['asc'][i], results['pbft'][i], results['bls'][i]
    ratio = a / max(p, 1)
    print(f'  {N:5d} | {a:10.0f} | {p:10.0f} | {b:10.0f} | {ratio:7.1f}x')

summary = {'node_counts': node_counts, 'rate': $RATE,
           'asc': results['asc'], 'pbft': results['pbft'], 'bls': results['bls']}
with open(f'{scale_dir}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\n  Saved: {scale_dir}/summary.json')
"
echo "  Plot: python analysis/plot_scalability.py"
echo "================================================================="
