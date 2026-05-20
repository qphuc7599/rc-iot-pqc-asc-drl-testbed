#!/bin/bash
#
# build.sh — Build pqc-benchmark static binary bang Docker
#
# Su dung: ./pqc/build.sh
# Output:  pqc/bin/pqc-benchmark (static binary, ~2-3MB)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================================="
echo "[BUILD] Compiling pqc-benchmark (liboqs ML-DSA, static)"
echo "========================================================="

cd "$SCRIPT_DIR"

# Build using Docker (chi build stage "builder")
docker build -f Dockerfile.build --target builder -t pqc-builder .

# Extract binary from container
CONTAINER_ID=$(docker create --entrypoint=true pqc-builder)
docker cp "$CONTAINER_ID:/pqc-benchmark" "$SCRIPT_DIR/bin/pqc-benchmark"
docker cp "$CONTAINER_ID:/tx-generator" "$SCRIPT_DIR/bin/tx-generator"
docker cp "$CONTAINER_ID:/ecdsa-benchmark" "$SCRIPT_DIR/bin/ecdsa-benchmark"
docker cp "$CONTAINER_ID:/tx-generator" "$SCRIPT_DIR/tx-generator"
docker rm "$CONTAINER_ID"

chmod +x "$SCRIPT_DIR/bin/pqc-benchmark"
chmod +x "$SCRIPT_DIR/bin/tx-generator"
chmod +x "$SCRIPT_DIR/bin/ecdsa-benchmark"
chmod +x "$SCRIPT_DIR/tx-generator"

echo ""
echo "[BUILD] Thanh cong!"
ls -lh "$SCRIPT_DIR/bin/pqc-benchmark"
ls -lh "$SCRIPT_DIR/bin/ecdsa-benchmark"
echo ""
echo "[BUILD] Test nhanh:"
"$SCRIPT_DIR/bin/pqc-benchmark" list
"$SCRIPT_DIR/bin/ecdsa-benchmark" list
echo ""
echo "[BUILD] Chay benchmark:"
echo "  docker exec rc-iot-testbed_iot_node_1_1 /opt/pqc/pqc-benchmark all --iterations 10"
echo "  docker exec rc-iot-testbed_iot_node_1_1 /opt/pqc/ecdsa-benchmark all --algo ecdsa-p256 --iterations 10"
echo "========================================================="
