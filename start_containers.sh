#!/bin/bash
#
# start_containers.sh — Khoi tao 100 containers theo batch
# Docker-compose v1 bi treo khi tao qua nhieu cung luc.
# Script nay tao theo batch 10 cai, cho moi batch xong roi tiep.
#
set -e

cd "$(dirname "$0")"

TOTAL=100
BATCH_SIZE=10
GATEWAY="gateway_node"

compose() {
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose "$@"
    else
        docker compose "$@"
    fi
}

echo "========================================================="
echo "[START] Khoi tao 1 Gateway + $TOTAL IoT containers"
echo "[START] Batch size: $BATCH_SIZE"
echo "========================================================="

# Buoc 1: Gateway truoc
echo "[1] Tao Gateway..."
compose up -d $GATEWAY 2>&1 | grep -v "is up-to-date" || true
sleep 2

# Buoc 2: IoT nodes theo batch
CREATED=0
for START in $(seq 1 $BATCH_SIZE $TOTAL); do
    END=$((START + BATCH_SIZE - 1))
    [ $END -gt $TOTAL ] && END=$TOTAL

    # Build service list cho batch nay
    SERVICES=""
    for i in $(seq $START $END); do
        SERVICES="$SERVICES iot_node_${i}"
    done

    echo "[BATCH] Creating nodes $START-$END..."
    compose up -d $SERVICES 2>&1 | grep -v "is up-to-date" || true

    # Cho het batch boot xong
    sleep 3

    # Dem running
    RUNNING=$(docker ps -q | wc -l)
    echo "   Running: $RUNNING containers"
done

# Buoc 3: Kiem tra
echo ""
echo "========================================================="
RUNNING=$(docker ps -q | wc -l)
EXITED=$(docker ps -a -q --filter status=exited | wc -l)
echo "[DONE] Running: $RUNNING | Exited: $EXITED"
echo "========================================================="
