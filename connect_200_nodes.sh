#!/bin/bash
#
# connect_200_nodes.sh — Connect 200 IoT + Gateway to NS-3
#
# Kien truc:
#   Container(veth, MAC=WiFi MAC) <-> bridge <-> tap(MAC=WiFi MAC) <-> NS-3 WiFi
#

set -e

NUM_IOT=100
SUBNET="10.1"      # /16 subnet for >200 nodes
GW_IP="${SUBNET}.0.100"

echo "========================================================="
echo "[CONNECT] Ket noi $NUM_IOT IoT + Gateway vao NS-3"
echo "========================================================="

# --- BUOC 1: Cho NS-3 tao tap devices ---
echo "[1/4] Cho NS-3 tao tap devices..."
TIMEOUT=90
ELAPSED=0
while ! ip link show tap0 &>/dev/null; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    [ $ELAPSED -ge $TIMEOUT ] && { echo "[ERROR] Timeout! Chay NS-3 truoc."; exit 1; }
done
echo "   tap0 OK. Cho them 10s cho $NUM_IOT tap devices..."
sleep 10

# Tat bridge iptables filtering
sysctl -w net.bridge.bridge-nf-call-iptables=0 >/dev/null 2>&1 || true
sysctl -w net.bridge.bridge-nf-call-ip6tables=0 >/dev/null 2>&1 || true

# --- BUOC 2: Tao bridge + veth (MAC matching) ---
echo "[2/4] Tao bridge + veth (MAC matching)..."
for i in $(seq 0 $NUM_IOT); do
    TAP="tap${i}"
    BR="br${i}"
    VETH="veth${i}"
    VETHC="veth${i}c"

    # Check tap exists
    if ! ip link show "$TAP" &>/dev/null; then
        echo "   WARN: $TAP not found, skipping"
        continue
    fi

    # Xoa cu
    ip link delete "$VETH" 2>/dev/null || true
    ip link delete "$BR" 2>/dev/null || true

    # Lay MAC cua tap (= WiFi MAC do NS-3 gan)
    TAP_MAC=$(cat "/sys/class/net/${TAP}/address")

    # Tao bridge
    ip link add "$BR" type bridge
    echo 0 > "/sys/class/net/${BR}/bridge/stp_state" 2>/dev/null || true
    ip link set "$BR" type bridge forward_delay 0
    sysctl -w "net.ipv6.conf.${BR}.disable_ipv6=1" >/dev/null 2>&1
    ip link set "$BR" up

    # Tao veth pair
    ip link add "$VETH" type veth peer name "$VETHC"
    sysctl -w "net.ipv6.conf.${VETH}.disable_ipv6=1" >/dev/null 2>&1
    sysctl -w "net.ipv6.conf.${VETHC}.disable_ipv6=1" >/dev/null 2>&1

    # Set veth MAC = tap/WiFi MAC
    ip link set "$VETHC" address "$TAP_MAC"
    ip link set "$VETH" up

    # Doi tap MAC thanh dummy
    DUMMY_MAC=$(printf "02:ff:ff:%02x:%02x:%02x" $(( i / 65536 )) $(( (i / 256) % 256 )) $(( i % 256 )))
    ip link set "$TAP" address "$DUMMY_MAC"

    # Gan vao bridge
    ip link set "$TAP" master "$BR"
    ip link set "$VETH" master "$BR"

    ip link set "$TAP" txqueuelen 1000 2>/dev/null || true
    sysctl -w "net.ipv6.conf.${TAP}.disable_ipv6=1" >/dev/null 2>&1

    (( i % 25 == 0 )) || (( i == NUM_IOT )) && \
        echo "   Node 0-$i OK"
done

# --- BUOC 3: Move veth vao Docker containers ---
echo "[3/4] Ket noi Docker containers..."
mkdir -p /var/run/netns

# Gateway (tap0 → veth0c)
PID_GW=$(docker inspect -f '{{.State.Pid}}' rc-iot-testbed_gateway_node_1 2>/dev/null || \
         docker inspect -f '{{.State.Pid}}' rc-iot-testbed-gateway_node-1 2>/dev/null || true)
[ -z "$PID_GW" ] || [ "$PID_GW" = "0" ] && { echo "[ERROR] Gateway container not found!"; exit 1; }

ln -sf "/proc/$PID_GW/ns/net" "/var/run/netns/$PID_GW"
ip link set veth0c netns "$PID_GW"
ip netns exec "$PID_GW" sysctl -w net.ipv6.conf.veth0c.disable_ipv6=1 >/dev/null 2>&1
ip netns exec "$PID_GW" ip addr add "${GW_IP}/16" dev veth0c 2>/dev/null || true
ip netns exec "$PID_GW" ip link set veth0c up
echo "   Gateway: $GW_IP"

# IoT Nodes (10.1.1.1 → 10.1.1.200)
FAIL=0
for i in $(seq 1 $NUM_IOT); do
    CONTAINER="rc-iot-testbed_iot_node_${i}_1"
    PID=$(docker inspect -f '{{.State.Pid}}' "$CONTAINER" 2>/dev/null || true)
    [ -z "$PID" ] || [ "$PID" = "0" ] && {
        CONTAINER="rc-iot-testbed-iot_node_${i}-1"
        PID=$(docker inspect -f '{{.State.Pid}}' "$CONTAINER" 2>/dev/null || true)
    }
    [ -z "$PID" ] || [ "$PID" = "0" ] && { FAIL=$((FAIL+1)); continue; }

    ln -sf "/proc/$PID/ns/net" "/var/run/netns/$PID"
    ip link set "veth${i}c" netns "$PID"
    ip netns exec "$PID" sysctl -w "net.ipv6.conf.veth${i}c.disable_ipv6=1" >/dev/null 2>&1

    # IP: 10.1.1.1 → 10.1.1.200
    OCTET3=$(( (i - 1) / 254 + 1 ))
    OCTET4=$(( (i - 1) % 254 + 1 ))
    NODE_IP="${SUBNET}.${OCTET3}.${OCTET4}"

    ip netns exec "$PID" ip addr add "${NODE_IP}/16" dev "veth${i}c" 2>/dev/null || true
    ip netns exec "$PID" ip link set "veth${i}c" up

    (( i % 25 == 0 )) || (( i == NUM_IOT )) && echo "   IoT 1-$i OK"
    sleep 0.02
done

# --- BUOC 4: Ket qua ---
echo "[4/4] Hoan tat!"
echo ""
echo "========================================================="
echo "  $((NUM_IOT - FAIL))/$NUM_IOT IoT + Gateway da ket noi"
echo "  Gateway: $GW_IP | IoT: ${SUBNET}.1.1-${SUBNET}.1.$NUM_IOT"
echo ""
echo "  Test: PID_1=\$(docker inspect -f '{{.State.Pid}}' rc-iot-testbed_iot_node_1_1)"
echo "        sudo ip netns exec \$PID_1 ping -c 3 $GW_IP"
echo "========================================================="
