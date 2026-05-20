#!/usr/bin/env python3
"""
generate.py — Tao docker-compose.yml cho RC-IoT Testbed

100 IoT containers heterogeneous, resource-constrained.
Host: 8GB RAM, 12 CPU cores (WSL2)

Strategies:
  - RAM: 32-64MB per container (Ubuntu needs ~20MB min, + PQC binary headroom)
  - CPU: use cpuset + cpu_quota to throttle. Higher quotas for boot stability,
         the actual IoT slowness is measured by PQC benchmark latency.
  - Total: 100 × 40MB avg = 4GB RAM (fits in 8GB WSL2)
"""

import json

NUM_NODES = 100

# Device profiles matching real IoT hardware
# ram_mb: container memory limit
# cpus: Docker --cpus limit (fraction of 1 core)
# Real chip specs in comments for paper reference
DEVICE_PROFILES = {
    "ESP32":      {"ram_mb": 32,  "cpus": 0.05, "count_pct": 0.25,
                   "spec": "240MHz Xtensa LX6, 520KB SRAM, WiFi/BLE, 160mW"},
    "ESP32-S3":   {"ram_mb": 32,  "cpus": 0.06, "count_pct": 0.10,
                   "spec": "240MHz Xtensa LX7, 512KB SRAM, WiFi/BLE+AI, 170mW"},
    "STM32L4-M4": {"ram_mb": 32,  "cpus": 0.03, "count_pct": 0.15,
                   "spec": "80MHz Cortex-M4, 256KB SRAM, ultra-low-power, 30mW"},
    "STM32F4-M4": {"ram_mb": 40,  "cpus": 0.08, "count_pct": 0.15,
                   "spec": "168MHz Cortex-M4+FPU, 192KB SRAM, 50mW"},
    "STM32H7-M7": {"ram_mb": 64,  "cpus": 0.15, "count_pct": 0.10,
                   "spec": "480MHz Cortex-M7, 1MB SRAM, L1 cache, 90mW"},
    "nRF52840":   {"ram_mb": 32,  "cpus": 0.02, "count_pct": 0.10,
                   "spec": "64MHz Cortex-M4, 256KB RAM, BLE/802.15.4, 23mW"},
    "RP2040":     {"ram_mb": 32,  "cpus": 0.04, "count_pct": 0.15,
                   "spec": "133MHz dual Cortex-M0+, 264KB SRAM, 45mW"},
}

# Build node assignment list
node_assignments = []
for dev_type, profile in DEVICE_PROFILES.items():
    count = max(1, int(NUM_NODES * profile["count_pct"]))
    for _ in range(count):
        node_assignments.append((dev_type, profile))
while len(node_assignments) < NUM_NODES:
    node_assignments.append(("ESP32", DEVICE_PROFILES["ESP32"]))
node_assignments = node_assignments[:NUM_NODES]

# Generate docker-compose.yml
with open('docker-compose.yml', 'w') as f:
    f.write("version: '3.8'\n")
    f.write("services:\n")

    # 1. Gateway node
    f.write("  gateway_node:\n")
    f.write("    image: ubuntu:22.04\n")
    f.write("    command: tail -f /dev/null\n")
    f.write("    volumes:\n")
    f.write("      - ./pqc/bin:/opt/pqc:ro\n")
    f.write("      - ./results:/opt/results\n")
    f.write("      - ./gateway:/opt/gateway:ro\n")
    f.write("    deploy:\n")
    f.write("      resources:\n")
    f.write("        limits:\n")
    f.write("          cpus: '0.5'\n")
    f.write("          memory: 256M\n\n")

    # 2. IoT nodes
    for i in range(NUM_NODES):
        dev_type, profile = node_assignments[i]
        node_id = i + 1
        cpus_str = f"{profile['cpus']:.3f}"
        ram_str = f"{profile['ram_mb']}M"
        f.write(f"  iot_node_{node_id}:\n")
        f.write(f"    image: ubuntu:22.04\n")
        f.write(f"    command: tail -f /dev/null\n")
        f.write(f"    labels:\n")
        f.write(f"      - \"iot.device_type={dev_type}\"\n")
        f.write(f"      - \"iot.node_id={node_id}\"\n")
        f.write("    volumes:\n")
        f.write("      - ./pqc/bin:/opt/pqc:ro\n")
        f.write("      - ./results:/opt/results\n")
        f.write("    deploy:\n")
        f.write("      resources:\n")
        f.write("        limits:\n")
        f.write(f"          cpus: '{cpus_str}'\n")
        f.write(f"          memory: {ram_str}\n")

# Save node map
node_map = {}
for i, (dev_type, profile) in enumerate(node_assignments):
    node_map[i + 1] = {
        "device_type": dev_type,
        "ram_mb": profile["ram_mb"],
        "cpus": profile["cpus"],
        "spec": profile["spec"],
    }
with open('results/node_map.json', 'w') as f:
    json.dump(node_map, f, indent=2)

# Summary
total_ram = sum(p["ram_mb"] for _, p in node_assignments)
total_cpu = sum(p["cpus"] for _, p in node_assignments)
print(f"Generated docker-compose.yml: 1 Gateway + {NUM_NODES} IoT nodes")
print(f"Total RAM: {total_ram}MB ({total_ram/1024:.1f}GB) | Total CPU: {total_cpu:.2f} cores")
print(f"\nDevice distribution:")
type_counts = {}
for dev_type, _ in node_assignments:
    type_counts[dev_type] = type_counts.get(dev_type, 0) + 1
for dev_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
    p = DEVICE_PROFILES[dev_type]
    print(f"  {dev_type:15s}: {count:3d} nodes | RAM={p['ram_mb']:3d}MB | CPU={p['cpus']:.3f}")
print(f"\nNode map saved to results/node_map.json")
