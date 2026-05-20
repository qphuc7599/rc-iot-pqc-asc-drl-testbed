#!/usr/bin/env python3
"""
sig-aggregator.py — Gateway: State Channel Aggregator

Chay trong Gateway container (512MB RAM, co Python).
Nhan State Channel update packets tu IoT nodes.
Moi packet chua tx_count giao dich off-chain + 1 ML-DSA signature tren state_hash.
Gateway verify 1 signature → cong nhan tx_count TXs → O(1) verification.

Packet format (State Channel, v2):
  [4B node_id] [8B first_tx_timestamp] [8B last_tx_timestamp]
  [8B timestamp] [4B tx_count] [32B state_hash] [4B sig_len] [sig]

Usage:
  python3 /opt/gateway/sig-aggregator.py --port 9000 --batch-size 50 --duration 120

Output:
  - stderr: realtime logs
  - stdout: CSV (batch_id, batch_size, verify_time_us, aggregate_time_us, tps)
"""

import socket
import struct
import time
import sys
import argparse
import hashlib
import json
import os
import math

# ML-DSA signature sizes (from liboqs)
SIG_SIZES = {
    "ML-DSA-44": 2420,
    "ML-DSA-65": 3309,
    "ML-DSA-87": 4627,
}
PK_SIZES = {
    "ML-DSA-44": 1312,
    "ML-DSA-65": 1952,
    "ML-DSA-87": 2592,
}

class StateChannelUpdate:
    """
    Parsed State Channel update from IoT node.
    Each packet represents tx_count off-chain transactions,
    verified by 1 ML-DSA signature on the state_hash root.
    """
    def __init__(self, raw_data):
        # v2 adds first/last logical-TX timestamps so the gateway can report
        # transaction-level latency directly from the simulation trace. Keep a
        # v1 parser for old tx-generator binaries in existing artifacts.
        old_hdr_size = 4 + 8 + 4 + 32 + 4  # = 52 bytes
        new_hdr_size = 4 + 8 + 8 + 8 + 4 + 32 + 4  # = 68 bytes
        if len(raw_data) < old_hdr_size:
            raise ValueError(f"Packet too small: {len(raw_data)} < {old_hdr_size}")

        parsed_new = False
        if len(raw_data) >= new_hdr_size:
            sig_len_new = struct.unpack('!I', raw_data[64:68])[0]
            expected_new = new_hdr_size + sig_len_new
            if sig_len_new > 0 and len(raw_data) == expected_new:
                first_us = struct.unpack('!Q', raw_data[4:12])[0]
                last_us = struct.unpack('!Q', raw_data[12:20])[0]
                timestamp_us = struct.unpack('!Q', raw_data[20:28])[0]
                if 0 < first_us <= last_us <= timestamp_us:
                    self.node_id = struct.unpack('!I', raw_data[0:4])[0]
                    self.first_tx_timestamp_us = first_us
                    self.last_tx_timestamp_us = last_us
                    self.timestamp_us = timestamp_us
                    self.tx_count = struct.unpack('!I', raw_data[28:32])[0]
                    self.state_hash = raw_data[32:64]
                    self.sig_len = sig_len_new
                    self.signature = raw_data[68:68 + self.sig_len]
                    self.has_tx_timestamp_range = True
                    hdr_size = new_hdr_size
                    parsed_new = True

        if not parsed_new:
            self.node_id = struct.unpack('!I', raw_data[0:4])[0]
            self.timestamp_us = struct.unpack('!Q', raw_data[4:12])[0]
            self.first_tx_timestamp_us = self.timestamp_us
            self.last_tx_timestamp_us = self.timestamp_us
            self.tx_count = struct.unpack('!I', raw_data[12:16])[0]  # off-chain TX count
            self.state_hash = raw_data[16:48]                         # SHA-256 state root
            self.sig_len = struct.unpack('!I', raw_data[48:52])[0]
            self.signature = raw_data[52:52 + self.sig_len]
            self.has_tx_timestamp_range = False
            hdr_size = old_hdr_size

        expected_size = hdr_size + self.sig_len
        if self.sig_len == 0:
            raise ValueError("Empty signature")
        if len(raw_data) != expected_size:
            raise ValueError(
                f"Malformed packet length: got {len(raw_data)}, expected {expected_size}"
            )
        if self.tx_count == 0:
            raise ValueError("Empty channel update")
        self.recv_time = time.monotonic()
        sent_time = self.timestamp_us / 1_000_000.0
        latency_s = self.recv_time - sent_time
        # Docker containers and the host share CLOCK_MONOTONIC in the normal
        # NS-3/tap setup. If a platform uses incompatible clock domains, avoid
        # reporting impossible latency values instead of polluting the summary.
        self.rx_latency_ms = latency_s * 1000.0 if -1.0 <= latency_s <= 3600.0 else None
        self.size = len(raw_data)

    def __repr__(self):
        return f"SC(node={self.node_id}, tx_count={self.tx_count}, sig_len={self.sig_len})"


class SignatureAggregator:
    """
    State Channel Aggregator — O(1) verification.

    Each incoming packet is a State Channel update containing:
    - tx_count: number of off-chain transactions in the batch
    - state_hash: SHA-256 rolling hash of all TX data
    - signature: 1 ML-DSA signature on state_hash

    Gateway verifies 1 signature → accepts tx_count TXs.
    This is the core of the O(1) State Channel architecture.
    """

    def __init__(self, batch_size=50, verify_delay_us=0.0, expected_sig_len=0):
        self.batch_size = batch_size  # channel updates per on-chain batch
        self.verify_delay_us = verify_delay_us
        self.expected_sig_len = expected_sig_len
        self.pending = []
        self.batches = []
        self.received_txs = 0
        self.accepted_txs = 0
        self.accepted_updates = 0
        self.total_failed = 0
        self.total_updates = 0       # total state channel updates received
        self.observed_sig_lens = {}
        self.failed_sig_lens = {}
        self.rx_latency_ms = []
        self.gateway_queue_ms = []
        self.settlement_latency_ms = []
        self.transaction_e2e_latency_ms = []

    @staticmethod
    def _percentiles(values):
        clean = sorted(v for v in values if v is not None and math.isfinite(v))
        if not clean:
            return {
                "count": 0,
                "mean": None,
                "p50": None,
                "p95": None,
                "p99": None,
                "max": None,
            }

        def pct(q):
            if len(clean) == 1:
                return clean[0]
            pos = (len(clean) - 1) * q
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return clean[lo]
            return clean[lo] + (clean[hi] - clean[lo]) * (pos - lo)

        return {
            "count": len(clean),
            "mean": sum(clean) / len(clean),
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": clean[-1],
        }

    def add_channel_update(self, update):
        """Queue a State Channel update; count it as accepted only after verify."""
        self.pending.append(update)
        self.received_txs += update.tx_count
        self.total_updates += 1
        self.observed_sig_lens[update.sig_len] = self.observed_sig_lens.get(update.sig_len, 0) + 1

    def should_aggregate(self):
        return len(self.pending) >= self.batch_size

    def aggregate_batch(self):
        """
        Aggregate State Channel updates into on-chain batch proof.
        Each update already represents N off-chain TXs.
        Gateway verifies 1 ML-DSA signature per update → O(1) per channel.
        """
        if not self.pending:
            return None

        batch = self.pending[:self.batch_size]
        self.pending = self.pending[self.batch_size:]

        t0 = time.monotonic()
        batch_rx_latency_ms = []
        batch_queue_ms = []

        # 1. Verify each State Channel update (1 signature per update = O(1))
        #    Each update represents tx_count off-chain TXs
        verified_updates = 0
        verified_txs = 0
        failed = 0
        accepted_updates = []
        for update in batch:
            # Verify one signature on the state_hash root. The Docker/NS-3
            # path validates packet structure and injects calibrated gateway
            # verification time from native benchmark measurements.
            if self.verify_delay_us > 0:
                time.sleep(self.verify_delay_us / 1_000_000.0)
            length_ok = update.sig_len > 0 and len(update.signature) == update.sig_len
            expected_ok = (self.expected_sig_len <= 0 or update.sig_len == self.expected_sig_len)
            if length_ok and expected_ok:
                verified_updates += 1
                verified_txs += update.tx_count  # Accept ALL off-chain TXs
                if update.rx_latency_ms is not None:
                    batch_rx_latency_ms.append(update.rx_latency_ms)
                    self.rx_latency_ms.append(update.rx_latency_ms)
                queue_ms = max(0.0, (t0 - update.recv_time) * 1000.0)
                batch_queue_ms.append(queue_ms)
                self.gateway_queue_ms.append(queue_ms)
                accepted_updates.append(update)
            else:
                failed += 1
                self.failed_sig_lens[update.sig_len] = self.failed_sig_lens.get(update.sig_len, 0) + 1

        verify_time = time.monotonic() - t0

        # 2. Aggregate: hash all state_hashes into 1 on-chain proof
        t1 = time.monotonic()
        hasher = hashlib.sha256()
        for update in batch:
            hasher.update(update.state_hash)
            hasher.update(update.signature)
        aggregated_proof = hasher.hexdigest()
        aggregate_time = time.monotonic() - t1
        commit_time = time.monotonic()
        batch_settlement_ms = []
        batch_tx_e2e_ms = []
        for update in accepted_updates:
            sent_time = update.timestamp_us / 1_000_000.0
            settlement_s = commit_time - sent_time
            if -1.0 <= settlement_s <= 3600.0:
                settlement_ms = settlement_s * 1000.0
                batch_settlement_ms.append(settlement_ms)
                self.settlement_latency_ms.append(settlement_ms)
            if update.has_tx_timestamp_range:
                first_s = update.first_tx_timestamp_us / 1_000_000.0
                last_s = update.last_tx_timestamp_us / 1_000_000.0
                if 0 < update.tx_count and first_s <= last_s <= sent_time:
                    if update.tx_count == 1:
                        tx_times = [first_s]
                    else:
                        step = (last_s - first_s) / (update.tx_count - 1)
                        tx_times = (first_s + step * idx for idx in range(update.tx_count))
                    for tx_time in tx_times:
                        e2e_s = commit_time - tx_time
                        if -1.0 <= e2e_s <= 3600.0:
                            e2e_ms = e2e_s * 1000.0
                            batch_tx_e2e_ms.append(e2e_ms)
                            self.transaction_e2e_latency_ms.append(e2e_ms)

        rx_stats = self._percentiles(batch_rx_latency_ms)
        queue_stats = self._percentiles(batch_queue_ms)
        settlement_stats = self._percentiles(batch_settlement_ms)
        tx_e2e_stats = self._percentiles(batch_tx_e2e_ms)

        self.accepted_updates += verified_updates
        self.accepted_txs += verified_txs
        self.total_failed += failed

        batch_info = {
            "batch_id": len(self.batches),
            "batch_size": len(batch),           # channel updates in this batch
            "verified_updates": verified_updates,
            "verified": verified_txs,            # total off-chain TXs verified
            "failed": failed,
            "verify_time_us": verify_time * 1e6,
            "aggregate_time_us": aggregate_time * 1e6,
            "rx_latency_ms_p50": rx_stats["p50"],
            "rx_latency_ms_p95": rx_stats["p95"],
            "gateway_queue_ms_p50": queue_stats["p50"],
            "gateway_queue_ms_p95": queue_stats["p95"],
            "settlement_latency_ms_p50": settlement_stats["p50"],
            "settlement_latency_ms_p95": settlement_stats["p95"],
            "transaction_e2e_ms_p50": tx_e2e_stats["p50"],
            "transaction_e2e_ms_p95": tx_e2e_stats["p95"],
            "proof": aggregated_proof[:16],
            "nodes": list(set(u.node_id for u in batch)),
            "timestamp": time.time(),
        }

        self.batches.append(batch_info)
        return batch_info


def _fmt_float(value):
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.3f}"


def run_gateway(port, batch_size, duration, output_file, pbft_delay=0.0,
                verify_delay_us=0.0, expected_sig_len=0,
                include_batches_in_summary=False):
    """Main gateway loop"""
    aggregator = SignatureAggregator(
        batch_size=batch_size,
        verify_delay_us=verify_delay_us,
        expected_sig_len=expected_sig_len,
    )

    # Setup UDP listener
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 8MB buffer — handles burst from 100+ nodes without kernel drops
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    sock.bind(('0.0.0.0', port))
    sock.setblocking(False)  # Non-blocking for buffer drain pattern

    print(f"[GW] Listening on UDP port {port}", file=sys.stderr)
    print(f"[GW] Batch size: {batch_size}, Duration: {duration}s", file=sys.stderr)
    if expected_sig_len > 0:
        print(f"[GW] Expected signature length: {expected_sig_len}B", file=sys.stderr)
    if verify_delay_us > 0:
        print(f"[GW] Injected verify latency: {verify_delay_us:.1f}us/update", file=sys.stderr)
    if pbft_delay > 0:
        print(f"[GW] Mode: ON-CHAIN PBFT | block_time={pbft_delay}s", file=sys.stderr)
        print(f"[GW] PBFT 3-phase consensus simulated per block", file=sys.stderr)
    else:
        print(f"[GW] Mode: State Channel O(1) | instant finality", file=sys.stderr)

    # CSV header
    print("batch_id,batch_size,verified,verify_time_us,aggregate_time_us,"
          "rx_latency_ms_p50,rx_latency_ms_p95,gateway_queue_ms_p50,"
          "gateway_queue_ms_p95,settlement_latency_ms_p50,"
          "settlement_latency_ms_p95,transaction_e2e_ms_p50,"
          "transaction_e2e_ms_p95,tps_instant,proof")

    start_time = time.monotonic()
    last_stats_time = start_time
    last_tx_count = 0

    import select

    while time.monotonic() - start_time < duration:
        # Wait for data with select (microsecond precision, unlike sleep)
        ready, _, _ = select.select([sock], [], [], 0.005)  # 5ms max wait

        if ready:
            # === DRAIN BUFFER: read ALL available State Channel updates ===
            while True:
                try:
                    data, addr = sock.recvfrom(8192)
                    update = StateChannelUpdate(data)
                    aggregator.add_channel_update(update)
                except BlockingIOError:
                    break  # buffer empty
                except ValueError as e:
                    print(f"[GW] Bad packet: {e}", file=sys.stderr)
                    continue

        # Aggregate when batch is full
        while aggregator.should_aggregate():
            batch = aggregator.aggregate_batch()
            if batch:
                # Instant TPS (since last batch)
                elapsed = time.monotonic() - start_time
                tps = aggregator.accepted_txs / elapsed if elapsed > 0 else 0

                print(f"{batch['batch_id']},"
                      f"{batch['batch_size']},"
                      f"{batch['verified']},"
                      f"{batch['verify_time_us']:.1f},"
                      f"{batch['aggregate_time_us']:.1f},"
                      f"{_fmt_float(batch['rx_latency_ms_p50'])},"
                      f"{_fmt_float(batch['rx_latency_ms_p95'])},"
                      f"{_fmt_float(batch['gateway_queue_ms_p50'])},"
                      f"{_fmt_float(batch['gateway_queue_ms_p95'])},"
                      f"{_fmt_float(batch['settlement_latency_ms_p50'])},"
                      f"{_fmt_float(batch['settlement_latency_ms_p95'])},"
                      f"{_fmt_float(batch['transaction_e2e_ms_p50'])},"
                      f"{_fmt_float(batch['transaction_e2e_ms_p95'])},"
                      f"{tps:.1f},"
                      f"{batch['proof']}")
                sys.stdout.flush()

                print(f"[GW] Batch {batch['batch_id']}: "
                      f"{batch['verified']} TX from {batch['verified_updates']} channels, "
                      f"TPS={tps:.1f}, proof={batch['proof']}",
                      file=sys.stderr)

                # PBFT consensus simulation: block the gateway for block_time
                # This simulates Pre-prepare → Prepare → Commit phases
                if pbft_delay > 0:
                    time.sleep(pbft_delay)

        # Periodic stats (every 5s)
        now = time.monotonic()
        if now - last_stats_time >= 5.0:
            elapsed = now - start_time
            tps = aggregator.accepted_txs / elapsed if elapsed > 0 else 0
            delta_tx = aggregator.accepted_txs - last_tx_count
            delta_tps = delta_tx / (now - last_stats_time)
            print(f"[GW] t={elapsed:.0f}s | Accepted: {aggregator.accepted_txs} tx | "
                  f"Received: {aggregator.received_txs} tx | "
                  f"Avg TPS: {tps:.1f} | Current TPS: {delta_tps:.1f} | "
                  f"Batches: {len(aggregator.batches)} | "
                  f"Pending: {len(aggregator.pending)}",
                  file=sys.stderr)
            last_stats_time = now
            last_tx_count = aggregator.accepted_txs

    # Commit trailing partial settlement batch so final TPS counts accepted
    # transactions, not merely received packets waiting in memory.
    if aggregator.pending:
        batch = aggregator.aggregate_batch()
        if batch:
            elapsed = time.monotonic() - start_time
            tps = aggregator.accepted_txs / elapsed if elapsed > 0 else 0
            print(f"{batch['batch_id']},"
                  f"{batch['batch_size']},"
                  f"{batch['verified']},"
                  f"{batch['verify_time_us']:.1f},"
                  f"{batch['aggregate_time_us']:.1f},"
                  f"{_fmt_float(batch['rx_latency_ms_p50'])},"
                  f"{_fmt_float(batch['rx_latency_ms_p95'])},"
                  f"{_fmt_float(batch['gateway_queue_ms_p50'])},"
                  f"{_fmt_float(batch['gateway_queue_ms_p95'])},"
                  f"{_fmt_float(batch['settlement_latency_ms_p50'])},"
                  f"{_fmt_float(batch['settlement_latency_ms_p95'])},"
                  f"{_fmt_float(batch['transaction_e2e_ms_p50'])},"
                  f"{_fmt_float(batch['transaction_e2e_ms_p95'])},"
                  f"{tps:.1f},"
                  f"{batch['proof']}")
            sys.stdout.flush()
            if pbft_delay > 0:
                time.sleep(pbft_delay)

    # Final summary
    elapsed = time.monotonic() - start_time
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"[GW] FINAL REPORT", file=sys.stderr)
    print(f"  Duration:       {elapsed:.1f}s", file=sys.stderr)
    print(f"  Channel updates:{aggregator.total_updates}", file=sys.stderr)
    print(f"  Received TX:    {aggregator.received_txs}", file=sys.stderr)
    print(f"  Accepted TX:    {aggregator.accepted_txs}", file=sys.stderr)
    print(f"  Avg TPS:        {aggregator.accepted_txs / elapsed:.1f}", file=sys.stderr)
    print(f"  Batches:        {len(aggregator.batches)}", file=sys.stderr)
    print(f"  Accepted updates:{aggregator.accepted_updates}", file=sys.stderr)
    print(f"  Failed:         {aggregator.total_failed}", file=sys.stderr)
    print(f"  Pending:        {len(aggregator.pending)}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Save summary JSON
    if output_file:
        summary = {
            "port": port,
            "duration_s": elapsed,
            "total_updates": aggregator.total_updates,  # packets received
            "received_tx": aggregator.received_txs,
            "total_tx": aggregator.accepted_txs,
            "avg_tps": aggregator.accepted_txs / elapsed if elapsed > 0 else 0,
            "total_batches": len(aggregator.batches),
            "batch_size": batch_size,
            "verified": aggregator.accepted_txs,
            "accepted_updates": aggregator.accepted_updates,
            "failed": aggregator.total_failed,
            "pbft_delay_s": pbft_delay,
            "verify_delay_us": verify_delay_us,
            "expected_sig_len": expected_sig_len,
            "observed_sig_lens": aggregator.observed_sig_lens,
            "failed_sig_lens": aggregator.failed_sig_lens,
            "rx_latency_ms": aggregator._percentiles(aggregator.rx_latency_ms),
            "gateway_queue_ms": aggregator._percentiles(aggregator.gateway_queue_ms),
            "settlement_latency_ms": aggregator._percentiles(aggregator.settlement_latency_ms),
            "transaction_e2e_latency_ms": aggregator._percentiles(aggregator.transaction_e2e_latency_ms),
        }
        if include_batches_in_summary:
            summary["batches"] = aggregator.batches

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        tmp_file = f"{output_file}.tmp"
        with open(tmp_file, 'w') as f:
            json.dump(summary, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, output_file)
        print(f"[GW] Summary saved to {output_file}", file=sys.stderr)

    sock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gateway Signature Aggregator')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--duration', type=int, default=120)
    parser.add_argument('--output', default='/opt/results/gateway_summary.json')
    parser.add_argument('--pbft-delay', type=float, default=0.0,
                        help='PBFT block time in seconds (0=off-chain, >0=on-chain PBFT)')
    parser.add_argument('--verify-delay-us', type=float, default=0.0,
                        help='Injected signature verification latency per update')
    parser.add_argument('--expected-sig-len', type=int, default=0,
                        help='Reject packets whose signature length differs (0=disabled)')
    parser.add_argument('--include-batches-in-summary', action='store_true',
                        help='Embed per-batch details in JSON summary. CSV output already contains batch details.')
    args = parser.parse_args()

    run_gateway(
        args.port, args.batch_size, args.duration, args.output,
        args.pbft_delay, args.verify_delay_us, args.expected_sig_len,
        args.include_batches_in_summary
    )
