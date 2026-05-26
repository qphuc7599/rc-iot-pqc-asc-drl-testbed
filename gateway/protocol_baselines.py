#!/usr/bin/env python3
"""Protocol-faithful baseline emulation for the RC-IoT gateway.

The gateway still receives real UDP traffic through NS-3.  This module models
the validator-side ordering/finality protocol that would run after those
transactions reach the relay/state-channel layer.  It keeps explicit state,
rounds, quorum thresholds, message counts, and latency split into:

* service_delay_s: blocking work that limits sustained throughput.
* finality_delay_s: protocol finality latency reported in settlement metrics.

Pipelined protocols such as Bullshark can have finality latency larger than
their per-batch service delay; modelling them as stop-and-wait would be less
faithful than this split.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class ProtocolCommitResult:
    mode: str
    sequence: int
    leader: int
    service_delay_s: float
    finality_delay_s: float
    messages_total: int
    bytes_total: int
    phases: List[Dict[str, object]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class ProtocolBaselineEngine:
    """Stateful emulation of baseline protocol commit paths."""

    MODES = {"none", "legacy_pbft", "pbft", "simplex", "bullshark", "hydra"}

    def __init__(
        self,
        mode: str = "none",
        n: int = 21,
        f: Optional[int] = None,
        signature_bytes: int = 72,
        tx_bytes: int = 256,
        rtt_ms: float = 4.0,
        bandwidth_mbps: float = 1000.0,
        verify_delay_us: float = 69.0,
        fixed_delay_s: float = 0.0,
        bullshark_round_ms: float = 100.0,
    ):
        mode = (mode or "none").lower()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported protocol mode: {mode}")
        if n <= 0:
            raise ValueError("Protocol participant count must be positive")

        self.mode = mode
        self.n = int(n)
        self.f = int(f) if f is not None and f >= 0 else (self.n - 1) // 3
        self.quorum = min(self.n, 2 * self.f + 1)
        self.signature_bytes = max(0, int(signature_bytes))
        self.tx_bytes = max(1, int(tx_bytes))
        self.rtt_s = max(0.0, float(rtt_ms) / 1000.0)
        self.bandwidth_bps = max(1.0, float(bandwidth_mbps) * 1_000_000.0)
        self.verify_delay_s = max(0.0, float(verify_delay_us) / 1_000_000.0)
        self.fixed_delay_s = max(0.0, float(fixed_delay_s))
        self.bullshark_round_s = max(self.rtt_s, float(bullshark_round_ms) / 1000.0)

        self.sequence = 0
        self.view = 0
        self.dag_round = 0
        self.snapshot = 0
        self.total_messages = 0
        self.total_bytes = 0
        self.total_service_delay_s = 0.0
        self.total_finality_delay_s = 0.0
        self.last_result: Optional[ProtocolCommitResult] = None

    def _serialization_s(self, byte_count: int) -> float:
        return max(0, int(byte_count)) * 8.0 / self.bandwidth_bps

    def _vote_verify_s(self, votes: int) -> float:
        return max(0, int(votes)) * self.verify_delay_s

    def _digest_vote_bytes(self) -> int:
        return 32 + self.signature_bytes

    def _qc_bytes(self) -> int:
        return 32 + self.quorum * self.signature_bytes

    def _block_bytes(self, verified_txs: int, updates: Iterable[object]) -> int:
        packet_bytes = sum(getattr(u, "size", 0) for u in updates)
        logical_payload = int(verified_txs) * self.tx_bytes
        # Use the larger of the logical block and observed packets so that
        # emulated signature-size modes remain honest about their wire payload.
        return max(logical_payload, packet_bytes) + 64

    def commit_batch(
        self,
        batch_id: int,
        updates: Iterable[object],
        verified_updates: int,
        verified_txs: int,
        proof: str = "",
    ) -> ProtocolCommitResult:
        updates = list(updates)
        if self.mode == "none" or verified_txs <= 0:
            result = ProtocolCommitResult(
                mode=self.mode,
                sequence=self.sequence,
                leader=0,
                service_delay_s=0.0,
                finality_delay_s=0.0,
                messages_total=0,
                bytes_total=0,
                notes="no validator-side finality protocol",
            )
        elif self.mode == "legacy_pbft":
            result = self._legacy_pbft(batch_id, updates, verified_txs)
        elif self.mode == "pbft":
            result = self._pbft(batch_id, updates, verified_txs)
        elif self.mode == "simplex":
            result = self._simplex(batch_id, updates, verified_txs)
        elif self.mode == "bullshark":
            result = self._bullshark(batch_id, updates, verified_txs)
        elif self.mode == "hydra":
            result = self._hydra(batch_id, updates, verified_txs)
        else:
            raise AssertionError(f"unhandled protocol mode {self.mode}")

        self.sequence += 1
        self.total_messages += result.messages_total
        self.total_bytes += result.bytes_total
        self.total_service_delay_s += result.service_delay_s
        self.total_finality_delay_s += result.finality_delay_s
        self.last_result = result
        return result

    def _legacy_pbft(self, batch_id: int, updates: List[object], verified_txs: int) -> ProtocolCommitResult:
        block_bytes = self._block_bytes(verified_txs, updates)
        messages = (self.n - 1) + 2 * self.n * max(0, self.n - 1)
        bytes_total = (
            (self.n - 1) * (block_bytes + self.signature_bytes)
            + 2 * self.n * max(0, self.n - 1) * self._digest_vote_bytes()
        )
        delay = self.fixed_delay_s
        phases = [
            {"name": "PRE-PREPARE", "messages": self.n - 1, "quorum": 1},
            {"name": "PREPARE", "messages": self.n * max(0, self.n - 1), "quorum": self.quorum},
            {"name": "COMMIT", "messages": self.n * max(0, self.n - 1), "quorum": self.quorum},
        ]
        return ProtocolCommitResult(
            mode="legacy_pbft",
            sequence=self.sequence,
            leader=(self.view % self.n) + 1,
            service_delay_s=delay,
            finality_delay_s=delay,
            messages_total=messages,
            bytes_total=bytes_total,
            phases=phases,
            notes="calibrated PBFT delay with explicit 3-phase accounting",
        )

    def _pbft(self, batch_id: int, updates: List[object], verified_txs: int) -> ProtocolCommitResult:
        block_bytes = self._block_bytes(verified_txs, updates)
        preprepare = self.n - 1
        prepare = self.n * max(0, self.n - 1)
        commit = self.n * max(0, self.n - 1)
        bytes_total = (
            preprepare * (block_bytes + self.signature_bytes)
            + (prepare + commit) * self._digest_vote_bytes()
        )
        finality = 3 * self.rtt_s + self._serialization_s(bytes_total) + self._vote_verify_s(2 * self.quorum)
        phases = [
            {"name": "PRE-PREPARE", "messages": preprepare, "quorum": 1},
            {"name": "PREPARE", "messages": prepare, "quorum": self.quorum},
            {"name": "COMMIT", "messages": commit, "quorum": self.quorum},
        ]
        return ProtocolCommitResult(
            mode="pbft",
            sequence=self.sequence,
            leader=(self.view % self.n) + 1,
            service_delay_s=finality,
            finality_delay_s=finality,
            messages_total=preprepare + prepare + commit,
            bytes_total=bytes_total,
            phases=phases,
            notes="full PBFT stop-and-wait commit",
        )

    def _simplex(self, batch_id: int, updates: List[object], verified_txs: int) -> ProtocolCommitResult:
        block_bytes = self._block_bytes(verified_txs, updates)
        leader = (self.view % self.n) + 1
        propose_msgs = self.n - 1
        vote_msgs = self.n - 1
        qc_msgs = self.n - 1
        bytes_total = (
            propose_msgs * (block_bytes + self._qc_bytes())
            + vote_msgs * self._digest_vote_bytes()
            + qc_msgs * self._qc_bytes()
        )
        qc_work = self._vote_verify_s(self.quorum)
        finality = 2 * self.rtt_s + self._serialization_s(bytes_total) + qc_work
        # The next leader can pipeline proposal construction while the previous
        # QC is disseminated, so the blocking service component is mostly QC
        # construction and serialization rather than the full finality delay.
        service = self._serialization_s(bytes_total) + qc_work
        phases = [
            {"name": "PROPOSE", "leader": leader, "messages": propose_msgs, "quorum": 1},
            {"name": "VOTE-QC", "messages": vote_msgs, "quorum": self.quorum},
            {"name": "QC-BROADCAST", "messages": qc_msgs, "quorum": self.quorum},
        ]
        self.view += 1
        return ProtocolCommitResult(
            mode="simplex",
            sequence=self.sequence,
            leader=leader,
            service_delay_s=service,
            finality_delay_s=finality,
            messages_total=propose_msgs + vote_msgs + qc_msgs,
            bytes_total=bytes_total,
            phases=phases,
            notes="rotating-leader BFT with proposal, quorum vote, and QC broadcast",
        )

    def _bullshark(self, batch_id: int, updates: List[object], verified_txs: int) -> ProtocolCommitResult:
        block_bytes = self._block_bytes(verified_txs, updates)
        rounds = 3
        start_round = self.dag_round + 1
        leader = (start_round % self.n) + 1
        vertex_meta = 96 + self.quorum * 32 + self.signature_bytes
        per_round_msgs = self.n * max(0, self.n - 1)
        messages = rounds * per_round_msgs
        bytes_total = messages * vertex_meta + (self.n - 1) * block_bytes
        # Bullshark/Narwhal-style DAG dissemination is pipelined: finality
        # waits for several DAG rounds, but the service bottleneck for a new
        # batch is one payload dissemination plus certificate checks.
        payload_service = self._serialization_s((self.n - 1) * block_bytes)
        metadata_service = self._serialization_s(per_round_msgs * vertex_meta) / max(1, self.n)
        qc_work = self._vote_verify_s(self.quorum) / max(1, self.n)
        service = payload_service + metadata_service + qc_work
        finality = rounds * self.bullshark_round_s + payload_service + self._vote_verify_s(self.quorum)
        phases = []
        for offset in range(rounds):
            r = start_round + offset
            phases.append(
                {
                    "name": "DAG-ROUND",
                    "round": r,
                    "messages": per_round_msgs,
                    "refs_per_vertex": self.quorum,
                    "leader": ((r % self.n) + 1),
                }
            )
        phases.append({"name": "COMMIT-LEADER", "round": start_round + 2, "leader": leader, "quorum": self.quorum})
        self.dag_round += rounds
        return ProtocolCommitResult(
            mode="bullshark",
            sequence=self.sequence,
            leader=leader,
            service_delay_s=service,
            finality_delay_s=finality,
            messages_total=messages,
            bytes_total=bytes_total,
            phases=phases,
            notes="pipelined DAG-BFT: vertices reference a quorum of the previous round; leader commits after support two rounds later",
        )

    def _hydra(self, batch_id: int, updates: List[object], verified_txs: int) -> ProtocolCommitResult:
        block_bytes = self._block_bytes(verified_txs, updates)
        leader = (self.snapshot % self.n) + 1
        propose = self.n - 1
        ack = self.n
        confirm = self.n - 1
        snapshot_cert = self._qc_bytes()
        bytes_total = (
            propose * (block_bytes + 64)
            + ack * self._digest_vote_bytes()
            + confirm * snapshot_cert
        )
        finality = self.rtt_s + self._serialization_s(bytes_total) + self._vote_verify_s(self.n)
        phases = [
            {"name": "COLLECT-TX", "snapshot": self.snapshot + 1, "messages": 0},
            {"name": "PROPOSE-SNAPSHOT", "leader": leader, "messages": propose, "quorum": 1},
            {"name": "ACK-SNAPSHOT", "messages": ack, "quorum": self.n},
            {"name": "CONFIRM-SNAPSHOT", "messages": confirm, "quorum": self.n},
        ]
        self.snapshot += 1
        return ProtocolCommitResult(
            mode="hydra",
            sequence=self.sequence,
            leader=leader,
            service_delay_s=finality,
            finality_delay_s=finality,
            messages_total=propose + ack + confirm,
            bytes_total=bytes_total,
            phases=phases,
            notes="Hydra-head style off-chain snapshot: open head assumed, each snapshot is acknowledged by all head members",
        )

    def summary(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "participants": self.n,
            "fault_tolerance": self.f,
            "quorum": self.quorum,
            "commits": self.sequence,
            "total_messages": self.total_messages,
            "total_bytes": self.total_bytes,
            "total_service_delay_s": self.total_service_delay_s,
            "total_finality_delay_s": self.total_finality_delay_s,
            "rtt_ms": self.rtt_s * 1000.0,
            "bandwidth_mbps": self.bandwidth_bps / 1_000_000.0,
            "signature_bytes": self.signature_bytes,
            "last_commit": self.last_result.to_dict() if self.last_result else None,
        }
