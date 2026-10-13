#!/usr/bin/env python3
"""Reproduce the analytical scaling tables used by the Q-Relay supplement.

This script is deliberately independent of the simulator.  It writes the
collision-free occupancy, committee-size, and compact settlement-cost tables
from the equations and constants stated in the manuscript.  The two measured
air-time endpoints are the rounded analytical values already reported in the
supplement; all sweeps below are deterministic transformations of them.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


PACKETS = {
    "ML-DSA-44": {
        "bytes": 2508,
        "frames": 2,
        "payload_ms": 3.344,
        "short_ms": 3.875,
        "long_ms": 4.084,
    },
    "ECDSA": {
        "bytes": 160,
        "frames": 1,
        "payload_ms": 0.213,
        "short_ms": 0.485,
        "long_ms": 0.590,
    },
}

NODE_COUNTS = (25, 50, 100, 150, 200)
BATCH_SIZES = (10, 50, 100)
COMMITTEE_SIZES = (4, 7, 10, 13, 16, 21, 31)


def quorum(n: int) -> tuple[int, int]:
    f = (n - 1) // 3
    q = math.ceil((n + f + 1) / 2)
    return f, q


def occupancy(n: int, records_per_second: float, airtime_ms: float, batch: int) -> float:
    return n * records_per_second * (airtime_ms / 1000.0) / batch


def saturation_nodes(records_per_second: float, airtime_ms: float, batch: int) -> float:
    return batch / (records_per_second * airtime_ms / 1000.0)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_airtime_rows() -> list[dict[str, object]]:
    return [
        {
            "packet": name,
            "bytes": spec["bytes"],
            "frames": spec["frames"],
            "payload_only_ms": spec["payload_ms"],
            "airtime_short_slot_ms": spec["short_ms"],
            "airtime_long_slot_ms": spec["long_ms"],
        }
        for name, spec in PACKETS.items()
    ]


def build_occupancy_rows(records_per_second: float, batch: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, spec in PACKETS.items():
        row: dict[str, object] = {"packet": name}
        for n in NODE_COUNTS:
            row[f"rho_short_N{n}"] = occupancy(n, records_per_second, spec["short_ms"], batch)
            row[f"rho_long_N{n}"] = occupancy(n, records_per_second, spec["long_ms"], batch)
        for b in BATCH_SIZES:
            row[f"Nstar_short_B{b}"] = saturation_nodes(records_per_second, spec["short_ms"], b)
            row[f"Nstar_long_B{b}"] = saturation_nodes(records_per_second, spec["long_ms"], b)
        rows.append(row)
    return rows


def build_committee_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n in COMMITTEE_SIZES:
        f, q = quorum(n)
        fallback_bytes = q * (2420 + 8) + 2
        tps_ref = 750_000 * 50 / (2420 * n + 50 * 256)
        rows.append(
            {
                "n": n,
                "f": f,
                "q_star": q,
                "q_fraction": q / n,
                "liveness_slack": n - f - q,
                "fallback_bytes": fallback_bytes,
                "threshold_certificate_bytes": 2420,
                "single_committee_tps_ref": tps_ref,
            }
        )
    return rows


def build_settlement_rows(records_per_state: int) -> list[dict[str, object]]:
    compact_fields = 32 + 8 + 32
    certificates = {
        "threshold_sigma": 2420,
        "fallback_14_signatures": 14 * (2420 + 8) + 2,
        "ecdsa_reference": 65,
    }
    rows: list[dict[str, object]] = []
    for name, certificate_bytes in certificates.items():
        calldata_bytes = compact_fields + certificate_bytes
        rows.append(
            {
                "certificate": name,
                "compact_calldata_bytes": calldata_bytes,
                "gas_at_16_per_byte": calldata_bytes * 16,
                "gas_at_40_per_byte": calldata_bytes * 40,
                "bytes_per_record": calldata_bytes / records_per_state,
            }
        )
    return rows


def validate_reported_values() -> None:
    committee = {row["n"]: row for row in build_committee_rows()}
    settlement = {row["certificate"]: row for row in build_settlement_rows(301)}
    assert committee[21]["f"] == 6
    assert committee[21]["q_star"] == 14
    assert committee[21]["fallback_bytes"] == 33_994
    assert round(float(committee[21]["single_committee_tps_ref"])) == 589
    assert settlement["threshold_sigma"]["compact_calldata_bytes"] == 2_492
    assert settlement["fallback_14_signatures"]["compact_calldata_bytes"] == 34_066
    assert settlement["ecdsa_reference"]["compact_calldata_bytes"] == 137


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision_R1_results/04_optional/overhead_scaling"),
    )
    parser.add_argument("--records-per-second", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    validate_reported_values()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    airtime = build_airtime_rows()
    write_csv(out / "airtime.csv", list(airtime[0]), airtime)

    occupancy_rows = build_occupancy_rows(args.records_per_second, args.batch_size)
    write_csv(out / "occupancy.csv", list(occupancy_rows[0]), occupancy_rows)

    committee_rows = build_committee_rows()
    write_csv(out / "committee_scaling.csv", list(committee_rows[0]), committee_rows)

    settlement_rows = build_settlement_rows(301)
    write_csv(out / "settlement_cost.csv", list(settlement_rows[0]), settlement_rows)

    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
