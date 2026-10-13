"""CLI for speed-v5 benchmark and conditionally authorized formal run."""

from __future__ import annotations

import argparse
import json

from drl.paper_recovery_security_speed_v5.trainer import (
    configure_speed_runtime,
    run_registered,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--formal", action="store_true")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    runtime = configure_speed_runtime()
    receipt = run_registered(
        run_kind="formal_3000" if args.formal else "benchmark_20",
        seed=int(args.seed),
        device=args.device,
    )
    print(json.dumps({"runtime": runtime, "receipt": receipt}, indent=2))


if __name__ == "__main__":
    main()
