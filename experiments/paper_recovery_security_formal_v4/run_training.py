"""CLI for the security-constrained v4 smoke and fixed-budget run."""

from __future__ import annotations

import argparse
import json

from drl.paper_recovery_security_v4.trainer import (
    configure_high_throughput_runtime,
    run_registered,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--qualification", action="store_true")
    group.add_argument("--formal", action="store_true")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    runtime = configure_high_throughput_runtime()
    run_kind = (
        "formal_3000"
        if args.formal
        else "qualification_8"
        if args.qualification
        else "smoke_2"
    )
    result = run_registered(
        run_kind=run_kind,
        seed=int(args.seed),
        device=args.device,
    )
    print(json.dumps({"runtime": runtime, "receipt": result}, indent=2))


if __name__ == "__main__":
    main()
