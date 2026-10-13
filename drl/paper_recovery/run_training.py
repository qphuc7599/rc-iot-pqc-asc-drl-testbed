"""Command-line entry point for registered development and formal PPO runs."""

from __future__ import annotations

import argparse
import json

from .trainer import run_registered_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("smoke", "profiler", "formal"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument(
        "--formal",
        action="store_true",
        help="required acknowledgement for a pre-authorized formal run",
    )
    args = parser.parse_args()
    result = run_registered_training(
        run_kind=args.kind,
        seed=args.seed,
        device=args.device,
        explicit_formal_flag=args.formal,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

