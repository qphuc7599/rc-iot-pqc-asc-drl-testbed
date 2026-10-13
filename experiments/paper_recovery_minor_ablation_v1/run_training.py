"""Registered CLI for the minor-revision PPO ablation campaign."""

from __future__ import annotations

import argparse
import json

from drl.paper_recovery_minor_ablation_v1.policy import ARMS
from drl.paper_recovery_minor_ablation_v1.trainer import run_registered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-kind", required=True, choices=("pilot_50", "formal_3000"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    receipt = run_registered(
        arm=args.arm,
        run_kind=args.run_kind,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
