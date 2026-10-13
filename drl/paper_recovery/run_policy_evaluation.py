"""CLI for final PPO or update-0 control evaluation after formal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import run_policy_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument(
        "--profile",
        default="canonical",
        choices=(
            "canonical",
            "ood_capability_kill",
            "ood_clustered_rf",
            "ood_bursty_overload",
            "ood_adaptive_byzantine",
            "ood_nakagami",
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--update-zero-control", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    panel = run_policy_panel(
        training_seed=args.training_seed,
        profile=args.profile,
        checkpoint_path=args.checkpoint,
        update_zero_control=args.update_zero_control,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": panel["status"],
                "policy": panel["policy"],
                "profile": panel["profile"],
                "training_seed": panel["training_seed"],
                "panel_hash": panel["panel_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

