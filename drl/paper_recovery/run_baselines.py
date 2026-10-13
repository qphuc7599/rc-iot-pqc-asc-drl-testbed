"""CLI for the frozen checkpoint-independent baseline panel."""

from __future__ import annotations

import argparse
import json

from .evaluation import run_baseline_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--selector", action="append", dest="selectors")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    args = parser.parse_args()
    panel = run_baseline_panel(
        profile=args.profile,
        selector_names=args.selectors,
        seeds=args.seeds,
    )
    print(
        json.dumps(
            {
                "status": panel["status"],
                "profile": panel["profile"],
                "selectors": panel["selectors"],
                "environment_seeds": panel["environment_seeds"],
                "panel_hash": panel["panel_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

