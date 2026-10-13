"""Fixed 9M-transition direct-PPO runner for compound IoT collapse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from drl.paper_recovery.config import PaperRecoveryConfig
from drl.paper_recovery.receipts import canonical_hash, sha256_file
from drl.paper_recovery_original_ppo.trainer import ReviewerAlignedPPOTrainer

from .config import multihazard_config
from .env import CompoundCollapseEnv


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "experiments/paper_recovery_multihazard_v1/protocol.json"
RESULT_ROOT = REPO_ROOT / "results/paper_recovery_multihazard_v1"


def multihazard_source_manifest(config: PaperRecoveryConfig) -> dict[str, Any]:
    paths = (
        REPO_ROOT / "drl/paper_recovery_original_ppo/policy.py",
        REPO_ROOT / "drl/paper_recovery_original_ppo/trainer.py",
        REPO_ROOT / "drl/paper_recovery/config.py",
        REPO_ROOT / "drl/paper_recovery/env.py",
        REPO_ROOT / "drl/paper_recovery/trainer.py",
        REPO_ROOT / "drl/paper_recovery_multihazard/config.py",
        REPO_ROOT / "drl/paper_recovery_multihazard/env.py",
        REPO_ROOT / "drl/paper_recovery_multihazard/trainer.py",
        PROTOCOL_PATH,
    )
    files = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in paths
    }
    return {
        "schema": "qrelay-direct-ppo-multihazard-source-1",
        "files": files,
        "source_hash": canonical_hash(files),
        "config_hash": canonical_hash(config.to_dict()),
    }


def run_multihazard(*, run_kind: str, seed: int, device: str = "auto") -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    budgets = {"profile": 1, "smoke": 2, "full_budget_3000": 3_000}
    expected_seed = {
        "profile": protocol["seeds"]["profile"],
        "smoke": protocol["seeds"]["smoke"],
        "full_budget_3000": protocol["seeds"]["full_budget"],
    }.get(run_kind)
    if expected_seed is None or int(seed) != int(expected_seed):
        raise RuntimeError("multihazard run kind/seed is not preregistered")
    output = RESULT_ROOT / f"{run_kind}_seed_{int(seed)}"
    trainer = ReviewerAlignedPPOTrainer(
        multihazard_config(training=True),
        seed=int(seed),
        device=device,
        environment_class=CompoundCollapseEnv,
        run_budgets=budgets,
        maximum_updates=3_000,
        source_manifest_builder=multihazard_source_manifest,
    )
    return trainer.train(
        updates=budgets[run_kind],
        run_kind=run_kind,
        output_directory=output,
    )
