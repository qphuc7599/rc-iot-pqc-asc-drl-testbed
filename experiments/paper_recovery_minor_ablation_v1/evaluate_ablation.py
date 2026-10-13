"""Frozen held-out evaluation for the IoT-73941-2026 controller ablation.

The script opens the registered 6572xx environment seeds only after all fixed-
budget checkpoints exist.  It evaluates the deterministic sequential decoder
for every training seed and records only per-environment summaries; training
and evaluation never use hidden Byzantine identities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import numpy as np

from drl.paper_recovery.receipts import (
    atomic_write_json,
    canonical_hash,
    sha256_file,
    utc_now,
)
from drl.paper_recovery_minor_ablation_v1.policy import (
    MinorRevisionAblationPPO,
)
from drl.paper_recovery_minor_ablation_v1.trainer import load_campaign_plan
from drl.paper_recovery_minor_ablation_v1.trainer import ablation_source_manifest
from drl.paper_recovery_security_v4.config import security_v4_config
from drl.paper_recovery_security_v4.policy import SecurityConstrainedPPO
from experiments.paper_recovery_security_speed_v5.evaluate_disaster import (
    PPO_NAME,
    MEAN_FIELDS,
    _evaluate_one,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
OUTPUT_ROOT = REPO_ROOT / "revision_R1_results/03_ablation/heldout"
OUTPUT_PATH = OUTPUT_ROOT / "evaluation.json"
VALIDATION_PATH = OUTPUT_ROOT / "validation.json"
PARENT_ROOT = REPO_ROOT / "results/paper_recovery_security_speed_v5/seed_657111"
PARENT_EVALUATION = (
    REPO_ROOT
    / "results/paper_recovery_security_speed_v5/heldout_disaster_seed_657111/evaluation.json"
)


class _DeterministicAblationAdapter:
    """Use the registered six-output security decoder without changing training code."""

    def __init__(self, policy: MinorRevisionAblationPPO):
        self.policy = policy

    def parameters(self):
        return self.policy.parameters()

    def act(self, observation, observation_sequence=None, *, deterministic=False, generator=None):
        return SecurityConstrainedPPO.act(
            self.policy,
            observation,
            observation_sequence,
            deterministic=deterministic,
            generator=generator,
        )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_parent() -> tuple[SecurityConstrainedPPO, dict[str, Any]]:
    receipt = _load_json(PARENT_ROOT / "completion_receipt.json")
    checkpoint_path = PARENT_ROOT / "ppo_final.pt"
    if receipt.get("updates_completed") != 3_000:
        raise RuntimeError("A0 parent update budget mismatch")
    if receipt.get("transitions_completed") != 9_000_000:
        raise RuntimeError("A0 parent transition budget mismatch")
    if sha256_file(checkpoint_path) != receipt.get("checkpoint_sha256"):
        raise RuntimeError("A0 parent checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy = SecurityConstrainedPPO(security_v4_config(training=True))
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    policy.eval()
    return policy, receipt


def _load_ablation(arm: str, seed: int) -> tuple[MinorRevisionAblationPPO, dict[str, Any]]:
    root = REPO_ROOT / f"revision_R1_results/03_ablation/formal_3000/{arm}/seed_{seed}"
    receipt = _load_json(root / "completion_receipt.json")
    checkpoint_path = root / "ppo_final.pt"
    required = {
        "status": "complete",
        "arm": arm,
        "seed": seed,
        "run_kind": "formal_3000",
        "updates_completed": 3_000,
        "transitions_completed": 9_000_000,
        "all_updates_finite": True,
        "all_required_modules_updated": True,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"{arm}/{seed} receipt mismatch: {key}")
    if receipt.get("worker_runtime", {}).get("all_workers_reaped") is not True:
        raise RuntimeError(f"{arm}/{seed} workers were not cleanly reaped")
    if sha256_file(checkpoint_path) != receipt.get("checkpoint_sha256"):
        raise RuntimeError(f"{arm}/{seed} checkpoint hash mismatch")
    if receipt.get("source_hash") != ablation_source_manifest(security_v4_config(training=True), arm=arm)["source_hash"]:
        raise RuntimeError(f"{arm}/{seed} formal training source drift")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("resumed") is not False:
        raise RuntimeError(f"{arm}/{seed} was resumed")
    if checkpoint.get("early_stopped") is not False:
        raise RuntimeError(f"{arm}/{seed} was early-stopped")
    if checkpoint.get("selected_by_evaluation") is not False:
        raise RuntimeError(f"{arm}/{seed} was selected by evaluation")
    policy = MinorRevisionAblationPPO(
        security_v4_config(training=True), arm=arm
    )
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    policy.eval()
    return policy, receipt


def _reuse_parent_rows(seeds: tuple[int, ...]) -> tuple[dict[str, Any], dict[str, str]]:
    document = _load_json(PARENT_EVALUATION)
    panel = document["panel"]
    if tuple(int(value) for value in panel["seeds"]) != seeds:
        raise RuntimeError("A0 parent held-out seed drift")
    rows = panel["deterministic_primary"]["per_seed"]
    hashes = panel["paired_exogenous_hashes"]
    if set(rows) != {str(seed) for seed in seeds}:
        raise RuntimeError("A0 parent held-out rows incomplete")
    return rows, hashes


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise RuntimeError("held-out ablation output exists; rerun forbidden")
    protocol = _load_json(PROTOCOL_PATH)
    plan = load_campaign_plan()
    if protocol.get("status") != "prospectively_frozen_before_any_pilot_or_formal_ablation":
        raise RuntimeError("ablation protocol is not frozen")
    seeds = tuple(
        int(value) for value in protocol["heldout_evaluation"]["environment_seeds"]
    )
    if len(seeds) != 20 or len(set(seeds)) != 20:
        raise RuntimeError("held-out environment seed contract drift")
    config = security_v4_config(training=False)
    if config.physics.horizon_steps != 2_000:
        raise RuntimeError("evaluation horizon drift")

    results: dict[str, dict[str, Any]] = {}
    checkpoint_receipts: dict[str, dict[str, Any]] = {}
    reference_rows, exogenous_hashes = _reuse_parent_rows(seeds)
    parent_policy, parent_receipt = _load_parent()
    del parent_policy

    # Require every planned final checkpoint before opening any new trajectory.
    prepared = {
        (arm, int(seed)): _load_ablation(arm, int(seed))
        for arm, registered_seeds in plan["formal_seeds"].items()
        for seed in registered_seeds
        if not (arm == "A0_full" and int(seed) == 657111)
    }

    for arm, registered_seeds in plan["formal_seeds"].items():
        results[arm] = {}
        for training_seed_value in registered_seeds:
            training_seed = int(training_seed_value)
            if arm == "A0_full" and training_seed == 657111:
                results[arm][str(training_seed)] = reference_rows
                checkpoint_receipts[f"{arm}/{training_seed}"] = {
                    "checkpoint_sha256": parent_receipt["checkpoint_sha256"],
                    "source_hash": parent_receipt["source_hash"],
                    "reused_frozen_parent_evaluation": True,
                }
                continue

            policy, receipt = prepared.pop((arm, training_seed))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            policy = policy.to(device)
            per_environment: dict[str, Any] = {}
            for environment_seed in seeds:
                summary, _, trace_hash = _evaluate_one(
                    PPO_NAME,
                    environment_seed,
                    config,
                    _DeterministicAblationAdapter(policy),
                    0,
                )
                if trace_hash != exogenous_hashes[str(environment_seed)]:
                    raise RuntimeError(
                        f"exogenous pairing drift: {arm}/{training_seed}/{environment_seed}"
                    )
                per_environment[str(environment_seed)] = summary
            results[arm][str(training_seed)] = per_environment
            checkpoint_receipts[f"{arm}/{training_seed}"] = {
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "source_hash": receipt["source_hash"],
                "reused_frozen_parent_evaluation": False,
            }
            del policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_rows = [
        row
        for arm_rows in results.values()
        for training_rows in arm_rows.values()
        for row in training_rows.values()
    ]
    expected_trajectories = len(plan["selected_arms"]) * plan["training_seeds_per_arm"] * len(seeds)
    checks = {
        "three_selected_arms": set(results) == set(plan["selected_arms"]),
        "two_training_seeds_per_arm": all(
            len(training_rows) == 2 for training_rows in results.values()
        ),
        "twenty_environment_seeds_per_checkpoint": all(
            len(environment_rows) == 20
            for training_rows in results.values()
            for environment_rows in training_rows.values()
        ),
        "exactly_120_trajectories": len(all_rows) == expected_trajectories,
        "all_trajectories_2000_steps": all(
            int(row["transition_count"]) == 2_000 for row in all_rows
        ),
        "no_forced_zero_injection": all(
            int(row["forced_zero_tps_steps"]) == 0 for row in all_rows
        ),
        "all_exogenous_streams_paired": len(exogenous_hashes) == 20,
        "all_scalar_metrics_finite": all(
            np.isfinite(float(row[field])) for row in all_rows for field in MEAN_FIELDS
        ),
    }
    validation = {
        "schema": "qrelay-minor-revision-ablation-heldout-validation-1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    validation["receipt_hash"] = canonical_hash(validation)
    if validation["status"] != "PASS":
        raise RuntimeError(f"ablation held-out validation failed: {checks}")

    payload = {
        "schema": "qrelay-minor-revision-ablation-heldout-evaluation-1",
        "created_at": utc_now(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "campaign_plan_sha256": sha256_file(PROTOCOL_PATH.with_name("campaign_plan.json")),
        "training_seeds_per_arm": plan["training_seeds_per_arm"],
        "scope": plan["inference"]["interpretation"],
        "evaluation_panel_note": plan["evaluation_panel_note"],
        "decoder": protocol["heldout_evaluation"]["primary_decoder"],
        "environment_seeds": list(seeds),
        "paired_exogenous_hashes": exogenous_hashes,
        "checkpoint_receipts": checkpoint_receipts,
        "results": results,
    }
    payload["receipt_hash"] = canonical_hash(payload)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    atomic_write_json(OUTPUT_PATH, payload)
    atomic_write_json(VALIDATION_PATH, validation)
    print(json.dumps({"status": "PASS", "trajectories": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
