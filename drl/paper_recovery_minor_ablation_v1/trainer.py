"""Fixed-budget, prospectively frozen minor-revision ablation trainer."""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from drl.paper_recovery.receipts import (
    aggregate_trajectory_hash,
    atomic_write_json,
    canonical_hash,
    runtime_manifest,
    sha256_file,
    utc_now,
)
from drl.paper_recovery_original_ppo.trainer import (
    DEPLETION_COST_LIMIT,
    DUAL_LEARNING_RATE,
    DUAL_MAXIMUM,
    FAILURE_COST_LIMIT,
)
from drl.paper_recovery_security_speed_v5.trainer import (
    LOGICAL_ENVIRONMENTS,
    PHYSICAL_WORKERS,
    SecuritySpeedTrainer,
    configure_speed_runtime,
)
from drl.paper_recovery_security_speed_v5.workers import (
    VectorizedPersistentRolloutWorkers,
)
from drl.paper_recovery_security_v4.config import security_v4_config
from drl.paper_recovery_security_v4.trainer import (
    ACTOR_LR,
    CRITIC_LR,
    PUBLIC_SECURITY_TAIL_LIMIT,
    SECURITY_DUAL_INITIAL,
    SECURITY_DUAL_LEARNING_RATE,
    SECURITY_DUAL_MAXIMUM,
)

from .policy import ARMS, MinorRevisionAblationPPO


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "experiments/paper_recovery_minor_ablation_v1/protocol.json"
CAMPAIGN_PLAN_PATH = PROTOCOL_PATH.with_name("campaign_plan.json")
RESULT_ROOT = REPO_ROOT / "revision_R1_results/03_ablation"

SOURCE_PATHS = (
    "drl/paper_recovery/config.py",
    "drl/paper_recovery/env.py",
    "drl/paper_recovery/policy.py",
    "drl/paper_recovery/receipts.py",
    "drl/paper_recovery_original_ppo/policy.py",
    "drl/paper_recovery_original_ppo/trainer.py",
    "drl/paper_recovery_longhorizon_v3/env.py",
    "drl/paper_recovery_longhorizon_v3/trainer.py",
    "drl/paper_recovery_hardcore_v5/config.py",
    "drl/paper_recovery_hardcore_v5/env.py",
    "drl/paper_recovery_hardcore_v5/policy.py",
    "drl/paper_recovery_security_v4/config.py",
    "drl/paper_recovery_security_v4/policy.py",
    "drl/paper_recovery_security_v4/trainer.py",
    "drl/paper_recovery_security_speed_v5/workers.py",
    "drl/paper_recovery_security_speed_v5/trainer.py",
    "drl/paper_recovery_minor_ablation_v1/__init__.py",
    "drl/paper_recovery_minor_ablation_v1/policy.py",
    "drl/paper_recovery_minor_ablation_v1/trainer.py",
    "experiments/paper_recovery_minor_ablation_v1/protocol.json",
    "experiments/paper_recovery_minor_ablation_v1/run_training.py",
    "experiments/paper_recovery_minor_ablation_v1/campaign_plan.json",
    "experiments/paper_recovery_minor_ablation_v1/run_session.py",
)


def ablation_source_manifest(config: Any, *, arm: str) -> dict[str, Any]:
    files = {relative: sha256_file(REPO_ROOT / relative) for relative in SOURCE_PATHS}
    document = {
        "schema": "qrelay-minor-revision-ablation-source-1",
        "arm": arm,
        "files": files,
        "scientific_config_hash": canonical_hash(config.to_dict()),
        "runtime_parallelism": {
            "physical_workers": PHYSICAL_WORKERS,
            "logical_environments": LOGICAL_ENVIRONMENTS,
        },
    }
    document["source_hash"] = canonical_hash(document)
    return document


def unique_parameters(modules: Iterable[torch.nn.Module]) -> list[torch.nn.Parameter]:
    values: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                values.append(parameter)
                seen.add(id(parameter))
    return values


class MinorRevisionAblationTrainer(SecuritySpeedTrainer):
    """Security-speed runtime with exactly one registered controller ablation."""

    def __init__(self, *, arm: str, seed: int, device: str = "cuda") -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown ablation arm: {arm}")
        super().__init__(seed=int(seed), device=device)
        self.arm = arm

        # Recreate the registered initialization point after the parent runtime
        # has been constructed.  This keeps A0 mathematically identical to the
        # security-v4 policy for the same seed.
        torch.manual_seed(self.seed + 6_000_067)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed + 6_000_067)
        self.policy = MinorRevisionAblationPPO(self.config, arm=arm).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=ACTOR_LR, eps=1e-5
        )
        critic_modules = (
            [self.policy.shared_critic]
            if self.policy.shared_critic is not None
            else [
                self.policy.reward_critic,
                self.policy.failure_cost_critic,
                self.policy.depletion_cost_critic,
                self.policy.security_cost_critic,
            ]
        )
        self.critic_parameters = unique_parameters(
            module for module in critic_modules if module is not None
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_parameters, lr=CRITIC_LR, eps=1e-5
        )
        self.security_dual = (
            0.0 if arm == "A3_no_security_critic" else SECURITY_DUAL_INITIAL
        )

    def _module_map(self) -> dict[str, torch.nn.Module]:
        if self.policy.shared_critic is not None:
            return {"actor": self.policy.actor, "shared_critic": self.policy.shared_critic}
        modules = {
            "actor": self.policy.actor,
            "reward_critic": self.policy.reward_critic,
            "failure_critic": self.policy.failure_cost_critic,
            "depletion_critic": self.policy.depletion_cost_critic,
        }
        if self.arm != "A3_no_security_critic":
            modules["security_critic"] = self.policy.security_cost_critic
        return modules

    def _update_duals(self, rollout: dict[str, Any]) -> None:
        self.failure_dual = float(
            np.clip(
                self.failure_dual
                + DUAL_LEARNING_RATE
                * (rollout["mean_failure_cost"] - FAILURE_COST_LIMIT),
                0.0,
                DUAL_MAXIMUM,
            )
        )
        self.depletion_dual = float(
            np.clip(
                self.depletion_dual
                + DUAL_LEARNING_RATE
                * (rollout["mean_depletion_exposure"] - DEPLETION_COST_LIMIT),
                0.0,
                DUAL_MAXIMUM,
            )
        )
        if self.arm == "A3_no_security_critic":
            self.security_dual = 0.0
        else:
            self.security_dual = float(
                np.clip(
                    self.security_dual
                    + SECURITY_DUAL_LEARNING_RATE
                    * (
                        rollout["mean_public_security_tail_cost"]
                        - PUBLIC_SECURITY_TAIL_LIMIT
                    ),
                    0.0,
                    SECURITY_DUAL_MAXIMUM,
                )
            )

    def train_ablation(
        self, *, updates: int, run_kind: str, output_directory: Path
    ) -> dict[str, Any]:
        expected = {"pilot_50": 50, "formal_3000": 3_000}
        if expected.get(run_kind) != int(updates):
            raise RuntimeError("minor-revision ablation budget mismatch")
        if output_directory.exists():
            raise RuntimeError("ablation output exists; overwrite/resume forbidden")
        output_directory.mkdir(parents=True)

        source = ablation_source_manifest(self.config, arm=self.arm)
        started_at = utc_now()
        started = time.monotonic()
        modules = self._module_map()
        initial_hashes = {
            name: self._module_hash(module) for name, module in modules.items()
        }
        progress: list[dict[str, Any]] = []
        workers = VectorizedPersistentRolloutWorkers(
            self.config,
            seed=self.seed,
            environment_class=self.environment_class,
        )
        close_receipt: dict[str, Any] = {}
        try:
            for update_index in range(updates):
                collect_started = time.monotonic()
                transitions, rollout = self.collect_update(
                    workers, update_index=update_index
                )
                collect_seconds = float(time.monotonic() - collect_started)
                optimise_started = time.monotonic()
                optimiser = self.optimise_update(transitions)
                optimise_seconds = float(time.monotonic() - optimise_started)
                self._update_duals(rollout)

                completed = update_index + 1
                transitions_completed = completed * 3_000
                elapsed = float(time.monotonic() - started)
                throughput = transitions_completed / max(elapsed, 1e-12)
                entry = {
                    **rollout,
                    **optimiser,
                    "arm": self.arm,
                    "collect_seconds": collect_seconds,
                    "optimise_seconds": optimise_seconds,
                    "failure_dual": self.failure_dual,
                    "depletion_dual": self.depletion_dual,
                    "security_dual": self.security_dual,
                    "elapsed_seconds": elapsed,
                    "rolling_transitions_per_second": throughput,
                    "estimated_remaining_seconds": (
                        (updates - completed) * 3_000 / max(throughput, 1e-12)
                    ),
                }
                progress.append(entry)
                atomic_write_json(
                    output_directory / "progress.json",
                    {
                        "schema": "qrelay-minor-revision-ablation-progress-1",
                        "arm": self.arm,
                        "seed": self.seed,
                        "run_kind": run_kind,
                        "updates_expected": updates,
                        "updates_completed": completed,
                        "transitions_completed": transitions_completed,
                        "rolling_transitions_per_second": throughput,
                        "estimated_remaining_seconds": entry[
                            "estimated_remaining_seconds"
                        ],
                        "entries": progress,
                    },
                )
        finally:
            close_receipt = workers.close()

        if len(progress) != updates:
            raise RuntimeError("ablation ended below the exact update budget")

        checkpoint_path = output_directory / (
            "ppo_final.pt" if run_kind == "formal_3000" else "pilot_final.pt"
        )
        torch.save(
            {
                "schema": "qrelay-minor-revision-ablation-checkpoint-1",
                "role": "ablation_pilot_not_reportable"
                if run_kind == "pilot_50"
                else "ablation_candidate_requires_heldout_evaluation",
                "paper_evidence_eligible": False,
                "arm": self.arm,
                "run_kind": run_kind,
                "seed": self.seed,
                "updates": updates,
                "transitions": updates * 3_000,
                "policy_state_dict": self.policy.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "failure_dual": self.failure_dual,
                "depletion_dual": self.depletion_dual,
                "security_dual": self.security_dual,
                "config": self.config.to_dict(),
                "source_manifest": source,
                "architecture_manifest": self.policy.architecture_manifest(),
                "runtime_parallelism": {
                    "physical_workers": PHYSICAL_WORKERS,
                    "logical_environments": LOGICAL_ENVIRONMENTS,
                },
                "resumed": False,
                "early_stopped": False,
                "selected_by_evaluation": False,
                "checkpoint_selection": "fixed_budget_final_only",
            },
            checkpoint_path,
        )
        final_hashes = {
            name: self._module_hash(module) for name, module in modules.items()
        }
        modules_updated = {
            name: final_hashes[name] != initial_hashes[name] for name in modules
        }
        logical_receipts = tuple(
            hasher.receipt() for hasher in self.trajectory_hashers
        )
        elapsed = float(time.monotonic() - started)
        receipt = {
            "schema": "qrelay-minor-revision-ablation-completion-1",
            "status": "complete",
            "role": "pilot_not_reportable"
            if run_kind == "pilot_50"
            else "candidate_requires_heldout_evaluation",
            "paper_evidence_eligible": False,
            "arm": self.arm,
            "seed": self.seed,
            "run_kind": run_kind,
            "started_at": started_at,
            "completed_at": utc_now(),
            "updates_completed": updates,
            "transitions_completed": updates * 3_000,
            "source_hash": source["source_hash"],
            "config_hash": canonical_hash(self.config.to_dict()),
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "trajectory_hash": aggregate_trajectory_hash(logical_receipts),
            "logical_trajectory_receipts": logical_receipts,
            "worker_runtime": close_receipt,
            "initial_module_hashes": initial_hashes,
            "final_module_hashes": final_hashes,
            "modules_updated": modules_updated,
            "all_required_modules_updated": all(modules_updated.values()),
            "all_updates_finite": all(row["all_finite"] for row in progress),
            "maximum_observed_kl": max(row["max_approx_kl"] for row in progress),
            "final_failure_dual": self.failure_dual,
            "final_depletion_dual": self.depletion_dual,
            "final_security_dual": self.security_dual,
            "elapsed_seconds": elapsed,
            "mean_transitions_per_second": updates * 3_000 / max(elapsed, 1e-12),
            "runtime": runtime_manifest(),
            "physical_workers": PHYSICAL_WORKERS,
            "logical_environments": LOGICAL_ENVIRONMENTS,
        }
        receipt["receipt_hash"] = canonical_hash(receipt)
        atomic_write_json(output_directory / "completion_receipt.json", receipt)
        return receipt


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def load_campaign_plan() -> dict[str, Any]:
    plan = json.loads(CAMPAIGN_PLAN_PATH.read_text(encoding="utf-8"))
    if sha256_file(PROTOCOL_PATH) != plan["original_protocol_sha256"]:
        raise RuntimeError("original ablation protocol hash drift")
    if plan.get("status") != "frozen_by_user_before_any_formal_ablation_run":
        raise RuntimeError("formal campaign plan is not frozen")
    if plan["updates_per_seed"] != 3000 or plan["transitions_per_update"] != 3000:
        raise RuntimeError("formal campaign changed the fixed training budget")
    if any(len(seeds) != 2 for seeds in plan["formal_seeds"].values()):
        raise RuntimeError("formal campaign requires two training seeds per arm")
    return plan


def run_registered(
    *, arm: str, run_kind: str, seed: int, device: str = "cuda"
) -> dict[str, Any]:
    protocol = load_protocol()
    if arm not in ARMS:
        raise RuntimeError("arm is not registered")
    if run_kind == "pilot_50":
        allowed = [int(value) for value in protocol["pilot_seeds"][arm]]
    elif run_kind == "formal_3000":
        plan = load_campaign_plan()
        if arm not in plan["selected_arms"]:
            raise RuntimeError("arm excluded from the user-selected formal campaign")
        allowed = [int(value) for value in plan["formal_seeds"][arm]]
    else:
        raise RuntimeError("unknown ablation run-kind")
    if int(seed) not in allowed:
        raise RuntimeError("arm/seed/run-kind tuple is not prospectively registered")
    if arm == "A0_full" and int(seed) == 657111:
        raise RuntimeError("A0 seed 657111 is the frozen completed parent; do not rerun")

    config = security_v4_config(training=True)
    if canonical_hash(config.to_dict()) != protocol["frozen_config_hash"]:
        raise RuntimeError("ablation scientific config drift")

    source = ablation_source_manifest(config, arm=arm)
    claims = RESULT_ROOT / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    claim_path = claims / f"{arm}_seed_{int(seed)}_{run_kind}.json"
    if claim_path.exists():
        raise RuntimeError("ablation seed already claimed")
    claim = {
        "schema": "qrelay-minor-revision-ablation-seed-claim-1",
        "arm": arm,
        "seed": int(seed),
        "run_kind": run_kind,
        "claimed_at": utc_now(),
        "source_hash": source["source_hash"],
        "config_hash": canonical_hash(config.to_dict()),
        "environment_constructed_before_claim": False,
        "resume_allowed": False,
    }
    claim["claim_hash"] = canonical_hash(claim)
    atomic_write_json(claim_path, claim)

    configure_speed_runtime()
    trainer = MinorRevisionAblationTrainer(arm=arm, seed=int(seed), device=device)
    updates = 50 if run_kind == "pilot_50" else 3_000
    output = RESULT_ROOT / run_kind / arm / f"seed_{int(seed)}"
    try:
        return trainer.train_ablation(
            updates=updates, run_kind=run_kind, output_directory=output
        )
    except BaseException as error:
        failure = {
            "schema": "qrelay-minor-revision-ablation-failure-1",
            "status": "spent_failed",
            "arm": arm,
            "seed": int(seed),
            "run_kind": run_kind,
            "failed_at": utc_now(),
            "error_type": type(error).__name__,
            "message": str(error),
            "source_hash": source["source_hash"],
            "resume_allowed": False,
        }
        failure["receipt_hash"] = canonical_hash(failure)
        atomic_write_json(
            RESULT_ROOT / f"{arm}_seed_{int(seed)}_{run_kind}_failure.json",
            failure,
        )
        raise
