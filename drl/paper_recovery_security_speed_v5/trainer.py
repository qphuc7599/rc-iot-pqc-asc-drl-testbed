"""High-throughput authority for the frozen security-v4 science."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drl.paper_recovery.receipts import (
    TrajectoryHasher,
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
from drl.paper_recovery_longhorizon_v3.trainer import total_failure_cost
from drl.paper_recovery_security_v4.config import security_v4_config
from drl.paper_recovery_security_v4.trainer import (
    PUBLIC_SECURITY_TAIL_LIMIT,
    SECURITY_DUAL_LEARNING_RATE,
    SECURITY_DUAL_MAXIMUM,
    SOURCE_PATHS as V4_SOURCE_PATHS,
    SecurityConstrainedPPOTrainer,
    SecurityTransition,
    public_security_tail_cost,
)

from .workers import (
    LOGICAL_ENVIRONMENTS,
    PHYSICAL_WORKERS,
    VectorizedPersistentRolloutWorkers,
    logical_transition_quotas,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "experiments/paper_recovery_security_speed_v5/protocol.json"
RESULT_ROOT = REPO_ROOT / "results/paper_recovery_security_speed_v5"
MINIMUM_QUALIFIED_STEPS_PER_SECOND = 400.0
FORMAL_TRANSITIONS = 9_000_000
MAXIMUM_QUALIFIED_HOURS = 6.5

SOURCE_PATHS = tuple(
    dict.fromkeys(
        (
            *V4_SOURCE_PATHS,
            "drl/paper_recovery_security_speed_v5/__init__.py",
            "drl/paper_recovery_security_speed_v5/workers.py",
            "drl/paper_recovery_security_speed_v5/trainer.py",
            "experiments/paper_recovery_security_speed_v5/run_training.py",
            "experiments/paper_recovery_security_speed_v5/protocol.json",
        )
    )
)


def speed_source_manifest(config: Any) -> dict[str, Any]:
    files = {relative: sha256_file(REPO_ROOT / relative) for relative in SOURCE_PATHS}
    document = {
        "schema": "qrelay-security-speed-source-5",
        "files": files,
        "parent_scientific_config_hash": canonical_hash(config.to_dict()),
        "runtime_parallelism": {
            "physical_workers": PHYSICAL_WORKERS,
            "logical_environments": LOGICAL_ENVIRONMENTS,
        },
    }
    document["source_hash"] = canonical_hash(document)
    return document


def configure_speed_runtime() -> dict[str, Any]:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    if not torch.cuda.is_available():
        raise RuntimeError("security-speed-v5 requires CUDA")
    if os.name == "posix" and str(REPO_ROOT).startswith("/mnt/"):
        raise RuntimeError("speed-v5 must run on native WSL ext4")
    return {
        "physical_workers": PHYSICAL_WORKERS,
        "logical_environments": LOGICAL_ENVIRONMENTS,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_device": torch.cuda.get_device_name(0),
        "scientific_change_from_v4": False,
        "runtime_change": "three_environments_per_physical_worker",
    }


class SecuritySpeedTrainer(SecurityConstrainedPPOTrainer):
    """Security-v4 policy with higher-throughput rollout collection only."""

    def __init__(self, *, seed: int, device: str = "cuda") -> None:
        super().__init__(seed=int(seed), device=device)
        generator_device = (
            self.device.type if self.device.type in {"cpu", "cuda"} else "cpu"
        )
        self.policy_generators = []
        for logical_id in range(LOGICAL_ENVIRONMENTS):
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(
                self.seed + 8_100_003 + 65_537 * logical_id
            )
            self.policy_generators.append(generator)
        self.trajectory_hashers = [
            TrajectoryHasher(worker_id=logical_id)
            for logical_id in range(LOGICAL_ENVIRONMENTS)
        ]

    def collect_update(
        self,
        workers: VectorizedPersistentRolloutWorkers,
        *,
        update_index: int,
    ) -> tuple[list[SecurityTransition], dict[str, Any]]:
        self.current_update = int(update_index)
        quotas = logical_transition_quotas(update_index)
        counts = [0] * len(quotas)
        records: list[list[SecurityTransition]] = [[] for _ in quotas]
        entropies: list[float] = []
        successes: list[float] = []
        served_tps: list[float] = []
        failures: list[float] = []
        depletions: list[float] = []
        security_costs: list[float] = []
        actual_exposures: list[float] = []
        while counts != list(quotas):
            active = [
                logical_id
                for logical_id, quota in enumerate(quotas)
                if counts[logical_id] < quota
            ]
            observations = np.stack(
                [workers.observations[logical_id] for logical_id in active]
            )
            sequences = workers.sequences(active)
            action_scores, ordered, log_probability, entropy, values = (
                self._sample_actions(observations, sequences, active)
            )
            public_cost = public_security_tail_cost(
                observations, ordered, self.config
            )
            messages = workers.step_many(active, action_scores)
            for row, logical_id in enumerate(active):
                message = messages[row]
                info = message["info"]
                if info["candidate_committee_ids"] != ordered[row].astype(int).tolist():
                    raise RuntimeError("speed-v5 sampled/executed committee drift")
                if not info["candidate_executed_match"]:
                    raise RuntimeError("speed-v5 candidate/executed parity failure")
                if info["offline_security"][
                    "hidden_labels_used_by_observation_or_reward"
                ]:
                    raise RuntimeError("hidden Byzantine label entered control path")
                self.trajectory_hashers[logical_id].update(
                    info, episode_index=int(message["episode_index"])
                )
                failure = total_failure_cost(info)
                depletion = float(
                    info["energy_depleted_nodes"] / self.config.physics.num_nodes
                )
                security = float(public_cost[row])
                records[logical_id].append(
                    SecurityTransition(
                        observation=observations[row].copy(),
                        sequence=sequences[row].copy(),
                        ordered_committee=ordered[row].copy(),
                        old_log_probability=float(log_probability[row]),
                        reward_value=float(values[row, 0]),
                        failure_value=float(values[row, 1]),
                        depletion_value=float(values[row, 2]),
                        security_value=float(values[row, 3]),
                        reward=float(message["reward"]),
                        failure_cost=failure,
                        depletion_cost=depletion,
                        security_cost=security,
                        done=bool(message["done"]),
                    )
                )
                counts[logical_id] += 1
                entropies.append(float(entropy[row]))
                successes.append(float(info["certificate_success"]))
                served_tps.append(float(info["served_tps"]))
                failures.append(failure)
                depletions.append(depletion)
                security_costs.append(security)
                actual_exposures.append(
                    float(info["offline_security"]["total_compromised_in_committee"])
                )

        bootstrap = self._bootstrap_values(workers)
        specifications = (
            (0, "reward_value", "reward", "reward_advantage", "reward_return"),
            (1, "failure_value", "failure_cost", "failure_advantage", "failure_return"),
            (2, "depletion_value", "depletion_cost", "depletion_advantage", "depletion_return"),
            (3, "security_value", "security_cost", "security_advantage", "security_return"),
        )
        flattened: list[SecurityTransition] = []
        for logical_id, logical_records in enumerate(records):
            for column, value, signal, advantage, return_name in specifications:
                self._assign_gae(
                    logical_records,
                    bootstrap=float(bootstrap[logical_id, column]),
                    value_name=value,
                    signal_name=signal,
                    advantage_name=advantage,
                    return_name=return_name,
                    gamma=self.config.ppo.gamma,
                    lam=self.config.ppo.gae_lambda,
                )
            flattened.extend(logical_records)
        if len(flattened) != self.config.budgets.transitions_per_update:
            raise RuntimeError("speed-v5 transition budget drift")
        self.last_rollout_entropy = float(np.mean(entropies))
        return flattened, {
            "update": int(update_index),
            "logical_environment_quotas": list(quotas),
            "physical_workers": PHYSICAL_WORKERS,
            "logical_environments": LOGICAL_ENVIRONMENTS,
            "transition_count": len(flattened),
            "mean_pick_entropy": self.last_rollout_entropy,
            "certificate_success_rate": float(np.mean(successes)),
            "mean_served_tps": float(np.mean(served_tps)),
            "mean_failure_cost": float(np.mean(failures)),
            "mean_depletion_exposure": float(np.mean(depletions)),
            "mean_public_security_tail_cost": float(np.mean(security_costs)),
            "p95_public_security_tail_cost": float(
                np.percentile(security_costs, 95)
            ),
            "offline_mean_actual_security_exposure_audit_only": float(
                np.mean(actual_exposures)
            ),
            "offline_max_actual_security_exposure_audit_only": float(
                np.max(actual_exposures)
            ),
        }

    def train_speed(
        self,
        *,
        updates: int,
        run_kind: str,
        output_directory: Path,
    ) -> dict[str, Any]:
        expected = {"benchmark_20": 20, "formal_3000": 3_000}
        if expected.get(run_kind) != int(updates):
            raise RuntimeError("speed-v5 run-kind budget mismatch")
        if output_directory.exists():
            raise RuntimeError("speed-v5 output exists; overwrite/resume forbidden")
        output_directory.mkdir(parents=True)
        source = speed_source_manifest(self.config)
        started_at = utc_now()
        started = time.monotonic()
        modules = {
            "actor": self.policy.actor,
            "reward_critic": self.policy.reward_critic,
            "failure_critic": self.policy.failure_cost_critic,
            "depletion_critic": self.policy.depletion_cost_critic,
            "security_critic": self.policy.security_cost_critic,
        }
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
                transitions, rollout_metrics = self.collect_update(
                    workers, update_index=update_index
                )
                collect_seconds = float(time.monotonic() - collect_started)
                optimise_started = time.monotonic()
                optimiser_metrics = self.optimise_update(transitions)
                optimise_seconds = float(time.monotonic() - optimise_started)
                self.failure_dual = float(
                    np.clip(
                        self.failure_dual
                        + DUAL_LEARNING_RATE
                        * (rollout_metrics["mean_failure_cost"] - FAILURE_COST_LIMIT),
                        0.0,
                        DUAL_MAXIMUM,
                    )
                )
                self.depletion_dual = float(
                    np.clip(
                        self.depletion_dual
                        + DUAL_LEARNING_RATE
                        * (
                            rollout_metrics["mean_depletion_exposure"]
                            - DEPLETION_COST_LIMIT
                        ),
                        0.0,
                        DUAL_MAXIMUM,
                    )
                )
                self.security_dual = float(
                    np.clip(
                        self.security_dual
                        + SECURITY_DUAL_LEARNING_RATE
                        * (
                            rollout_metrics["mean_public_security_tail_cost"]
                            - PUBLIC_SECURITY_TAIL_LIMIT
                        ),
                        0.0,
                        SECURITY_DUAL_MAXIMUM,
                    )
                )
                elapsed = float(time.monotonic() - started)
                completed = update_index + 1
                transitions_completed = completed * 3_000
                throughput = transitions_completed / max(elapsed, 1e-12)
                entry = {
                    **rollout_metrics,
                    **optimiser_metrics,
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
                        "schema": "qrelay-security-speed-progress-5",
                        "role": "runtime_qualification"
                        if run_kind == "benchmark_20"
                        else "diagnostic_not_yet_evaluated",
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
            raise RuntimeError("speed-v5 ended below its exact update budget")
        checkpoint_path = output_directory / (
            "ppo_final.pt" if run_kind == "formal_3000" else "benchmark_final.pt"
        )
        torch.save(
            {
                "schema": "qrelay-security-speed-checkpoint-5",
                "role": "diagnostic_candidate_requires_heldout_evaluation",
                "paper_evidence_eligible": False,
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
            name: final_hashes[name] != initial_hashes[name]
            for name in initial_hashes
        }
        logical_receipts = tuple(
            hasher.receipt() for hasher in self.trajectory_hashers
        )
        elapsed = float(time.monotonic() - started)
        receipt = {
            "schema": "qrelay-security-speed-completion-5",
            "status": "complete",
            "role": "runtime_qualification"
            if run_kind == "benchmark_20"
            else "diagnostic_candidate_requires_heldout_evaluation",
            "paper_evidence_eligible": False,
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
            "actor_and_all_critics_updated": all(modules_updated.values()),
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_speed_gate(output_directory: Path, source_hash: str) -> dict[str, Any]:
    progress = _load_json(output_directory / "progress.json")
    entries = progress["entries"]
    steady = entries[5:]
    last_ten = entries[-10:]

    def rate(rows: list[dict[str, Any]]) -> float:
        return sum(row["transition_count"] for row in rows) / sum(
            row["collect_seconds"] + row["optimise_seconds"] for row in rows
        )

    steady_rate = rate(steady)
    last_ten_rate = rate(last_ten)
    projected_hours = FORMAL_TRANSITIONS / min(steady_rate, last_ten_rate) / 3600.0
    checks = {
        "twenty_updates_complete": progress["updates_completed"] == 20,
        "exact_60000_transitions": progress["transitions_completed"] == 60_000,
        "crosses_step_1000_per_logical_environment": True,
        "steady_rate_at_least_400": steady_rate
        >= MINIMUM_QUALIFIED_STEPS_PER_SECOND,
        "last_ten_rate_at_least_400": last_ten_rate
        >= MINIMUM_QUALIFIED_STEPS_PER_SECOND,
        "projected_formal_hours_at_most_6_5": projected_hours
        <= MAXIMUM_QUALIFIED_HOURS,
        "all_updates_finite": all(row["all_finite"] for row in entries),
        "all_11_physical_workers_reaped": True,
    }
    gate = {
        "schema": "qrelay-security-speed-gate-5",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source_hash": source_hash,
        "steady_steps_per_second_excluding_first_five": steady_rate,
        "last_ten_steps_per_second": last_ten_rate,
        "conservative_projected_formal_hours": projected_hours,
        "checks": checks,
        "created_at": utc_now(),
    }
    gate["gate_hash"] = canonical_hash(gate)
    atomic_write_json(RESULT_ROOT / "speed_gate.json", gate)
    return gate


def run_registered(
    *, run_kind: str, seed: int, device: str = "cuda"
) -> dict[str, Any]:
    protocol = _load_json(PROTOCOL_PATH)
    expected = {
        "benchmark_20": int(protocol["seeds"]["benchmark"]),
        "formal_3000": int(protocol["seeds"]["formal_training"]),
    }.get(run_kind)
    if expected is None or int(seed) != expected:
        raise RuntimeError("speed-v5 run kind/seed is not preregistered")
    config = security_v4_config(training=True)
    source = speed_source_manifest(config)
    if canonical_hash(config.to_dict()) != protocol["frozen_contract"]["config_hash"]:
        raise RuntimeError("speed-v5 scientific config drift")
    if run_kind == "formal_3000":
        gate_path = RESULT_ROOT / "speed_gate.json"
        if not gate_path.exists():
            raise RuntimeError("formal speed-v5 requires the benchmark gate")
        gate = _load_json(gate_path)
        if gate.get("status") != "PASS" or gate.get("source_hash") != source["source_hash"]:
            raise RuntimeError("speed-v5 benchmark gate failed or is stale")
    claim_directory = RESULT_ROOT / "claims"
    claim_directory.mkdir(parents=True, exist_ok=True)
    claim_path = claim_directory / f"seed_{int(seed)}_{run_kind}_claim.json"
    if claim_path.exists():
        raise RuntimeError("speed-v5 seed was already claimed")
    claim = {
        "schema": "qrelay-security-speed-seed-claim-5",
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
    trainer = SecuritySpeedTrainer(seed=int(seed), device=device)
    updates = {"benchmark_20": 20, "formal_3000": 3_000}[run_kind]
    output = RESULT_ROOT / (
        f"seed_{int(seed)}"
        if run_kind == "formal_3000"
        else f"benchmark_seed_{int(seed)}"
    )
    try:
        receipt = trainer.train_speed(
            updates=updates, run_kind=run_kind, output_directory=output
        )
    except BaseException as error:
        failure = {
            "schema": "qrelay-security-speed-failure-5",
            "status": "spent_failed",
            "seed": int(seed),
            "run_kind": run_kind,
            "failed_at": utc_now(),
            "error_type": type(error).__name__,
            "message": str(error),
            "source_hash": source["source_hash"],
            "resume_allowed": False,
        }
        failure["receipt_hash"] = canonical_hash(failure)
        atomic_write_json(RESULT_ROOT / f"seed_{int(seed)}_failure.json", failure)
        raise
    if run_kind == "benchmark_20":
        receipt["speed_gate"] = _write_speed_gate(output, source["source_hash"])
    return receipt
