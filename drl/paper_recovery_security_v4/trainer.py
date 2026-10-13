"""Fixed-budget security-constrained PPO trainer for Q-Relay.

This campaign keeps the hardcore-v5 disaster environment unchanged.  The new
scientific variable is confined to the controller: a public response-risk
critic, a primal-dual constraint, and a public admission rule.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drl.paper_recovery.config import PaperRecoveryConfig
from drl.paper_recovery.receipts import (
    aggregate_trajectory_hash,
    atomic_write_json,
    canonical_hash,
    runtime_manifest,
    sha256_file,
    utc_now,
)
from drl.paper_recovery.trainer import (
    PersistentRolloutWorkers,
    worker_transition_quotas,
)
from drl.paper_recovery_original_ppo.trainer import (
    DEPLETION_COST_LIMIT,
    DUAL_LEARNING_RATE,
    DUAL_MAXIMUM,
    FAILURE_COST_LIMIT,
    ReviewerAlignedPPOTrainer,
)
from drl.paper_recovery_longhorizon_v3.trainer import total_failure_cost
from drl.paper_recovery_hardcore_v5.env import HardcoreCascadeEnv

from .config import security_v4_config
from .policy import SecurityConstrainedPPO


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "experiments/paper_recovery_security_formal_v4/protocol.json"
RESULT_ROOT = REPO_ROOT / "results/paper_recovery_security_formal_v4"

ACTOR_LR = 3e-5
CRITIC_LR = 2e-4
CRITIC_LOSS_COEFFICIENT = 0.5
INITIAL_ENTROPY_COEFFICIENT = 0.010
MIN_ENTROPY_COEFFICIENT = 0.001
MAX_ENTROPY_COEFFICIENT = 0.050
TARGET_PER_PICK_ENTROPY = 1.75
INITIAL_KL_COEFFICIENT = 0.10
MIN_KL_COEFFICIENT = 0.01
MAX_KL_COEFFICIENT = 20.0

# The cost is a posterior-predictive probability, not a hidden Byzantine count.
# Five percent is a prospective operational risk target; it is intentionally
# below the uninformative-prior tail of 0.1085 at the first decision.
PUBLIC_SECURITY_TAIL_LIMIT = 0.05
SECURITY_DUAL_INITIAL = 0.50
SECURITY_DUAL_LEARNING_RATE = 0.05
SECURITY_DUAL_MAXIMUM = 20.0


SOURCE_PATHS = (
    "drl/paper_recovery/config.py",
    "drl/paper_recovery/env.py",
    "drl/paper_recovery/policy.py",
    "drl/paper_recovery/trainer.py",
    "drl/paper_recovery/receipts.py",
    "drl/paper_recovery_original_ppo/policy.py",
    "drl/paper_recovery_original_ppo/trainer.py",
    "drl/paper_recovery_longhorizon_v3/env.py",
    "drl/paper_recovery_longhorizon_v3/trainer.py",
    "drl/paper_recovery_hardcore_v5/config.py",
    "drl/paper_recovery_hardcore_v5/env.py",
    "drl/paper_recovery_hardcore_v5/policy.py",
    "drl/paper_recovery_security_v4/__init__.py",
    "drl/paper_recovery_security_v4/config.py",
    "drl/paper_recovery_security_v4/policy.py",
    "drl/paper_recovery_security_v4/trainer.py",
    "experiments/paper_recovery_security_formal_v4/run_training.py",
    "experiments/paper_recovery_security_formal_v4/protocol.json",
)


def security_source_manifest(config: PaperRecoveryConfig) -> dict[str, Any]:
    files = {relative: sha256_file(REPO_ROOT / relative) for relative in SOURCE_PATHS}
    document = {
        "schema": "qrelay-security-constrained-source-4",
        "files": files,
        "config_hash": canonical_hash(config.to_dict()),
    }
    document["source_hash"] = canonical_hash(document)
    return document


def configure_high_throughput_runtime() -> dict[str, Any]:
    """Use all rollout workers while preventing nested BLAS oversubscription."""

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
        raise RuntimeError("security-v4 formal training requires CUDA")
    if os.name == "posix" and str(REPO_ROOT).startswith("/mnt/"):
        raise RuntimeError("formal runtime must be staged on native WSL ext4")
    return {
        "filesystem_requirement": "native_wsl_ext4",
        "persistent_workers": 11,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "blas_threads_per_process": 1,
        "cuda_device": torch.cuda.get_device_name(0),
        "training_action_mode": "stochastic_exact_Plackett_Luce",
        "primary_deployment_mode": "deterministic_public_risk_top_k",
        "stochastic_audit_mode": "same_temperature_as_training",
    }


def poisson_binomial_tail_from_probabilities(
    probabilities: np.ndarray, *, fault_budget: int
) -> np.ndarray:
    """Exact independent posterior-predictive P(X > f), vectorized by batch."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("probabilities must have [batch, committee] shape")
    values = np.clip(values, 0.0, 1.0)
    batch, committee = values.shape
    distribution = np.zeros((batch, committee + 1), dtype=np.float64)
    distribution[:, 0] = 1.0
    for index in range(committee):
        probability = values[:, index : index + 1]
        previous = distribution.copy()
        distribution *= 1.0 - probability
        distribution[:, 1:] += previous[:, :-1] * probability
    return np.sum(distribution[:, int(fault_budget) + 1 :], axis=1)


def public_security_tail_cost(
    observations: np.ndarray,
    ordered_committees: np.ndarray,
    config: PaperRecoveryConfig,
) -> np.ndarray:
    """Compute security cost from pre-action public Beta response posteriors."""

    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    ordered = np.asarray(ordered_committees, dtype=np.int64)
    if ordered.ndim == 1:
        ordered = ordered[None, :]
    boundary = config.physics.num_nodes * config.observations.per_node_features
    nodes = obs[:, :boundary].reshape(
        obs.shape[0],
        config.physics.num_nodes,
        config.observations.per_node_features,
    )
    response_mean = np.take_along_axis(nodes[:, :, 2], ordered, axis=1)
    return poisson_binomial_tail_from_probabilities(
        1.0 - response_mean,
        fault_budget=config.physics.fault_budget,
    )


@dataclass
class SecurityTransition:
    observation: np.ndarray
    sequence: np.ndarray
    ordered_committee: np.ndarray
    old_log_probability: float
    reward_value: float
    failure_value: float
    depletion_value: float
    security_value: float
    reward: float
    failure_cost: float
    depletion_cost: float
    security_cost: float
    done: bool
    reward_advantage: float = 0.0
    failure_advantage: float = 0.0
    depletion_advantage: float = 0.0
    security_advantage: float = 0.0
    reward_return: float = 0.0
    failure_return: float = 0.0
    depletion_return: float = 0.0
    security_return: float = 0.0


class SecurityConstrainedPPOTrainer(ReviewerAlignedPPOTrainer):
    """Four-critic primal-dual PPO with a public chance-risk constraint."""

    def __init__(self, *, seed: int, device: str = "cuda") -> None:
        config = security_v4_config(training=True)
        ReviewerAlignedPPOTrainer.__init__(
            self,
            config,
            seed=int(seed),
            device=device,
            environment_class=HardcoreCascadeEnv,
            run_budgets={
                "smoke_2": 2,
                "qualification_8": 8,
                "formal_3000": 3_000,
            },
            maximum_updates=3_000,
            source_manifest_builder=security_source_manifest,
        )
        torch.manual_seed(self.seed + 6_000_067)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed + 6_000_067)
        self.policy = SecurityConstrainedPPO(config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=ACTOR_LR, eps=1e-5
        )
        self.critic_parameters = list(
            itertools.chain(
                self.policy.reward_critic.parameters(),
                self.policy.failure_cost_critic.parameters(),
                self.policy.depletion_cost_critic.parameters(),
                self.policy.security_cost_critic.parameters(),
            )
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_parameters, lr=CRITIC_LR, eps=1e-5
        )
        self.security_dual = SECURITY_DUAL_INITIAL
        self.entropy_coefficient = INITIAL_ENTROPY_COEFFICIENT
        self.kl_coefficient = INITIAL_KL_COEFFICIENT
        self.current_update = 0
        self.last_rollout_entropy = 0.0

    def _sample_actions(
        self,
        observations: np.ndarray,
        sequences: np.ndarray,
        worker_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        seq = torch.as_tensor(sequences, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, reward, failure, depletion, security, masks = self.policy(obs, seq)
            eligible = masks.sum(dim=1)
            if torch.any(eligible < self.config.physics.committee_size):
                raise RuntimeError(
                    "security-v4 rollout failed closed below k: "
                    + str(eligible.cpu().tolist())
                )
            policy_logits = logits / max(self.policy.temperature, 1e-6)
            action_scores, ordered, log_probability, entropy = (
                self.policy.sample_ordered(
                    policy_logits,
                    masks,
                    [self.policy_generators[index] for index in worker_ids],
                )
            )
            values = torch.stack((reward, failure, depletion, security), dim=1)
        return (
            action_scores.cpu().numpy(),
            ordered.cpu().numpy(),
            log_probability.cpu().numpy(),
            entropy.cpu().numpy(),
            values.cpu().numpy(),
        )

    def _bootstrap_values(self, workers: PersistentRolloutWorkers) -> np.ndarray:
        observations = np.stack(workers.observations)
        sequences = workers.sequences(list(range(len(workers.observations))))
        with torch.no_grad():
            _, reward, failure, depletion, security, _ = self.policy(
                torch.as_tensor(observations, dtype=torch.float32, device=self.device),
                torch.as_tensor(sequences, dtype=torch.float32, device=self.device),
            )
        return torch.stack((reward, failure, depletion, security), dim=1).cpu().numpy()

    def collect_update(
        self, workers: PersistentRolloutWorkers, *, update_index: int
    ) -> tuple[list[SecurityTransition], dict[str, Any]]:
        self.current_update = int(update_index)
        quotas = worker_transition_quotas(update_index)
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
                index for index, quota in enumerate(quotas) if counts[index] < quota
            ]
            observations = np.stack([workers.observations[index] for index in active])
            sequences = workers.sequences(active)
            action_scores, ordered, log_probability, entropy, values = (
                self._sample_actions(observations, sequences, active)
            )
            public_cost = public_security_tail_cost(
                observations, ordered, self.config
            )
            messages = workers.step_many(active, action_scores)
            for row, worker_id in enumerate(active):
                message = messages[row]
                info = message["info"]
                if info["candidate_committee_ids"] != ordered[row].astype(int).tolist():
                    raise RuntimeError("security-v4 sampled/executed committee drift")
                if not info["candidate_executed_match"]:
                    raise RuntimeError("security-v4 candidate/executed parity failure")
                if info["offline_security"][
                    "hidden_labels_used_by_observation_or_reward"
                ]:
                    raise RuntimeError("hidden Byzantine label entered control path")
                self.trajectory_hashers[worker_id].update(
                    info, episode_index=int(message["episode_index"])
                )
                failure = total_failure_cost(info)
                depletion = float(
                    info["energy_depleted_nodes"] / self.config.physics.num_nodes
                )
                security = float(public_cost[row])
                records[worker_id].append(
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
                counts[worker_id] += 1
                entropies.append(float(entropy[row]))
                successes.append(float(info["certificate_success"]))
                served_tps.append(float(info["served_tps"]))
                failures.append(failure)
                depletions.append(depletion)
                security_costs.append(security)
                # Hidden truth is logged only for offline audit; it never enters
                # a transition, reward, critic target, mask, or policy input.
                actual_exposures.append(
                    float(info["offline_security"]["total_compromised_in_committee"])
                )

        bootstrap = self._bootstrap_values(workers)
        specifications = (
            (0, "reward_value", "reward", "reward_advantage", "reward_return"),
            (
                1,
                "failure_value",
                "failure_cost",
                "failure_advantage",
                "failure_return",
            ),
            (
                2,
                "depletion_value",
                "depletion_cost",
                "depletion_advantage",
                "depletion_return",
            ),
            (
                3,
                "security_value",
                "security_cost",
                "security_advantage",
                "security_return",
            ),
        )
        flattened: list[SecurityTransition] = []
        for worker_id, worker_records in enumerate(records):
            for column, value, signal, advantage, return_name in specifications:
                self._assign_gae(
                    worker_records,
                    bootstrap=float(bootstrap[worker_id, column]),
                    value_name=value,
                    signal_name=signal,
                    advantage_name=advantage,
                    return_name=return_name,
                    gamma=self.config.ppo.gamma,
                    lam=self.config.ppo.gae_lambda,
                )
            flattened.extend(worker_records)
        if len(flattened) != self.config.budgets.transitions_per_update:
            raise RuntimeError("security-v4 transition budget drift")
        self.last_rollout_entropy = float(np.mean(entropies))
        return flattened, {
            "update": int(update_index),
            "worker_quotas": list(quotas),
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

    def optimise_update(
        self, transitions: list[SecurityTransition]
    ) -> dict[str, Any]:
        observations = torch.as_tensor(
            np.stack([row.observation for row in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        sequences = torch.as_tensor(
            np.stack([row.sequence for row in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        ordered = torch.as_tensor(
            np.stack([row.ordered_committee for row in transitions]),
            dtype=torch.long,
            device=self.device,
        )
        old_log_probability = torch.as_tensor(
            [row.old_log_probability for row in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = tuple(
            torch.as_tensor(
                [getattr(row, name) for row in transitions],
                dtype=torch.float32,
                device=self.device,
            )
            for name in (
                "reward_advantage",
                "failure_advantage",
                "depletion_advantage",
                "security_advantage",
            )
        )
        combined = (
            advantages[0]
            - self.failure_dual * advantages[1]
            - self.depletion_dual * advantages[2]
            - self.security_dual * advantages[3]
        )
        combined = (combined - combined.mean()) / combined.std(
            unbiased=False
        ).clamp_min(1e-8)
        returns = tuple(
            torch.as_tensor(
                [getattr(row, name) for row in transitions],
                dtype=torch.float32,
                device=self.device,
            )
            for name in (
                "reward_return",
                "failure_return",
                "depletion_return",
                "security_return",
            )
        )
        names = (
            "actor_loss",
            "reward_critic_loss",
            "failure_critic_loss",
            "depletion_critic_loss",
            "security_critic_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "actor_gradient_norm",
            "critic_gradient_norm",
            "actor_step_allowed",
        )
        metrics: dict[str, list[float]] = {name: [] for name in names}
        actor_update_open = True
        batch_size = len(transitions)
        for _ in range(self.config.budgets.epochs_per_update):
            permutation = torch.randperm(batch_size, generator=self.shuffle_generator)
            for start in range(0, batch_size, self.config.budgets.minibatch_size):
                indices = permutation[
                    start : start + self.config.budgets.minibatch_size
                ].to(self.device)
                logits, reward, failure, depletion, security, valid = self.policy(
                    observations[indices], sequences[indices]
                )
                policy_logits = logits / max(self.policy.temperature, 1e-6)
                log_probability, entropy = self.policy.plackett_luce_stats(
                    policy_logits, ordered[indices], valid
                )
                log_ratio = log_probability - old_log_probability[indices]
                ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    torch.abs(ratio - 1.0) > self.config.ppo.clip_epsilon
                ).float().mean()
                actor_step_allowed = bool(
                    actor_update_open
                    and float(approx_kl.item())
                    <= 1.5 * self.config.ppo.target_kl
                )
                if not actor_step_allowed:
                    actor_update_open = False
                unclipped = ratio * combined[indices]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.ppo.clip_epsilon,
                    1.0 + self.config.ppo.clip_epsilon,
                ) * combined[indices]
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                actor_objective = (
                    actor_loss
                    + self.kl_coefficient * approx_kl
                    - self.entropy_coefficient * entropy.mean()
                )
                if not actor_step_allowed:
                    actor_objective = 0.0 * actor_objective
                critic_losses = (
                    torch.nn.functional.smooth_l1_loss(
                        reward, returns[0][indices]
                    ),
                    torch.nn.functional.smooth_l1_loss(
                        failure, returns[1][indices]
                    ),
                    torch.nn.functional.smooth_l1_loss(
                        depletion, returns[2][indices]
                    ),
                    torch.nn.functional.smooth_l1_loss(
                        security, returns[3][indices]
                    ),
                )
                loss = actor_objective + CRITIC_LOSS_COEFFICIENT * sum(
                    critic_losses
                )
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                actor_gradient = torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(),
                    self.config.ppo.gradient_clip_norm,
                )
                critic_gradient = torch.nn.utils.clip_grad_norm_(
                    self.critic_parameters,
                    self.config.ppo.gradient_clip_norm,
                )
                if actor_step_allowed:
                    self.actor_optimizer.step()
                self.critic_optimizer.step()
                values = (
                    actor_loss,
                    *critic_losses,
                    entropy.mean(),
                    approx_kl,
                    clip_fraction,
                    actor_gradient,
                    critic_gradient,
                    float(actor_step_allowed),
                )
                for name, value in zip(names, values, strict=True):
                    metrics[name].append(
                        float(value.item()) if torch.is_tensor(value) else float(value)
                    )

        maximum_kl = float(np.max(metrics["approx_kl"]))
        if maximum_kl > self.config.ppo.target_kl:
            self.kl_coefficient = min(
                MAX_KL_COEFFICIENT, self.kl_coefficient * 1.6
            )
        elif maximum_kl < 0.5 * self.config.ppo.target_kl:
            self.kl_coefficient = max(
                MIN_KL_COEFFICIENT, self.kl_coefficient / 1.25
            )
        entropy_error = TARGET_PER_PICK_ENTROPY - self.last_rollout_entropy
        self.entropy_coefficient = float(
            np.clip(
                self.entropy_coefficient * np.exp(0.04 * entropy_error),
                MIN_ENTROPY_COEFFICIENT,
                MAX_ENTROPY_COEFFICIENT,
            )
        )
        summary = {
            name: float(np.mean(values)) for name, values in metrics.items()
        }
        summary.update(
            {
                "max_approx_kl": maximum_kl,
                "all_finite": bool(
                    all(np.isfinite(values).all() for values in metrics.values())
                ),
                "optimizer_minibatches": len(metrics["actor_loss"]),
                "actor_minibatches_skipped_by_kl_guard": int(
                    len(metrics["actor_step_allowed"])
                    - sum(metrics["actor_step_allowed"])
                ),
                "entropy_coefficient": self.entropy_coefficient,
                "kl_coefficient": self.kl_coefficient,
            }
        )
        return summary

    def train_fixed(
        self,
        *,
        updates: int,
        run_kind: str,
        output_directory: Path,
    ) -> dict[str, Any]:
        if {
            "smoke_2": 2,
            "qualification_8": 8,
            "formal_3000": 3_000,
        }.get(run_kind) != int(updates):
            raise RuntimeError("security-v4 run-kind budget mismatch")
        if output_directory.exists():
            raise RuntimeError("security-v4 output exists; overwrite/resume forbidden")
        output_directory.mkdir(parents=True)
        source = security_source_manifest(self.config)
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
        workers = PersistentRolloutWorkers(
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
                transitions_completed = (
                    completed * self.config.budgets.transitions_per_update
                )
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
                        (updates - completed)
                        * self.config.budgets.transitions_per_update
                        / max(throughput, 1e-12)
                    ),
                }
                progress.append(entry)
                atomic_write_json(
                    output_directory / "progress.json",
                    {
                        "schema": "qrelay-security-constrained-progress-4",
                        "role": "diagnostic_not_yet_heldout_evaluated",
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
            raise RuntimeError("security-v4 training ended below its exact budget")
        checkpoint_path = output_directory / (
            "ppo_final.pt" if run_kind == "formal_3000" else f"{run_kind}_final.pt"
        )
        torch.save(
            {
                "schema": "qrelay-security-constrained-checkpoint-4",
                "role": "diagnostic_candidate_requires_heldout_evaluation",
                "paper_evidence_eligible": False,
                "run_kind": run_kind,
                "seed": self.seed,
                "updates": updates,
                "transitions": updates
                * self.config.budgets.transitions_per_update,
                "policy_state_dict": self.policy.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "failure_dual": self.failure_dual,
                "depletion_dual": self.depletion_dual,
                "security_dual": self.security_dual,
                "public_security_tail_limit": PUBLIC_SECURITY_TAIL_LIMIT,
                "config": self.config.to_dict(),
                "source_manifest": source,
                "architecture_manifest": self.policy.architecture_manifest(),
                "deployment": {
                    "primary": "deterministic_public_risk_top_k",
                    "stochastic_audit": "exact_prefix_conditioned_Plackett_Luce",
                    "hidden_label_gate": False,
                    "public_admission_below_k": "minimum_risk_refill_exact_deficit",
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
        worker_receipts = tuple(
            hasher.receipt() for hasher in self.trajectory_hashers
        )
        elapsed = float(time.monotonic() - started)
        receipt = {
            "schema": "qrelay-security-constrained-completion-4",
            "status": "complete",
            "role": "diagnostic_candidate_requires_heldout_evaluation",
            "paper_evidence_eligible": False,
            "seed": self.seed,
            "run_kind": run_kind,
            "started_at": started_at,
            "completed_at": utc_now(),
            "updates_completed": updates,
            "transitions_completed": updates
            * self.config.budgets.transitions_per_update,
            "source_hash": source["source_hash"],
            "config_hash": source["config_hash"],
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "trajectory_hash": aggregate_trajectory_hash(worker_receipts),
            "worker_trajectory_receipts": worker_receipts,
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
            "mean_transitions_per_second": (
                updates * self.config.budgets.transitions_per_update
            )
            / max(elapsed, 1e-12),
            "runtime": runtime_manifest(),
            "primary_deployment_contract": "deterministic_public_risk_top_k",
            "stochastic_audit_contract": "exact_Plackett_Luce_temperature_0.55",
        }
        receipt["receipt_hash"] = canonical_hash(receipt)
        atomic_write_json(output_directory / "completion_receipt.json", receipt)
        return receipt


def _load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def run_registered(
    *, run_kind: str, seed: int, device: str = "cuda"
) -> dict[str, Any]:
    protocol = _load_protocol()
    expected = {
        "smoke_2": int(protocol["seeds"]["smoke"]),
        "qualification_8": int(protocol["seeds"]["qualification"]),
        "formal_3000": int(protocol["seeds"]["formal_training"]),
    }.get(run_kind)
    if expected is None or int(seed) != expected:
        raise RuntimeError("security-v4 run kind/seed is not preregistered")
    config = security_v4_config(training=True)
    source = security_source_manifest(config)
    if source["config_hash"] != protocol["frozen_contract"]["config_hash"]:
        raise RuntimeError("security-v4 config drifted after protocol freeze")
    claim_directory = RESULT_ROOT / "claims"
    claim_directory.mkdir(parents=True, exist_ok=True)
    claim_path = claim_directory / f"seed_{int(seed)}_{run_kind}_claim.json"
    if claim_path.exists():
        raise RuntimeError("security-v4 seed was already claimed")
    claim = {
        "schema": "qrelay-security-seed-claim-4",
        "seed": int(seed),
        "run_kind": run_kind,
        "claimed_at": utc_now(),
        "source_hash": source["source_hash"],
        "config_hash": source["config_hash"],
        "environment_constructed_before_claim": False,
    }
    claim["claim_hash"] = canonical_hash(claim)
    atomic_write_json(claim_path, claim)
    trainer = SecurityConstrainedPPOTrainer(seed=int(seed), device=device)
    updates = {"smoke_2": 2, "qualification_8": 8, "formal_3000": 3_000}[
        run_kind
    ]
    output = RESULT_ROOT / (
        f"seed_{int(seed)}"
        if run_kind == "formal_3000"
        else f"{run_kind}_seed_{int(seed)}"
    )
    return trainer.train_fixed(
        updates=updates, run_kind=run_kind, output_directory=output
    )
