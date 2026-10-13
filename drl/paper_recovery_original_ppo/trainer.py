"""Fixed-budget constrained PPO trainer for the reviewer-aligned direct actor."""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drl.paper_recovery.config import PaperRecoveryConfig
from drl.paper_recovery.env import PaperRecoveryEnv
from drl.paper_recovery.receipts import (
    TrajectoryHasher,
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
from drl.paper_recovery_original_ppo.policy import ReviewerAlignedDirectPPO


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "results/paper_recovery_original_ppo_v1"
PROTOCOL_PATH = REPO_ROOT / "experiments/paper_recovery_original_ppo_v1/protocol.json"
ACTOR_LR = 5e-5
CRITIC_LR = 3e-4
ENTROPY_COEFFICIENT = 0.01
CRITIC_LOSS_COEFFICIENT = 0.5
FAILURE_COST_LIMIT = 0.01
DEPLETION_COST_LIMIT = 0.02
DUAL_INITIAL = 0.1
DUAL_LEARNING_RATE = 0.02
DUAL_MAXIMUM = 10.0


def development_config() -> PaperRecoveryConfig:
    base = PaperRecoveryConfig()
    observations = replace(base.observations, temporal_delta_window=32)
    budgets = replace(
        base.budgets,
        smoke_updates=2,
        profiler_updates=64,
        formal_updates=300,
        epochs_per_update=4,
        minibatch_size=64,
        formal_transitions_per_seed=900_000,
    )
    ppo = replace(
        base.ppo,
        learning_rate=ACTOR_LR,
        clip_epsilon=0.20,
        entropy_coefficient_mean_per_pick=ENTROPY_COEFFICIENT,
        gradient_clip_norm=0.5,
    )
    return replace(
        base,
        observations=observations,
        budgets=budgets,
        ppo=ppo,
        training_mode=True,
        scenario_profile="training_primary",
        formal_training_authorized=False,
    )


def source_manifest(config: PaperRecoveryConfig) -> dict[str, Any]:
    paths = (
        REPO_ROOT / "drl/paper_recovery_original_ppo/policy.py",
        REPO_ROOT / "drl/paper_recovery_original_ppo/trainer.py",
        REPO_ROOT / "drl/paper_recovery/env.py",
        REPO_ROOT / "drl/paper_recovery/config.py",
        REPO_ROOT / "drl/paper_recovery/trainer.py",
        PROTOCOL_PATH,
    )
    files = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in paths
    }
    return {
        "schema": "qrelay-development-original-ppo-source-1",
        "files": files,
        "source_hash": canonical_hash(files),
        "config_hash": canonical_hash(config.to_dict()),
    }


@dataclass
class DirectTransition:
    observation: np.ndarray
    sequence: np.ndarray
    ordered_committee: np.ndarray
    old_log_probability: float
    reward_value: float
    failure_value: float
    depletion_value: float
    reward: float
    failure_cost: float
    depletion_cost: float
    done: bool
    reward_advantage: float = 0.0
    failure_advantage: float = 0.0
    depletion_advantage: float = 0.0
    reward_return: float = 0.0
    failure_return: float = 0.0
    depletion_return: float = 0.0


class ReviewerAlignedPPOTrainer:
    """Pure PPO plus independent public failure/depletion cost value functions."""

    def __init__(
        self,
        config: PaperRecoveryConfig,
        *,
        seed: int,
        device: str = "auto",
        environment_class: type[PaperRecoveryEnv] = PaperRecoveryEnv,
        run_budgets: dict[str, int] | None = None,
        maximum_updates: int = 300,
        source_manifest_builder: Any = source_manifest,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.environment_class = environment_class
        self.run_budgets = run_budgets or {
            "smoke": 2,
            "qualification": 64,
            "diagnostic_300": 300,
        }
        self.maximum_updates = int(maximum_updates)
        self.source_manifest_builder = source_manifest_builder
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        self.policy = ReviewerAlignedDirectPPO(config).to(self.device)
        critic_parameters = list(
            itertools.chain(
                self.policy.reward_critic.parameters(),
                self.policy.failure_cost_critic.parameters(),
                self.policy.depletion_cost_critic.parameters(),
            )
        )
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=ACTOR_LR
        )
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=CRITIC_LR)
        self.critic_parameters = critic_parameters
        generator_device = self.device.type if self.device.type in {"cpu", "cuda"} else "cpu"
        self.policy_generators: list[torch.Generator] = []
        for worker_id in range(config.budgets.persistent_workers):
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(self.seed + 8_100_003 + 65_537 * worker_id)
            self.policy_generators.append(generator)
        self.shuffle_generator = torch.Generator(device="cpu")
        self.shuffle_generator.manual_seed(self.seed + 9_100_001)
        self.trajectory_hashers = [
            TrajectoryHasher(worker_id=worker_id)
            for worker_id in range(config.budgets.persistent_workers)
        ]
        self.failure_dual = DUAL_INITIAL
        self.depletion_dual = DUAL_INITIAL

    def _sample_actions(
        self,
        observations: np.ndarray,
        sequences: np.ndarray,
        worker_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        seq = torch.as_tensor(sequences, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, reward, failure, depletion, masks = self.policy(obs, seq)
            eligible = masks.sum(dim=1)
            if torch.any(eligible < self.config.physics.committee_size):
                raise RuntimeError(
                    f"direct-PPO rollout failed closed below k: {eligible.cpu().tolist()}"
                )
            policy_logits = logits / max(self.policy.temperature, 1e-6)
            action_rows = []
            for row, worker_id in enumerate(worker_ids):
                uniforms = torch.rand(
                    (self.config.physics.num_nodes,),
                    dtype=torch.float64,
                    device=self.device,
                    generator=self.policy_generators[worker_id],
                ).clamp_(1e-12, 1.0 - 1e-12)
                gumbel = -torch.log(-torch.log(uniforms))
                action_rows.append(
                    (policy_logits[row].to(torch.float64) + gumbel).masked_fill(
                        ~masks[row], -torch.inf
                    )
                )
            action_scores = torch.stack(action_rows)
            ordered = torch.topk(
                action_scores,
                k=self.config.physics.committee_size,
                dim=1,
                sorted=True,
            ).indices
            log_probability, entropy = self.policy.plackett_luce_stats(
                policy_logits, ordered, masks
            )
            values = torch.stack((reward, failure, depletion), dim=1)
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
            _, reward, failure, depletion, _ = self.policy(
                torch.as_tensor(observations, dtype=torch.float32, device=self.device),
                torch.as_tensor(sequences, dtype=torch.float32, device=self.device),
            )
        return torch.stack((reward, failure, depletion), dim=1).cpu().numpy()

    @staticmethod
    def _assign_gae(
        records: list[DirectTransition],
        *,
        bootstrap: float,
        value_name: str,
        signal_name: str,
        advantage_name: str,
        return_name: str,
        gamma: float,
        lam: float,
    ) -> None:
        advantage = 0.0
        next_value = float(bootstrap)
        for transition in reversed(records):
            value = float(getattr(transition, value_name))
            signal = float(getattr(transition, signal_name))
            nonterminal = 0.0 if transition.done else 1.0
            delta = signal + gamma * next_value * nonterminal - value
            advantage = delta + gamma * lam * nonterminal * advantage
            setattr(transition, advantage_name, float(advantage))
            setattr(transition, return_name, float(advantage + value))
            next_value = value

    def collect_update(
        self,
        workers: PersistentRolloutWorkers,
        *,
        update_index: int,
    ) -> tuple[list[DirectTransition], dict[str, Any]]:
        quotas = worker_transition_quotas(update_index)
        counts = [0] * len(quotas)
        records: list[list[DirectTransition]] = [[] for _ in quotas]
        entropies: list[float] = []
        successes: list[float] = []
        tps: list[float] = []
        failures: list[float] = []
        depletions: list[float] = []
        while counts != list(quotas):
            active = [index for index, quota in enumerate(quotas) if counts[index] < quota]
            observations = np.stack([workers.observations[index] for index in active])
            sequences = workers.sequences(active)
            action_scores, ordered, log_probability, entropy, values = self._sample_actions(
                observations, sequences, active
            )
            messages = workers.step_many(active, action_scores)
            for row, worker_id in enumerate(active):
                message = messages[row]
                info = message["info"]
                if info["candidate_committee_ids"] != ordered[row].astype(int).tolist():
                    raise RuntimeError("direct-PPO sampled/executed committee order drift")
                if not info["candidate_executed_match"]:
                    raise RuntimeError("direct-PPO candidate/executed parity failure")
                if info["offline_security"]["hidden_labels_used_by_observation_or_reward"]:
                    raise RuntimeError("reviewer-forbidden hidden label entered control path")
                self.trajectory_hashers[worker_id].update(
                    info, episode_index=int(message["episode_index"])
                )
                failure_cost = float(not info["certificate_success"])
                depletion_cost = float(
                    info["energy_depleted_nodes"] / self.config.physics.num_nodes
                )
                records[worker_id].append(
                    DirectTransition(
                        observation=observations[row].copy(),
                        sequence=sequences[row].copy(),
                        ordered_committee=ordered[row].copy(),
                        old_log_probability=float(log_probability[row]),
                        reward_value=float(values[row, 0]),
                        failure_value=float(values[row, 1]),
                        depletion_value=float(values[row, 2]),
                        reward=float(message["reward"]),
                        failure_cost=failure_cost,
                        depletion_cost=depletion_cost,
                        done=bool(message["done"]),
                    )
                )
                counts[worker_id] += 1
                entropies.append(float(entropy[row]))
                successes.append(float(info["certificate_success"]))
                tps.append(float(info["served_tps"]))
                failures.append(failure_cost)
                depletions.append(depletion_cost)

        bootstrap = self._bootstrap_values(workers)
        gamma = self.config.ppo.gamma
        lam = self.config.ppo.gae_lambda
        flattened: list[DirectTransition] = []
        specifications = (
            (0, "reward_value", "reward", "reward_advantage", "reward_return"),
            (1, "failure_value", "failure_cost", "failure_advantage", "failure_return"),
            (2, "depletion_value", "depletion_cost", "depletion_advantage", "depletion_return"),
        )
        for worker_id, worker_records in enumerate(records):
            for column, value, signal, advantage, return_name in specifications:
                self._assign_gae(
                    worker_records,
                    bootstrap=float(bootstrap[worker_id, column]),
                    value_name=value,
                    signal_name=signal,
                    advantage_name=advantage,
                    return_name=return_name,
                    gamma=gamma,
                    lam=lam,
                )
            flattened.extend(worker_records)
        if len(flattened) != self.config.budgets.transitions_per_update:
            raise RuntimeError("direct-PPO transition budget drift")
        return flattened, {
            "update": int(update_index),
            "worker_quotas": list(quotas),
            "transition_count": len(flattened),
            "mean_pick_entropy": float(np.mean(entropies)),
            "certificate_success_rate": float(np.mean(successes)),
            "mean_served_tps": float(np.mean(tps)),
            "mean_failure_cost": float(np.mean(failures)),
            "mean_depletion_exposure": float(np.mean(depletions)),
        }

    def optimise_update(self, transitions: list[DirectTransition]) -> dict[str, Any]:
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
        reward_advantage = torch.as_tensor(
            [row.reward_advantage for row in transitions], dtype=torch.float32, device=self.device
        )
        failure_advantage = torch.as_tensor(
            [row.failure_advantage for row in transitions], dtype=torch.float32, device=self.device
        )
        depletion_advantage = torch.as_tensor(
            [row.depletion_advantage for row in transitions], dtype=torch.float32, device=self.device
        )
        combined_advantage = (
            reward_advantage
            - self.failure_dual * failure_advantage
            - self.depletion_dual * depletion_advantage
        )
        combined_advantage = (
            combined_advantage - combined_advantage.mean()
        ) / combined_advantage.std(unbiased=False).clamp_min(1e-8)
        returns = tuple(
            torch.as_tensor(
                [getattr(row, name) for row in transitions],
                dtype=torch.float32,
                device=self.device,
            )
            for name in ("reward_return", "failure_return", "depletion_return")
        )

        metrics: dict[str, list[float]] = {
            name: []
            for name in (
                "actor_loss",
                "reward_critic_loss",
                "failure_critic_loss",
                "depletion_critic_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "actor_gradient_norm",
                "critic_gradient_norm",
                "actor_step_allowed",
            )
        }
        batch_size = len(transitions)
        actor_update_open = True
        for _ in range(self.config.budgets.epochs_per_update):
            permutation = torch.randperm(batch_size, generator=self.shuffle_generator)
            for start in range(0, batch_size, self.config.budgets.minibatch_size):
                indices = permutation[
                    start : start + self.config.budgets.minibatch_size
                ].to(self.device)
                log_probability, entropy, reward, failure, depletion = (
                    self.policy.evaluate_ordered_committees(
                        observations[indices], sequences[indices], ordered[indices]
                    )
                )
                log_ratio = log_probability - old_log_probability[indices]
                ratio = torch.exp(log_ratio)
                with torch.no_grad():
                    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        torch.abs(ratio - 1.0) > self.config.ppo.clip_epsilon
                    ).float().mean()
                actor_step_allowed = bool(
                    actor_update_open
                    and float(approx_kl.item()) <= self.config.ppo.target_kl
                )
                if not actor_step_allowed:
                    actor_update_open = False
                unclipped = ratio * combined_advantage[indices]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.ppo.clip_epsilon,
                    1.0 + self.config.ppo.clip_epsilon,
                ) * combined_advantage[indices]
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                critic_losses = (
                    torch.nn.functional.smooth_l1_loss(reward, returns[0][indices]),
                    torch.nn.functional.smooth_l1_loss(failure, returns[1][indices]),
                    torch.nn.functional.smooth_l1_loss(depletion, returns[2][indices]),
                )
                actor_objective = actor_loss - ENTROPY_COEFFICIENT * entropy.mean()
                if not actor_step_allowed:
                    actor_objective = 0.0 * actor_objective
                loss = actor_objective + CRITIC_LOSS_COEFFICIENT * sum(critic_losses)
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                actor_gradient = torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), self.config.ppo.gradient_clip_norm
                )
                critic_gradient = torch.nn.utils.clip_grad_norm_(
                    self.critic_parameters, self.config.ppo.gradient_clip_norm
                )
                if actor_step_allowed:
                    self.actor_optimizer.step()
                self.critic_optimizer.step()
                for name, value in zip(
                    (
                        "actor_loss",
                        "reward_critic_loss",
                        "failure_critic_loss",
                        "depletion_critic_loss",
                        "entropy",
                        "approx_kl",
                        "clip_fraction",
                        "actor_gradient_norm",
                        "critic_gradient_norm",
                        "actor_step_allowed",
                    ),
                    (
                        actor_loss,
                        *critic_losses,
                        entropy.mean(),
                        approx_kl,
                        clip_fraction,
                        actor_gradient,
                        critic_gradient,
                        float(actor_step_allowed),
                    ),
                    strict=True,
                ):
                    metrics[name].append(
                        float(value.item()) if torch.is_tensor(value) else float(value)
                    )
        summary = {name: float(np.mean(values)) for name, values in metrics.items()}
        summary["max_approx_kl"] = float(np.max(metrics["approx_kl"]))
        summary["all_finite"] = bool(
            all(np.isfinite(values).all() for values in metrics.values())
        )
        summary["optimizer_minibatches"] = len(metrics["actor_loss"])
        summary["actor_minibatches_skipped_by_kl_guard"] = int(
            len(metrics["actor_step_allowed"])
            - sum(metrics["actor_step_allowed"])
        )
        return summary

    @staticmethod
    def _module_hash(module: torch.nn.Module) -> str:
        return canonical_hash(
            {
                name: value.detach().cpu().numpy().tolist()
                for name, value in module.named_parameters()
            }
        )

    def train(
        self,
        *,
        updates: int,
        run_kind: str,
        output_directory: Path,
    ) -> dict[str, Any]:
        expected = self.run_budgets
        if run_kind not in expected or int(updates) != expected[run_kind]:
            raise ValueError("direct-PPO run kind/update budget mismatch")
        if int(updates) > self.maximum_updates:
            raise RuntimeError("direct-PPO run exceeds its registered update cap")
        if output_directory.exists():
            raise RuntimeError("direct-PPO output exists; overwrite forbidden")
        output_directory.mkdir(parents=True)
        source = self.source_manifest_builder(self.config)
        started_at = utc_now()
        started = time.monotonic()
        initial_hashes = {
            "actor": self._module_hash(self.policy.actor),
            "reward_critic": self._module_hash(self.policy.reward_critic),
            "failure_critic": self._module_hash(self.policy.failure_cost_critic),
            "depletion_critic": self._module_hash(self.policy.depletion_cost_critic),
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
                progress.append(
                    {
                        **rollout_metrics,
                        **optimiser_metrics,
                        "collect_seconds": collect_seconds,
                        "optimise_seconds": optimise_seconds,
                        "failure_dual": self.failure_dual,
                        "depletion_dual": self.depletion_dual,
                        "elapsed_seconds": float(time.monotonic() - started),
                    }
                )
                atomic_write_json(
                    output_directory / "progress.json",
                    {
                        "schema": "qrelay-development-original-ppo-progress-1",
                        "role": "development_only_not_for_manuscript",
                        "seed": self.seed,
                        "run_kind": run_kind,
                        "updates_expected": updates,
                        "updates_completed": update_index + 1,
                        "entries": progress,
                    },
                )
        finally:
            close_receipt = workers.close()

        checkpoint_path = output_directory / f"{run_kind}_development.pt"
        torch.save(
            {
                "schema": "qrelay-development-original-ppo-checkpoint-1",
                "role": "development_only_not_for_manuscript",
                "paper_evidence_eligible": False,
                "run_kind": run_kind,
                "seed": self.seed,
                "updates": updates,
                "transitions": updates * self.config.budgets.transitions_per_update,
                "policy_state_dict": self.policy.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "failure_dual": self.failure_dual,
                "depletion_dual": self.depletion_dual,
                "config": self.config.to_dict(),
                "source_manifest": source,
                "architecture_manifest": self.policy.architecture_manifest(),
                "resumed": False,
                "selected_by_evaluation": False,
            },
            checkpoint_path,
        )
        final_hashes = {
            "actor": self._module_hash(self.policy.actor),
            "reward_critic": self._module_hash(self.policy.reward_critic),
            "failure_critic": self._module_hash(self.policy.failure_cost_critic),
            "depletion_critic": self._module_hash(self.policy.depletion_cost_critic),
        }
        updates_by_module = {
            name: bool(final_hashes[name] != initial_hashes[name]) for name in initial_hashes
        }
        worker_receipts = tuple(hasher.receipt() for hasher in self.trajectory_hashers)
        receipt = {
            "schema": "qrelay-development-original-ppo-completion-1",
            "status": "complete",
            "role": "development_only_not_for_manuscript",
            "paper_evidence_eligible": False,
            "seed": self.seed,
            "run_kind": run_kind,
            "started_at": started_at,
            "completed_at": utc_now(),
            "updates_completed": updates,
            "transitions_completed": updates * self.config.budgets.transitions_per_update,
            "source_hash": source["source_hash"],
            "config_hash": source["config_hash"],
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "trajectory_hash": aggregate_trajectory_hash(worker_receipts),
            "worker_runtime": close_receipt,
            "initial_module_hashes": initial_hashes,
            "final_module_hashes": final_hashes,
            "modules_updated": updates_by_module,
            "actor_and_all_critics_updated": bool(all(updates_by_module.values())),
            "all_updates_finite": bool(all(row["all_finite"] for row in progress)),
            "maximum_observed_kl": float(max(row["max_approx_kl"] for row in progress)),
            "final_failure_dual": self.failure_dual,
            "final_depletion_dual": self.depletion_dual,
            "elapsed_seconds": float(time.monotonic() - started),
            "runtime": runtime_manifest(),
        }
        receipt["receipt_hash"] = canonical_hash(receipt)
        atomic_write_json(output_directory / "completion_receipt.json", receipt)
        return receipt


def run_development(
    *,
    run_kind: str,
    seed: int,
    device: str = "auto",
) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    expected_seed = {
        "smoke": protocol["seeds"].get(
            "smoke_kl_stability",
            protocol["seeds"].get(
                "smoke_replacement_after_masked_nan",
                protocol["seeds"].get(
                    "smoke_replacement_after_spent_oom", protocol["seeds"]["smoke"]
                ),
            ),
        ),
        "qualification": protocol["seeds"]["qualification_training"],
        "diagnostic_300": protocol["seeds"]["conditional_300"],
    }.get(run_kind)
    if expected_seed is None or int(seed) != int(expected_seed):
        raise RuntimeError("direct-PPO run kind/seed is not preregistered")
    if run_kind == "diagnostic_300":
        gate_path = RESULT_ROOT / "qualification_gate.json"
        if not gate_path.exists():
            raise RuntimeError("direct-PPO 300-update run requires qualification gate")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") != "PASS":
            raise RuntimeError("direct-PPO qualification gate did not pass")
    updates = {"smoke": 2, "qualification": 64, "diagnostic_300": 300}[run_kind]
    config = development_config()
    trainer = ReviewerAlignedPPOTrainer(config, seed=int(seed), device=device)
    return trainer.train(
        updates=updates,
        run_kind=run_kind,
        output_directory=RESULT_ROOT / f"{run_kind}_seed_{int(seed)}",
    )
