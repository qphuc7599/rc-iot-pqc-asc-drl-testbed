"""Causally corrected constrained-PPO trainer for compound IoT collapse."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drl.paper_recovery.config import PaperRecoveryConfig
from drl.paper_recovery.receipts import canonical_hash, sha256_file
from drl.paper_recovery.trainer import PersistentRolloutWorkers, worker_transition_quotas
from drl.paper_recovery_multihazard.env import CompoundCollapseEnv
from drl.paper_recovery_original_ppo.trainer import ReviewerAlignedPPOTrainer

from .config import multihazard_v2_config
from .policy import CausalDirectPPO


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "experiments/paper_recovery_multihazard_v2/protocol.json"
RESULT_ROOT = REPO_ROOT / "results/paper_recovery_multihazard_v2"
ACTOR_LR = 1e-4
CRITIC_LR = 3e-4
ENTROPY_COEFFICIENT = 0.01
CRITIC_LOSS_COEFFICIENT = 0.5
PUBLIC_MEMBER_CREDIT_COEFFICIENT = 0.15


def controllable_failure_cost(info: dict[str, Any]) -> float:
    """Do not charge the actor for a registered exogenous gateway blackout."""

    total_failure = not bool(info["certificate_success"])
    forced = bool(info["compound_collapse"]["forced_gateway_outage_this_step"])
    return float(total_failure and not forced)


def public_member_quality(
    info: dict[str, Any], ordered_committee: np.ndarray
) -> np.ndarray:
    """Public post-action response signal used only for within-list credit."""

    closing = set(int(node) for node in info["closing_certificate_node_ids"])
    responded = set(int(node) for node in info["responded_node_ids"])
    quality = np.asarray(
        [1.0 if int(node) in closing else 0.5 if int(node) in responded else 0.0
         for node in ordered_committee],
        dtype=np.float32,
    )
    return quality - np.float32(np.mean(quality))


def v2_source_manifest(config: PaperRecoveryConfig) -> dict[str, Any]:
    paths = (
        REPO_ROOT / "drl/paper_recovery/config.py",
        REPO_ROOT / "drl/paper_recovery/env.py",
        REPO_ROOT / "drl/paper_recovery/trainer.py",
        REPO_ROOT / "drl/paper_recovery_original_ppo/policy.py",
        REPO_ROOT / "drl/paper_recovery_original_ppo/trainer.py",
        REPO_ROOT / "drl/paper_recovery_multihazard/config.py",
        REPO_ROOT / "drl/paper_recovery_multihazard/env.py",
        REPO_ROOT / "drl/paper_recovery_multihazard_v2/config.py",
        REPO_ROOT / "drl/paper_recovery_multihazard_v2/policy.py",
        REPO_ROOT / "drl/paper_recovery_multihazard_v2/trainer.py",
        PROTOCOL_PATH,
    )
    files = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in paths
    }
    return {
        "schema": "qrelay-causal-direct-ppo-multihazard-source-2",
        "files": files,
        "source_hash": canonical_hash(files),
        "config_hash": canonical_hash(config.to_dict()),
    }


@dataclass
class CausalTransition:
    observation: np.ndarray
    sequence: np.ndarray
    ordered_committee: np.ndarray
    centered_member_quality: np.ndarray
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


class CausalPPOTrainer(ReviewerAlignedPPOTrainer):
    """V1 runner with a corrected actor/critic graph and public list credit."""

    def __init__(self, config: PaperRecoveryConfig, *, seed: int, device: str = "auto"):
        budgets = {"smoke": 2, "qualification_500": 500, "full_budget_3000": 3_000}
        super().__init__(
            config,
            seed=seed,
            device=device,
            environment_class=CompoundCollapseEnv,
            run_budgets=budgets,
            maximum_updates=3_000,
            source_manifest_builder=v2_source_manifest,
        )
        # Re-seeding isolates v2 initialization from the discarded v1 module.
        torch.manual_seed(self.seed + 2_000_003)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed + 2_000_003)
        self.policy = CausalDirectPPO(config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=ACTOR_LR
        )
        self.critic_parameters = list(
            itertools.chain(
                self.policy.reward_critic.parameters(),
                self.policy.failure_cost_critic.parameters(),
                self.policy.depletion_cost_critic.parameters(),
            )
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_parameters, lr=CRITIC_LR
        )

    def collect_update(
        self, workers: PersistentRolloutWorkers, *, update_index: int
    ) -> tuple[list[CausalTransition], dict[str, Any]]:
        quotas = worker_transition_quotas(update_index)
        counts = [0] * len(quotas)
        records: list[list[CausalTransition]] = [[] for _ in quotas]
        entropies: list[float] = []
        successes: list[float] = []
        tps: list[float] = []
        controllable_failures: list[float] = []
        total_failures: list[float] = []
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
                    raise RuntimeError("causal PPO sampled/executed committee drift")
                if not info["candidate_executed_match"]:
                    raise RuntimeError("causal PPO candidate/executed parity failure")
                if info["offline_security"]["hidden_labels_used_by_observation_or_reward"]:
                    raise RuntimeError("reviewer-forbidden hidden label entered control path")
                self.trajectory_hashers[worker_id].update(
                    info, episode_index=int(message["episode_index"])
                )
                failure = controllable_failure_cost(info)
                total_failure = float(not info["certificate_success"])
                depletion = float(
                    info["energy_depleted_nodes"] / self.config.physics.num_nodes
                )
                records[worker_id].append(
                    CausalTransition(
                        observation=observations[row].copy(),
                        sequence=sequences[row].copy(),
                        ordered_committee=ordered[row].copy(),
                        centered_member_quality=public_member_quality(info, ordered[row]),
                        old_log_probability=float(log_probability[row]),
                        reward_value=float(values[row, 0]),
                        failure_value=float(values[row, 1]),
                        depletion_value=float(values[row, 2]),
                        reward=float(message["reward"]),
                        failure_cost=failure,
                        depletion_cost=depletion,
                        done=bool(message["done"]),
                    )
                )
                counts[worker_id] += 1
                entropies.append(float(entropy[row]))
                successes.append(float(info["certificate_success"]))
                tps.append(float(info["served_tps"]))
                controllable_failures.append(failure)
                total_failures.append(total_failure)
                depletions.append(depletion)

        bootstrap = self._bootstrap_values(workers)
        specifications = (
            (0, "reward_value", "reward", "reward_advantage", "reward_return"),
            (1, "failure_value", "failure_cost", "failure_advantage", "failure_return"),
            (2, "depletion_value", "depletion_cost", "depletion_advantage", "depletion_return"),
        )
        flattened: list[CausalTransition] = []
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
            raise RuntimeError("causal PPO transition budget drift")
        return flattened, {
            "update": int(update_index),
            "worker_quotas": list(quotas),
            "transition_count": len(flattened),
            "mean_pick_entropy": float(np.mean(entropies)),
            "certificate_success_rate": float(np.mean(successes)),
            "mean_served_tps": float(np.mean(tps)),
            "mean_failure_cost": float(np.mean(controllable_failures)),
            "mean_total_failure_rate": float(np.mean(total_failures)),
            "mean_depletion_exposure": float(np.mean(depletions)),
        }

    def optimise_update(self, transitions: list[CausalTransition]) -> dict[str, Any]:
        observations = torch.as_tensor(
            np.stack([row.observation for row in transitions]),
            dtype=torch.float32, device=self.device,
        )
        sequences = torch.as_tensor(
            np.stack([row.sequence for row in transitions]),
            dtype=torch.float32, device=self.device,
        )
        ordered = torch.as_tensor(
            np.stack([row.ordered_committee for row in transitions]),
            dtype=torch.long, device=self.device,
        )
        member_quality = torch.as_tensor(
            np.stack([row.centered_member_quality for row in transitions]),
            dtype=torch.float32, device=self.device,
        )
        old_log_probability = torch.as_tensor(
            [row.old_log_probability for row in transitions],
            dtype=torch.float32, device=self.device,
        )
        advantages = tuple(
            torch.as_tensor(
                [getattr(row, name) for row in transitions],
                dtype=torch.float32, device=self.device,
            )
            for name in ("reward_advantage", "failure_advantage", "depletion_advantage")
        )
        combined = advantages[0] - self.failure_dual * advantages[1] - self.depletion_dual * advantages[2]
        combined = (combined - combined.mean()) / combined.std(unbiased=False).clamp_min(1e-8)
        returns = tuple(
            torch.as_tensor(
                [getattr(row, name) for row in transitions],
                dtype=torch.float32, device=self.device,
            )
            for name in ("reward_return", "failure_return", "depletion_return")
        )
        names = (
            "actor_loss", "member_credit_loss", "reward_critic_loss",
            "failure_critic_loss", "depletion_critic_loss", "entropy",
            "approx_kl", "clip_fraction", "actor_gradient_norm",
            "critic_gradient_norm", "actor_step_allowed",
        )
        metrics: dict[str, list[float]] = {name: [] for name in names}
        batch_size = len(transitions)
        actor_update_open = True
        for _ in range(self.config.budgets.epochs_per_update):
            permutation = torch.randperm(batch_size, generator=self.shuffle_generator)
            for start in range(0, batch_size, self.config.budgets.minibatch_size):
                indices = permutation[start : start + self.config.budgets.minibatch_size].to(self.device)
                logits, reward, failure, depletion, valid = self.policy(
                    observations[indices], sequences[indices]
                )
                policy_logits = logits / max(self.policy.temperature, 1e-6)
                log_probability, entropy = self.policy.plackett_luce_stats(
                    policy_logits, ordered[indices], valid
                )
                log_ratio = log_probability - old_log_probability[indices]
                ratio = torch.exp(log_ratio)
                with torch.no_grad():
                    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        torch.abs(ratio - 1.0) > self.config.ppo.clip_epsilon
                    ).float().mean()
                actor_step_allowed = bool(
                    actor_update_open and float(approx_kl.item()) <= self.config.ppo.target_kl
                )
                if not actor_step_allowed:
                    actor_update_open = False
                unclipped = ratio * combined[indices]
                clipped = torch.clamp(
                    ratio, 1.0 - self.config.ppo.clip_epsilon,
                    1.0 + self.config.ppo.clip_epsilon,
                ) * combined[indices]
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                selected_logits = policy_logits.gather(1, ordered[indices])
                member_credit_loss = -(
                    member_quality[indices] * selected_logits
                ).mean()
                actor_objective = (
                    actor_loss
                    + PUBLIC_MEMBER_CREDIT_COEFFICIENT * member_credit_loss
                    - ENTROPY_COEFFICIENT * entropy.mean()
                )
                if not actor_step_allowed:
                    actor_objective = 0.0 * actor_objective
                critic_losses = (
                    torch.nn.functional.smooth_l1_loss(reward, returns[0][indices]),
                    torch.nn.functional.smooth_l1_loss(failure, returns[1][indices]),
                    torch.nn.functional.smooth_l1_loss(depletion, returns[2][indices]),
                )
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
                values = (
                    actor_loss, member_credit_loss, *critic_losses, entropy.mean(),
                    approx_kl, clip_fraction, actor_gradient, critic_gradient,
                    float(actor_step_allowed),
                )
                for name, value in zip(names, values, strict=True):
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
            len(metrics["actor_step_allowed"]) - sum(metrics["actor_step_allowed"])
        )
        return summary


def run_v2(*, run_kind: str, seed: int, device: str = "auto") -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    expected_seed: int | tuple[int, ...] | None = {
        "smoke": protocol["seeds"]["smoke"],
        "qualification_500": protocol["seeds"]["qualification_training"],
        "full_budget_3000": tuple(
            [protocol["seeds"]["conditional_full_training"]]
            + protocol["seeds"].get("conditional_full_replacements", [])
        ),
    }.get(run_kind)
    allowed = expected_seed if isinstance(expected_seed, tuple) else (expected_seed,)
    if expected_seed is None or int(seed) not in tuple(int(value) for value in allowed):
        raise RuntimeError("PPO-v2 run kind/seed is not preregistered")
    if run_kind == "full_budget_3000":
        gate_path = RESULT_ROOT / "qualification_gate.json"
        if not gate_path.exists():
            raise RuntimeError("PPO-v2 full run requires qualification gate")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") != "PASS":
            raise RuntimeError("PPO-v2 qualification gate did not pass")
        primary_seed = int(protocol["seeds"]["conditional_full_training"])
        if int(seed) != primary_seed:
            aborted = RESULT_ROOT / f"full_budget_3000_seed_{primary_seed}" / "ABORTED.md"
            if not aborted.exists():
                raise RuntimeError("replacement seed requires primary infrastructure audit")
    budgets = {"smoke": 2, "qualification_500": 500, "full_budget_3000": 3_000}
    trainer = CausalPPOTrainer(
        multihazard_v2_config(training=True), seed=int(seed), device=device
    )
    return trainer.train(
        updates=budgets[run_kind],
        run_kind=run_kind,
        output_directory=RESULT_ROOT / f"{run_kind}_seed_{int(seed)}",
    )
