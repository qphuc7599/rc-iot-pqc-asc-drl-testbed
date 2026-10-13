"""Stable constrained-PPO trainer for the controllable-cascade study."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drl.paper_recovery.config import PaperRecoveryConfig
from drl.paper_recovery.receipts import canonical_hash, sha256_file
from drl.paper_recovery.trainer import PersistentRolloutWorkers, worker_transition_quotas
from drl.paper_recovery_multihazard_v2.trainer import (
    CausalTransition,
    public_member_quality,
)
from drl.paper_recovery_original_ppo.trainer import ReviewerAlignedPPOTrainer

from .config import longhorizon_config
from .env import LongHorizonCommitteeEnv
from .policy import LongHorizonDirectPPO


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "experiments/paper_recovery_longhorizon_v3/protocol.json"
RESULT_ROOT = REPO_ROOT / "results/paper_recovery_longhorizon_v3"
ACTOR_LR = 3e-5
CRITIC_LR = 2e-4
CRITIC_LOSS_COEFFICIENT = 0.5
INITIAL_ENTROPY_COEFFICIENT = 0.02
MIN_ENTROPY_COEFFICIENT = 0.005
MAX_ENTROPY_COEFFICIENT = 0.12
TARGET_PER_PICK_ENTROPY = 2.50
INITIAL_KL_COEFFICIENT = 0.10
MIN_KL_COEFFICIENT = 0.01
MAX_KL_COEFFICIENT = 20.0
MEMBER_CREDIT_INITIAL = 0.04
MEMBER_CREDIT_DECAY_UPDATES = 200


def source_manifest(config: PaperRecoveryConfig) -> dict[str, Any]:
    paths = (
        REPO_ROOT / "drl/paper_recovery/config.py",
        REPO_ROOT / "drl/paper_recovery/env.py",
        REPO_ROOT / "drl/paper_recovery/trainer.py",
        REPO_ROOT / "drl/paper_recovery/baselines.py",
        REPO_ROOT / "drl/paper_recovery_original_ppo/policy.py",
        REPO_ROOT / "drl/paper_recovery_original_ppo/trainer.py",
        REPO_ROOT / "drl/paper_recovery_longhorizon_v3/config.py",
        REPO_ROOT / "drl/paper_recovery_longhorizon_v3/baselines.py",
        REPO_ROOT / "drl/paper_recovery_longhorizon_v3/env.py",
        REPO_ROOT / "drl/paper_recovery_longhorizon_v3/policy.py",
        REPO_ROOT / "drl/paper_recovery_longhorizon_v3/trainer.py",
        PROTOCOL_PATH,
    )
    files = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in paths
    }
    return {
        "schema": "qrelay-reviewer-longhorizon-source-3",
        "files": files,
        "source_hash": canonical_hash(files),
        "config_hash": canonical_hash(config.to_dict()),
    }


def total_failure_cost(info: dict[str, Any]) -> float:
    """All failures are controllable because this environment has no hard blackout."""

    if info["long_horizon"]["forced_zero_tps"]:
        raise RuntimeError("long-horizon protocol forbids forced zero-TPS intervals")
    return float(not info["certificate_success"])


class LongHorizonPPOTrainer(ReviewerAlignedPPOTrainer):
    """Direct PPO with bounded exploration and decaying public list credit."""

    def __init__(
        self, config: PaperRecoveryConfig, *, seed: int, device: str = "auto"
    ):
        budgets = {
            "smoke": 2,
            "qualification_500": 500,
            "conditional_1000": 1_000,
            "full_budget_3000": 3_000,
        }
        super().__init__(
            config,
            seed=seed,
            device=device,
            environment_class=LongHorizonCommitteeEnv,
            run_budgets=budgets,
            maximum_updates=3_000,
            source_manifest_builder=source_manifest,
        )
        torch.manual_seed(self.seed + 3_000_017)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed + 3_000_017)
        self.policy = LongHorizonDirectPPO(config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=ACTOR_LR, eps=1e-5
        )
        self.critic_parameters = list(
            itertools.chain(
                self.policy.reward_critic.parameters(),
                self.policy.failure_cost_critic.parameters(),
                self.policy.depletion_cost_critic.parameters(),
            )
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_parameters, lr=CRITIC_LR, eps=1e-5
        )
        self.entropy_coefficient = INITIAL_ENTROPY_COEFFICIENT
        self.kl_coefficient = INITIAL_KL_COEFFICIENT
        self.current_update = 0
        self.last_rollout_entropy = 0.0

    def collect_update(
        self, workers: PersistentRolloutWorkers, *, update_index: int
    ) -> tuple[list[CausalTransition], dict[str, Any]]:
        self.current_update = int(update_index)
        quotas = worker_transition_quotas(update_index)
        counts = [0] * len(quotas)
        records: list[list[CausalTransition]] = [[] for _ in quotas]
        entropies: list[float] = []
        successes: list[float] = []
        tps: list[float] = []
        failures: list[float] = []
        depletions: list[float] = []
        backlog: list[float] = []
        reserve: list[float] = []
        while counts != list(quotas):
            active = [
                index for index, quota in enumerate(quotas)
                if counts[index] < quota
            ]
            observations = np.stack(
                [workers.observations[index] for index in active]
            )
            sequences = workers.sequences(active)
            action_scores, ordered, log_probability, entropy, values = (
                self._sample_actions(observations, sequences, active)
            )
            messages = workers.step_many(active, action_scores)
            for row, worker_id in enumerate(active):
                message = messages[row]
                info = message["info"]
                if info["candidate_committee_ids"] != ordered[row].astype(int).tolist():
                    raise RuntimeError("long-horizon PPO sampled/executed committee drift")
                if not info["candidate_executed_match"]:
                    raise RuntimeError("long-horizon candidate/executed parity failure")
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
                records[worker_id].append(
                    CausalTransition(
                        observation=observations[row].copy(),
                        sequence=sequences[row].copy(),
                        ordered_committee=ordered[row].copy(),
                        centered_member_quality=public_member_quality(
                            info, ordered[row]
                        ),
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
                failures.append(failure)
                depletions.append(depletion)
                backlog.append(float(info["long_horizon"]["backlog_normalized"]))
                reserve.append(float(info["long_horizon"]["network_reserve"]))

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
            raise RuntimeError("long-horizon transition budget drift")
        self.last_rollout_entropy = float(np.mean(entropies))
        return flattened, {
            "update": int(update_index),
            "worker_quotas": list(quotas),
            "transition_count": len(flattened),
            "mean_pick_entropy": self.last_rollout_entropy,
            "certificate_success_rate": float(np.mean(successes)),
            "mean_served_tps": float(np.mean(tps)),
            "mean_failure_cost": float(np.mean(failures)),
            "mean_total_failure_rate": float(np.mean(failures)),
            "mean_depletion_exposure": float(np.mean(depletions)),
            "mean_backlog_normalized": float(np.mean(backlog)),
            "mean_network_reserve": float(np.mean(reserve)),
        }

    def _member_credit_coefficient(self) -> float:
        remaining = max(
            0.0,
            1.0 - self.current_update / float(MEMBER_CREDIT_DECAY_UPDATES),
        )
        return float(MEMBER_CREDIT_INITIAL * remaining)

    def optimise_update(self, transitions: list[CausalTransition]) -> dict[str, Any]:
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
        member_quality = torch.as_tensor(
            np.stack([row.centered_member_quality for row in transitions]),
            dtype=torch.float32,
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
            )
        )
        combined = (
            advantages[0]
            - self.failure_dual * advantages[1]
            - self.depletion_dual * advantages[2]
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
            for name in ("reward_return", "failure_return", "depletion_return")
        )
        names = (
            "actor_loss",
            "member_credit_loss",
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
        metrics: dict[str, list[float]] = {name: [] for name in names}
        actor_update_open = True
        credit_coefficient = self._member_credit_coefficient()
        batch_size = len(transitions)
        for _ in range(self.config.budgets.epochs_per_update):
            permutation = torch.randperm(
                batch_size, generator=self.shuffle_generator
            )
            for start in range(0, batch_size, self.config.budgets.minibatch_size):
                indices = permutation[
                    start : start + self.config.budgets.minibatch_size
                ].to(self.device)
                logits, reward, failure, depletion, valid = self.policy(
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
                with torch.no_grad():
                    observed_kl = float(approx_kl.item())
                actor_step_allowed = bool(
                    actor_update_open
                    and observed_kl <= 1.5 * self.config.ppo.target_kl
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
                selected_logits = policy_logits.gather(1, ordered[indices])
                member_credit_loss = -(
                    member_quality[indices] * selected_logits
                ).mean()
                actor_objective = (
                    actor_loss
                    + credit_coefficient * member_credit_loss
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
                    member_credit_loss,
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
        summary = {name: float(np.mean(values)) for name, values in metrics.items()}
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
                "member_credit_coefficient": credit_coefficient,
            }
        )
        return summary


def run_longhorizon(
    *, run_kind: str, seed: int, device: str = "auto"
) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    expected = {
        "smoke": protocol["seeds"]["smoke"],
        "qualification_500": protocol["seeds"]["qualification_500_training"],
        "conditional_1000": protocol["seeds"]["conditional_1000_training"],
        "full_budget_3000": protocol["seeds"]["conditional_full_training"],
    }.get(run_kind)
    if expected is None or int(seed) != int(expected):
        raise RuntimeError("long-horizon run kind/seed is not preregistered")
    if run_kind == "qualification_500":
        gate_path = RESULT_ROOT / "baseline_headroom_gate.json"
        if not gate_path.exists():
            raise RuntimeError("qualification requires the baseline-headroom gate")
        headroom = json.loads(gate_path.read_text(encoding="utf-8"))
        if headroom.get("status") != "PASS":
            raise RuntimeError("baseline-headroom gate did not pass")
    if run_kind == "conditional_1000":
        gate_path = RESULT_ROOT / "qualification_500_gate.json"
        if not gate_path.exists():
            raise RuntimeError("1000-update run requires the frozen 500 gate")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("advance_to_1000") is not True:
            raise RuntimeError("500-update evidence did not authorize 1000 updates")
    if run_kind == "full_budget_3000":
        raise RuntimeError(
            "formal/full 3000 remains locked; this development namespace only "
            "qualifies 500 or prospectively 1000 updates"
        )
    config = longhorizon_config(training=True)
    trainer = LongHorizonPPOTrainer(config, seed=int(seed), device=device)
    updates = {"smoke": 2, "qualification_500": 500, "conditional_1000": 1_000}[
        run_kind
    ]
    return trainer.train(
        updates=updates,
        run_kind=run_kind,
        output_directory=RESULT_ROOT / f"{run_kind}_seed_{int(seed)}",
    )
