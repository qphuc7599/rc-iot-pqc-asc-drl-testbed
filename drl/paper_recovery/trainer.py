"""Fixed-budget PPO runner with eleven persistent response-engine workers."""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import PaperRecoveryConfig
from .env import PaperRecoveryEnv
from .policy import ResidualPPOPolicy, TemporalObservationBuffer
from .receipts import (
    RESULT_ROOT,
    TrainingReceipt,
    TrajectoryHasher,
    aggregate_trajectory_hash,
    assert_formal_runtime_authorized,
    atomic_write_json,
    canonical_hash,
    claim_formal_seed_before_environment,
    recovery_source_manifest,
    runtime_manifest,
    sha256_file,
    update_formal_seed_status,
    utc_now,
)


def worker_transition_quotas(update_index: int, workers: int = 11) -> tuple[int, ...]:
    """Rotate eight 273-step and three 272-step slices without changing 3,000."""

    if workers != 11:
        raise ValueError("the frozen contract requires exactly eleven workers")
    large_start = (int(update_index) * 8) % workers
    large = {(large_start + offset) % workers for offset in range(8)}
    quotas = tuple(273 if worker in large else 272 for worker in range(workers))
    if sum(quotas) != 3_000 or quotas.count(273) != 8 or quotas.count(272) != 3:
        raise RuntimeError("ragged rollout allocation drift")
    return quotas


def _worker_main(
    connection: Any,
    config: PaperRecoveryConfig,
    worker_id: int,
    base_seed: int,
    environment_class: type[PaperRecoveryEnv] = PaperRecoveryEnv,
) -> None:
    """Own one environment for the lifetime of a smoke/profiler/formal run."""

    try:
        initial_seed = int(base_seed) + 1_009 * int(worker_id)
        env = environment_class(config)
        episode_index = 0
        observation, reset_info = env.reset(seed=initial_seed)
        connection.send(
            {
                "kind": "ready",
                "pid": os.getpid(),
                "worker_id": int(worker_id),
                "episode_index": episode_index,
                "observation": observation,
                "reset_info": reset_info,
                "environment_manifest": env.model_manifest(),
            }
        )
        while True:
            command = connection.recv()
            if command["kind"] == "step":
                stepped_episode = episode_index
                observation, reward, terminated, truncated, info = env.step(
                    command["action_scores"]
                )
                done = bool(terminated or truncated)
                next_reset_info = None
                if done:
                    episode_index += 1
                    episode_seed = initial_seed + 1_000_003 * episode_index
                    observation, next_reset_info = env.reset(seed=episode_seed)
                connection.send(
                    {
                        "kind": "transition",
                        "worker_id": int(worker_id),
                        "episode_index": int(stepped_episode),
                        "observation": observation,
                        "reward": float(reward),
                        "done": done,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "info": info,
                        "reset_info": next_reset_info,
                    }
                )
            elif command["kind"] == "close":
                connection.send({"kind": "closed", "worker_id": int(worker_id)})
                break
            else:
                raise RuntimeError(f"unknown worker command: {command['kind']}")
    except BaseException as error:  # propagate process failures to the controller
        try:
            connection.send(
                {
                    "kind": "error",
                    "worker_id": int(worker_id),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
        finally:
            raise
    finally:
        connection.close()


class PersistentRolloutWorkers:
    """Eleven spawned workers retained across every PPO update."""

    def __init__(
        self,
        config: PaperRecoveryConfig,
        *,
        seed: int,
        environment_class: type[PaperRecoveryEnv] = PaperRecoveryEnv,
    ):
        count = config.budgets.persistent_workers
        if count != 11:
            raise ValueError("the paper-recovery runtime requires exactly 11 workers")
        context = mp.get_context("spawn")
        self.connections: list[Any] = []
        self.processes: list[mp.Process] = []
        self.observations: list[np.ndarray] = []
        self.episode_indices = [0] * count
        self.histories = [
            TemporalObservationBuffer(config.observations.temporal_delta_window)
            for _ in range(count)
        ]
        self.worker_manifests: list[dict[str, Any]] = []
        self.pids: list[int] = []
        for worker_id in range(count):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker_main,
                args=(child, config, worker_id, int(seed), environment_class),
                daemon=False,
                name=f"qrelay-recovery-worker-{worker_id}",
            )
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        ready = [connection.recv() for connection in self.connections]
        for expected, message in enumerate(ready):
            if message.get("kind") != "ready" or message.get("worker_id") != expected:
                self.close(force=True)
                raise RuntimeError(f"worker {expected} failed startup: {message}")
            observation = np.asarray(message["observation"], dtype=np.float32)
            self.observations.append(observation)
            self.histories[expected].reset(observation)
            self.worker_manifests.append(message["environment_manifest"])
            self.pids.append(int(message["pid"]))

    def sequences(self, worker_ids: list[int]) -> np.ndarray:
        return np.stack([self.histories[index].array() for index in worker_ids])

    def step_many(
        self, worker_ids: list[int], action_scores: np.ndarray
    ) -> list[dict[str, Any]]:
        for row, worker_id in enumerate(worker_ids):
            self.connections[worker_id].send(
                {"kind": "step", "action_scores": np.asarray(action_scores[row])}
            )
        values: list[dict[str, Any]] = []
        for worker_id in worker_ids:
            message = self.connections[worker_id].recv()
            if message.get("kind") == "error":
                raise RuntimeError(f"worker {worker_id} failed: {message}")
            if message.get("kind") != "transition":
                raise RuntimeError(f"worker protocol drift: {message}")
            observation = np.asarray(message["observation"], dtype=np.float32)
            self.observations[worker_id] = observation
            if message["done"]:
                self.episode_indices[worker_id] = int(message["episode_index"]) + 1
                self.histories[worker_id].reset(observation)
            else:
                self.histories[worker_id].append(observation)
            values.append(message)
        return values

    def close(self, *, force: bool = False) -> dict[str, Any]:
        if not force:
            for connection, process in zip(self.connections, self.processes, strict=True):
                if process.is_alive():
                    connection.send({"kind": "close"})
            for connection, process in zip(self.connections, self.processes, strict=True):
                if process.is_alive() and connection.poll(5.0):
                    connection.recv()
        for connection in self.connections:
            connection.close()
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        return {
            "worker_pids": list(self.pids),
            "all_workers_reaped": all(not process.is_alive() for process in self.processes),
            "exit_codes": [process.exitcode for process in self.processes],
        }


@dataclass
class _Transition:
    observation: np.ndarray
    sequence: np.ndarray
    ordered_committee: np.ndarray
    old_log_probability: float
    value: float
    reward: float
    done: bool
    advantage: float = 0.0
    return_value: float = 0.0


class PaperRecoveryPPOTrainer:
    """PPO implementation frozen before formal seeds are opened."""

    def __init__(
        self,
        config: PaperRecoveryConfig,
        *,
        seed: int,
        device: str = "auto",
    ):
        self.config = config
        self.seed = int(seed)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        self.policy = ResidualPPOPolicy(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.ppo.learning_rate
        )
        self.policy_generators: list[torch.Generator] = []
        generator_device = self.device.type if self.device.type in {"cpu", "cuda"} else "cpu"
        for worker_id in range(config.budgets.persistent_workers):
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(self.seed + 8_000_003 + 65_537 * worker_id)
            self.policy_generators.append(generator)
        self.shuffle_generator = torch.Generator(device="cpu")
        self.shuffle_generator.manual_seed(self.seed + 9_000_001)
        self.trajectory_hashers = [
            TrajectoryHasher(worker_id=worker_id)
            for worker_id in range(config.budgets.persistent_workers)
        ]

    def _sample_actions(
        self,
        observations: np.ndarray,
        sequences: np.ndarray,
        worker_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        seq = torch.as_tensor(sequences, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            scores, values, masks, _ = self.policy(obs, seq)
            eligible = masks.sum(dim=-1)
            if torch.any(eligible < self.config.physics.committee_size):
                counts = eligible.cpu().tolist()
                raise RuntimeError(
                    "training action failed closed because fewer than k nodes were "
                    f"observable-eligible: {counts}"
                )
            policy_logits = scores / max(self.config.ppo.gumbel_temperature, 1e-6)
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
                dim=-1,
                sorted=True,
            ).indices
            log_probability, entropy = self.policy.plackett_luce_stats(
                policy_logits, ordered, masks
            )
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
            _, values, _, _ = self.policy(
                torch.as_tensor(observations, dtype=torch.float32, device=self.device),
                torch.as_tensor(sequences, dtype=torch.float32, device=self.device),
            )
        return values.cpu().numpy()

    def collect_update(
        self,
        workers: PersistentRolloutWorkers,
        *,
        update_index: int,
    ) -> tuple[list[_Transition], dict[str, Any]]:
        quotas = worker_transition_quotas(update_index)
        counts = [0] * len(quotas)
        records: list[list[_Transition]] = [[] for _ in quotas]
        entropy_values: list[float] = []
        success_values: list[float] = []
        tps_values: list[float] = []
        while counts != list(quotas):
            active = [index for index, quota in enumerate(quotas) if counts[index] < quota]
            observations = np.stack([workers.observations[index] for index in active])
            sequences = workers.sequences(active)
            action_scores, ordered, log_probability, entropy, values = self._sample_actions(
                observations, sequences, active
            )
            transitions = workers.step_many(active, action_scores)
            for row, worker_id in enumerate(active):
                message = transitions[row]
                info = message["info"]
                expected_order = ordered[row].astype(int).tolist()
                if info["candidate_committee_ids"] != expected_order:
                    raise RuntimeError("policy order and environment committee diverged")
                if not info["candidate_executed_match"]:
                    raise RuntimeError("candidate/executed parity contract was violated")
                self.trajectory_hashers[worker_id].update(
                    info, episode_index=int(message["episode_index"])
                )
                records[worker_id].append(
                    _Transition(
                        observation=observations[row].copy(),
                        sequence=sequences[row].copy(),
                        ordered_committee=ordered[row].copy(),
                        old_log_probability=float(log_probability[row]),
                        value=float(values[row]),
                        reward=float(message["reward"]),
                        done=bool(message["done"]),
                    )
                )
                counts[worker_id] += 1
                entropy_values.append(float(entropy[row]))
                success_values.append(float(info["certificate_success"]))
                tps_values.append(float(info["served_tps"]))

        bootstrap = self._bootstrap_values(workers)
        gamma = self.config.ppo.gamma
        lam = self.config.ppo.gae_lambda
        flattened: list[_Transition] = []
        for worker_id, worker_records in enumerate(records):
            advantage = 0.0
            next_value = float(bootstrap[worker_id])
            for transition in reversed(worker_records):
                nonterminal = 0.0 if transition.done else 1.0
                delta = transition.reward + gamma * next_value * nonterminal - transition.value
                advantage = delta + gamma * lam * nonterminal * advantage
                transition.advantage = float(advantage)
                transition.return_value = float(advantage + transition.value)
                next_value = transition.value
            flattened.extend(worker_records)
        if len(flattened) != self.config.budgets.transitions_per_update:
            raise RuntimeError("rollout transition budget drift")
        telemetry = {
            "update": int(update_index),
            "worker_quotas": list(quotas),
            "transition_count": len(flattened),
            "mean_pick_entropy": float(np.mean(entropy_values)),
            "certificate_success_rate": float(np.mean(success_values)),
            "mean_served_tps": float(np.mean(tps_values)),
        }
        return flattened, telemetry

    def optimise_update(self, transitions: list[_Transition]) -> dict[str, Any]:
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
        returns = torch.as_tensor(
            [row.return_value for row in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = torch.as_tensor(
            [row.advantage for row in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

        metrics: dict[str, list[float]] = {
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "gradient_norm": [],
        }
        batch_size = len(transitions)
        mini = self.config.budgets.minibatch_size
        for _epoch in range(self.config.budgets.epochs_per_update):
            permutation = torch.randperm(batch_size, generator=self.shuffle_generator)
            for start in range(0, batch_size, mini):
                indices_cpu = permutation[start : start + mini]
                indices = indices_cpu.to(self.device)
                new_log_probability, entropy, values = self.policy.evaluate_ordered_committees(
                    observations[indices], sequences[indices], ordered[indices]
                )
                log_ratio = new_log_probability - old_log_probability[indices]
                ratio = torch.exp(log_ratio)
                unclipped = ratio * advantages[indices]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.ppo.clip_epsilon,
                    1.0 + self.config.ppo.clip_epsilon,
                ) * advantages[indices]
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                critic_loss = torch.nn.functional.mse_loss(values, returns[indices])
                loss = (
                    actor_loss
                    + self.config.ppo.value_coefficient * critic_loss
                    - self.config.ppo.entropy_coefficient_mean_per_pick * entropy.mean()
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.ppo.gradient_clip_norm
                )
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        torch.abs(ratio - 1.0) > self.config.ppo.clip_epsilon
                    ).float().mean()
                metrics["actor_loss"].append(float(actor_loss.item()))
                metrics["critic_loss"].append(float(critic_loss.item()))
                metrics["entropy"].append(float(entropy.mean().item()))
                metrics["approx_kl"].append(float(approx_kl.item()))
                metrics["clip_fraction"].append(float(clip_fraction.item()))
                metrics["gradient_norm"].append(float(gradient_norm.item()))
        summary = {key: float(np.mean(value)) for key, value in metrics.items()}
        summary["max_approx_kl"] = float(np.max(metrics["approx_kl"]))
        summary["target_kl"] = float(self.config.ppo.target_kl)
        summary["target_kl_exceeded"] = bool(
            summary["max_approx_kl"] > self.config.ppo.target_kl
        )
        summary["epochs_executed"] = self.config.budgets.epochs_per_update
        summary["optimizer_minibatches"] = len(metrics["actor_loss"])
        summary["all_finite"] = bool(
            all(np.isfinite(value).all() for value in metrics.values())
        )
        return summary

    def train(
        self,
        *,
        updates: int,
        run_kind: str,
        output_directory: Path,
    ) -> dict[str, Any]:
        if run_kind not in {"smoke", "profiler", "formal"}:
            raise ValueError("run_kind must be smoke, profiler or formal")
        expected = {
            "smoke": self.config.budgets.smoke_updates,
            "profiler": self.config.budgets.profiler_updates,
            "formal": self.config.budgets.formal_updates,
        }[run_kind]
        if int(updates) != expected:
            raise ValueError(f"{run_kind} requires exactly {expected} updates")
        output_directory.mkdir(parents=True, exist_ok=True)
        started_at = utc_now()
        started_monotonic = time.monotonic()
        source = recovery_source_manifest(self.config)
        progress: list[dict[str, Any]] = []
        workers = PersistentRolloutWorkers(self.config, seed=self.seed)
        initial_parameter_hash = canonical_hash(
            {name: value.detach().cpu().numpy().tolist() for name, value in self.policy.named_parameters()}
        )
        initial_actor_hash = canonical_hash(
            {
                name: value.detach().cpu().numpy().tolist()
                for module_name, module in (
                    ("temporal_encoder", self.policy.temporal_encoder),
                    ("actor", self.policy.actor),
                )
                for name, value in module.named_parameters(prefix=module_name)
            }
        )
        initial_critic_hash = canonical_hash(
            {
                name: value.detach().cpu().numpy().tolist()
                for module_name, module in (
                    ("critic_node_encoder", self.policy.critic_node_encoder),
                    ("critic_head", self.policy.critic_head),
                )
                for name, value in module.named_parameters(prefix=module_name)
            }
        )
        close_receipt: dict[str, Any] = {}
        try:
            for update_index in range(updates):
                rollout, rollout_metrics = self.collect_update(
                    workers, update_index=update_index
                )
                optimiser_metrics = self.optimise_update(rollout)
                entry = {
                    **rollout_metrics,
                    **optimiser_metrics,
                    "elapsed_seconds": float(time.monotonic() - started_monotonic),
                }
                progress.append(entry)
                atomic_write_json(
                    output_directory / "progress.json",
                    {
                        "schema": "qrelay-paper-recovery-training-progress-1",
                        "run_kind": run_kind,
                        "seed": self.seed,
                        "updates_expected": updates,
                        "updates_completed": update_index + 1,
                        "transitions_completed": (update_index + 1)
                        * self.config.budgets.transitions_per_update,
                        "entries": progress,
                    },
                )
        finally:
            close_receipt = workers.close()

        checkpoint_name = "ppo_final.pt" if run_kind == "formal" else f"{run_kind}_development.pt"
        checkpoint_path = output_directory / checkpoint_name
        checkpoint_payload = {
            "schema": "qrelay-paper-recovery-checkpoint-1",
            "run_kind": run_kind,
            "seed": self.seed,
            "updates": updates,
            "transitions": updates * self.config.budgets.transitions_per_update,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "source_manifest": source,
            "architecture_manifest": self.policy.architecture_manifest(),
            "checkpoint_selection": "fixed_budget_final_only",
            "resumed": False,
            "early_stopped": False,
            "selected_by_evaluation": False,
        }
        torch.save(checkpoint_payload, checkpoint_path)
        worker_receipts = tuple(hasher.receipt() for hasher in self.trajectory_hashers)
        receipt = TrainingReceipt(
            schema="qrelay-paper-recovery-training-receipt-1",
            status="complete",
            seed=self.seed,
            run_kind=run_kind,
            started_at=started_at,
            completed_at=utc_now(),
            updates_completed=updates,
            transitions_completed=updates * self.config.budgets.transitions_per_update,
            source_hash=source["source_hash"],
            config_hash=canonical_hash(self.config.to_dict()),
            checkpoint_path=checkpoint_path.relative_to(Path(__file__).resolve().parents[2]).as_posix(),
            checkpoint_sha256=sha256_file(checkpoint_path),
            trajectory_hash=aggregate_trajectory_hash(worker_receipts),
            worker_trajectory_receipts=worker_receipts,
        ).to_dict()
        final_parameter_hash = canonical_hash(
            {name: value.detach().cpu().numpy().tolist() for name, value in self.policy.named_parameters()}
        )
        final_actor_hash = canonical_hash(
            {
                name: value.detach().cpu().numpy().tolist()
                for module_name, module in (
                    ("temporal_encoder", self.policy.temporal_encoder),
                    ("actor", self.policy.actor),
                )
                for name, value in module.named_parameters(prefix=module_name)
            }
        )
        final_critic_hash = canonical_hash(
            {
                name: value.detach().cpu().numpy().tolist()
                for module_name, module in (
                    ("critic_node_encoder", self.policy.critic_node_encoder),
                    ("critic_head", self.policy.critic_head),
                )
                for name, value in module.named_parameters(prefix=module_name)
            }
        )
        actor_updated = bool(final_actor_hash != initial_actor_hash)
        critic_updated = bool(final_critic_hash != initial_critic_hash)
        document = {
            **receipt,
            "runtime": runtime_manifest(),
            "environment_manifest_hashes": sorted(
                {manifest["manifest_hash"] for manifest in workers.worker_manifests}
            ),
            "worker_runtime": close_receipt,
            "initial_parameter_hash": initial_parameter_hash,
            "final_parameter_hash": final_parameter_hash,
            "initial_actor_hash": initial_actor_hash,
            "final_actor_hash": final_actor_hash,
            "initial_critic_hash": initial_critic_hash,
            "final_critic_hash": final_critic_hash,
            "actor_updated": actor_updated,
            "critic_updated": critic_updated,
            "actor_and_critic_updated": bool(actor_updated and critic_updated),
            "elapsed_seconds": float(time.monotonic() - started_monotonic),
            "progress_path": (output_directory / "progress.json").relative_to(
                Path(__file__).resolve().parents[2]
            ).as_posix(),
        }
        document["receipt_hash"] = canonical_hash(
            {key: value for key, value in document.items() if key != "receipt_hash"}
        )
        atomic_write_json(output_directory / "completion_receipt.json", document)
        return document


def run_registered_training(
    *,
    run_kind: str,
    seed: int,
    device: str = "auto",
    explicit_formal_flag: bool = False,
) -> dict[str, Any]:
    """Public entry point; formal claims occur before environment construction."""

    base = PaperRecoveryConfig()
    seed = int(seed)
    training_config = replace(
        base,
        training_mode=True,
        scenario_profile="training_primary",
    )
    if run_kind in {"smoke", "profiler"}:
        if seed not in base.seeds.development:
            raise RuntimeError("smoke/profiler runs accept development seeds only")
    elif run_kind == "formal":
        assert_formal_runtime_authorized()
        claim_formal_seed_before_environment(
            config=training_config,
            seed=seed,
            explicit_formal_flag=explicit_formal_flag,
        )
    else:
        raise ValueError("run_kind must be smoke, profiler or formal")
    updates = {
        "smoke": base.budgets.smoke_updates,
        "profiler": base.budgets.profiler_updates,
        "formal": base.budgets.formal_updates,
    }[run_kind]
    output_directory = RESULT_ROOT / f"{run_kind}_seed_{seed}"
    try:
        trainer = PaperRecoveryPPOTrainer(training_config, seed=seed, device=device)
        receipt = trainer.train(
            updates=updates, run_kind=run_kind, output_directory=output_directory
        )
    except BaseException:
        if run_kind == "formal":
            update_formal_seed_status(seed, status="spent_failed")
        raise
    if run_kind == "formal":
        update_formal_seed_status(
            seed,
            status="completed",
            receipt_path=(output_directory / "completion_receipt.json").relative_to(
                Path(__file__).resolve().parents[2]
            ).as_posix(),
        )
    return receipt
