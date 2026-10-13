"""Vectorized rollout workers: 33 logical environments in 11 processes."""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

import numpy as np

from drl.paper_recovery.config import PaperRecoveryConfig
from drl.paper_recovery.env import PaperRecoveryEnv
from drl.paper_recovery.policy import TemporalObservationBuffer


PHYSICAL_WORKERS = 11
LOGICAL_ENVIRONMENTS = 33


def logical_transition_quotas(
    update_index: int, logical_environments: int = LOGICAL_ENVIRONMENTS
) -> tuple[int, ...]:
    """Rotate 30x91 and 3x90 slices while preserving exactly 3,000."""

    if logical_environments != LOGICAL_ENVIRONMENTS:
        raise ValueError("speed-v5 contract requires exactly 33 logical environments")
    base, remainder = divmod(3_000, logical_environments)
    start = (int(update_index) * remainder) % logical_environments
    larger = {
        (start + offset) % logical_environments for offset in range(remainder)
    }
    quotas = tuple(
        base + 1 if logical_id in larger else base
        for logical_id in range(logical_environments)
    )
    if sum(quotas) != 3_000 or quotas.count(91) != 30 or quotas.count(90) != 3:
        raise RuntimeError("speed-v5 logical quota drift")
    return quotas


def _vector_worker_main(
    connection: Any,
    config: PaperRecoveryConfig,
    logical_ids: tuple[int, ...],
    base_seed: int,
    environment_class: type[PaperRecoveryEnv],
) -> None:
    """Own three independent environments in one long-lived process."""

    environments: dict[int, PaperRecoveryEnv] = {}
    episode_indices: dict[int, int] = {}
    initial_seeds: dict[int, int] = {}
    try:
        ready = []
        for logical_id in logical_ids:
            initial_seed = int(base_seed) + 1_009 * int(logical_id)
            environment = environment_class(config)
            observation, reset_info = environment.reset(seed=initial_seed)
            environments[logical_id] = environment
            episode_indices[logical_id] = 0
            initial_seeds[logical_id] = initial_seed
            ready.append(
                {
                    "logical_id": logical_id,
                    "episode_index": 0,
                    "observation": observation,
                    "reset_info": reset_info,
                    "environment_manifest": environment.model_manifest(),
                }
            )
        connection.send(
            {
                "kind": "ready",
                "pid": os.getpid(),
                "logical_ids": list(logical_ids),
                "records": ready,
            }
        )
        while True:
            command = connection.recv()
            if command["kind"] == "step_many":
                transitions = []
                for item in command["items"]:
                    logical_id = int(item["logical_id"])
                    environment = environments[logical_id]
                    stepped_episode = episode_indices[logical_id]
                    observation, reward, terminated, truncated, info = environment.step(
                        item["action_scores"]
                    )
                    done = bool(terminated or truncated)
                    next_reset_info = None
                    if done:
                        episode_indices[logical_id] += 1
                        episode_seed = (
                            initial_seeds[logical_id]
                            + 1_000_003 * episode_indices[logical_id]
                        )
                        observation, next_reset_info = environment.reset(
                            seed=episode_seed
                        )
                    transitions.append(
                        {
                            "logical_id": logical_id,
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
                connection.send({"kind": "transitions", "records": transitions})
            elif command["kind"] == "close":
                connection.send({"kind": "closed", "logical_ids": list(logical_ids)})
                break
            else:
                raise RuntimeError(f"unknown speed-v5 worker command: {command['kind']}")
    except BaseException as error:
        try:
            connection.send(
                {
                    "kind": "error",
                    "logical_ids": list(logical_ids),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
        finally:
            raise
    finally:
        connection.close()


class VectorizedPersistentRolloutWorkers:
    """Interface-compatible pool with 33 logical and 11 physical workers."""

    def __init__(
        self,
        config: PaperRecoveryConfig,
        *,
        seed: int,
        environment_class: type[PaperRecoveryEnv] = PaperRecoveryEnv,
    ) -> None:
        context = mp.get_context("spawn")
        groups = tuple(
            tuple(range(physical_id, LOGICAL_ENVIRONMENTS, PHYSICAL_WORKERS))
            for physical_id in range(PHYSICAL_WORKERS)
        )
        if any(len(group) != 3 for group in groups):
            raise RuntimeError("speed-v5 physical/logical grouping drift")
        self.groups = groups
        self.logical_to_physical = {
            logical_id: physical_id
            for physical_id, group in enumerate(groups)
            for logical_id in group
        }
        self.connections: list[Any] = []
        self.processes: list[mp.Process] = []
        self.observations: list[np.ndarray | None] = [None] * LOGICAL_ENVIRONMENTS
        self.episode_indices = [0] * LOGICAL_ENVIRONMENTS
        self.histories = [
            TemporalObservationBuffer(config.observations.temporal_delta_window)
            for _ in range(LOGICAL_ENVIRONMENTS)
        ]
        self.worker_manifests: list[dict[str, Any] | None] = [
            None
        ] * LOGICAL_ENVIRONMENTS
        self.pids: list[int] = []
        for physical_id, group in enumerate(groups):
            parent, child = context.Pipe()
            process = context.Process(
                target=_vector_worker_main,
                args=(child, config, group, int(seed), environment_class),
                daemon=False,
                name=f"qrelay-speed-v5-worker-{physical_id}",
            )
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        ready = [connection.recv() for connection in self.connections]
        for physical_id, message in enumerate(ready):
            if message.get("kind") != "ready":
                self.close(force=True)
                raise RuntimeError(
                    f"speed-v5 physical worker {physical_id} failed startup: {message}"
                )
            if tuple(message["logical_ids"]) != groups[physical_id]:
                self.close(force=True)
                raise RuntimeError("speed-v5 ready logical-id drift")
            self.pids.append(int(message["pid"]))
            for record in message["records"]:
                logical_id = int(record["logical_id"])
                observation = np.asarray(record["observation"], dtype=np.float32)
                self.observations[logical_id] = observation
                self.histories[logical_id].reset(observation)
                self.worker_manifests[logical_id] = record["environment_manifest"]
        if any(value is None for value in self.observations):
            self.close(force=True)
            raise RuntimeError("speed-v5 did not initialize all logical observations")

    def sequences(self, logical_ids: list[int]) -> np.ndarray:
        return np.stack([self.histories[index].array() for index in logical_ids])

    def step_many(
        self, logical_ids: list[int], action_scores: np.ndarray
    ) -> list[dict[str, Any]]:
        items_by_physical: dict[int, list[dict[str, Any]]] = {}
        for row, logical_id in enumerate(logical_ids):
            physical_id = self.logical_to_physical[logical_id]
            items_by_physical.setdefault(physical_id, []).append(
                {
                    "logical_id": logical_id,
                    "action_scores": np.asarray(action_scores[row]),
                }
            )
        for physical_id, items in items_by_physical.items():
            self.connections[physical_id].send(
                {"kind": "step_many", "items": items}
            )
        by_logical: dict[int, dict[str, Any]] = {}
        for physical_id in items_by_physical:
            message = self.connections[physical_id].recv()
            if message.get("kind") == "error":
                raise RuntimeError(f"speed-v5 worker failed: {message}")
            if message.get("kind") != "transitions":
                raise RuntimeError(f"speed-v5 worker protocol drift: {message}")
            for record in message["records"]:
                logical_id = int(record["logical_id"])
                observation = np.asarray(record["observation"], dtype=np.float32)
                self.observations[logical_id] = observation
                if record["done"]:
                    self.episode_indices[logical_id] = int(record["episode_index"]) + 1
                    self.histories[logical_id].reset(observation)
                else:
                    self.histories[logical_id].append(observation)
                by_logical[logical_id] = record
        if set(by_logical) != set(logical_ids):
            raise RuntimeError("speed-v5 transition response set drift")
        return [by_logical[logical_id] for logical_id in logical_ids]

    def close(self, *, force: bool = False) -> dict[str, Any]:
        if not force:
            for connection, process in zip(
                self.connections, self.processes, strict=True
            ):
                if process.is_alive():
                    connection.send({"kind": "close"})
            for connection, process in zip(
                self.connections, self.processes, strict=True
            ):
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
            "physical_worker_pids": list(self.pids),
            "physical_workers": PHYSICAL_WORKERS,
            "logical_environments": LOGICAL_ENVIRONMENTS,
            "all_workers_reaped": all(
                not process.is_alive() for process in self.processes
            ),
            "exit_codes": [process.exitcode for process in self.processes],
        }
