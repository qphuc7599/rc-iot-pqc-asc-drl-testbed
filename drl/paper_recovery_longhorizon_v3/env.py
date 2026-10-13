"""Controllable long-horizon IoT cascade layered on the original event.

The canonical event is unchanged: 30 of 100 nodes are killed permanently at
control step 1000.  Unlike the discarded compound-blackout diagnostic, this
environment never forces every controller to zero throughput.  Committee
choices instead alter future node fatigue and shared RF-cell congestion, while
public telemetry is delayed.  This creates a genuine partially observable,
long-horizon control problem without exposing the event clock, hidden
Byzantine identities, or a future realization.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from drl.paper_recovery.env import PaperRecoveryEnv, _canonical_hash, _sha256_bytes


TELEMETRY_KEYS = (
    "trust_alpha",
    "trust_beta",
    "link_alpha",
    "link_beta",
    "latency_ewma_ms",
    "service_ewma",
    "response_fast_ewma",
    "response_slow_ewma",
    "response_miss_streak",
    "observed_reachable",
)


class LongHorizonCommitteeEnv(PaperRecoveryEnv):
    """Five-cell response engine with action-dependent future capacity."""

    MODEL_VERSION = "qrelay-reviewer-controllable-cascade-engine-3"
    NUM_CELLS = 5

    def _init_state_arrays(self) -> None:
        super()._init_state_arrays()
        self.cell_ids = np.arange(self.num_nodes, dtype=np.int64) % self.NUM_CELLS
        self.cell_health = np.ones(self.NUM_CELLS, dtype=np.float64)
        self.cell_debt = np.zeros(self.NUM_CELLS, dtype=np.float64)
        self.node_fatigue = np.zeros(self.num_nodes, dtype=np.float64)
        self.observed_node_fatigue = self.node_fatigue.copy()
        self.observed_cell_headroom = np.ones(self.NUM_CELLS, dtype=np.float64)
        self._public_state_queue: list[dict[str, np.ndarray]] = []
        self._telemetry_truth: dict[str, np.ndarray] | None = None
        self._telemetry_queue: list[dict[str, np.ndarray]] = []
        self.telemetry_delay_steps = 4
        self.external_arrival_tps = 500.0
        self.backlog_transactions = 0.0
        self.maximum_backlog_transactions = 1_200_000.0
        self.disaster_family = "wind_grid_congestion"
        self.family_parameters: dict[str, int | float | bool] = {}
        self.affected_cells = np.asarray([], dtype=np.int64)
        self.aftershock_cells = np.asarray([], dtype=np.int64)
        self.aftershock_step = 1_300
        self.aftershock_triggered = False
        self._round_aftershock = False
        self._pending_selected = np.asarray([], dtype=np.int64)

    @staticmethod
    def _family_contract(family: str) -> dict[str, int | float | bool]:
        contracts: dict[str, dict[str, int | float | bool]] = {
            "wind_grid_congestion": {
                "telemetry_delay_steps": 4,
                "affected_cells": 2,
                "shock_floor_percent": 52,
                "health_recovery_per_mille": 7,
                "aftershock_gap": 320,
                "aftershock_cells": 1,
                "aftershock_floor_percent": 58,
                "bursty_arrivals": False,
            },
            "flooded_edge_mesh": {
                "telemetry_delay_steps": 7,
                "affected_cells": 2,
                "shock_floor_percent": 46,
                "health_recovery_per_mille": 5,
                "aftershock_gap": 360,
                "aftershock_cells": 1,
                "aftershock_floor_percent": 50,
                "bursty_arrivals": False,
            },
            "emergency_uplink_surge": {
                "telemetry_delay_steps": 5,
                "affected_cells": 1,
                "shock_floor_percent": 55,
                "health_recovery_per_mille": 8,
                "aftershock_gap": 240,
                "aftershock_cells": 2,
                "aftershock_floor_percent": 62,
                "bursty_arrivals": True,
            },
            "rolling_power_brownout": {
                "telemetry_delay_steps": 8,
                "affected_cells": 2,
                "shock_floor_percent": 49,
                "health_recovery_per_mille": 6,
                "aftershock_gap": 285,
                "aftershock_cells": 1,
                "aftershock_floor_percent": 54,
                "bursty_arrivals": True,
            },
            "heldout_typhoon_backhaul": {
                "telemetry_delay_steps": 6,
                "affected_cells": 2,
                "shock_floor_percent": 43,
                "health_recovery_per_mille": 4,
                "aftershock_gap": 305,
                "aftershock_cells": 2,
                "aftershock_floor_percent": 48,
                "bursty_arrivals": True,
            },
        }
        return contracts[family].copy()

    def _configure_primary_scenario(self, seed: int) -> None:
        super()._configure_primary_scenario(seed)
        self.event_step = self.config.physics.evaluation_event_step
        if self.config.training_mode:
            families = (
                "wind_grid_congestion",
                "flooded_edge_mesh",
                "emergency_uplink_surge",
                "rolling_power_brownout",
            )
            self.disaster_family = families[
                int(self.rng_scenario.integers(0, len(families)))
            ]
        else:
            self.disaster_family = "heldout_typhoon_backhaul"
        self.family_parameters = self._family_contract(self.disaster_family)
        self.telemetry_delay_steps = int(
            self.family_parameters["telemetry_delay_steps"]
        )
        affected_count = int(self.family_parameters["affected_cells"])
        self.affected_cells = np.sort(
            self.rng_scenario.choice(
                np.arange(self.NUM_CELLS, dtype=np.int64),
                size=affected_count,
                replace=False,
            )
        )
        remaining = np.setdiff1d(
            np.arange(self.NUM_CELLS, dtype=np.int64), self.affected_cells
        )
        aftershock_count = min(
            int(self.family_parameters["aftershock_cells"]), len(remaining)
        )
        self.aftershock_cells = np.sort(
            self.rng_scenario.choice(
                remaining, size=aftershock_count, replace=False
            )
        ).astype(np.int64)
        self.aftershock_step = self.event_step + int(
            self.family_parameters["aftershock_gap"]
        )
        self.scenario_record.update(
            {
                "event_step": int(self.event_step),
                "original_30_percent_kill_retained": True,
                "controllable_cascade_family": self.disaster_family,
                "controllable_cascade_parameters": copy.deepcopy(
                    self.family_parameters
                ),
                "public_rf_cells": int(self.NUM_CELLS),
                "affected_cells": self.affected_cells.tolist(),
                "aftershock_cells": self.aftershock_cells.tolist(),
                "aftershock_step": int(self.aftershock_step),
                "unavoidable_zero_tps_window": False,
                "committee_choice_changes_future_capacity": True,
                "delayed_public_telemetry_steps": self.telemetry_delay_steps,
            }
        )
        self.scenario_record["scenario_hash"] = _canonical_hash(
            self.scenario_record
        )

    def _sample_offered_load(self) -> float:
        bursty = bool(self.family_parameters.get("bursty_arrivals", False))
        previous = float(getattr(self, "external_arrival_tps", 500.0))
        if bursty:
            high = self.rng_load.random() < (0.72 if previous > 720.0 else 0.10)
            mean = 900.0 if high else 430.0
            return float(
                np.clip(self.rng_load.normal(mean, 75.0), 250.0, 1_150.0)
            )
        return float(np.clip(self.rng_load.normal(500.0, 45.0), 320.0, 760.0))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        self.external_arrival_tps = float(self.last_offered_load_tps)
        self.last_offered_load_tps = self.external_arrival_tps
        self.observed_node_fatigue = self.node_fatigue.copy()
        self.observed_cell_headroom = self._actual_cell_headroom()
        observation = self._get_obs()
        info["observation_hash"] = _sha256_bytes(observation.tobytes())
        return observation, info

    def _actual_cell_headroom(self) -> np.ndarray:
        return np.clip(self.cell_health * (1.0 - 0.72 * self.cell_debt), 0.05, 1.0)

    def _recover_physical_capacity(self) -> None:
        rate = float(self.family_parameters.get("health_recovery_per_mille", 6)) / 1_000.0
        self.cell_health += rate * (1.0 - self.cell_health)
        self.cell_health = np.clip(self.cell_health, 0.05, 1.0)

    def _apply_registered_event(self) -> list[int]:
        killed = super()._apply_registered_event()
        if self.step_count == self.event_step:
            floor = float(self.family_parameters["shock_floor_percent"]) / 100.0
            self.cell_health[self.affected_cells] = np.minimum(
                self.cell_health[self.affected_cells], floor
            )
        return killed

    def _start_churn_events(self) -> list[int]:
        correlated = super()._start_churn_events()
        self._round_aftershock = False
        if self.step_count == self.aftershock_step:
            floor = float(
                self.family_parameters["aftershock_floor_percent"]
            ) / 100.0
            self.cell_health[self.aftershock_cells] = np.minimum(
                self.cell_health[self.aftershock_cells], floor
            )
            self.aftershock_triggered = True
            self._round_aftershock = True
        return correlated

    def _preallocate_response_draws(self, regime: str) -> dict[str, Any]:
        draws = super()._preallocate_response_draws(regime)
        cell = self.cell_ids
        headroom = self._actual_cell_headroom()[cell]
        fatigue = np.clip(self.node_fatigue, 0.0, 1.0)
        draws["signing_ms"] = np.asarray(draws["signing_ms"]) * (
            1.0 + 1.10 * fatigue
        )
        draws["network_delay_ms"] = np.asarray(draws["network_delay_ms"]) * (
            1.0 + 1.20 * (1.0 - headroom) + 0.35 * fatigue
        )
        return draws

    def _capture_telemetry(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(getattr(self, name)).copy() for name in TELEMETRY_KEYS
        }

    def _restore_telemetry(self, snapshot: dict[str, np.ndarray]) -> None:
        for name, value in snapshot.items():
            setattr(self, name, value.copy())

    def _update_observable_telemetry(
        self,
        selected: np.ndarray,
        round_result: dict[str, Any],
        actual_available: np.ndarray,
    ) -> None:
        visible = self._capture_telemetry()
        if self._telemetry_truth is not None:
            self._restore_telemetry(self._telemetry_truth)
        super()._update_observable_telemetry(
            selected, round_result, actual_available
        )
        truth = self._capture_telemetry()
        self._telemetry_truth = truth
        self._restore_telemetry(visible)
        self._telemetry_queue.append(truth)
        if len(self._telemetry_queue) > self.telemetry_delay_steps:
            self._restore_telemetry(self._telemetry_queue.pop(0))

    def _publish_delayed_capacity(self) -> None:
        self._public_state_queue.append(
            {
                "node_fatigue": self.node_fatigue.copy(),
                "cell_headroom": self._actual_cell_headroom().copy(),
            }
        )
        if len(self._public_state_queue) > self.telemetry_delay_steps:
            visible = self._public_state_queue.pop(0)
            self.observed_node_fatigue = visible["node_fatigue"]
            self.observed_cell_headroom = visible["cell_headroom"]

    def _update_endogenous_capacity(
        self, selected: np.ndarray, responded_ids: list[int], offered_load: float
    ) -> None:
        load_ratio = float(np.clip(offered_load / 1_000.0, 0.2, 1.6))
        selected_mask = np.zeros(self.num_nodes, dtype=np.float64)
        selected_mask[selected] = 1.0
        responded_mask = np.zeros(self.num_nodes, dtype=np.float64)
        responded_mask[np.asarray(responded_ids, dtype=np.int64)] = 1.0
        self.node_fatigue = np.clip(
            0.90 * self.node_fatigue
            + (0.055 + 0.035 * load_ratio) * selected_mask
            + 0.025 * responded_mask,
            0.0,
            1.0,
        )
        counts = np.bincount(self.cell_ids[selected], minlength=self.NUM_CELLS)
        fractions = counts.astype(np.float64) / max(self.committee_size, 1)
        self.cell_debt = np.clip(
            0.90 * self.cell_debt
            + 0.78 * np.square(fractions) * (0.55 + 0.45 * load_ratio),
            0.0,
            1.0,
        )

    def _get_obs(self) -> np.ndarray:
        observation = super()._get_obs()
        boundary = self.num_nodes * self.config.observations.per_node_features
        node = observation[:boundary].reshape(self.num_nodes, -1).copy()
        node[:, 6] = self.cell_ids / max(self.NUM_CELLS - 1, 1)
        node[:, 7] = np.clip(1.0 - self.observed_node_fatigue, 0.0, 1.0)
        node[:, 8] = self.observed_cell_headroom[self.cell_ids]
        backlog_ratio = float(
            np.clip(
                self.backlog_transactions / self.maximum_backlog_transactions,
                0.0,
                1.0,
            )
        )
        global_obs = observation[boundary:].copy()
        global_obs[0] = np.clip(self.last_offered_load_tps / 1_200.0, 0.0, 1.0)
        global_obs[2] = float(np.mean(self.observed_cell_headroom))
        global_obs[5] = backlog_ratio
        value = np.concatenate((node.reshape(-1), global_obs)).astype(np.float32)
        return np.clip(value, 0.0, 1.0)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._recover_physical_capacity()
        candidate, _ = self.select_committee(action)
        self._pending_selected = candidate.copy()
        external_arrival = float(self.external_arrival_tps)
        effective_demand = float(
            np.clip(
                external_arrival + self.backlog_transactions / 300.0,
                0.0,
                1_600.0,
            )
        )
        self.last_offered_load_tps = effective_demand
        observation, reward, terminated, truncated, info = super().step(action)
        next_external = float(self.last_offered_load_tps)
        interval_seconds = self.config.physics.control_interval_seconds
        self.backlog_transactions = float(
            np.clip(
                self.backlog_transactions
                + interval_seconds * (external_arrival - float(info["served_tps"])),
                0.0,
                self.maximum_backlog_transactions,
            )
        )
        self.external_arrival_tps = next_external
        self.last_offered_load_tps = float(
            np.clip(
                next_external + self.backlog_transactions / 300.0,
                0.0,
                1_600.0,
            )
        )
        self._update_endogenous_capacity(
            candidate, info["responded_node_ids"], effective_demand
        )
        self._publish_delayed_capacity()
        backlog_ratio = float(
            np.clip(
                self.backlog_transactions / self.maximum_backlog_transactions,
                0.0,
                1.0,
            )
        )
        network_reserve = float(np.mean(self._actual_cell_headroom()))
        selected_fatigue = float(
            np.mean(self.node_fatigue[candidate]) if len(candidate) else 1.0
        )
        info["reward_components"]["public_backlog"] = -18.0 * backlog_ratio
        info["reward_components"]["public_network_reserve"] = -6.0 * (
            1.0 - network_reserve
        )
        info["reward_components"]["selected_fatigue"] = -2.0 * selected_fatigue
        reward = float(sum(info["reward_components"].values()))
        info["reward_total"] = reward
        info["reward_recomputed"] = reward
        info["long_horizon"] = {
            "family": self.disaster_family,
            "forced_zero_tps": False,
            "aftershock_this_step": bool(self._round_aftershock),
            "aftershock_step": int(self.aftershock_step),
            "backlog_transactions": self.backlog_transactions,
            "backlog_normalized": backlog_ratio,
            "external_arrival_tps": external_arrival,
            "effective_demand_tps": effective_demand,
            "telemetry_delay_steps": int(self.telemetry_delay_steps),
            "mean_node_fatigue": float(np.mean(self.node_fatigue)),
            "selected_mean_fatigue": selected_fatigue,
            "cell_health": self.cell_health.tolist(),
            "cell_debt": self.cell_debt.tolist(),
            "cell_headroom": self._actual_cell_headroom().tolist(),
            "network_reserve": network_reserve,
            "affected_cells": self.affected_cells.tolist(),
            "aftershock_cells": self.aftershock_cells.tolist(),
        }
        observation = self._get_obs()
        info["post_observation_hash"] = _sha256_bytes(observation.tobytes())
        info["state_hash"] = _canonical_hash(
            {
                "base_state_hash": info["state_hash"],
                "backlog_transactions": self.backlog_transactions,
                "external_arrival_tps": self.external_arrival_tps,
                "node_fatigue": self.node_fatigue.tolist(),
                "cell_health": self.cell_health.tolist(),
                "cell_debt": self.cell_debt.tolist(),
                "telemetry_queue_depth": len(self._telemetry_queue),
                "public_state_queue_depth": len(self._public_state_queue),
            }
        )
        info["transition_hash"] = _canonical_hash(info)
        self.last_info = info
        return observation, reward, terminated, truncated, info

    def model_manifest(self) -> dict[str, Any]:
        manifest = super().model_manifest()
        manifest.update(
            {
                "model_version": self.MODEL_VERSION,
                "original_kill_contract": "30_percent_permanent_at_step_1000",
                "unavoidable_zero_tps_window": False,
                "committee_action_changes_future": [
                    "per_node_fatigue",
                    "shared_rf_cell_congestion_debt",
                    "queue_backlog",
                ],
                "partial_observability": "delayed_public_telemetry_only",
                "actor_disaster_flag": False,
                "actor_hidden_byzantine_labels": False,
                "future_realization_input": False,
            }
        )
        manifest["manifest_hash"] = _canonical_hash(manifest)
        return manifest
