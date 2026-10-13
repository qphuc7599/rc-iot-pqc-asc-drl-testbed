"""Policy-dependent delayed cascading recovery without a forced blackout.

The original event remains a permanent 30/100-node kill at step 1000.  The
hardness comes from causal state: repeated service creates long-lived node
fatigue, shared-cell congestion spills into neighbouring cells, overloaded
nodes enter thermal cooldown, queues persist, and all capacity telemetry is
delayed.  Random tapes are consumed for every node before actions are applied,
so common-random-number comparisons remain exact.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from drl.paper_recovery.env import PaperRecoveryEnv, _canonical_hash, _sha256_bytes
from drl.paper_recovery_longhorizon_v3.env import LongHorizonCommitteeEnv


class HardcoreCascadeEnv(LongHorizonCommitteeEnv):
    MODEL_VERSION = "qrelay-reviewer-hardcore-delayed-cascade-engine-5"
    MAX_SERVICE_AGE = 240.0

    def _init_state_arrays(self) -> None:
        super()._init_state_arrays()
        self.maximum_backlog_transactions = 12_000_000.0
        self.service_age = np.full(self.num_nodes, self.MAX_SERVICE_AGE)
        self.observed_service_age = self.service_age.copy()
        self.thermal_trip_count = np.zeros(self.num_nodes, dtype=np.int64)
        self.total_expired_transactions = 0.0
        self._round_new_thermal_trips: list[int] = []

    @staticmethod
    def _family_contract(family: str) -> dict[str, int | float | bool]:
        contracts: dict[str, dict[str, int | float | bool]] = {
            "heatwave_edge_saturation": {
                "telemetry_delay_steps": 12,
                "affected_cells": 2,
                "shock_floor_percent": 38,
                "health_recovery_per_mille": 18,
                "aftershock_gap": 220,
                "aftershock_cells": 1,
                "aftershock_floor_percent": 42,
                "high_load_probability": 16,
            },
            "flooded_backhaul_repair": {
                "telemetry_delay_steps": 18,
                "affected_cells": 2,
                "shock_floor_percent": 32,
                "health_recovery_per_mille": 12,
                "aftershock_gap": 260,
                "aftershock_cells": 2,
                "aftershock_floor_percent": 36,
                "high_load_probability": 13,
            },
            "emergency_uplink_thermal": {
                "telemetry_delay_steps": 14,
                "affected_cells": 1,
                "shock_floor_percent": 40,
                "health_recovery_per_mille": 16,
                "aftershock_gap": 180,
                "aftershock_cells": 2,
                "aftershock_floor_percent": 45,
                "high_load_probability": 22,
            },
            "rolling_microgrid_instability": {
                "telemetry_delay_steps": 22,
                "affected_cells": 2,
                "shock_floor_percent": 35,
                "health_recovery_per_mille": 10,
                "aftershock_gap": 300,
                "aftershock_cells": 1,
                "aftershock_floor_percent": 34,
                "high_load_probability": 18,
            },
            "heldout_compound_cyclone": {
                "telemetry_delay_steps": 20,
                "affected_cells": 2,
                "shock_floor_percent": 30,
                "health_recovery_per_mille": 9,
                "aftershock_gap": 240,
                "aftershock_cells": 2,
                "aftershock_floor_percent": 32,
                "high_load_probability": 20,
            },
        }
        return contracts[family].copy()

    def _configure_primary_scenario(self, seed: int) -> None:
        # Run the base joint kill/Byzantine assignment first, then replace only
        # the long-horizon family contract.  Both train and evaluation retain
        # the exact original event time requested for this diagnostic.
        PaperRecoveryEnv._configure_primary_scenario(self, seed)
        self.event_step = self.config.physics.evaluation_event_step
        if self.config.training_mode:
            families = (
                "heatwave_edge_saturation",
                "flooded_backhaul_repair",
                "emergency_uplink_thermal",
                "rolling_microgrid_instability",
            )
            self.disaster_family = families[
                int(self.rng_scenario.integers(0, len(families)))
            ]
        else:
            self.disaster_family = "heldout_compound_cyclone"
        self.family_parameters = self._family_contract(self.disaster_family)
        self.telemetry_delay_steps = int(
            self.family_parameters["telemetry_delay_steps"]
        )
        cells = np.arange(self.NUM_CELLS, dtype=np.int64)
        affected_count = int(self.family_parameters["affected_cells"])
        self.affected_cells = np.sort(
            self.rng_scenario.choice(cells, size=affected_count, replace=False)
        )
        remaining = np.setdiff1d(cells, self.affected_cells)
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
                "hardcore_family": self.disaster_family,
                "hardcore_parameters": copy.deepcopy(self.family_parameters),
                "public_rf_cells": int(self.NUM_CELLS),
                "affected_cells": self.affected_cells.tolist(),
                "aftershock_cells": self.aftershock_cells.tolist(),
                "aftershock_step": int(self.aftershock_step),
                "unavoidable_zero_tps_window": False,
                "committee_choice_changes_future_capacity": True,
                "endogenous_state_half_life_steps": {
                    "node_fatigue": 57,
                    "cell_debt": 86,
                },
                "delayed_public_telemetry_steps": self.telemetry_delay_steps,
                "network_delay_enters_edge_queue_before_certificate_close": True,
            }
        )
        self.scenario_record["scenario_hash"] = _canonical_hash(
            self.scenario_record
        )

    def _sample_offered_load(self) -> float:
        previous = float(getattr(self, "external_arrival_tps", 720.0))
        base_probability = float(
            self.family_parameters.get("high_load_probability", 18)
        ) / 100.0
        high_probability = 0.82 if previous > 820.0 else base_probability
        high = bool(self.rng_load.random() < high_probability)
        mean = 1_080.0 if high else 680.0
        deviation = 95.0 if high else 70.0
        return float(
            np.clip(self.rng_load.normal(mean, deviation), 480.0, 1_400.0)
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        self.observed_service_age = self.service_age.copy()
        observation = self._get_obs()
        info["observation_hash"] = _sha256_bytes(observation.tobytes())
        return observation, info

    def _actual_cell_headroom(self) -> np.ndarray:
        return np.clip(
            self.cell_health * (1.0 - 0.82 * self.cell_debt), 0.03, 1.0
        )

    def _recover_physical_capacity(self) -> None:
        rate = float(
            self.family_parameters.get("health_recovery_per_mille", 12)
        ) / 10_000.0
        self.cell_health += rate * (1.0 - self.cell_health)
        self.cell_health = np.clip(self.cell_health, 0.03, 1.0)

    def _preallocate_response_draws(self, regime: str) -> dict[str, Any]:
        draws = PaperRecoveryEnv._preallocate_response_draws(self, regime)
        cells = self.cell_ids
        headroom = self._actual_cell_headroom()[cells]
        fatigue = np.clip(self.node_fatigue, 0.0, 1.0)
        queue_pressure = float(
            np.clip(
                self.backlog_transactions / self.maximum_backlog_transactions,
                0.0,
                1.0,
            )
        )
        # Congested radio/backhaul work queues delay when a signature becomes
        # ready, so the calibrated certificate-close path is affected rather
        # than merely changing a telemetry-only network-delay field.
        edge_queue_multiplier = (
            1.0
            + 0.75 * fatigue
            + 1.40 * (1.0 - headroom)
            + 0.25 * queue_pressure
        )
        draws["signing_ms"] = np.asarray(draws["signing_ms"]) * edge_queue_multiplier
        draws["network_delay_ms"] = np.asarray(draws["network_delay_ms"]) * (
            1.0 + 1.60 * (1.0 - headroom) + 0.35 * fatigue
        )
        draws["hardcore_edge_queue_multiplier"] = edge_queue_multiplier
        return draws

    def _publish_delayed_capacity(self) -> None:
        self._public_state_queue.append(
            {
                "node_fatigue": self.node_fatigue.copy(),
                "cell_headroom": self._actual_cell_headroom().copy(),
                "service_age": self.service_age.copy(),
            }
        )
        if len(self._public_state_queue) > self.telemetry_delay_steps:
            visible = self._public_state_queue.pop(0)
            self.observed_node_fatigue = visible["node_fatigue"]
            self.observed_cell_headroom = visible["cell_headroom"]
            self.observed_service_age = visible["service_age"]

    def _update_endogenous_capacity(
        self, selected: np.ndarray, responded_ids: list[int], offered_load: float
    ) -> None:
        load_ratio = float(np.clip(offered_load / 1_000.0, 0.4, 1.6))
        selected_mask = np.zeros(self.num_nodes, dtype=np.float64)
        selected_mask[selected] = 1.0
        responded_mask = np.zeros(self.num_nodes, dtype=np.float64)
        responded_mask[np.asarray(responded_ids, dtype=np.int64)] = 1.0
        self.node_fatigue = np.clip(
            0.988 * self.node_fatigue
            + (0.006 + 0.008 * load_ratio) * selected_mask
            + 0.003 * responded_mask,
            0.0,
            1.0,
        )
        self.service_age = np.minimum(
            self.service_age + 1.0, self.MAX_SERVICE_AGE
        )
        self.service_age[selected] = 0.0
        counts = np.bincount(self.cell_ids[selected], minlength=self.NUM_CELLS)
        fractions = counts.astype(np.float64) / max(self.committee_size, 1)
        squared = np.square(fractions)
        spillover = np.roll(squared, 1) + np.roll(squared, -1)
        self.cell_debt = np.clip(
            0.992 * self.cell_debt
            + 0.030 * squared * (0.55 + 0.45 * load_ratio)
            + 0.006 * spillover * (0.55 + 0.45 * load_ratio),
            0.0,
            1.0,
        )
        # Consume a full-node duration tape every step.  The selected/fatigued
        # mask decides which pre-drawn durations take effect, preserving paired
        # RNG streams across policies while retaining an action consequence.
        durations = self.rng_thermal.integers(10, 31, size=self.num_nodes)
        tripped = selected[self.node_fatigue[selected] > 0.995]
        if len(tripped):
            self.outage_remaining[tripped] = np.maximum(
                self.outage_remaining[tripped], durations[tripped]
            )
            self.thermal_trip_count[tripped] += 1
        self._round_new_thermal_trips = tripped.astype(int).tolist()

    def _get_obs(self) -> np.ndarray:
        observation = super()._get_obs()
        boundary = self.num_nodes * self.config.observations.per_node_features
        node = observation[:boundary].reshape(self.num_nodes, -1).copy()
        node[:, 6] = self.cell_ids / max(self.NUM_CELLS - 1, 1)
        node[:, 7] = np.clip(1.0 - self.observed_node_fatigue, 0.0, 1.0)
        node[:, 8] = self.observed_cell_headroom[self.cell_ids]
        node[:, 11] = np.clip(
            self.observed_service_age / self.MAX_SERVICE_AGE, 0.0, 1.0
        )
        queue_pressure = float(
            np.clip(
                self.backlog_transactions / self.maximum_backlog_transactions,
                0.0,
                1.0,
            )
        )
        global_obs = observation[boundary:].copy()
        global_obs[0] = np.clip(self.last_offered_load_tps / 1_600.0, 0.0, 1.0)
        global_obs[2] = float(np.mean(self.observed_cell_headroom))
        global_obs[5] = queue_pressure
        return np.clip(
            np.concatenate((node.reshape(-1), global_obs)).astype(np.float32),
            0.0,
            1.0,
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        queue_pressure = float(
            np.clip(
                self.backlog_transactions / self.maximum_backlog_transactions,
                0.0,
                1.0,
            )
        )
        interval = self.config.physics.control_interval_seconds
        expired = max(self.backlog_transactions - 9_000_000.0, 0.0) * 0.002
        expired = min(expired, self.backlog_transactions)
        self.backlog_transactions -= expired
        self.total_expired_transactions += expired
        active_thermal = int(
            np.sum((self.outage_remaining > 0) & self.physical_alive)
        )
        info["reward_components"]["hardcore_queue_pressure"] = -20.0 * queue_pressure
        info["reward_components"]["hardcore_cell_debt"] = -8.0 * float(
            np.mean(self.cell_debt)
        )
        info["reward_components"]["hardcore_thermal_unavailability"] = (
            -15.0 * active_thermal / self.num_nodes
        )
        info["reward_components"]["hardcore_expired_transactions"] = (
            -4.0 * float(np.clip(expired / max(interval * 1_000.0, 1.0), 0.0, 1.0))
        )
        reward = float(sum(info["reward_components"].values()))
        info["reward_total"] = reward
        info["reward_recomputed"] = reward
        info["long_horizon"].update(
            {
                "hardcore": True,
                "service_age": self.service_age.tolist(),
                "new_thermal_trip_ids": self._round_new_thermal_trips,
                "active_thermal_unavailable_nodes": active_thermal,
                "thermal_trip_count_total": int(np.sum(self.thermal_trip_count)),
                "expired_transactions_this_step": float(expired),
                "expired_transactions_total": float(
                    self.total_expired_transactions
                ),
                "queue_pressure": queue_pressure,
                "backlog_transactions": self.backlog_transactions,
            }
        )
        observation = self._get_obs()
        info["post_observation_hash"] = _sha256_bytes(observation.tobytes())
        info["state_hash"] = _canonical_hash(
            {
                "parent_state_hash": info["state_hash"],
                "service_age": self.service_age.tolist(),
                "thermal_trip_count": self.thermal_trip_count.tolist(),
                "total_expired_transactions": self.total_expired_transactions,
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
                "action_dependent_future": [
                    "long_memory_node_fatigue",
                    "neighbour_spilling_rf_cell_debt",
                    "thermal_cooldown",
                    "service_age",
                    "persistent_queue_and_expiration",
                ],
                "partial_observability": "12_to_22_step_delayed_public_telemetry",
                "actor_disaster_flag": False,
                "actor_hidden_byzantine_labels": False,
                "future_realization_input": False,
            }
        )
        manifest["manifest_hash"] = _canonical_hash(manifest)
        return manifest
