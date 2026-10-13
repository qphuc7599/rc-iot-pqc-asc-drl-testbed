"""Compound cascading IoT collapse layered on the canonical 30% node kill.

The original event is retained exactly: 30 nodes are killed permanently at
control step 1000.  The extension adds public, stateful consequences without
giving the policy an event clock, disaster flag, hidden Byzantine label, or
future realization.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from drl.paper_recovery.env import (
    PaperRecoveryEnv,
    _canonical_hash,
    _sha256_bytes,
)


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


class CompoundCollapseEnv(PaperRecoveryEnv):
    """Realistic recovery problem with unavoidable collapse and endogenous debt."""

    MODEL_VERSION = "qrelay-compound-collapse-response-engine-1"

    def _init_state_arrays(self) -> None:
        super()._init_state_arrays()
        self.disaster_family = "regional_grid_backhaul"
        self.family_parameters: dict[str, int | float | bool] = {}
        self.gateway_blackout_remaining = 0
        self.gateway_recovery_remaining = 0
        self.aftershock_step = 1_300
        self.aftershock_ids = np.asarray([], dtype=np.int64)
        self.aftershock_triggered = False
        self.telemetry_delay_steps = 4
        self._telemetry_truth: dict[str, np.ndarray] | None = None
        self._telemetry_queue: list[dict[str, np.ndarray]] = []
        self.backlog_transactions = 0.0
        self.maximum_backlog_transactions = 1_200_000.0
        self.external_arrival_tps = 520.0
        self.recovery_debt_ms = 0.0
        self._round_forced_outage = False
        self._round_aftershock = False
        self._pending_selected = np.asarray([], dtype=np.int64)

    def _family_contract(self, family: str) -> dict[str, int | float | bool]:
        contracts: dict[str, dict[str, int | float | bool]] = {
            "regional_grid_backhaul": {
                "blackout_steps": 55,
                "recovery_ramp_steps": 220,
                "telemetry_delay_steps": 4,
                "aftershock_gap": 330,
                "aftershock_nodes": 18,
                "aftershock_duration": 80,
                "aftershock_gateway_steps": 8,
                "bursty_arrivals": False,
            },
            "gateway_power_brownout": {
                "blackout_steps": 40,
                "recovery_ramp_steps": 300,
                "telemetry_delay_steps": 3,
                "aftershock_gap": 260,
                "aftershock_nodes": 14,
                "aftershock_duration": 60,
                "aftershock_gateway_steps": 16,
                "bursty_arrivals": True,
            },
            "flooded_mesh_partition": {
                "blackout_steps": 75,
                "recovery_ramp_steps": 240,
                "telemetry_delay_steps": 7,
                "aftershock_gap": 360,
                "aftershock_nodes": 25,
                "aftershock_duration": 120,
                "aftershock_gateway_steps": 10,
                "bursty_arrivals": False,
            },
            "emergency_traffic_aftershock": {
                "blackout_steps": 35,
                "recovery_ramp_steps": 180,
                "telemetry_delay_steps": 5,
                "aftershock_gap": 220,
                "aftershock_nodes": 20,
                "aftershock_duration": 90,
                "aftershock_gateway_steps": 20,
                "bursty_arrivals": True,
            },
            "compound_cascade": {
                "blackout_steps": 90,
                "recovery_ramp_steps": 340,
                "telemetry_delay_steps": 8,
                "aftershock_gap": 280,
                "aftershock_nodes": 28,
                "aftershock_duration": 140,
                "aftershock_gateway_steps": 24,
                "bursty_arrivals": True,
            },
            "heldout_cyclone_gridlock": {
                "blackout_steps": 100,
                "recovery_ramp_steps": 380,
                "telemetry_delay_steps": 6,
                "aftershock_gap": 310,
                "aftershock_nodes": 26,
                "aftershock_duration": 150,
                "aftershock_gateway_steps": 30,
                "bursty_arrivals": True,
            },
        }
        return contracts[family].copy()

    def _configure_primary_scenario(self, seed: int) -> None:
        super()._configure_primary_scenario(seed)
        # Latest user contract fixes the original 30% kill at step 1000 even in
        # training; no event time or phase flag enters the observation.
        self.event_step = self.config.physics.evaluation_event_step
        if self.config.scenario_profile == "multihazard_training":
            families = (
                "regional_grid_backhaul",
                "gateway_power_brownout",
                "flooded_mesh_partition",
                "emergency_traffic_aftershock",
                "compound_cascade",
            )
            self.disaster_family = families[int(self.rng_scenario.integers(0, len(families)))]
        else:
            self.disaster_family = "heldout_cyclone_gridlock"
        self.family_parameters = self._family_contract(self.disaster_family)
        self.telemetry_delay_steps = int(
            self.family_parameters["telemetry_delay_steps"]
        )
        self.aftershock_step = self.event_step + int(
            self.family_parameters["aftershock_gap"]
        )
        survivors = np.setdiff1d(
            np.arange(self.num_nodes, dtype=np.int64),
            self.kill_ids,
            assume_unique=True,
        )
        aftershock_count = min(
            int(self.family_parameters["aftershock_nodes"]), len(survivors)
        )
        self.aftershock_ids = np.sort(
            self.rng_scenario.choice(survivors, size=aftershock_count, replace=False)
        ).astype(np.int64)
        self.scenario_record.update(
            {
                "event_step": int(self.event_step),
                "compound_disaster_family": self.disaster_family,
                "compound_parameters": copy.deepcopy(self.family_parameters),
                "aftershock_step": int(self.aftershock_step),
                "aftershock_ids": self.aftershock_ids.tolist(),
                "persistent_backlog": True,
                "delayed_public_telemetry_steps": self.telemetry_delay_steps,
                "forced_gateway_collapse": True,
                "original_30_percent_kill_retained": True,
            }
        )
        self.scenario_record["scenario_hash"] = _canonical_hash(self.scenario_record)

    def _sample_offered_load(self) -> float:
        bursty = bool(self.family_parameters.get("bursty_arrivals", False))
        previous = float(getattr(self, "external_arrival_tps", 520.0))
        if bursty:
            high = self.rng_load.random() < (0.65 if previous > 700.0 else 0.08)
            mean = 850.0 if high else 420.0
            return float(np.clip(self.rng_load.normal(mean, 70.0), 240.0, 1_100.0))
        return float(np.clip(self.rng_load.normal(480.0, 50.0), 300.0, 720.0))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        self.external_arrival_tps = float(self.last_offered_load_tps)
        self.last_offered_load_tps = self.external_arrival_tps
        observation = self._get_obs()
        info["observation_hash"] = _sha256_bytes(observation.tobytes())
        return observation, info

    def _apply_registered_event(self) -> list[int]:
        killed = super()._apply_registered_event()
        if self.step_count == self.event_step:
            self.gateway_blackout_remaining = int(
                self.family_parameters["blackout_steps"]
            )
            self.gateway_recovery_remaining = int(
                self.family_parameters["recovery_ramp_steps"]
            )
        return killed

    def _start_churn_events(self) -> list[int]:
        correlated = super()._start_churn_events()
        self._round_aftershock = False
        if self.step_count == self.aftershock_step:
            duration = int(self.family_parameters["aftershock_duration"]) + 1
            self.outage_remaining[self.aftershock_ids] = np.maximum(
                self.outage_remaining[self.aftershock_ids], duration
            )
            self.gateway_blackout_remaining = max(
                self.gateway_blackout_remaining,
                int(self.family_parameters["aftershock_gateway_steps"]),
            )
            self.gateway_recovery_remaining = max(
                self.gateway_recovery_remaining,
                int(self.family_parameters["recovery_ramp_steps"]) // 2,
            )
            self.aftershock_triggered = True
            self._round_aftershock = True
            return correlated + self.aftershock_ids.astype(int).tolist()
        return correlated

    def _simulate_certificate_round(
        self,
        selected: np.ndarray,
        actual_available: np.ndarray,
        active_byzantine: np.ndarray,
        draws: dict[str, Any],
    ) -> dict[str, Any]:
        self._round_forced_outage = self.gateway_blackout_remaining > 0
        if self._round_forced_outage:
            self.gateway_blackout_remaining -= 1
            return {
                "success": False,
                "round_latency_ms": self.config.physics.certificate_deadline_ms,
                "responded_ids": [],
                "nonresponded_ids": selected.astype(int).tolist(),
                "closing_ids": [],
                "response_arrivals_ms": [],
                "observable_response_margin": 0,
                "bytes_transmitted": 0,
            }
        result = super()._simulate_certificate_round(
            selected, actual_available, active_byzantine, draws
        )
        ramp_steps = max(int(self.family_parameters["recovery_ramp_steps"]), 1)
        if self.gateway_recovery_remaining > 0:
            progress = 1.0 - self.gateway_recovery_remaining / ramp_steps
            capacity_factor = float(np.clip(0.22 + 0.78 * progress, 0.22, 1.0))
            self.gateway_recovery_remaining -= 1
        else:
            capacity_factor = 1.0
        adjusted_latency = (
            float(result["round_latency_ms"]) / capacity_factor
            + float(self.recovery_debt_ms)
        )
        result["round_latency_ms"] = adjusted_latency
        if bool(result["success"]) and adjusted_latency > self.config.physics.certificate_deadline_ms:
            result["success"] = False
            result["closing_ids"] = []
        return result

    def _capture_telemetry(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)).copy() for name in TELEMETRY_KEYS}

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
        super()._update_observable_telemetry(selected, round_result, actual_available)
        truth = self._capture_telemetry()
        self._telemetry_truth = truth
        self._restore_telemetry(visible)
        self._telemetry_queue.append(truth)
        if len(self._telemetry_queue) > self.telemetry_delay_steps:
            self._restore_telemetry(self._telemetry_queue.pop(0))

    def _apply_energy(self, served_tps: float, responded_ids: list[int]) -> np.ndarray:
        drain = super()._apply_energy(served_tps, responded_ids)
        ids = self._pending_selected
        if len(ids):
            extra_per_node = 2.0e-5 * np.clip(self.recovery_debt_ms / 100.0, 0.0, 3.0)
            extra = np.zeros(self.num_nodes, dtype=np.float64)
            eligible = ids[self.physical_alive[ids]]
            extra[eligible] = extra_per_node
            self.batteries = np.maximum(0.0, self.batteries - extra)
            drain = drain + extra
            soc = self.batteries / np.maximum(self.node_bat_cap, 1e-12)
            newly_depleted = self.physical_alive & (
                soc <= self.config.physics.battery_cutoff_ratio
            )
            self.energy_depleted |= newly_depleted
            self.physical_alive[newly_depleted] = False
        return drain

    def _get_obs(self) -> np.ndarray:
        observation = super()._get_obs()
        backlog_ratio = np.clip(
            self.backlog_transactions / self.maximum_backlog_transactions,
            0.0,
            1.0,
        )
        observation[-1] = np.float32(backlog_ratio)
        return observation

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        candidate, _ = self.select_committee(action)
        self._pending_selected = candidate.copy()
        if self.step_count > 0 and len(candidate) == self.committee_size:
            current = np.zeros(self.num_nodes, dtype=bool)
            current[candidate] = True
            swaps = 0.5 * float(np.sum(current != self.in_committee))
            self.recovery_debt_ms = float(
                np.clip(0.94 * self.recovery_debt_ms + 1.6 * swaps, 0.0, 180.0)
            )
        external_arrival = float(self.external_arrival_tps)
        clearance_tps = self.backlog_transactions / 300.0
        effective_demand = float(np.clip(external_arrival + clearance_tps, 0.0, 1_600.0))
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
            np.clip(next_external + self.backlog_transactions / 300.0, 0.0, 1_600.0)
        )
        backlog_ratio = float(
            np.clip(
                self.backlog_transactions / self.maximum_backlog_transactions,
                0.0,
                1.0,
            )
        )
        recovery_ratio = float(np.clip(self.recovery_debt_ms / 180.0, 0.0, 1.0))
        info["reward_components"]["public_backlog"] = -24.0 * backlog_ratio
        info["reward_components"]["recovery_debt"] = -8.0 * recovery_ratio
        reward = float(sum(info["reward_components"].values()))
        info["reward_total"] = reward
        info["reward_recomputed"] = reward
        info["compound_collapse"] = {
            "family": self.disaster_family,
            "forced_gateway_outage_this_step": bool(self._round_forced_outage),
            "gateway_blackout_remaining": int(self.gateway_blackout_remaining),
            "gateway_recovery_remaining": int(self.gateway_recovery_remaining),
            "aftershock_this_step": bool(self._round_aftershock),
            "aftershock_step": int(self.aftershock_step),
            "backlog_transactions": self.backlog_transactions,
            "backlog_normalized": backlog_ratio,
            "external_arrival_tps": external_arrival,
            "effective_demand_tps": effective_demand,
            "telemetry_delay_steps": int(self.telemetry_delay_steps),
            "recovery_debt_ms": self.recovery_debt_ms,
        }
        observation = self._get_obs()
        info["post_observation_hash"] = _sha256_bytes(observation.tobytes())
        compound_state = {
            "base_state_hash": info["state_hash"],
            "backlog_transactions": self.backlog_transactions,
            "external_arrival_tps": self.external_arrival_tps,
            "gateway_blackout_remaining": self.gateway_blackout_remaining,
            "gateway_recovery_remaining": self.gateway_recovery_remaining,
            "recovery_debt_ms": self.recovery_debt_ms,
            "telemetry_queue_depth": len(self._telemetry_queue),
        }
        info["state_hash"] = _canonical_hash(compound_state)
        info["transition_hash"] = _canonical_hash(info)
        self.last_info = info
        return observation, reward, terminated, truncated, info

    def model_manifest(self) -> dict[str, Any]:
        manifest = super().model_manifest()
        manifest.update(
            {
                "model_version": self.MODEL_VERSION,
                "original_kill_contract": "30_percent_permanent_at_step_1000",
                "compound_state": [
                    "queue_backlog",
                    "gateway_recovery_ramp",
                    "committee_recovery_debt",
                    "delayed_public_node_telemetry",
                    "exogenous_aftershock",
                ],
                "forced_collapse": "all_methods_zero_tps_during_gateway_blackout",
                "actor_disaster_flag": False,
                "actor_hidden_labels": False,
            }
        )
        manifest["manifest_hash"] = _canonical_hash(manifest)
        return manifest
