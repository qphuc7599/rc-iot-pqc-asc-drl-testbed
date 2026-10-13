"""Standalone Q-Relay response-level environment for paper recovery.

The implementation follows the registered causal order exactly:

    lagged observation -> action -> event/kill -> response -> telemetry update

Hidden compromise identities are retained only in the scenario receipt and
offline security diagnostics.  They never enter observations or rewards.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import PaperRecoveryConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    )


@dataclass(frozen=True)
class DeviceProfile:
    sign_ms: float
    verify_ms: float
    cpu: float
    battery_wh: float
    mcu_idle_mw: float
    mcu_active_mw: float
    canonical_count: int


DEVICE_PROFILES: dict[str, DeviceProfile] = {
    "ESP32": DeviceProfile(24.645, 8.885, 0.333, 7.4, 16.0, 160.0, 25),
    "ESP32-S3": DeviceProfile(21.359, 7.700, 0.400, 7.4, 17.0, 170.0, 10),
    "STM32L4-M4": DeviceProfile(49.289, 17.770, 0.200, 7.2, 3.0, 30.0, 15),
    "STM32F4-M4": DeviceProfile(23.471, 8.462, 0.533, 7.4, 5.0, 50.0, 15),
    "STM32H7-M7": DeviceProfile(4.518, 1.629, 1.000, 20.0, 9.0, 90.0, 10),
    "RP2040": DeviceProfile(158.907, 44.000, 0.267, 7.4, 4.5, 45.0, 15),
    "nRF52840": DeviceProfile(61.611, 22.213, 0.133, 0.72, 0.023, 23.0, 10),
}


class PaperRecoveryEnv(gym.Env):
    """Observable-only committee-selection environment with frozen physics."""

    metadata = {"render_modes": []}
    MODEL_VERSION = "qrelay-paper-recovery-response-engine-1"
    CAUSAL_ORDER = (
        "lagged_observation",
        "candidate_equals_executed_action",
        "event_and_permanent_kill",
        "response_and_certificate_close",
        "reward_and_energy",
        "telemetry_update_for_next_observation",
    )

    def __init__(self, config: PaperRecoveryConfig | None = None):
        super().__init__()
        self.config = config or PaperRecoveryConfig()
        p = self.config.physics
        o = self.config.observations
        self.num_nodes = p.num_nodes
        self.committee_size = p.committee_size
        self.fault_budget = p.fault_budget
        self.quorum = p.quorum
        self.max_steps = p.horizon_steps
        self._base_seed = 0

        calibration_path = REPO_ROOT / p.calibration_relative_path
        if not calibration_path.is_file():
            raise FileNotFoundError(f"missing frozen response calibration: {calibration_path}")
        calibration_hash = _sha256_file(calibration_path)
        if calibration_hash != p.calibration_sha256:
            raise RuntimeError(
                f"response calibration hash mismatch: {calibration_hash}"
            )
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if not calibration.get("all_gates_pass", False):
            raise RuntimeError("frozen response calibration did not pass its gates")
        self.calibration_path = calibration_path
        self.calibration_sha256 = calibration_hash
        self.calibration_models = calibration["models"]

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_nodes,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.config.observation_size,),
            dtype=np.float32,
        )
        self._init_static_device_arrays()
        self._init_state_arrays()

        if o.per_node_features != 13 or o.global_features != 6:
            raise ValueError("the frozen policy interface requires 13 node and 6 global features")

    @staticmethod
    def safe_quorum(committee_size: int) -> int:
        f = max((int(committee_size) - 1) // 3, 0)
        return int(math.ceil((int(committee_size) + f + 1) / 2.0))

    def _scaled_device_names(self) -> np.ndarray:
        names = list(DEVICE_PROFILES)
        canonical = np.asarray(
            [DEVICE_PROFILES[name].canonical_count for name in names], dtype=np.float64
        )
        exact = canonical * self.num_nodes / canonical.sum()
        counts = np.floor(exact).astype(int)
        remainder = self.num_nodes - int(counts.sum())
        order = np.argsort(-(exact - counts), kind="stable")
        counts[order[:remainder]] += 1
        values: list[str] = []
        for name, count in zip(names, counts, strict=True):
            values.extend([name] * int(count))
        return np.asarray(values, dtype=object)

    def _init_static_device_arrays(self) -> None:
        self._canonical_device_names = self._scaled_device_names()
        profiles = [DEVICE_PROFILES[str(name)] for name in self._canonical_device_names]
        self._canonical_type_ids = np.asarray(
            [list(DEVICE_PROFILES).index(str(name)) for name in self._canonical_device_names],
            dtype=np.int16,
        )
        self._canonical_sign_ms = np.asarray([profile.sign_ms for profile in profiles])
        self._canonical_verify_ms = np.asarray([profile.verify_ms for profile in profiles])
        self._canonical_cpu = np.asarray([profile.cpu for profile in profiles])
        self._canonical_battery_wh = np.asarray([profile.battery_wh for profile in profiles])
        self._canonical_idle_mw = np.asarray([profile.mcu_idle_mw for profile in profiles])
        self._canonical_active_mw = np.asarray([profile.mcu_active_mw for profile in profiles])
        self.node_distance_m = np.asarray(
            [10.0 + 10.0 * (index % 5) for index in range(self.num_nodes)],
            dtype=np.float64,
        )
        angles = np.arange(self.num_nodes, dtype=np.float64) * (
            math.pi * (3.0 - math.sqrt(5.0))
        )
        self.node_x_m = self.node_distance_m * np.cos(angles)
        self.node_y_m = self.node_distance_m * np.sin(angles)

    def _assign_device_arrays(self, permutation: np.ndarray) -> None:
        self.device_names = self._canonical_device_names[permutation].copy()
        self.type_ids = self._canonical_type_ids[permutation].copy()
        self.node_sign_ms = self._canonical_sign_ms[permutation].copy()
        self.node_verify_ms = self._canonical_verify_ms[permutation].copy()
        self.node_cpu = self._canonical_cpu[permutation].copy()
        self.node_bat_cap = self._canonical_battery_wh[permutation].copy()
        self.node_mcu_idle_mw = self._canonical_idle_mw[permutation].copy()
        self.node_mcu_active_mw = self._canonical_active_mw[permutation].copy()

    def _init_state_arrays(self) -> None:
        n = self.num_nodes
        self.step_count = 0
        self.event_step = self.config.physics.evaluation_event_step
        self.physical_alive = np.ones(n, dtype=bool)
        self.permanent_dead = np.zeros(n, dtype=bool)
        self.observed_reachable = np.ones(n, dtype=bool)
        self.outage_remaining = np.zeros(n, dtype=np.int32)
        self.batteries = getattr(self, "node_bat_cap", self._canonical_battery_wh).copy()
        self.energy_depleted = np.zeros(n, dtype=bool)
        self.is_byzantine = np.zeros(n, dtype=bool)
        self.kill_ids = np.asarray([], dtype=np.int64)
        self.rf_blackout_ids = np.asarray([], dtype=np.int64)
        self.last_event_rf_blackout_ids: list[int] = []
        self.trust_alpha = np.full(n, 8.0, dtype=np.float64)
        self.trust_beta = np.full(n, 2.0, dtype=np.float64)
        self.link_alpha = np.full(n, 18.0, dtype=np.float64)
        self.link_beta = np.full(n, 2.0, dtype=np.float64)
        base_sign = getattr(self, "node_sign_ms", self._canonical_sign_ms)
        self.latency_ewma_ms = base_sign + 20.0
        self.service_ewma = np.zeros(n, dtype=np.float64)
        self.response_fast_ewma = np.full(n, 0.8, dtype=np.float64)
        self.response_slow_ewma = np.full(n, 0.8, dtype=np.float64)
        self.response_miss_streak = np.zeros(n, dtype=np.int16)
        self.in_committee = np.zeros(n, dtype=bool)
        self.last_turnover_ratio = 0.0
        self.last_service_success = 1.0
        self.last_offered_load_tps = 2_000.0
        self.current_temperature_c = 25.0
        self._link_regime_state = "moderate"
        self.scenario_record: dict[str, Any] = {}
        self.last_info: dict[str, Any] = {}

    def _spawn_rng_streams(self, seed: int) -> None:
        sequence = np.random.SeedSequence(int(seed))
        children = sequence.spawn(9)
        (
            self.rng_scenario,
            self.rng_compute,
            self.rng_link,
            self.rng_backoff,
            self.rng_churn,
            self.rng_load,
            self.rng_energy,
            self.rng_observation,
            self.rng_thermal,
        ) = [np.random.default_rng(child) for child in children]

    def _configure_primary_scenario(self, seed: int) -> None:
        p = self.config.physics
        if self.config.training_mode or self.config.scenario_profile == "training_primary":
            self.event_step = int(
                self.rng_scenario.integers(
                    p.training_event_step_low, p.training_event_step_high + 1
                )
            )
        else:
            self.event_step = p.evaluation_event_step

        all_ids = np.arange(self.num_nodes, dtype=np.int64)
        cluster_center_id: int | None = None
        if self.config.scenario_profile == "ood_capability_kill":
            # Capability-targeted means the highest public CPU-capability nodes
            # are removed; ties use the seeded permutation order.
            tie_break = self.rng_scenario.random(self.num_nodes)
            order = np.lexsort((tie_break, -self.node_cpu))
            self.kill_ids = np.sort(order[: p.permanent_kill_count]).astype(np.int64)
        elif self.config.scenario_profile == "ood_clustered_rf":
            cluster_center_id = int(self.rng_scenario.integers(0, self.num_nodes))
            distance_to_center = np.hypot(
                self.node_x_m - self.node_x_m[cluster_center_id],
                self.node_y_m - self.node_y_m[cluster_center_id],
            )
            spatial_order = np.lexsort((all_ids, distance_to_center))
            self.kill_ids = np.sort(
                spatial_order[: p.permanent_kill_count]
            ).astype(np.int64)
            surviving_order = [
                int(node_id)
                for node_id in spatial_order
                if int(node_id) not in set(self.kill_ids.tolist())
            ]
            self.rf_blackout_ids = np.sort(
                np.asarray(surviving_order[: p.ood_rf_blackout_count], dtype=np.int64)
            )
        else:
            self.kill_ids = np.sort(
                self.rng_scenario.choice(
                    all_ids, size=p.permanent_kill_count, replace=False
                ).astype(np.int64)
            )
        survivors = np.setdiff1d(all_ids, self.kill_ids, assume_unique=True)
        killed_byzantine = np.sort(
            self.rng_scenario.choice(
                self.kill_ids, size=p.killed_byzantine_count, replace=False
            ).astype(np.int64)
        )
        survivor_byzantine = np.sort(
            self.rng_scenario.choice(
                survivors, size=p.surviving_byzantine_count, replace=False
            ).astype(np.int64)
        )
        byzantine_ids = np.sort(
            np.concatenate((killed_byzantine, survivor_byzantine))
        ).astype(np.int64)
        self.is_byzantine[byzantine_ids] = True

        record = {
            "schema": "qrelay-paper-recovery-scenario-1",
            "seed": int(seed),
            "profile": self.config.scenario_profile,
            "training_mode": bool(self.config.training_mode),
            "event_step": int(self.event_step),
            "permanent_kill_ids": self.kill_ids.tolist(),
            "byzantine_ids": byzantine_ids.tolist(),
            "killed_byzantine_ids": killed_byzantine.tolist(),
            "surviving_byzantine_ids": survivor_byzantine.tolist(),
            "pre_event_byzantine_fraction": float(len(byzantine_ids) / self.num_nodes),
            "post_event_byzantine_fraction": float(len(survivor_byzantine) / len(survivors)),
            "post_event_promotion": False,
            "primary_behaviour": p.primary_byzantine_behaviour,
            "contagion": False,
            "dormant_activation": False,
            "hidden_labels_observed_or_rewarded": False,
            "causal_order": list(self.CAUSAL_ORDER),
            "cluster_center_id": cluster_center_id,
            "rf_blackout_ids": self.rf_blackout_ids.tolist(),
            "rf_blackout_duration_steps": (
                p.ood_rf_blackout_duration_steps
                if self.config.scenario_profile == "ood_clustered_rf"
                else 0
            ),
            "ood_dormant_on_off_latency_mimicry": bool(
                self.config.scenario_profile == "ood_adaptive_byzantine"
            ),
        }
        record["scenario_hash"] = _canonical_hash(record)
        self.scenario_record = record

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is None:
            seed = self._base_seed
            self._base_seed += 1
        self._spawn_rng_streams(int(seed))
        permutation = self.rng_scenario.permutation(self.num_nodes)
        self._assign_device_arrays(permutation)
        self._init_state_arrays()
        self.batteries = self.node_bat_cap.copy()
        self.latency_ewma_ms = self.node_sign_ms + 20.0
        self._configure_primary_scenario(int(seed))
        if self.config.scenario_profile == "ood_bursty_overload":
            self.last_offered_load_tps = self._sample_bursty_load()
        else:
            self.last_offered_load_tps = self._sample_offered_load()
        observation = self._get_obs()
        return observation, {
            "scenario_record": copy.deepcopy(self.scenario_record),
            "observation_hash": _sha256_bytes(observation.tobytes()),
            "config_hash": _canonical_hash(self.config.to_dict()),
        }

    def _sample_bursty_load(self) -> float:
        previous = float(getattr(self, "last_offered_load_tps", 450.0))
        high = self.rng_load.random() < (0.90 if previous > 700.0 else 0.18)
        mean = 1_150.0 if high else 380.0
        return float(np.clip(self.rng_load.normal(mean, 90.0), 150.0, 1_600.0))

    def _sample_offered_load(self) -> float:
        mode = self.config.offered_load_mode
        if mode == "saturated":
            return 2_000.0
        if mode == "stationary":
            return float(np.clip(self.rng_load.normal(550.0, 35.0), 300.0, 800.0))
        return self._sample_bursty_load()

    def _current_link_regime(self) -> str:
        if self.config.link_regime != "mixed":
            return self.config.link_regime
        previous = self._link_regime_state
        if self.rng_link.random() < 0.03:
            previous = str(
                self.rng_link.choice(
                    ("clean", "moderate", "poor"), p=(0.35, 0.45, 0.20)
                )
            )
        self._link_regime_state = previous
        return previous

    def _start_churn_events(self) -> list[int]:
        self.outage_remaining = np.maximum(self.outage_remaining - 1, 0)
        if not self.config.physics.background_churn:
            return []
        starts = (
            (self.outage_remaining == 0)
            & self.physical_alive
            & (self.rng_churn.random(self.num_nodes) < 5e-4)
        )
        durations = self.rng_churn.integers(2, 11, size=self.num_nodes)
        self.outage_remaining[starts] = durations[starts]
        correlated_ids: list[int] = []
        if (
            self.config.physics.correlated_outages
            and self.rng_churn.random() < 1e-3
        ):
            # The outage schedule is exogenous and therefore sampled from the
            # fixed node universe, never from a policy-dependent alive set.
            # Applying a scheduled outage to a dead node is a harmless no-op;
            # most importantly, every selector consumes the same RNG stream
            # and receives the same scheduled identifiers.
            count = min(int(self.rng_churn.integers(8, 26)), self.num_nodes)
            correlated_ids = self.rng_churn.choice(
                np.arange(self.num_nodes, dtype=np.int64),
                size=count,
                replace=False,
            ).astype(int).tolist()
            duration = int(self.rng_churn.integers(3, 13))
            self.outage_remaining[correlated_ids] = np.maximum(
                self.outage_remaining[correlated_ids], duration
            )
        return correlated_ids

    def _apply_registered_event(self) -> list[int]:
        self.last_event_rf_blackout_ids = []
        if self.step_count != self.event_step:
            return []
        ids = self.kill_ids[self.physical_alive[self.kill_ids]]
        self.physical_alive[ids] = False
        self.permanent_dead[ids] = True
        # A killed node becomes permanently unavailable, but its stored battery
        # energy does not physically disappear.  Keeping that energy in the
        # ledger makes pre-energy - measured drain - post-energy close exactly;
        # eligibility is controlled independently by ``physical_alive`` and
        # lagged reachability telemetry.
        if self.config.scenario_profile == "ood_clustered_rf" and len(self.rf_blackout_ids):
            # Add one because churn bookkeeping decrements at the start of the
            # same registered event round.
            duration = self.config.physics.ood_rf_blackout_duration_steps + 1
            self.outage_remaining[self.rf_blackout_ids] = np.maximum(
                self.outage_remaining[self.rf_blackout_ids], duration
            )
            self.last_event_rf_blackout_ids = self.rf_blackout_ids.astype(int).tolist()
        return ids.astype(int).tolist()

    def observable_action_mask(self, observation: np.ndarray | None = None) -> np.ndarray:
        if observation is None:
            soc = self.batteries / np.maximum(self.node_bat_cap, 1e-12)
            return self.observed_reachable & (soc > self.config.physics.battery_cutoff_ratio)
        values = np.asarray(observation, dtype=np.float32)
        node = values[: self.num_nodes * 13].reshape(self.num_nodes, 13)
        return (node[:, 1] > 0.5) & (
            node[:, 0] > self.config.physics.battery_cutoff_ratio
        )

    def select_committee(
        self, action_scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(action_scores, dtype=np.float64).reshape(-1)
        if scores.shape != (self.num_nodes,):
            raise ValueError(f"expected {self.num_nodes} action scores")
        mask = self.observable_action_mask()
        if int(mask.sum()) < self.committee_size:
            return np.asarray([], dtype=np.int64), mask
        safe_scores = np.nan_to_num(
            scores, nan=-np.inf, posinf=1e12, neginf=-1e12
        )
        order = np.lexsort((np.arange(self.num_nodes), -safe_scores))
        selected = order[mask[order]][: self.committee_size]
        return selected.astype(np.int64), mask

    def _active_byzantine_mask(self) -> np.ndarray:
        if self.config.scenario_profile != "ood_adaptive_byzantine":
            return self.is_byzantine & self.physical_alive
        # Registered OOD-only behaviour.  No such mode enters primary training.
        ids = np.flatnonzero(self.is_byzantine)
        active = np.zeros(self.num_nodes, dtype=bool)
        phases = (self.step_count + ids * 17) % 120
        active[ids] = (phases >= 80) | (
            (self.step_count >= self.event_step - 30)
            & (self.step_count <= self.event_step + 90)
        )
        return active & self.physical_alive

    def _byzantine_delay_ms(self) -> np.ndarray:
        values = np.zeros(self.num_nodes, dtype=np.float64)
        if self.config.scenario_profile != "ood_adaptive_byzantine":
            return values
        ids = np.flatnonzero(self.is_byzantine & self.physical_alive)
        if len(ids):
            delayed = ids[((self.step_count + ids * 17) % 120) < 40]
            values[delayed] = self.rng_backoff.uniform(180.0, 330.0, size=len(delayed))
        return values

    def _preallocate_response_draws(self, regime: str) -> dict[str, Any]:
        p = self.config.physics
        model = self.calibration_models[regime]
        distance_keys = [
            str(int(round(value / 10.0) * 10)) for value in self.node_distance_m
        ]
        shared_delivery = float(self.rng_link.normal())
        logits = np.asarray(
            [model["marginal_logit_intercept_by_distance_m"][key] for key in distance_keys]
        ) + float(model["shared_logit_sigma"]) * shared_delivery
        success_probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        if self.config.scenario_profile == "ood_nakagami":
            fading = self.rng_link.gamma(shape=1.5, scale=1.0 / 1.5, size=self.num_nodes)
            success_probability = np.clip(success_probability * np.minimum(fading, 1.2), 0.0, 1.0)
        link_uniform = self.rng_link.random(self.num_nodes)

        cv = 0.35
        sigma = math.sqrt(math.log(1.0 + cv * cv))
        mu = np.log(np.maximum(self.node_sign_ms, 1e-6)) - 0.5 * sigma * sigma
        signing_ms = self.rng_compute.lognormal(mu, sigma)
        network_log_mean = np.asarray(
            [model["network_delay_log_mean_by_distance_m"][key] for key in distance_keys]
        )
        network_log_sigma = np.asarray(
            [model["network_delay_individual_log_sigma_by_distance_m"][key] for key in distance_keys]
        )
        shared_network = float(self.rng_link.normal())
        network_delay_ms = np.exp(
            network_log_mean
            + float(model["network_delay_shared_log_sigma"]) * shared_network
            + network_log_sigma * self.rng_link.normal(size=self.num_nodes)
        )
        residuals = model["certificate_close_residual_ms_samples"]
        close_residual_ms = float(residuals[int(self.rng_link.integers(0, len(residuals)))])
        return {
            "success_probability": success_probability,
            "link_success": link_uniform < success_probability,
            "signing_ms": signing_ms,
            "network_delay_ms": network_delay_ms,
            "certificate_close_intercept_ms": float(model["certificate_close_intercept_ms"]),
            "certificate_close_per_qth_ready_slope": float(
                model["certificate_close_per_qth_ready_slope"]
            ),
            "certificate_close_residual_ms": close_residual_ms,
            "byzantine_delay_ms": self._byzantine_delay_ms(),
            "gateway_verify_ms": p.gateway_verify_ms,
        }

    def _simulate_certificate_round(
        self,
        selected: np.ndarray,
        actual_available: np.ndarray,
        active_byzantine: np.ndarray,
        draws: dict[str, Any],
    ) -> dict[str, Any]:
        p = self.config.physics
        response_candidates: list[tuple[float, int]] = []
        for node_id in selected:
            index = int(node_id)
            if (
                not actual_available[index]
                or active_byzantine[index]
                or not draws["link_success"][index]
            ):
                continue
            ready_ms = (
                draws["signing_ms"][index]
                + draws["byzantine_delay_ms"][index]
            )
            response_candidates.append((float(ready_ms), index))
        response_candidates.sort(key=lambda item: (item[0], item[1]))
        raw_arrivals = {
            index: float(ready_ms + draws["network_delay_ms"][index])
            for ready_ms, index in response_candidates
        }

        success = False
        latency_ms = p.certificate_deadline_ms
        closing_ids: list[int] = []
        arrivals: list[tuple[float, int]] = []
        if len(response_candidates) >= self.quorum:
            q_candidates = response_candidates[: self.quorum]
            calibrated_close = (
                draws["certificate_close_intercept_ms"]
                + draws["certificate_close_per_qth_ready_slope"]
                * q_candidates[-1][0]
                + draws["certificate_close_residual_ms"]
            )
            physical_lower_bound = max(
                ready_ms
                + (self.quorum - rank) * p.gateway_verify_ms
                + 1.0
                for rank, (ready_ms, _) in enumerate(q_candidates)
            )
            calibrated_close = max(calibrated_close, physical_lower_bound)
            if calibrated_close <= p.certificate_deadline_ms:
                success = True
                latency_ms = float(calibrated_close)
                closing_ids = [int(index) for _, index in q_candidates]
                for rank, (_, index) in enumerate(q_candidates):
                    planned = (
                        calibrated_close
                        - 1.0
                        - (self.quorum - rank) * p.gateway_verify_ms
                    )
                    arrivals.append((float(planned), int(index)))
                for _, index in response_candidates[self.quorum :]:
                    arrivals.append(
                        (float(max(raw_arrivals[index], calibrated_close)), int(index))
                    )
            else:
                arrivals = sorted((raw_arrivals[index], index) for _, index in response_candidates)
                for rank in range(self.quorum - 1, len(arrivals)):
                    arrivals[rank] = (
                        max(arrivals[rank][0], p.certificate_deadline_ms + 1.0),
                        arrivals[rank][1],
                    )
        else:
            arrivals = [(raw_arrivals[index], index) for _, index in response_candidates]
        arrivals.sort(key=lambda item: (item[0], item[1]))
        before_deadline = [
            item for item in arrivals if item[0] <= p.certificate_deadline_ms
        ]
        responded = [int(item[1]) for item in before_deadline]
        nonresponded = sorted(set(selected.astype(int).tolist()) - set(responded))
        return {
            "success": bool(success),
            "round_latency_ms": float(latency_ms),
            "responded_ids": responded,
            "nonresponded_ids": nonresponded,
            "closing_ids": closing_ids,
            "response_arrivals_ms": [
                {"node_id": int(index), "arrival_ms": float(value)}
                for value, index in arrivals
            ],
            "observable_response_margin": max(len(before_deadline) - self.quorum, 0),
            "bytes_transmitted": len(response_candidates) * p.ml_dsa_signature_bytes,
        }

    def _apply_energy(self, served_tps: float, responded_ids: list[int]) -> np.ndarray:
        p = self.config.physics
        interval_h = p.control_interval_seconds / 3600.0
        online = self.physical_alive & (self.outage_remaining == 0)
        baseline_mw = self.node_mcu_idle_mw + 0.05 * 290.0
        drain = np.zeros(self.num_nodes, dtype=np.float64)
        drain[online] = baseline_mw[online] * interval_h / 1000.0
        ids = np.asarray(responded_ids, dtype=np.int64)
        if len(ids):
            rounds_s = max(float(served_tps), 0.0) / p.batch_size
            busy = np.clip(rounds_s * self.node_sign_ms[ids] / 1000.0, 0.0, 1.0)
            active_extra = np.maximum(
                self.node_mcu_active_mw[ids] - self.node_mcu_idle_mw[ids], 0.0
            )
            drain[ids] += active_extra * busy * interval_h / 1000.0
            tx_duty = np.clip(
                rounds_s * p.ml_dsa_signature_bytes * 8.0 / p.phy_rate_bps,
                0.0,
                1.0,
            )
            drain[ids] += 660.0 * tx_duty * interval_h / 1000.0
        self.batteries = np.maximum(0.0, self.batteries - drain)
        soc = self.batteries / np.maximum(self.node_bat_cap, 1e-12)
        newly_depleted = self.physical_alive & (soc <= p.battery_cutoff_ratio)
        self.energy_depleted |= newly_depleted
        self.physical_alive[newly_depleted] = False
        return drain

    def _update_observable_telemetry(
        self,
        selected: np.ndarray,
        round_result: dict[str, Any],
        actual_available: np.ndarray,
    ) -> None:
        responded = np.asarray(round_result["responded_ids"], dtype=np.int64)
        nonresponded = np.asarray(round_result["nonresponded_ids"], dtype=np.int64)
        if len(responded):
            self.trust_alpha[responded] += 1.0
            self.link_alpha[responded] += 1.0
            arrival_by_node = {
                row["node_id"]: row["arrival_ms"]
                for row in round_result["response_arrivals_ms"]
            }
            values = np.asarray([arrival_by_node[int(index)] for index in responded])
            self.latency_ewma_ms[responded] = (
                0.90 * self.latency_ewma_ms[responded] + 0.10 * values
            )
        if len(nonresponded):
            self.trust_beta[nonresponded] += 1.0
            self.link_beta[nonresponded] += 1.0
        if len(selected):
            indicator = np.zeros(self.num_nodes, dtype=np.float64)
            indicator[responded] = 1.0
            self.response_fast_ewma[selected] = (
                0.70 * self.response_fast_ewma[selected] + 0.30 * indicator[selected]
            )
            self.response_slow_ewma[selected] = (
                0.97 * self.response_slow_ewma[selected] + 0.03 * indicator[selected]
            )
        if len(responded):
            self.response_miss_streak[responded] = 0
        if len(nonresponded):
            self.response_miss_streak[nonresponded] = np.minimum(
                self.response_miss_streak[nonresponded] + 1, 8
            )
        self.service_ewma *= 0.98
        if len(selected):
            self.service_ewma[selected] += 0.02
        self.observed_reachable = actual_available.copy()

    def _get_obs(self) -> np.ndarray:
        p = self.config.physics
        o = self.config.observations
        soc = np.clip(self.batteries / np.maximum(self.node_bat_cap, 1e-12), 0.0, 1.0)
        trust_mean = self.trust_alpha / (self.trust_alpha + self.trust_beta)
        trust_var = (
            self.trust_alpha
            * self.trust_beta
            / (
                (self.trust_alpha + self.trust_beta) ** 2
                * (self.trust_alpha + self.trust_beta + 1.0)
            )
        )
        trust_lcb = np.clip(
            trust_mean - o.trust_lcb_z * np.sqrt(trust_var), 0.0, 1.0
        )
        link_mean = self.link_alpha / (self.link_alpha + self.link_beta)
        latency_score = np.clip(1.0 / (1.0 + self.latency_ewma_ms / 100.0), 0.0, 1.0)
        uncertainty = np.clip(
            1.0 / np.sqrt(self.trust_alpha + self.trust_beta), 0.0, 1.0
        )
        node = np.column_stack(
            (
                soc,
                self.observed_reachable.astype(np.float64),
                trust_mean,
                self.node_cpu,
                link_mean,
                latency_score,
                self.in_committee.astype(np.float64),
                np.clip(self.service_ewma, 0.0, 1.0),
                uncertainty,
                trust_lcb,
                np.clip(self.response_fast_ewma, 0.0, 1.0),
                np.clip(self.response_slow_ewma, 0.0, 1.0),
                np.clip(self.response_miss_streak / 8.0, 0.0, 1.0),
            )
        )
        global_obs = np.asarray(
            [
                np.clip(self.last_offered_load_tps / 1_200.0, 0.0, 1.0),
                float(np.mean(self.observed_reachable)),
                np.clip(self.last_turnover_ratio, 0.0, 1.0),
                np.clip(self.last_service_success, 0.0, 1.0),
                float(np.mean(link_mean)),
                np.clip((self.current_temperature_c - 10.0) / 55.0, 0.0, 1.0),
            ],
            dtype=np.float64,
        )
        observation = np.concatenate((node.reshape(-1), global_obs)).astype(np.float32)
        if observation.shape != (self.config.observation_size,):
            raise RuntimeError("observation schema drift")
        return np.clip(observation, 0.0, 1.0)

    def _state_receipt(self) -> dict[str, Any]:
        state = {
            "step_count": int(self.step_count),
            "physical_alive": self.physical_alive.astype(np.uint8).tolist(),
            "permanent_dead": self.permanent_dead.astype(np.uint8).tolist(),
            "observed_reachable": self.observed_reachable.astype(np.uint8).tolist(),
            "outage_remaining": self.outage_remaining.astype(int).tolist(),
            "batteries": self.batteries.tolist(),
            "trust_alpha": self.trust_alpha.tolist(),
            "trust_beta": self.trust_beta.tolist(),
            "link_alpha": self.link_alpha.tolist(),
            "link_beta": self.link_beta.tolist(),
            "latency_ewma_ms": self.latency_ewma_ms.tolist(),
            "service_ewma": self.service_ewma.tolist(),
            "response_fast_ewma": self.response_fast_ewma.tolist(),
            "response_slow_ewma": self.response_slow_ewma.tolist(),
            "response_miss_streak": self.response_miss_streak.astype(int).tolist(),
            "in_committee": self.in_committee.astype(np.uint8).tolist(),
            "last_turnover_ratio": float(self.last_turnover_ratio),
            "last_service_success": float(self.last_service_success),
            "last_offered_load_tps": float(self.last_offered_load_tps),
        }
        return {"state_hash": _canonical_hash(state), "state": state}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        p = self.config.physics
        r = self.config.reward
        pre_observation = self._get_obs()
        pre_energy_wh = float(np.sum(self.batteries))
        previous_committee = self.in_committee.copy()

        candidate, observable_mask = self.select_committee(action)
        executed = candidate.copy()  # immutable no-shield contract
        killed_ids = self._apply_registered_event()
        rf_blackout_ids = list(self.last_event_rf_blackout_ids)
        correlated_outage_ids = self._start_churn_events()
        actual_available = (
            self.physical_alive
            & (self.outage_remaining == 0)
            & (
                self.batteries / np.maximum(self.node_bat_cap, 1e-12)
                > p.battery_cutoff_ratio
            )
        )
        active_byzantine = self._active_byzantine_mask()
        regime = self._current_link_regime()
        draws = self._preallocate_response_draws(regime)

        underfilled = len(executed) < self.committee_size
        if underfilled:
            round_result = {
                "success": False,
                "round_latency_ms": p.certificate_deadline_ms,
                "responded_ids": [],
                "nonresponded_ids": executed.astype(int).tolist(),
                "closing_ids": [],
                "response_arrivals_ms": [],
                "observable_response_margin": 0,
                "bytes_transmitted": 0,
            }
        else:
            round_result = self._simulate_certificate_round(
                executed, actual_available, active_byzantine, draws
            )

        capacity_tps = (
            1_000.0 * p.batch_size / max(round_result["round_latency_ms"], 1e-12)
            if round_result["success"]
            else 0.0
        )
        offered_load = float(self.last_offered_load_tps)
        served_tps = (
            capacity_tps
            if self.config.offered_load_mode == "saturated"
            else min(capacity_tps, offered_load)
        )
        next_offered_load = (
            self._sample_bursty_load()
            if self.config.scenario_profile == "ood_bursty_overload"
            else self._sample_offered_load()
        )
        exogenous_draw_receipt = {
            "step": int(self.step_count),
            "link_regime": regime,
            "event_killed_node_ids": killed_ids,
            "event_rf_blackout_node_ids": rf_blackout_ids,
            "correlated_outage_node_ids": correlated_outage_ids,
            "success_probability": np.asarray(draws["success_probability"]).tolist(),
            "link_success": np.asarray(draws["link_success"], dtype=np.uint8).tolist(),
            "signing_ms": np.asarray(draws["signing_ms"]).tolist(),
            "network_delay_ms": np.asarray(draws["network_delay_ms"]).tolist(),
            "certificate_close_intercept_ms": draws["certificate_close_intercept_ms"],
            "certificate_close_per_qth_ready_slope": draws[
                "certificate_close_per_qth_ready_slope"
            ],
            "certificate_close_residual_ms": draws["certificate_close_residual_ms"],
            "byzantine_delay_ms": np.asarray(draws["byzantine_delay_ms"]).tolist(),
            "next_offered_load_tps": float(next_offered_load),
        }
        exogenous_draw_hash = _canonical_hash(exogenous_draw_receipt)

        drain = self._apply_energy(served_tps, round_result["responded_ids"])
        prior_unavailable_ratio = float(
            np.sum(previous_committee & ~actual_available)
            / max(self.committee_size, 1)
        )
        current_committee = np.zeros(self.num_nodes, dtype=bool)
        current_committee[executed] = True
        swaps = int(np.sum(previous_committee != current_committee))
        turnover_ratio = swaps / max(2.0 * self.committee_size, 1.0)
        margin_ratio = round_result["observable_response_margin"] / max(
            self.committee_size - self.quorum, 1
        )
        reserve_shortfall_ratio = max(
            r.reserve_target_responses_above_q
            - round_result["observable_response_margin"],
            0,
        ) / max(r.reserve_target_responses_above_q, 1)
        reward_components = {
            "service": r.service_weight
            * float(np.clip(served_tps / r.service_normalizer_tps, 0.0, 2.0)),
            "success": r.success_bonus if round_result["success"] else r.failure_penalty,
            "response_margin": r.response_margin_weight * margin_ratio,
            "reserve_shortfall": r.reserve_shortfall_weight * reserve_shortfall_ratio,
            "unavailability": r.unavailability_weight * prior_unavailable_ratio,
            "turnover": r.turnover_weight * turnover_ratio,
            "energy": r.energy_weight
            * float(np.clip(float(np.sum(drain)) / r.energy_normalizer_wh, 0.0, 4.0)),
        }
        reward = float(sum(reward_components.values()))

        # Only now expose response-derived telemetry to the next decision.
        self._update_observable_telemetry(executed, round_result, actual_available)
        self.in_committee = current_committee
        self.last_turnover_ratio = float(turnover_ratio)
        self.last_service_success = float(round_result["success"])
        self.last_offered_load_tps = float(next_offered_load)

        compromised_selected = int(np.sum(self.is_byzantine[executed])) if len(executed) else 0
        active_selected = int(np.sum(active_byzantine[executed])) if len(executed) else 0
        closing_reason = "closed_q_responses_before_deadline"
        if underfilled:
            closing_reason = "fail_closed_fewer_than_k_observable_eligible"
        elif not round_result["success"]:
            if len(round_result["responded_ids"]) < self.quorum:
                closing_reason = "fewer_than_q_valid_responses_before_deadline"
            else:
                closing_reason = "qth_close_latency_exceeded_deadline"
        if not round_result["success"] and served_tps != 0.0:
            raise RuntimeError("positive TPS emitted despite failed closing condition")

        self.step_count += 1
        terminated = bool(self.observable_action_mask().sum() < self.committee_size)
        truncated = bool(self.step_count >= self.max_steps)
        observation = self._get_obs()
        state_receipt = self._state_receipt()
        post_energy_wh = float(np.sum(self.batteries))
        info: dict[str, Any] = {
            "schema": "qrelay-paper-recovery-transition-1",
            "step": int(self.step_count - 1),
            "causal_order": list(self.CAUSAL_ORDER),
            "scenario_hash": self.scenario_record["scenario_hash"],
            "exogenous_draw_hash": exogenous_draw_hash,
            "pre_observation_hash": _sha256_bytes(pre_observation.tobytes()),
            "post_observation_hash": _sha256_bytes(observation.tobytes()),
            "observable_action_mask": observable_mask.astype(bool).tolist(),
            "observable_eligible_count": int(observable_mask.sum()),
            "candidate_committee_ids": candidate.astype(int).tolist(),
            "executed_committee_ids": executed.astype(int).tolist(),
            "candidate_executed_match": bool(np.array_equal(candidate, executed)),
            "committee_size_realized": int(len(executed)),
            "signing_quorum": int(self.quorum),
            "event_killed_node_ids": killed_ids,
            "event_killed": len(killed_ids),
            "event_rf_blackout_node_ids": rf_blackout_ids,
            "correlated_outage_node_ids": correlated_outage_ids,
            "certificate_success": bool(round_result["success"]),
            "closing_condition": closing_reason,
            "round_latency_ms": float(round_result["round_latency_ms"]),
            "certificate_deadline_ms": float(p.certificate_deadline_ms),
            "responded_node_ids": round_result["responded_ids"],
            "nonresponding_node_ids": round_result["nonresponded_ids"],
            "closing_certificate_node_ids": round_result["closing_ids"],
            "observable_response_margin": int(round_result["observable_response_margin"]),
            "bytes_transmitted": int(round_result["bytes_transmitted"]),
            "offered_load_tps": offered_load,
            "capacity_tps": float(capacity_tps),
            "served_tps": float(served_tps),
            "tps": float(served_tps),
            "link_regime": regime,
            "energy": {
                "pre_total_wh": pre_energy_wh,
                "drain_total_wh": float(np.sum(drain)),
                "post_total_wh": post_energy_wh,
                "per_node_drain_wh": drain.tolist(),
                "mass_balance_error_wh": float(pre_energy_wh - float(np.sum(drain)) - post_energy_wh),
            },
            "reward_components": {key: float(value) for key, value in reward_components.items()},
            "reward_total": reward,
            "reward_recomputed": float(sum(reward_components.values())),
            "prior_committee_unavailability_ratio": prior_unavailable_ratio,
            "committee_turnover_ratio": float(turnover_ratio),
            "offline_security": {
                "hidden_labels_used_by_observation_or_reward": False,
                "total_compromised_in_committee": compromised_selected,
                "active_withholders_in_committee": active_selected,
                "security_budget_exceeded": bool(compromised_selected > self.fault_budget),
            },
            "physical_alive": int(np.sum(self.physical_alive)),
            "observed_reachable": int(np.sum(self.observed_reachable)),
            "actual_reachable": int(np.sum(actual_available)),
            "energy_depleted_nodes": int(np.sum(self.energy_depleted)),
            "state_hash": state_receipt["state_hash"],
        }
        info["transition_hash"] = _canonical_hash(info)
        self.last_info = info
        return observation, reward, terminated, truncated, info

    def model_manifest(self) -> dict[str, Any]:
        manifest = {
            "schema": "qrelay-paper-recovery-environment-manifest-1",
            "model_version": self.MODEL_VERSION,
            "config": self.config.to_dict(),
            "calibration": {
                "path": str(self.calibration_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": self.calibration_sha256,
                "status": "heldout_ns3_calibration_reused_without_rerun",
            },
            "causal_order": list(self.CAUSAL_ORDER),
            "candidate_equals_executed": True,
            "execution_shield": False,
            "masked_filler": False,
            "hidden_labels_observed_or_rewarded": False,
            "primary_contagion": False,
            "primary_dormant_activation": False,
            "primary_event_time_input": False,
            "primary_disaster_flag": False,
            "device_mix": {
                name: profile.canonical_count for name, profile in DEVICE_PROFILES.items()
            },
            "observation_features": {
                "per_node": list(self.config.observations.per_node_names),
                "global": list(self.config.observations.global_names),
                "forbidden": list(self.config.observations.forbidden_inputs),
            },
            "reward_contract": self.config.reward.__dict__,
        }
        manifest["manifest_hash"] = _canonical_hash(manifest)
        return manifest
