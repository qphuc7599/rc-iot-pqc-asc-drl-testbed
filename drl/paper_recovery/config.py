"""Frozen scientific configuration for the Q-Relay recovery experiment."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class PhysicsConfig:
    num_nodes: int = 100
    committee_size: int = 21
    fault_budget: int = 6
    quorum: int = 14
    batch_size: int = 50
    horizon_steps: int = 2_000
    evaluation_event_step: int = 1_000
    training_event_step_low: int = 900
    training_event_step_high: int = 1_100
    permanent_kill_count: int = 30
    byzantine_count: int = 20
    killed_byzantine_count: int = 6
    surviving_byzantine_count: int = 14
    control_interval_seconds: float = 60.0
    certificate_deadline_ms: float = 500.0
    phy_rate_bps: float = 6_000_000.0
    ml_dsa_signature_bytes: int = 2_420
    application_item_bytes: int = 256
    gateway_verify_ms: float = 0.796
    battery_cutoff_ratio: float = 0.05
    background_churn: bool = True
    correlated_outages: bool = True
    ood_rf_blackout_duration_steps: int = 120
    ood_rf_blackout_count: int = 25
    primary_byzantine_behaviour: str = "withholding"
    packet_batching_mode: str = "rolling_commitment_one_signature_per_B_records"
    calibration_relative_path: str = "results/v12_ns3_final_validation/calibration.json"
    calibration_sha256: str = "3fb293cafbff33489db5a3e3cee007a0ebd730fbc08a30a81a4e74cb0c61c03c"


@dataclass(frozen=True)
class ObservationConfig:
    per_node_features: int = 13
    global_features: int = 6
    temporal_delta_window: int = 5
    trust_lcb_z: float = 2.0
    forbidden_inputs: tuple[str, ...] = (
        "byzantine_identity",
        "byzantine_count",
        "episode_progress",
        "event_time",
        "disaster_flag",
        "future_kill_membership",
        "hidden_disappearance_reason",
    )
    per_node_names: tuple[str, ...] = (
        "soc",
        "observed_reachable_previous_round",
        "response_beta_mean",
        "declared_compute_capability",
        "link_success_beta_mean",
        "response_latency_score",
        "previous_committee_membership",
        "recent_service_ewma",
        "response_posterior_uncertainty",
        "response_trust_lcb",
        "deadline_response_fast_ewma",
        "deadline_response_slow_ewma",
        "consecutive_deadline_miss_streak_normalized",
    )
    global_names: tuple[str, ...] = (
        "offered_load_normalized",
        "observed_reachable_ratio",
        "previous_turnover_ratio",
        "previous_service_success",
        "mean_link_success_posterior",
        "ambient_temperature_normalized",
    )


@dataclass(frozen=True)
class RewardConfig:
    service_weight: float = 48.0
    success_bonus: float = 8.0
    failure_penalty: float = -100.0
    response_margin_weight: float = 4.0
    reserve_target_responses_above_q: int = 4
    reserve_shortfall_weight: float = -30.0
    unavailability_weight: float = -6.0
    turnover_weight: float = -0.10
    energy_weight: float = -0.25
    service_normalizer_tps: float = 700.0
    energy_normalizer_wh: float = 0.05
    phase_conditioned: bool = False
    hidden_labels_used: bool = False


@dataclass(frozen=True)
class SeedConfig:
    development: tuple[int, ...] = (630001, 630002, 630003)
    formal_training: tuple[int, ...] = (630101, 630102, 630103, 630104, 630105, 630106)
    formal_replacements: tuple[int, ...] = (630111, 630112, 630113, 630114, 630115, 630116)
    canonical_evaluation: tuple[int, ...] = tuple(range(630201, 630221))
    ood_evaluation: tuple[int, ...] = tuple(range(630301, 630321))


@dataclass(frozen=True)
class BudgetConfig:
    formal_updates: int = 3_000
    transitions_per_update: int = 3_000
    persistent_workers: int = 11
    worker_large_slice: int = 273
    large_slice_workers: int = 8
    worker_small_slice: int = 272
    small_slice_workers: int = 3
    smoke_updates: int = 2
    profiler_updates: int = 64
    epochs_per_update: int = 10
    minibatch_size: int = 1_024
    formal_transitions_per_seed: int = 9_000_000
    max_formal_hours_per_seed: float = 12.0


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    entropy_coefficient_mean_per_pick: float = 0.03
    value_coefficient: float = 0.5
    gradient_clip_norm: float = 0.5
    target_kl: float = 0.03
    gumbel_temperature: float = 0.7
    actor_hidden: tuple[int, int] = (128, 64)
    critic_node_hidden: tuple[int, int] = (128, 128)
    critic_head_hidden: int = 128
    residual_score_scale: float = 2.0
    checkpoint_rule: str = "fixed_budget_final_only"
    resume_allowed: bool = False
    early_stop_allowed: bool = False
    best_checkpoint_allowed: bool = False


@dataclass(frozen=True)
class PaperRecoveryConfig:
    """One serialisable contract covering physics, observations, reward and budget."""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    observations: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    budgets: BudgetConfig = field(default_factory=BudgetConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    scenario_profile: str = "canonical"
    training_mode: bool = False
    offered_load_mode: str = "saturated"
    link_regime: str = "mixed"
    sensitivity_mode: bool = False
    formal_training_authorized: bool = False
    schema: str = "qrelay-paper-recovery-config-1"

    def __post_init__(self) -> None:
        p = self.physics
        expected_f = (p.committee_size - 1) // 3
        expected_q = math.ceil((p.committee_size + expected_f + 1) / 2)
        if p.fault_budget != expected_f or p.quorum != expected_q:
            raise ValueError(
                f"unsafe quorum contract: expected f={expected_f}, Q={expected_q}"
            )
        if p.killed_byzantine_count + p.surviving_byzantine_count != p.byzantine_count:
            raise ValueError("joint Byzantine partition does not sum to Byzantine count")
        if p.permanent_kill_count >= p.num_nodes - p.committee_size + 1:
            raise ValueError("permanent kill leaves fewer than k physical survivors")
        survivors = p.num_nodes - p.permanent_kill_count
        if not self.sensitivity_mode:
            if not math.isclose(p.byzantine_count / p.num_nodes, 0.20):
                raise ValueError("pre-event Byzantine prevalence must be 20%")
            if not math.isclose(p.surviving_byzantine_count / survivors, 0.20):
                raise ValueError("post-event Byzantine prevalence must be 20%")
        else:
            if not (0 <= p.killed_byzantine_count <= p.permanent_kill_count):
                raise ValueError("sensitivity killed-Byzantine count is infeasible")
            if not (0 <= p.surviving_byzantine_count <= survivors):
                raise ValueError("sensitivity surviving-Byzantine count is infeasible")
        if p.primary_byzantine_behaviour != "withholding":
            raise ValueError("primary paper scenario permits withholding only")
        if not 0 <= p.ood_rf_blackout_count <= p.num_nodes - p.permanent_kill_count:
            raise ValueError("OOD RF blackout count exceeds the surviving population")
        if self.observations.per_node_features != len(self.observations.per_node_names):
            raise ValueError("per-node observation schema length mismatch")
        if self.observations.global_features != len(self.observations.global_names):
            raise ValueError("global observation schema length mismatch")
        b = self.budgets
        if b.large_slice_workers * b.worker_large_slice + b.small_slice_workers * b.worker_small_slice != b.transitions_per_update:
            raise ValueError("ragged worker allocation must total exactly 3,000 transitions")
        if b.large_slice_workers + b.small_slice_workers != b.persistent_workers:
            raise ValueError("ragged worker counts must total 11")
        if b.formal_updates * b.transitions_per_update != b.formal_transitions_per_seed:
            raise ValueError("formal transition budget mismatch")
        if self.scenario_profile not in {
            "canonical",
            "training_primary",
            "ood_capability_kill",
            "ood_clustered_rf",
            "ood_bursty_overload",
            "ood_adaptive_byzantine",
            "ood_nakagami",
            "multihazard_training",
            "multihazard_evaluation",
        }:
            raise ValueError(f"unsupported scenario profile: {self.scenario_profile}")
        if self.offered_load_mode not in {"saturated", "stationary", "bursty"}:
            raise ValueError("unsupported offered-load mode")
        if self.link_regime not in {"mixed", "clean", "moderate", "poor"}:
            raise ValueError("unsupported link regime")
        if self.formal_training_authorized:
            raise ValueError(
                "formal training authorization is controlled by the pre-training gate, "
                "not by constructing PaperRecoveryConfig"
            )

    @property
    def observation_size(self) -> int:
        return (
            self.physics.num_nodes * self.observations.per_node_features
            + self.observations.global_features
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def with_overrides(self, **kwargs: Any) -> "PaperRecoveryConfig":
        """Return a validated top-level override for tests and registered panels."""

        return replace(self, **kwargs)
