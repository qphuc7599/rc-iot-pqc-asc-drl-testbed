"""Frozen public schema and optimizer contract for hardcore v5."""

from __future__ import annotations

from dataclasses import replace

from drl.paper_recovery.config import PaperRecoveryConfig


NODE_FEATURES = (
    "soc",
    "observed_reachable_previous_round",
    "response_beta_mean",
    "declared_compute_capability",
    "link_success_beta_mean",
    "response_latency_score",
    "public_rf_cell_id_normalized",
    "public_node_recovery_headroom_delayed",
    "public_cell_capacity_headroom_delayed",
    "response_trust_lcb",
    "deadline_response_fast_ewma",
    "public_service_age_delayed",
    "consecutive_deadline_miss_streak_normalized",
)

GLOBAL_FEATURES = (
    "effective_demand_with_backlog_normalized",
    "observed_reachable_ratio",
    "public_network_reserve_delayed",
    "previous_service_success",
    "mean_link_success_posterior",
    "public_queue_pressure_normalized",
)


def hardcore_config(*, training: bool) -> PaperRecoveryConfig:
    base = PaperRecoveryConfig()
    observations = replace(
        base.observations,
        temporal_delta_window=64,
        per_node_names=NODE_FEATURES,
        global_names=GLOBAL_FEATURES,
    )
    budgets = replace(
        base.budgets,
        smoke_updates=2,
        profiler_updates=500,
        formal_updates=3_000,
        epochs_per_update=2,
        minibatch_size=1_024,
        formal_transitions_per_seed=9_000_000,
    )
    ppo = replace(
        base.ppo,
        learning_rate=3e-5,
        gamma=0.995,
        gae_lambda=0.97,
        clip_epsilon=0.10,
        entropy_coefficient_mean_per_pick=0.025,
        target_kl=0.015,
        gumbel_temperature=0.9,
    )
    return replace(
        base,
        observations=observations,
        budgets=budgets,
        ppo=ppo,
        scenario_profile=(
            "multihazard_training" if training else "multihazard_evaluation"
        ),
        training_mode=bool(training),
        offered_load_mode="stationary",
        formal_training_authorized=False,
        schema="qrelay-reviewer-hardcore-config-5",
    )
