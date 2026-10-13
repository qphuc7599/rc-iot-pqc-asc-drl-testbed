"""Configuration factory for the fixed 30%-kill compound IoT collapse."""

from __future__ import annotations

from dataclasses import replace

from drl.paper_recovery.config import PaperRecoveryConfig


def multihazard_config(*, training: bool) -> PaperRecoveryConfig:
    base = PaperRecoveryConfig()
    observations = replace(
        base.observations,
        temporal_delta_window=5,
        global_names=(
            "effective_demand_with_backlog_normalized",
            "observed_reachable_ratio",
            "previous_turnover_ratio",
            "previous_service_success",
            "mean_link_success_posterior",
            "public_queue_backlog_normalized",
        ),
    )
    budgets = replace(
        base.budgets,
        smoke_updates=2,
        profiler_updates=64,
        formal_updates=3_000,
        epochs_per_update=4,
        minibatch_size=1024,
        formal_transitions_per_seed=9_000_000,
    )
    ppo = replace(
        base.ppo,
        learning_rate=5e-5,
        clip_epsilon=0.20,
        entropy_coefficient_mean_per_pick=0.01,
        gradient_clip_norm=0.5,
        target_kl=0.03,
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
    )
