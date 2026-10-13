"""Prospectively frozen public-input contract for security-constrained PPO v4."""

from __future__ import annotations

from dataclasses import replace

from drl.paper_recovery.config import PaperRecoveryConfig
from drl.paper_recovery_hardcore_v5.config import GLOBAL_FEATURES, NODE_FEATURES


def security_v4_config(*, training: bool) -> PaperRecoveryConfig:
    """Keep the v5 disaster physics and observations; sharpen only the controller."""

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
        profiler_updates=2,
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
        entropy_coefficient_mean_per_pick=0.010,
        target_kl=0.015,
        # A lower, still stochastic temperature reduces deployment-time tail
        # exploration.  Training and stochastic audit use the same value.
        gumbel_temperature=0.55,
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
        schema="qrelay-security-constrained-config-4",
    )
