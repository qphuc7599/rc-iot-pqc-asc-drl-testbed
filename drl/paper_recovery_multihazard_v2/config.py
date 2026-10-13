"""Frozen configuration for the 500-update PPO-v2 qualification."""

from __future__ import annotations

from dataclasses import replace

from drl.paper_recovery.config import PaperRecoveryConfig
from drl.paper_recovery_multihazard.config import multihazard_config


def multihazard_v2_config(*, training: bool) -> PaperRecoveryConfig:
    """Keep v1 physics/reward, but cover the longest public telemetry delay."""

    base = multihazard_config(training=training)
    observations = replace(base.observations, temporal_delta_window=8)
    budgets = replace(
        base.budgets,
        smoke_updates=2,
        profiler_updates=500,
        formal_updates=3_000,
        epochs_per_update=4,
        minibatch_size=1_024,
        formal_transitions_per_seed=9_000_000,
    )
    ppo = replace(
        base.ppo,
        learning_rate=1e-4,
        clip_epsilon=0.20,
        entropy_coefficient_mean_per_pick=0.01,
        gradient_clip_norm=0.5,
        target_kl=0.03,
    )
    return replace(base, observations=observations, budgets=budgets, ppo=ppo)
