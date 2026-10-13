"""Frozen 500-update constrained PPO trainer for hardcore v5."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drl.paper_recovery.config import PaperRecoveryConfig
from drl.paper_recovery.receipts import canonical_hash, sha256_file
from drl.paper_recovery.trainer import PersistentRolloutWorkers
from drl.paper_recovery_original_ppo.trainer import ReviewerAlignedPPOTrainer
from drl.paper_recovery_longhorizon_v3.trainer import (
    CRITIC_LR,
    INITIAL_ENTROPY_COEFFICIENT,
    INITIAL_KL_COEFFICIENT,
    LongHorizonPPOTrainer,
)

from .config import hardcore_config
from .env import HardcoreCascadeEnv
from .policy import HardcoreSetAwarePPO


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "experiments/paper_recovery_hardcore_v5/protocol.json"
RESULT_ROOT = REPO_ROOT / "results/paper_recovery_hardcore_v5"
ACTOR_LR = 3e-5


def source_manifest(config: PaperRecoveryConfig) -> dict[str, Any]:
    paths = (
        REPO_ROOT / "drl/paper_recovery/config.py",
        REPO_ROOT / "drl/paper_recovery/env.py",
        REPO_ROOT / "drl/paper_recovery/trainer.py",
        REPO_ROOT / "drl/paper_recovery_original_ppo/policy.py",
        REPO_ROOT / "drl/paper_recovery_original_ppo/trainer.py",
        REPO_ROOT / "drl/paper_recovery_longhorizon_v3/env.py",
        REPO_ROOT / "drl/paper_recovery_longhorizon_v3/trainer.py",
        REPO_ROOT / "drl/paper_recovery_hardcore_v5/config.py",
        REPO_ROOT / "drl/paper_recovery_hardcore_v5/env.py",
        REPO_ROOT / "drl/paper_recovery_hardcore_v5/baselines.py",
        REPO_ROOT / "drl/paper_recovery_hardcore_v5/policy.py",
        REPO_ROOT / "drl/paper_recovery_hardcore_v5/trainer.py",
        PROTOCOL_PATH,
    )
    files = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in paths
    }
    return {
        "schema": "qrelay-reviewer-hardcore-source-5",
        "files": files,
        "source_hash": canonical_hash(files),
        "config_hash": canonical_hash(config.to_dict()),
    }


class HardcorePPOTrainer(LongHorizonPPOTrainer):
    def __init__(
        self, config: PaperRecoveryConfig, *, seed: int, device: str = "auto"
    ):
        budgets = {"smoke": 2, "qualification_500": 500}
        ReviewerAlignedPPOTrainer.__init__(
            self,
            config,
            seed=seed,
            device=device,
            environment_class=HardcoreCascadeEnv,
            run_budgets=budgets,
            maximum_updates=500,
            source_manifest_builder=source_manifest,
        )
        torch.manual_seed(self.seed + 5_000_057)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed + 5_000_057)
        self.policy = HardcoreSetAwarePPO(config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=ACTOR_LR, eps=1e-5
        )
        self.critic_parameters = list(
            itertools.chain(
                self.policy.reward_critic.parameters(),
                self.policy.failure_cost_critic.parameters(),
                self.policy.depletion_cost_critic.parameters(),
            )
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_parameters, lr=CRITIC_LR, eps=1e-5
        )
        self.entropy_coefficient = 0.025
        self.kl_coefficient = INITIAL_KL_COEFFICIENT
        self.current_update = 0
        self.last_rollout_entropy = 0.0

    def _member_credit_coefficient(self) -> float:
        return 0.0

    def _sample_actions(
        self,
        observations: np.ndarray,
        sequences: np.ndarray,
        worker_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        seq = torch.as_tensor(sequences, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, reward, failure, depletion, masks = self.policy(obs, seq)
            eligible = masks.sum(dim=1)
            if torch.any(eligible < self.config.physics.committee_size):
                raise RuntimeError(
                    f"hardcore rollout failed closed below k: {eligible.cpu().tolist()}"
                )
            policy_logits = logits / max(self.policy.temperature, 1e-6)
            generators = [self.policy_generators[index] for index in worker_ids]
            action_scores, ordered, log_probability, entropy = (
                self.policy.sample_ordered(policy_logits, masks, generators)
            )
            values = torch.stack((reward, failure, depletion), dim=1)
        return (
            action_scores.cpu().numpy(),
            ordered.cpu().numpy(),
            log_probability.cpu().numpy(),
            entropy.cpu().numpy(),
            values.cpu().numpy(),
        )


def run_hardcore(*, run_kind: str, seed: int, device: str = "auto") -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    expected = {
        "smoke": protocol["seeds"]["smoke"],
        "qualification_500": protocol["seeds"]["qualification_500_training"],
    }.get(run_kind)
    if expected is None or int(seed) != int(expected):
        raise RuntimeError("hardcore run kind/seed is not preregistered")
    hardness_path = RESULT_ROOT / "baseline_hardness_gate_corrigendum_1.json"
    if not hardness_path.exists():
        raise RuntimeError("hardcore PPO requires the baseline hardness gate")
    hardness = json.loads(hardness_path.read_text(encoding="utf-8"))
    if hardness.get("status") != "PASS":
        raise RuntimeError("baseline hardness gate did not pass")
    config = hardcore_config(training=True)
    trainer = HardcorePPOTrainer(config, seed=int(seed), device=device)
    updates = {"smoke": 2, "qualification_500": 500}[run_kind]
    return trainer.train(
        updates=updates,
        run_kind=run_kind,
        output_directory=RESULT_ROOT / f"{run_kind}_seed_{int(seed)}",
    )
