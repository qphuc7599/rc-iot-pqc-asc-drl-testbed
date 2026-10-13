"""Controlled PPO ablations for the IoT-73941-2026 minor revision.

Every arm retains the registered public observation contract, environment,
ordered Plackett--Luce action, and training budget.  Exactly one controller
component is removed per arm.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from drl.paper_recovery_hardcore_v5.policy import (
    HardcoreCritic,
)
from drl.paper_recovery_security_v4.policy import PublicSecurityActor, SecurityConstrainedPPO


ARMS = (
    "A0_full",
    "A1_no_history",
    "A2_no_admission",
    "A3_no_security_critic",
    "A4_no_cell_diversity",
    "A5_single_critic",
)


class AblationPublicSecurityActor(PublicSecurityActor):
    """Full actor with two explicit single-component switches."""

    def __init__(self, config: Any, *, no_admission: bool, no_cell_diversity: bool):
        super().__init__(config)
        self.no_admission = bool(no_admission)
        self.no_cell_diversity = bool(no_cell_diversity)

    @property
    def diversity_weight(self) -> Tensor:
        if self.no_cell_diversity:
            # Preserve the parameter and actor size while fixing lambda_cell=0.
            return self.raw_diversity_weight * 0.0
        return super().diversity_weight

    def public_admission_mask(
        self, current_nodes: Tensor, valid_mask: Tensor
    ) -> Tensor:
        if self.no_admission:
            return valid_mask
        return super().public_admission_mask(current_nodes, valid_mask)


class SharedFourHeadCritic(HardcoreCritic):
    """One shared critic network with reward and three cost heads."""

    def __init__(self, config: Any):
        super().__init__(config)
        final = self.head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("unexpected HardcoreCritic head")
        self.head[-1] = nn.Linear(final.in_features, 4)


class ZeroSecurityCritic(nn.Module):
    """Parameter-free zero baseline used only by A3."""

    def forward(
        self,
        current_nodes: Tensor,
        current_global: Tensor,
        node_sequence: Tensor,
        global_sequence: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        del current_nodes, node_sequence, global_sequence, valid_mask
        return current_global[:, 0] * 0.0


class MinorRevisionAblationPPO(SecurityConstrainedPPO):
    """Security-v4 policy with one frozen ablation selected by ``arm``."""

    MODEL_SCHEMA = "qrelay-minor-revision-ablation-policy-1"

    def __init__(self, config: Any, *, arm: str):
        if arm not in ARMS:
            raise ValueError(f"unknown ablation arm: {arm}")
        super().__init__(config)
        self.arm = arm
        self.no_history = arm == "A1_no_history"
        actor_state = self.actor.state_dict()
        self.actor = AblationPublicSecurityActor(
            config,
            no_admission=arm == "A2_no_admission",
            no_cell_diversity=arm == "A4_no_cell_diversity",
        )
        self.actor.load_state_dict(actor_state, strict=True)
        self.shared_critic: SharedFourHeadCritic | None = None
        if arm == "A5_single_critic":
            self.shared_critic = SharedFourHeadCritic(config)
            # Aliases make checkpoint inspection uniform.  Optimisation deduplicates
            # parameters by identity, so the shared network is updated once.
            self.reward_critic = self.shared_critic
            self.failure_cost_critic = self.shared_critic
            self.depletion_cost_critic = self.shared_critic
            self.security_cost_critic = self.shared_critic
        elif arm == "A3_no_security_critic":
            self.security_cost_critic = ZeroSecurityCritic()

    @staticmethod
    def _zero_lag_differences(
        current_nodes: Tensor,
        current_global: Tensor,
        node_sequence: Tensor,
        global_sequence: Tensor,
    ) -> tuple[Tensor, Tensor]:
        history = node_sequence.shape[2]
        return (
            current_nodes.unsqueeze(2).expand(-1, -1, history, -1),
            current_global.unsqueeze(1).expand(-1, history, -1),
        )

    def forward(
        self,
        observation: Tensor,
        observation_sequence: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        observation, observation_sequence = self.contract.normalise(
            observation, observation_sequence
        )
        current_nodes, current_global, node_sequence, global_sequence, valid = (
            self.contract.split(observation, observation_sequence)
        )
        if self.no_history:
            node_sequence, global_sequence = self._zero_lag_differences(
                current_nodes, current_global, node_sequence, global_sequence
            )
        inputs = (
            current_nodes,
            current_global,
            node_sequence,
            global_sequence,
            valid,
        )
        admitted = self.actor.public_admission_mask(current_nodes, valid)
        logits = self.actor(*inputs)
        if self.shared_critic is not None:
            values = self.shared_critic(*inputs)
            reward, failure, depletion, security = values.unbind(dim=1)
        else:
            reward = self.reward_critic(*inputs)
            failure = self.failure_cost_critic(*inputs)
            depletion = self.depletion_cost_critic(*inputs)
            security = self.security_cost_critic(*inputs)
        return logits, reward, failure, depletion, security, admitted

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "schema": self.MODEL_SCHEMA,
            "arm": self.arm,
            "scientific_parent": "qrelay-security-constrained-direct-ppo-4",
            "environment_change": False,
            "public_observation_change": False,
            "decoder_change": False,
            "history_lag_differences_zeroed": self.arm == "A1_no_history",
            "public_admission_removed": self.arm == "A2_no_admission",
            "security_critic_removed_and_lambda_s_fixed_zero": self.arm
            == "A3_no_security_critic",
            "lambda_cell_fixed_zero": self.arm == "A4_no_cell_diversity",
            "single_shared_four_head_critic": self.arm == "A5_single_critic",
            "hidden_inputs": False,
            "detector_oracle": False,
            "post_action_shield": False,
            "sampling": {
                "distribution": "exact_prefix_conditioned_Plackett_Luce",
                "temperature": self.temperature,
                "committee_size": self.committee_size,
            },
        }
