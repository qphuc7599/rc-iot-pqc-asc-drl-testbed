"""Causally corrected direct full-committee PPO.

The action remains an exact ordered masked Plackett--Luce committee.  This
version removes two action-induced state variables from the actor and makes
all value/cost critics parameter-disjoint from the actor and from each other.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from drl.paper_recovery.config import PaperRecoveryConfig
from drl.paper_recovery_original_ppo.policy import (
    ReviewerAlignedDirectPPO,
    _InputContract,
    _masked_max,
    _masked_mean,
)


ACTOR_EXCLUDED_PUBLIC_FEATURES = (
    "previous_committee_membership",
    "recent_service_ewma",
)


class CausalSetActor(nn.Module):
    """Original-style per-node actor without action-induced self-lock inputs."""

    def __init__(self, config: PaperRecoveryConfig):
        super().__init__()
        self.contract = _InputContract(config)
        names = self.contract.config.observations.per_node_names
        self.included_indices = tuple(
            index for index, name in enumerate(names)
            if name not in ACTOR_EXCLUDED_PUBLIC_FEATURES
        )
        self.included_names = tuple(names[index] for index in self.included_indices)
        node_input = 2 * (
            len(self.included_indices) + self.contract.global_features
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.node_head = nn.Linear(64, 1)
        nn.init.orthogonal_(self.node_head.weight, gain=0.01)
        nn.init.zeros_(self.node_head.bias)

    def forward(
        self,
        current_nodes: Tensor,
        current_global: Tensor,
        node_sequence: Tensor,
        global_sequence: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if torch.any(valid_mask.sum(dim=1) == 0):
            raise RuntimeError("causal direct-PPO scorer has no eligible node")
        indices = torch.as_tensor(
            self.included_indices, dtype=torch.long, device=current_nodes.device
        )
        current_public = current_nodes.index_select(-1, indices)
        oldest_public = node_sequence[:, :, 0, :].index_select(-1, indices)
        node_delta = current_public - oldest_public
        global_delta = current_global - global_sequence[:, 0, :]
        nodes = current_nodes.shape[1]
        features = torch.cat(
            (
                current_public,
                node_delta,
                current_global.unsqueeze(1).expand(-1, nodes, -1),
                global_delta.unsqueeze(1).expand(-1, nodes, -1),
            ),
            dim=-1,
        )
        return self.node_head(self.node_encoder(features)).squeeze(-1).masked_fill(
            ~valid_mask, -torch.inf
        )


class IndependentDeltaCritic(nn.Module):
    """Cheap independent critic over all public current/delta features."""

    def __init__(self, config: PaperRecoveryConfig):
        super().__init__()
        self.contract = _InputContract(config)
        node_input = 2 * (
            self.contract.node_features + self.contract.global_features
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * 64 + 2 * self.contract.global_features, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        current_nodes: Tensor,
        current_global: Tensor,
        node_sequence: Tensor,
        global_sequence: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        node_delta = current_nodes - node_sequence[:, :, 0, :]
        global_delta = current_global - global_sequence[:, 0, :]
        nodes = current_nodes.shape[1]
        features = torch.cat(
            (
                current_nodes,
                node_delta,
                current_global.unsqueeze(1).expand(-1, nodes, -1),
                global_delta.unsqueeze(1).expand(-1, nodes, -1),
            ),
            dim=-1,
        )
        encoded = self.node_encoder(features)
        context = torch.cat(
            (
                _masked_mean(encoded, valid_mask),
                _masked_max(encoded, valid_mask),
                current_global,
                global_delta,
            ),
            dim=-1,
        )
        return self.head(context).squeeze(-1)


class CausalDirectPPO(ReviewerAlignedDirectPPO):
    """Paper-trajectory direct PPO with corrected causal optimization."""

    MODEL_SCHEMA = "qrelay-development-causal-direct-ppo-2"

    def __init__(self, config: PaperRecoveryConfig):
        nn.Module.__init__(self)
        self.config = config
        self.contract = _InputContract(config)
        self.num_nodes = config.physics.num_nodes
        self.committee_size = config.physics.committee_size
        self.temperature = float(config.ppo.gumbel_temperature)
        self.actor = CausalSetActor(config)
        self.reward_critic = IndependentDeltaCritic(config)
        self.failure_cost_critic = IndependentDeltaCritic(config)
        self.depletion_cost_critic = IndependentDeltaCritic(config)

    def forward(
        self,
        observation: Tensor,
        observation_sequence: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        observation, observation_sequence = self.contract.normalise(
            observation, observation_sequence
        )
        inputs = self.contract.split(observation, observation_sequence)
        return (
            self.actor(*inputs),
            self.reward_critic(*inputs),
            self.failure_cost_critic(*inputs),
            self.depletion_cost_critic(*inputs),
            inputs[-1],
        )

    def architecture_manifest(self) -> dict[str, Any]:
        modules = {
            "actor": self.actor,
            "reward": self.reward_critic,
            "failure": self.failure_cost_critic,
            "depletion": self.depletion_cost_critic,
        }
        ids = {
            name: {id(parameter) for parameter in module.parameters()}
            for name, module in modules.items()
        }
        disjoint = all(
            not ids[left] & ids[right]
            for index, left in enumerate(ids)
            for right in tuple(ids)[index + 1 :]
        )
        return {
            "schema": self.MODEL_SCHEMA,
            "scientific_parent": "original_paper_direct_full_committee_ppo",
            "actor": {
                "per_node_mlp": [128, 64, 1],
                "history_delta_window": self.contract.history,
                "included_public_node_features": list(self.actor.included_names),
                "excluded_action_induced_features": list(
                    ACTOR_EXCLUDED_PUBLIC_FEATURES
                ),
                "fmg_prior": False,
                "residual": False,
            },
            "critics": {
                "independent_reward_failure_depletion": True,
                "actor_and_all_critics_parameter_disjoint": disjoint,
                "public_current_plus_delta": True,
            },
            "sampling": {
                "distribution": "exact_masked_ordered_plackett_luce",
                "sampler": "gumbel_top_k",
                "committee_size": self.committee_size,
                "temperature": self.temperature,
            },
            "hidden_inputs": False,
            "detector_oracle": False,
            "execution_shield": False,
        }
