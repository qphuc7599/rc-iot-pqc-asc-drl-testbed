"""Causal multi-scale direct PPO for unordered-effect committee control."""

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


class MultiScalePublicEncoder(nn.Module):
    """Shared per-node encoder over four causal public-history scales."""

    def __init__(self, config: PaperRecoveryConfig, *, hidden: int = 128):
        super().__init__()
        self.contract = _InputContract(config)
        # Current value plus deltas from four preregistered past positions.
        self.scales = (0, 15, 27, 30)
        if self.contract.history != 32:
            raise ValueError("multi-scale encoder requires exactly 32 public steps")
        inputs = 5 * (
            self.contract.node_features + self.contract.global_features
        )
        self.network = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 64),
            nn.SiLU(),
        )

    def forward(
        self,
        current_nodes: Tensor,
        current_global: Tensor,
        node_sequence: Tensor,
        global_sequence: Tensor,
    ) -> Tensor:
        nodes = current_nodes.shape[1]
        node_parts = [current_nodes]
        global_parts = [current_global]
        for index in self.scales:
            node_parts.append(current_nodes - node_sequence[:, :, index, :])
            global_parts.append(current_global - global_sequence[:, index, :])
        expanded_global = [
            value.unsqueeze(1).expand(-1, nodes, -1) for value in global_parts
        ]
        return self.network(torch.cat((*node_parts, *expanded_global), dim=-1))


class MultiScaleSetActor(nn.Module):
    """Original-paper set scorer with a 32-step causal public history."""

    def __init__(self, config: PaperRecoveryConfig):
        super().__init__()
        self.encoder = MultiScalePublicEncoder(config)
        self.head = nn.Linear(64, 1)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        current_nodes: Tensor,
        current_global: Tensor,
        node_sequence: Tensor,
        global_sequence: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if torch.any(valid_mask.sum(dim=1) == 0):
            raise RuntimeError("long-horizon actor has no eligible node")
        encoded = self.encoder(
            current_nodes, current_global, node_sequence, global_sequence
        )
        return self.head(encoded).squeeze(-1).masked_fill(~valid_mask, -torch.inf)


class IndependentMultiScaleCritic(nn.Module):
    """Parameter-disjoint permutation-invariant public critic."""

    def __init__(self, config: PaperRecoveryConfig):
        super().__init__()
        self.encoder = MultiScalePublicEncoder(config, hidden=96)
        global_context = 5 * config.observations.global_features
        self.head = nn.Sequential(
            nn.Linear(2 * 64 + global_context, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.scales = self.encoder.scales

    def forward(
        self,
        current_nodes: Tensor,
        current_global: Tensor,
        node_sequence: Tensor,
        global_sequence: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        encoded = self.encoder(
            current_nodes, current_global, node_sequence, global_sequence
        )
        global_parts = [current_global]
        for index in self.scales:
            global_parts.append(current_global - global_sequence[:, index, :])
        context = torch.cat(
            (
                _masked_mean(encoded, valid_mask),
                _masked_max(encoded, valid_mask),
                *global_parts,
            ),
            dim=-1,
        )
        return self.head(context).squeeze(-1)


class LongHorizonDirectPPO(ReviewerAlignedDirectPPO):
    """Pure direct PPO; FMG is neither an input nor an action anchor."""

    MODEL_SCHEMA = "qrelay-reviewer-multiscale-direct-ppo-3"

    def __init__(self, config: PaperRecoveryConfig):
        nn.Module.__init__(self)
        self.config = config
        self.contract = _InputContract(config)
        self.num_nodes = config.physics.num_nodes
        self.committee_size = config.physics.committee_size
        self.temperature = float(config.ppo.gumbel_temperature)
        self.actor = MultiScaleSetActor(config)
        self.reward_critic = IndependentMultiScaleCritic(config)
        self.failure_cost_critic = IndependentMultiScaleCritic(config)
        self.depletion_cost_critic = IndependentMultiScaleCritic(config)

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

    @staticmethod
    def plackett_luce_per_pick(
        policy_logits: Tensor,
        ordered_committee: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return exact ordered per-pick log probabilities and entropies."""

        if ordered_committee.dim() == 1:
            ordered_committee = ordered_committee.unsqueeze(0)
        batch, picks = ordered_committee.shape
        if policy_logits.shape != valid_mask.shape or policy_logits.shape[0] != batch:
            raise ValueError("long-horizon Plackett--Luce shape mismatch")
        if not torch.all(valid_mask.gather(1, ordered_committee)):
            raise ValueError("long-horizon committee contains an invalid node")
        one_hot = torch.zeros(
            (batch, picks, policy_logits.shape[1]),
            dtype=torch.bool,
            device=policy_logits.device,
        )
        one_hot.scatter_(2, ordered_committee.unsqueeze(-1), True)
        if torch.any(one_hot.sum(dim=(1, 2)) != picks):
            raise ValueError("long-horizon committee repeats a node")
        removed = torch.cat(
            (
                torch.zeros_like(one_hot[:, :1]),
                one_hot[:, :-1].cumsum(dim=1) > 0,
            ),
            dim=1,
        )
        remaining = valid_mask.unsqueeze(1) & ~removed
        prefix_logits = policy_logits.unsqueeze(1).masked_fill(~remaining, -1e9)
        prefix_logp = torch.log_softmax(prefix_logits, dim=-1)
        selected_logp = prefix_logp.gather(
            2, ordered_committee.unsqueeze(-1)
        ).squeeze(-1)
        probability = prefix_logp.exp()
        entropy = -(probability * prefix_logp).masked_fill(
            ~remaining, 0.0
        ).sum(dim=-1)
        return selected_logp, entropy

    @classmethod
    def plackett_luce_stats(
        cls,
        policy_logits: Tensor,
        ordered_committee: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        per_pick, entropy = cls.plackett_luce_per_pick(
            policy_logits, ordered_committee, valid_mask
        )
        return per_pick.sum(dim=-1), entropy.mean(dim=-1)

    def architecture_manifest(self) -> dict[str, Any]:
        modules = {
            "actor": self.actor,
            "reward": self.reward_critic,
            "failure": self.failure_cost_critic,
            "depletion": self.depletion_cost_critic,
        }
        parameters = {
            name: {id(value) for value in module.parameters()}
            for name, module in modules.items()
        }
        disjoint = all(
            not parameters[left] & parameters[right]
            for index, left in enumerate(parameters)
            for right in tuple(parameters)[index + 1 :]
        )
        return {
            "schema": self.MODEL_SCHEMA,
            "scientific_parent": "original_paper_direct_full_committee_ppo",
            "actor": {
                "per_node_shared_mlp": [128, 64, 1],
                "public_history_steps": self.contract.history,
                "history_scales": list(self.actor.encoder.scales),
                "fmg_prior": False,
                "residual": False,
            },
            "critics": {
                "permutation_invariant_mean_max_pool": True,
                "independent_reward_failure_depletion": True,
                "actor_and_all_critics_parameter_disjoint": disjoint,
            },
            "sampling": {
                "distribution": "exact_masked_ordered_plackett_luce",
                "sampler": "gumbel_top_k",
                "committee_size": self.committee_size,
                "temperature": self.temperature,
            },
            "optimization": {
                "constant_member_auxiliary": False,
                "decaying_public_member_auxiliary": True,
                "adaptive_entropy": True,
                "adaptive_kl_penalty": True,
            },
            "hidden_inputs": False,
            "detector_oracle": False,
            "execution_shield": False,
        }
