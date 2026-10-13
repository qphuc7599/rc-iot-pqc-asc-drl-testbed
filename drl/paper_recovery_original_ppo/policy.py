"""Direct full-committee PPO upgraded from the original Q-Relay actor.

The actor remains a pure PPO controller: it scores every publicly eligible node
and samples all k members from an exact ordered masked Plackett-Luce policy.
There is no FMG prior, detector oracle, execution shield, or masked filler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from drl.paper_recovery.config import PaperRecoveryConfig


ACTOR_NODE_HIDDEN = 64
ACTOR_GLOBAL_HIDDEN = 32
CRITIC_NODE_HIDDEN = 32
CRITIC_GLOBAL_HIDDEN = 16
CRITIC_EMBED = 64


@dataclass(frozen=True)
class DirectCommitteeAction:
    """One complete, auditable direct-PPO committee action."""

    action_scores: np.ndarray
    ordered_committee: np.ndarray
    observable_mask: np.ndarray
    ordered_log_probability: float
    mean_per_pick_entropy: float
    reward_value: float
    failure_cost_value: float
    depletion_cost_value: float
    deterministic: bool


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_max(values: Tensor, mask: Tensor) -> Tensor:
    maximum = values.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(dim=1)
    return torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))


class _InputContract:
    """Public observation reshaping shared by independent network modules."""

    def __init__(self, config: PaperRecoveryConfig):
        self.config = config
        self.num_nodes = config.physics.num_nodes
        self.node_features = config.observations.per_node_features
        self.global_features = config.observations.global_features
        self.history = config.observations.temporal_delta_window

    def normalise(
        self, observation: Tensor, sequence: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        if observation.dim() != 2 or observation.shape[-1] != self.config.observation_size:
            raise ValueError("direct-PPO observation schema mismatch")
        if sequence is None:
            sequence = observation.unsqueeze(1).expand(-1, self.history, -1)
        elif sequence.dim() == 2:
            sequence = sequence.unsqueeze(0)
        if sequence.dim() != 3:
            raise ValueError("direct-PPO sequence must have [batch,time,observation]")
        if sequence.shape[0] != observation.shape[0]:
            raise ValueError("direct-PPO observation/sequence batch mismatch")
        if sequence.shape[1] != self.history:
            raise ValueError("direct-PPO public history length mismatch")
        if sequence.shape[2] != self.config.observation_size:
            raise ValueError("direct-PPO temporal observation schema mismatch")
        return observation, sequence

    def split(
        self, observation: Tensor, sequence: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        boundary = self.num_nodes * self.node_features
        current_nodes = observation[:, :boundary].reshape(
            observation.shape[0], self.num_nodes, self.node_features
        )
        current_global = observation[:, boundary:]
        node_sequence = sequence[:, :, :boundary].reshape(
            sequence.shape[0], self.history, self.num_nodes, self.node_features
        ).permute(0, 2, 1, 3)
        global_sequence = sequence[:, :, boundary:]
        valid_mask = (current_nodes[:, :, 1] > 0.5) & (
            current_nodes[:, :, 0] > self.config.physics.battery_cutoff_ratio
        )
        return current_nodes, current_global, node_sequence, global_sequence, valid_mask


class DirectSetActor(nn.Module):
    """Original-style per-node MLP with a cheap public temporal delta."""

    def __init__(self, config: PaperRecoveryConfig):
        super().__init__()
        self.contract = _InputContract(config)
        node_input = 2 * (
            self.contract.node_features + self.contract.global_features
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.node_head = nn.Sequential(
            nn.Linear(64, 1),
        )
        nn.init.orthogonal_(self.node_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.node_head[-1].bias)

    def encode(
        self,
        current_nodes: Tensor,
        current_global: Tensor,
        node_sequence: Tensor,
        global_sequence: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        _, nodes, _, _ = node_sequence.shape
        if torch.any(valid_mask.sum(dim=1) == 0):
            raise RuntimeError("direct-PPO scorer has no publicly eligible node")
        node_delta = current_nodes - node_sequence[:, :, 0, :]
        global_delta = current_global - global_sequence[:, 0, :]
        expanded_global = current_global.unsqueeze(1).expand(-1, nodes, -1)
        expanded_global_delta = global_delta.unsqueeze(1).expand(-1, nodes, -1)
        features_out = torch.cat(
            (
                current_nodes,
                node_delta,
                expanded_global,
                expanded_global_delta,
            ),
            dim=-1,
        )
        encoded = self.node_encoder(features_out)
        logits = self.node_head(encoded).squeeze(-1).masked_fill(
            ~valid_mask, -torch.inf
        )
        critic_context = torch.cat(
            (
                _masked_mean(encoded, valid_mask),
                _masked_max(encoded, valid_mask),
                current_global,
                global_delta,
            ),
            dim=-1,
        )
        return logits, critic_context

    def forward(
        self,
        current_nodes: Tensor,
        current_global: Tensor,
        node_sequence: Tensor,
        global_sequence: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        return self.encode(
            current_nodes,
            current_global,
            node_sequence,
            global_sequence,
            valid_mask,
        )[0]


class IndependentPooledCritic(nn.Module):
    """One independent public-history critic for reward or a safety cost."""

    def __init__(self, config: PaperRecoveryConfig):
        super().__init__()
        self.contract = _InputContract(config)
        self.node_gru = nn.GRU(
            self.contract.node_features, CRITIC_NODE_HIDDEN, batch_first=True
        )
        self.global_gru = nn.GRU(
            self.contract.global_features, CRITIC_GLOBAL_HIDDEN, batch_first=True
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(CRITIC_NODE_HIDDEN + self.contract.node_features, CRITIC_EMBED),
            nn.SiLU(),
            nn.Linear(CRITIC_EMBED, CRITIC_EMBED),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(
                2 * CRITIC_EMBED
                + CRITIC_GLOBAL_HIDDEN
                + self.contract.global_features,
                128,
            ),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        current_nodes: Tensor,
        current_global: Tensor,
        node_sequence: Tensor,
        global_sequence: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        batch, nodes, history, features = node_sequence.shape
        flattened = node_sequence.reshape(batch * nodes, history, features)
        _, node_hidden = self.node_gru(flattened)
        node_hidden = node_hidden[-1].reshape(batch, nodes, CRITIC_NODE_HIDDEN)
        encoded = self.node_encoder(torch.cat((node_hidden, current_nodes), dim=-1))
        _, global_hidden = self.global_gru(global_sequence)
        pooled = torch.cat(
            (
                _masked_mean(encoded, valid_mask),
                _masked_max(encoded, valid_mask),
                global_hidden[-1],
                current_global,
            ),
            dim=-1,
        )
        return self.head(pooled).squeeze(-1)


class PublicValueHead(nn.Module):
    """Independent value/cost head on the actor's public temporal context."""

    def __init__(self, input_features: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, context: Tensor) -> Tensor:
        return self.head(context).squeeze(-1)


class ReviewerAlignedDirectPPO(nn.Module):
    """Original direct committee PPO with reviewer-aligned public information."""

    MODEL_SCHEMA = "qrelay-development-reviewer-aligned-direct-ppo-1"

    def __init__(self, config: PaperRecoveryConfig | None = None):
        super().__init__()
        self.config = config or PaperRecoveryConfig()
        self.contract = _InputContract(self.config)
        self.num_nodes = self.config.physics.num_nodes
        self.committee_size = self.config.physics.committee_size
        self.temperature = float(self.config.ppo.gumbel_temperature)
        self.actor = DirectSetActor(self.config)
        critic_context = 2 * 64 + 2 * self.contract.global_features
        self.reward_critic = PublicValueHead(critic_context)
        self.failure_cost_critic = PublicValueHead(critic_context)
        self.depletion_cost_critic = PublicValueHead(critic_context)

    def forward(
        self,
        observation: Tensor,
        observation_sequence: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        observation, observation_sequence = self.contract.normalise(
            observation, observation_sequence
        )
        inputs = self.contract.split(observation, observation_sequence)
        logits, public_context = self.actor.encode(*inputs)
        reward_value = self.reward_critic(public_context)
        failure_cost_value = self.failure_cost_critic(public_context)
        depletion_cost_value = self.depletion_cost_critic(public_context)
        valid_mask = inputs[-1]
        return (
            logits,
            reward_value,
            failure_cost_value,
            depletion_cost_value,
            valid_mask,
        )

    @staticmethod
    def plackett_luce_stats(
        policy_logits: Tensor,
        ordered_committee: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Stable batched exact statistics for an ordered Top-k action."""
        if policy_logits.dim() != 2 or valid_mask.shape != policy_logits.shape:
            raise ValueError("direct-PPO logits/mask shape mismatch")
        if ordered_committee.dim() == 1:
            ordered_committee = ordered_committee.unsqueeze(0)
        if ordered_committee.shape[0] != policy_logits.shape[0]:
            raise ValueError("direct-PPO committee batch mismatch")
        if ordered_committee.numel() == 0:
            raise ValueError("direct-PPO committee cannot be empty")
        if torch.any(ordered_committee < 0) or torch.any(
            ordered_committee >= policy_logits.shape[1]
        ):
            raise ValueError("direct-PPO committee node is out of range")
        if not torch.all(valid_mask.gather(1, ordered_committee)):
            raise ValueError("direct-PPO committee uses an invalid node")

        batch, picks = ordered_committee.shape
        selected_one_hot = torch.zeros(
            (batch, picks, policy_logits.shape[1]),
            dtype=torch.bool,
            device=policy_logits.device,
        )
        selected_one_hot.scatter_(2, ordered_committee.unsqueeze(-1), True)
        if torch.any(selected_one_hot.sum(dim=(1, 2)) != picks):
            raise ValueError("direct-PPO committee repeats a node")
        removed_before = torch.cat(
            (
                torch.zeros_like(selected_one_hot[:, :1]),
                selected_one_hot[:, :-1].cumsum(dim=1) > 0,
            ),
            dim=1,
        )
        remaining = valid_mask.unsqueeze(1) & ~removed_before
        prefix_logits = policy_logits.unsqueeze(1).masked_fill(~remaining, -1e9)
        prefix_log_probability = torch.log_softmax(prefix_logits, dim=-1)
        ordered_log_probability = prefix_log_probability.gather(
            2, ordered_committee.unsqueeze(-1)
        ).squeeze(-1)
        log_probability = ordered_log_probability.sum(dim=-1)
        prefix_probability = prefix_log_probability.exp()
        prefix_entropy = -(
            prefix_probability * prefix_log_probability
        ).masked_fill(~remaining, 0.0).sum(dim=-1)
        return log_probability, prefix_entropy.mean(dim=-1)

    def evaluate_ordered_committees(
        self,
        observation: Tensor,
        observation_sequence: Tensor,
        ordered_committee: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        logits, reward, failure, depletion, valid = self(
            observation, observation_sequence
        )
        log_probability, entropy = self.plackett_luce_stats(
            logits / max(self.temperature, 1e-6), ordered_committee, valid
        )
        return log_probability, entropy, reward, failure, depletion

    def act(
        self,
        observation: np.ndarray,
        observation_sequence: np.ndarray | None = None,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> DirectCommitteeAction:
        device = next(self.parameters()).device
        obs = torch.as_tensor(observation, dtype=torch.float32, device=device)
        sequence = None
        if observation_sequence is not None:
            sequence = torch.as_tensor(
                observation_sequence, dtype=torch.float32, device=device
            )
        with torch.no_grad():
            logits, reward, failure, depletion, valid = self(obs, sequence)
            if int(valid[0].sum().item()) < self.committee_size:
                raise RuntimeError("direct-PPO action failed closed below committee size")
            policy_logits = logits / max(self.temperature, 1e-6)
            if deterministic:
                score_row = policy_logits[0].cpu().numpy()
                mask_row = valid[0].cpu().numpy()
                order = np.lexsort(
                    (np.arange(self.num_nodes, dtype=np.int64), -score_row)
                )
                order = order[mask_row[order]][: self.committee_size]
                ordered = torch.as_tensor(order, dtype=torch.long, device=device).unsqueeze(0)
                action_scores = policy_logits.to(torch.float64)
            else:
                uniforms = torch.rand(
                    policy_logits.shape,
                    dtype=torch.float64,
                    device=device,
                    generator=generator,
                ).clamp_(1e-12, 1.0 - 1e-12)
                gumbel = -torch.log(-torch.log(uniforms))
                action_scores = (policy_logits.to(torch.float64) + gumbel).masked_fill(
                    ~valid, -torch.inf
                )
                ordered = torch.topk(
                    action_scores,
                    k=self.committee_size,
                    dim=1,
                    sorted=True,
                ).indices
            log_probability, entropy = self.plackett_luce_stats(
                policy_logits, ordered, valid
            )
        return DirectCommitteeAction(
            action_scores=action_scores[0].cpu().numpy(),
            ordered_committee=ordered[0].cpu().numpy().astype(np.int64),
            observable_mask=valid[0].cpu().numpy().astype(bool),
            ordered_log_probability=float(log_probability[0].item()),
            mean_per_pick_entropy=float(entropy[0].item()),
            reward_value=float(reward[0].item()),
            failure_cost_value=float(failure[0].item()),
            depletion_cost_value=float(depletion[0].item()),
            deterministic=bool(deterministic),
        )

    def architecture_manifest(self) -> dict[str, Any]:
        critic_ids = {
            name: sorted(id(parameter) for parameter in module.parameters())
            for name, module in (
                ("reward", self.reward_critic),
                ("failure", self.failure_cost_critic),
                ("depletion", self.depletion_cost_critic),
            )
        }
        return {
            "schema": self.MODEL_SCHEMA,
            "scientific_parent": "original_paper_direct_full_committee_ppo",
            "actor": {
                "per_node_mlp": [128, 64, 1],
                "history": self.contract.history,
                "temporal_feature": "current_minus_oldest_public_observation",
                "set_context": "shared_per_node_scorer",
                "head": [128, 64, 1],
                "fmg_prior": False,
                "residual_bound": False,
            },
            "critics": {
                "names": list(critic_ids),
                "parameters_disjoint": not bool(
                    set(critic_ids["reward"])
                    & (set(critic_ids["failure"]) | set(critic_ids["depletion"]))
                    or set(critic_ids["failure"]) & set(critic_ids["depletion"])
                ),
                "shared_public_temporal_encoder": True,
                "independent_value_cost_heads": True,
                "pooling": ["eligible_mean", "eligible_max"],
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
            "masked_filler": False,
        }
