"""Long-memory set-aware PPO for the hardcore delayed cascade."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from drl.paper_recovery_original_ppo.policy import (
    DirectCommitteeAction,
    ReviewerAlignedDirectPPO,
    _InputContract,
    _masked_max,
    _masked_mean,
)


class LongMemoryEncoder(nn.Module):
    """Cheap causal 64-step public-history encoder at six time scales."""

    def __init__(self, config: Any, *, hidden: int = 128):
        super().__init__()
        self.contract = _InputContract(config)
        self.scales = (0, 15, 31, 47, 62)
        if self.contract.history != 64:
            raise ValueError("hardcore encoder requires exactly 64 public steps")
        inputs = 6 * (
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
            value.unsqueeze(1).expand(-1, nodes, -1)
            for value in global_parts
        ]
        return self.network(
            torch.cat((*node_parts, *expanded_global), dim=-1)
        )


class HardcoreSetActor(nn.Module):
    """Public scorer with monotone recovery/recency inductive structure."""

    def __init__(self, config: Any):
        super().__init__()
        self.encoder = LongMemoryEncoder(config)
        self.head = nn.Linear(64, 1)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)
        self.raw_node_headroom_weight = nn.Parameter(torch.tensor(0.0))
        self.raw_cell_headroom_weight = nn.Parameter(torch.tensor(0.0))
        self.raw_service_age_weight = nn.Parameter(torch.tensor(0.0))
        self.raw_diversity_weight = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _positive(value: Tensor) -> Tensor:
        return torch.nn.functional.softplus(value)

    @property
    def node_headroom_weight(self) -> Tensor:
        return self._positive(self.raw_node_headroom_weight)

    @property
    def cell_headroom_weight(self) -> Tensor:
        return self._positive(self.raw_cell_headroom_weight)

    @property
    def service_age_weight(self) -> Tensor:
        return self._positive(self.raw_service_age_weight)

    @property
    def diversity_weight(self) -> Tensor:
        return self._positive(self.raw_diversity_weight)

    def forward(
        self,
        current_nodes: Tensor,
        current_global: Tensor,
        node_sequence: Tensor,
        global_sequence: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if torch.any(valid_mask.sum(dim=1) == 0):
            raise RuntimeError("hardcore actor has no eligible node")
        encoded = self.encoder(
            current_nodes, current_global, node_sequence, global_sequence
        )
        learned = self.head(encoded).squeeze(-1)
        score = (
            learned
            + self.node_headroom_weight * current_nodes[:, :, 7]
            + self.cell_headroom_weight * current_nodes[:, :, 8]
            + self.service_age_weight * current_nodes[:, :, 11]
        )
        return score.masked_fill(~valid_mask, -torch.inf)


class HardcoreCritic(nn.Module):
    """Parameter-disjoint long-memory permutation-invariant critic."""

    def __init__(self, config: Any):
        super().__init__()
        self.encoder = LongMemoryEncoder(config, hidden=96)
        global_context = 6 * config.observations.global_features
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


class HardcoreSetAwarePPO(ReviewerAlignedDirectPPO):
    """Pure direct PPO with exact prefix-conditioned committee sampling."""

    MODEL_SCHEMA = "qrelay-reviewer-hardcore-setaware-ppo-5"
    NUM_CELLS = 5

    def __init__(self, config: Any):
        nn.Module.__init__(self)
        self.config = config
        self.contract = _InputContract(config)
        self.num_nodes = config.physics.num_nodes
        self.committee_size = config.physics.committee_size
        self.temperature = float(config.ppo.gumbel_temperature)
        self.actor = HardcoreSetActor(config)
        self.reward_critic = HardcoreCritic(config)
        self.failure_cost_critic = HardcoreCritic(config)
        self.depletion_cost_critic = HardcoreCritic(config)
        self.register_buffer(
            "node_cell_ids",
            torch.arange(self.num_nodes, dtype=torch.long) % self.NUM_CELLS,
            persistent=True,
        )

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

    def plackett_luce_per_pick(
        self,
        policy_logits: Tensor,
        ordered_committee: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if ordered_committee.dim() == 1:
            ordered_committee = ordered_committee.unsqueeze(0)
        batch, picks = ordered_committee.shape
        if policy_logits.shape != valid_mask.shape or policy_logits.shape[0] != batch:
            raise ValueError("hardcore Plackett--Luce shape mismatch")
        if not torch.all(valid_mask.gather(1, ordered_committee)):
            raise ValueError("hardcore committee contains an invalid node")
        if any(torch.unique(row).numel() != picks for row in ordered_committee):
            raise ValueError("hardcore committee repeats a node")
        one_hot = torch.zeros(
            (batch, picks, policy_logits.shape[1]),
            dtype=torch.bool,
            device=policy_logits.device,
        )
        one_hot.scatter_(2, ordered_committee.unsqueeze(-1), True)
        removed = torch.cat(
            (
                torch.zeros_like(one_hot[:, :1]),
                one_hot[:, :-1].cumsum(dim=1) > 0,
            ),
            dim=1,
        )
        remaining = valid_mask.unsqueeze(1) & ~removed
        node_cells = self.node_cell_ids.to(policy_logits.device)
        selected_cells = node_cells[ordered_committee]
        selected_cell_one_hot = torch.nn.functional.one_hot(
            selected_cells, num_classes=self.NUM_CELLS
        ).to(policy_logits.dtype)
        prefix_cell_counts = torch.cat(
            (
                torch.zeros_like(selected_cell_one_hot[:, :1]),
                selected_cell_one_hot[:, :-1].cumsum(dim=1),
            ),
            dim=1,
        )
        prefix_count_by_candidate = prefix_cell_counts[:, :, node_cells]
        adjusted = (
            policy_logits.unsqueeze(1)
            - self.actor.diversity_weight * prefix_count_by_candidate
        ).masked_fill(~remaining, -1e9)
        logp = torch.log_softmax(adjusted, dim=-1)
        probability = logp.exp()
        selected_logp = logp.gather(
            2, ordered_committee.unsqueeze(-1)
        ).squeeze(-1)
        entropy = -(probability * logp).masked_fill(~remaining, 0.0).sum(dim=-1)
        return selected_logp, entropy

    def plackett_luce_stats(
        self,
        policy_logits: Tensor,
        ordered_committee: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        per_pick, entropy = self.plackett_luce_per_pick(
            policy_logits, ordered_committee, valid_mask
        )
        return per_pick.sum(dim=-1), entropy.mean(dim=-1)

    def sample_ordered(
        self,
        policy_logits: Tensor,
        valid_mask: Tensor,
        generators: list[torch.Generator],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = policy_logits.shape[0]
        remaining = valid_mask.clone()
        cell_counts = torch.zeros(
            (batch, self.NUM_CELLS),
            dtype=policy_logits.dtype,
            device=policy_logits.device,
        )
        node_cells = self.node_cell_ids.to(policy_logits.device)
        tapes = []
        for generator in generators:
            uniforms = torch.rand(
                (self.committee_size, self.num_nodes),
                dtype=torch.float64,
                device=policy_logits.device,
                generator=generator,
            ).clamp_(1e-12, 1.0 - 1e-12)
            tapes.append(-torch.log(-torch.log(uniforms)))
        gumbels = torch.stack(tapes, dim=0)
        picks: list[Tensor] = []
        for pick in range(self.committee_size):
            adjusted = (
                policy_logits
                - self.actor.diversity_weight * cell_counts[:, node_cells]
            ).masked_fill(~remaining, -torch.inf)
            selected = torch.argmax(
                adjusted.to(torch.float64) + gumbels[:, pick, :], dim=1
            ).to(torch.long)
            picks.append(selected)
            cells = node_cells[selected]
            cell_counts.scatter_add_(
                1,
                cells[:, None],
                torch.ones(
                    (batch, 1),
                    dtype=policy_logits.dtype,
                    device=policy_logits.device,
                ),
            )
            remaining = remaining.scatter(1, selected[:, None], False)
        ordered = torch.stack(picks, dim=1)
        log_probability, entropy = self.plackett_luce_stats(
            policy_logits, ordered, valid_mask
        )
        action_scores = torch.full(
            policy_logits.shape,
            -1e12,
            dtype=torch.float64,
            device=policy_logits.device,
        )
        ranks = torch.arange(
            self.committee_size, dtype=torch.float64, device=policy_logits.device
        )
        action_scores.scatter_(
            1,
            ordered,
            (self.committee_size - ranks)[None, :].expand(batch, -1),
        )
        return action_scores, ordered, log_probability, entropy

    def deterministic_order(
        self, policy_logits: Tensor, valid_mask: Tensor
    ) -> Tensor:
        batch = policy_logits.shape[0]
        remaining = valid_mask.clone()
        counts = torch.zeros(
            (batch, self.NUM_CELLS),
            dtype=policy_logits.dtype,
            device=policy_logits.device,
        )
        node_cells = self.node_cell_ids.to(policy_logits.device)
        picks: list[Tensor] = []
        for _ in range(self.committee_size):
            adjusted = (
                policy_logits - self.actor.diversity_weight * counts[:, node_cells]
            ).masked_fill(~remaining, -torch.inf)
            selected = torch.argmax(adjusted, dim=1)
            picks.append(selected)
            cells = node_cells[selected]
            counts.scatter_add_(
                1,
                cells[:, None],
                torch.ones(
                    (batch, 1),
                    dtype=policy_logits.dtype,
                    device=policy_logits.device,
                ),
            )
            remaining = remaining.scatter(1, selected[:, None], False)
        return torch.stack(picks, dim=1)

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
        sequence = None if observation_sequence is None else torch.as_tensor(
            observation_sequence, dtype=torch.float32, device=device
        )
        with torch.no_grad():
            logits, reward, failure, depletion, valid = self(obs, sequence)
            if int(valid[0].sum().item()) < self.committee_size:
                raise RuntimeError("hardcore action failed closed below k")
            policy_logits = logits / max(self.temperature, 1e-6)
            if deterministic:
                ordered = self.deterministic_order(policy_logits, valid)
                log_probability, entropy = self.plackett_luce_stats(
                    policy_logits, ordered, valid
                )
                action_scores = torch.full(
                    policy_logits.shape,
                    -1e12,
                    dtype=torch.float64,
                    device=device,
                )
                ranks = torch.arange(
                    self.committee_size, dtype=torch.float64, device=device
                )
                action_scores.scatter_(
                    1, ordered, (self.committee_size - ranks)[None, :]
                )
            else:
                if generator is None:
                    generator = torch.Generator(device=device)
                action_scores, ordered, log_probability, entropy = self.sample_ordered(
                    policy_logits, valid, [generator]
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
        modules = {
            "actor": self.actor,
            "reward": self.reward_critic,
            "failure": self.failure_cost_critic,
            "depletion": self.depletion_cost_critic,
        }
        parameter_ids = {
            name: {id(value) for value in module.parameters()}
            for name, module in modules.items()
        }
        disjoint = all(
            not parameter_ids[left] & parameter_ids[right]
            for index, left in enumerate(parameter_ids)
            for right in tuple(parameter_ids)[index + 1 :]
        )
        return {
            "schema": self.MODEL_SCHEMA,
            "scientific_parent": "original_paper_direct_full_committee_ppo",
            "actor": {
                "per_node_shared_mlp": [128, 64, 1],
                "public_history_steps": self.contract.history,
                "history_scales": list(self.actor.encoder.scales),
                "decoder": "exact_prefix_conditioned_Plackett_Luce",
                "positive_public_heads": [
                    "node_recovery_headroom",
                    "cell_capacity_headroom",
                    "service_age",
                ],
                "fmg_prior": False,
                "residual": False,
            },
            "critics": {
                "independent_reward_failure_depletion": True,
                "actor_and_all_critics_parameter_disjoint": disjoint,
                "pooling": ["eligible_mean", "eligible_max"],
            },
            "sampling": {
                "distribution": "exact_setaware_ordered_plackett_luce",
                "committee_size": self.committee_size,
                "temperature": self.temperature,
            },
            "hidden_inputs": False,
            "detector_oracle": False,
            "execution_shield": False,
        }
