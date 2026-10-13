"""Residual PPO policy for the frozen Q-Relay recovery contract.

The actor is permutation equivariant: one shared MLP scores every node.  Its
final residual layer is initialised to zero, so deterministic update-0 is
exactly the transparent feature-matched-greedy (FMG) prior.  Stochastic
committees use a masked Plackett--Luce distribution sampled by Gumbel Top-k.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .config import PaperRecoveryConfig


@dataclass(frozen=True)
class PolicyAction:
    """One auditable ordered committee decision."""

    action_scores: np.ndarray
    ordered_committee: np.ndarray
    observable_mask: np.ndarray
    ordered_log_probability: float
    mean_per_pick_entropy: float
    value: float
    deterministic: bool


class TemporalObservationBuffer:
    """Fixed-width observable history; no hidden simulator state is retained."""

    def __init__(self, window: int):
        if int(window) < 1:
            raise ValueError("temporal window must be positive")
        self.window = int(window)
        self._values: deque[np.ndarray] = deque(maxlen=self.window)

    def reset(self, observation: np.ndarray) -> None:
        value = np.asarray(observation, dtype=np.float32).copy()
        self._values.clear()
        for _ in range(self.window):
            self._values.append(value.copy())

    def append(self, observation: np.ndarray) -> None:
        value = np.asarray(observation, dtype=np.float32).copy()
        if not self._values:
            self.reset(value)
        else:
            self._values.append(value)

    def array(self) -> np.ndarray:
        if not self._values:
            raise RuntimeError("temporal buffer has not been reset")
        values = list(self._values)
        while len(values) < self.window:
            values.insert(0, values[0].copy())
        return np.stack(values, axis=0).astype(np.float32, copy=False)


class ResidualPPOPolicy(nn.Module):
    """Shared per-node residual actor and permutation-invariant pooled critic."""

    MODEL_SCHEMA = "qrelay-paper-recovery-residual-ppo-policy-1"

    def __init__(self, config: PaperRecoveryConfig | None = None):
        super().__init__()
        self.config = config or PaperRecoveryConfig()
        p = self.config.physics
        o = self.config.observations
        h1, h2 = self.config.ppo.actor_hidden
        c1, c2 = self.config.ppo.critic_node_hidden
        self.num_nodes = p.num_nodes
        self.committee_size = p.committee_size
        self.node_features = o.per_node_features
        self.global_features = o.global_features
        self.temperature = float(self.config.ppo.gumbel_temperature)
        self.residual_scale = float(self.config.ppo.residual_score_scale)
        self.temporal_hidden = 64

        # For each node: current-minus-window-start and current-minus-previous.
        self.temporal_encoder = nn.Sequential(
            nn.Linear(2 * self.node_features, self.temporal_hidden),
            nn.Tanh(),
        )
        node_input = self.node_features + self.global_features + self.temporal_hidden
        self.actor = nn.Sequential(
            nn.Linear(node_input, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )
        self.critic_node_encoder = nn.Sequential(
            nn.Linear(node_input, c1),
            nn.ReLU(),
            nn.Linear(c1, c2),
            nn.ReLU(),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(2 * c2 + self.global_features + self.temporal_hidden, self.config.ppo.critic_head_hidden),
            nn.ReLU(),
            nn.Linear(self.config.ppo.critic_head_hidden, 1),
        )
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def _normalise_inputs(
        self, observation: Tensor, observation_sequence: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        if observation.dim() != 2 or observation.shape[-1] != self.config.observation_size:
            raise ValueError("observation shape does not match frozen schema")
        if observation_sequence is None:
            observation_sequence = observation.unsqueeze(1).expand(
                -1, self.config.observations.temporal_delta_window, -1
            )
        elif observation_sequence.dim() == 2:
            observation_sequence = observation_sequence.unsqueeze(0)
        if observation_sequence.dim() != 3:
            raise ValueError("observation_sequence must have [batch, time, observation]")
        if observation_sequence.shape[0] != observation.shape[0]:
            raise ValueError("observation and sequence batch sizes differ")
        if observation_sequence.shape[-1] != self.config.observation_size:
            raise ValueError("temporal observation schema mismatch")
        return observation, observation_sequence

    def _split_observation(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        boundary = self.num_nodes * self.node_features
        node = observation[:, :boundary].reshape(
            observation.shape[0], self.num_nodes, self.node_features
        )
        global_obs = observation[:, boundary:]
        return node, global_obs

    def _node_temporal_context(
        self, observation: Tensor, observation_sequence: Tensor
    ) -> tuple[Tensor, Tensor]:
        del observation
        boundary = self.num_nodes * self.node_features
        node_sequence = observation_sequence[:, :, :boundary].reshape(
            observation_sequence.shape[0],
            observation_sequence.shape[1],
            self.num_nodes,
            self.node_features,
        ).permute(0, 2, 1, 3)
        current = node_sequence[:, :, -1]
        previous = node_sequence[:, :, -2] if node_sequence.shape[2] > 1 else current
        temporal_delta = torch.cat(
            (current - node_sequence[:, :, 0], current - previous), dim=-1
        )
        per_node = self.temporal_encoder(temporal_delta)
        return per_node, current

    @staticmethod
    def _fmg_prior(node: Tensor, global_obs: Tensor) -> Tensor:
        """Transparent prior registered in the manuscript and baseline code."""

        load = global_obs[:, 0].unsqueeze(1)
        return (
            1.30 * node[:, :, 9]
            + 0.90 * node[:, :, 4]
            + (1.10 + 0.40 * load) * node[:, :, 5]
            + 0.55 * node[:, :, 3]
            + 0.45 * node[:, :, 0]
            - 0.35 * node[:, :, 7]
            - 0.20 * node[:, :, 8]
            + 0.10 * node[:, :, 6]
            + 0.80 * node[:, :, 10]
            + 0.40 * node[:, :, 11]
            - 0.80 * node[:, :, 12]
        )

    def forward(
        self,
        observation: Tensor,
        observation_sequence: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        observation, observation_sequence = self._normalise_inputs(
            observation, observation_sequence
        )
        node, global_obs = self._split_observation(observation)
        temporal, _ = self._node_temporal_context(observation, observation_sequence)
        global_expanded = global_obs.unsqueeze(1).expand(-1, self.num_nodes, -1)
        node_input = torch.cat((node, global_expanded, temporal), dim=-1)
        residual = self.actor(node_input).squeeze(-1)
        prior = self._fmg_prior(node, global_obs)
        scores = prior + self.residual_scale * torch.tanh(residual)
        valid_mask = (node[:, :, 1] > 0.5) & (
            node[:, :, 0] > self.config.physics.battery_cutoff_ratio
        )
        masked_scores = scores.masked_fill(~valid_mask, -torch.inf)

        encoded = self.critic_node_encoder(node_input)
        mask_float = valid_mask.unsqueeze(-1).to(encoded.dtype)
        pooled_mean = (encoded * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        masked_encoded = encoded.masked_fill(~valid_mask.unsqueeze(-1), -torch.inf)
        pooled_max = masked_encoded.max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        temporal_global = (temporal * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        value = self.critic_head(
            torch.cat((pooled_mean, pooled_max, global_obs, temporal_global), dim=-1)
        ).squeeze(-1)
        return masked_scores, value, valid_mask, prior

    @staticmethod
    def plackett_luce_stats(
        policy_logits: Tensor,
        ordered_committee: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Exact ordered log probability and mean conditional pick entropy.

        Entropy is conditioned on the sampled prefix and averaged across picks,
        which is the quantity multiplied by the registered entropy coefficient.
        """

        if policy_logits.dim() != 2 or valid_mask.shape != policy_logits.shape:
            raise ValueError("policy logits and mask must be matching rank-2 tensors")
        if ordered_committee.dim() == 1:
            ordered_committee = ordered_committee.unsqueeze(0)
        if ordered_committee.shape[0] != policy_logits.shape[0]:
            raise ValueError("ordered committee batch size mismatch")
        if ordered_committee.numel() == 0:
            zeros = policy_logits.sum(dim=-1) * 0.0
            return zeros, zeros
        if torch.any(ordered_committee < 0) or torch.any(
            ordered_committee >= policy_logits.shape[1]
        ):
            raise ValueError("ordered committee contains an out-of-range node")

        remaining = valid_mask.clone()
        log_probability = torch.zeros(
            policy_logits.shape[0], dtype=policy_logits.dtype, device=policy_logits.device
        )
        entropy_sum = torch.zeros_like(log_probability)
        for pick in range(ordered_committee.shape[1]):
            chosen = ordered_committee[:, pick]
            if not torch.all(remaining.gather(1, chosen.unsqueeze(1)).squeeze(1)):
                raise ValueError("ordered committee repeats or selects an ineligible node")
            conditional_logits = policy_logits.masked_fill(~remaining, -torch.inf)
            conditional_log_probs = torch.log_softmax(conditional_logits, dim=-1)
            conditional_probs = torch.softmax(conditional_logits, dim=-1)
            log_probability = log_probability + conditional_log_probs.gather(
                1, chosen.unsqueeze(1)
            ).squeeze(1)
            safe_conditional_log_probs = conditional_log_probs.masked_fill(~remaining, 0.0)
            entropy_sum = entropy_sum - (
                conditional_probs * safe_conditional_log_probs
            ).sum(dim=-1)
            remaining = remaining.scatter(1, chosen.unsqueeze(1), False)
        return log_probability, entropy_sum / float(ordered_committee.shape[1])

    def evaluate_ordered_committees(
        self,
        observation: Tensor,
        observation_sequence: Tensor,
        ordered_committee: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scores, value, valid_mask, _ = self.forward(observation, observation_sequence)
        policy_logits = scores / max(self.temperature, 1e-6)
        log_probability, mean_pick_entropy = self.plackett_luce_stats(
            policy_logits, ordered_committee, valid_mask
        )
        return log_probability, mean_pick_entropy, value

    def act(
        self,
        observation: np.ndarray,
        observation_sequence: np.ndarray | None = None,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> PolicyAction:
        device = next(self.parameters()).device
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        sequence_tensor = None
        if observation_sequence is not None:
            sequence_tensor = torch.as_tensor(
                observation_sequence, dtype=torch.float32, device=device
            )
        with torch.no_grad():
            scores, value, valid_mask, _ = self.forward(obs_tensor, sequence_tensor)
            eligible = int(valid_mask[0].sum().item())
            policy_logits = scores / max(self.temperature, 1e-6)
            if eligible < self.committee_size:
                ordered = torch.empty((1, 0), dtype=torch.long, device=device)
                action_scores = policy_logits
                log_probability = torch.zeros(1, device=device)
                entropy = torch.zeros(1, device=device)
            else:
                if deterministic:
                    action_scores = policy_logits
                    score_row = action_scores[0].detach().cpu().numpy()
                    mask_row = valid_mask[0].detach().cpu().numpy()
                    stable_order = np.lexsort(
                        (np.arange(self.num_nodes, dtype=np.int64), -score_row)
                    )
                    stable_order = stable_order[mask_row[stable_order]][
                        : self.committee_size
                    ]
                    ordered = torch.as_tensor(
                        stable_order,
                        dtype=torch.long,
                        device=device,
                    ).unsqueeze(0)
                else:
                    uniforms = torch.rand(
                        policy_logits.shape,
                        # Float64 prevents the upper-tail clamp collisions that
                        # can occur in float32 and make two Gumbel scores equal.
                        # The ordered sample remains exactly Gumbel Top-k from
                        # the registered float32 policy logits.
                        dtype=torch.float64,
                        device=device,
                        generator=generator,
                    ).clamp_(1e-12, 1.0 - 1e-12)
                    gumbel = -torch.log(-torch.log(uniforms))
                    action_scores = (policy_logits.to(torch.float64) + gumbel).masked_fill(
                        ~valid_mask, -torch.inf
                    )
                    ordered = torch.topk(
                        action_scores, k=self.committee_size, dim=-1, sorted=True
                    ).indices
                log_probability, entropy = self.plackett_luce_stats(
                    policy_logits, ordered, valid_mask
                )
        return PolicyAction(
            action_scores=action_scores[0].cpu().numpy().astype(np.float64),
            ordered_committee=ordered[0].cpu().numpy().astype(np.int64),
            observable_mask=valid_mask[0].cpu().numpy().astype(bool),
            ordered_log_probability=float(log_probability[0].item()),
            mean_per_pick_entropy=float(entropy[0].item()),
            value=float(value[0].item()),
            deterministic=bool(deterministic),
        )

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "schema": self.MODEL_SCHEMA,
            "actor": {
                "sharing": "one_mlp_shared_across_nodes",
                "hidden": list(self.config.ppo.actor_hidden),
                "output": 1,
                "bounded_residual": f"{self.residual_scale}*tanh(raw_residual)",
                "output_layer_zero_initialised": True,
                "transparent_prior": "feature_matched_greedy",
            },
            "temporal": {
                "window": self.config.observations.temporal_delta_window,
                "features": ["current_minus_window_start", "current_minus_previous"],
                "hidden": self.temporal_hidden,
            },
            "critic": {
                "node_hidden": list(self.config.ppo.critic_node_hidden),
                "pooling": ["eligible_mean", "eligible_max"],
                "head_hidden": self.config.ppo.critic_head_hidden,
            },
            "sampling": {
                "distribution": "exact_masked_ordered_plackett_luce",
                "sampler": "gumbel_top_k",
                "temperature": self.temperature,
                "entropy": "mean_conditional_entropy_per_pick",
            },
            "execution_shield": False,
            "masked_filler": False,
        }


def stack_temporal_sequences(values: Iterable[Sequence[np.ndarray]]) -> np.ndarray:
    """Convert a batch of observable histories to a stable float32 tensor."""

    return np.asarray([[np.asarray(item, dtype=np.float32) for item in row] for row in values])
