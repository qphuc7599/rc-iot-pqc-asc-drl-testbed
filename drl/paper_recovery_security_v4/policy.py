"""Public-telemetry security-constrained direct PPO for Q-Relay.

The actor never receives a Byzantine identity, disaster flag, future outcome,
or simulator state.  Its admission rule is a deterministic function of the
same delayed public response telemetry supplied to every selector.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from drl.paper_recovery_original_ppo.policy import DirectCommitteeAction
from drl.paper_recovery_hardcore_v5.policy import (
    HardcoreCritic,
    HardcoreSetActor,
    HardcoreSetAwarePPO,
)


TRUST_LCB_ADMISSION_FLOOR = 0.45
MISS_STREAK_ADMISSION_STEPS = 2


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(float(value))))


class PublicSecurityActor(HardcoreSetActor):
    """Long-memory scorer with monotone public reliability structure."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.committee_size = int(config.physics.committee_size)
        self.raw_trust_lcb_weight = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), dtype=torch.float32)
        )
        self.raw_fast_response_weight = nn.Parameter(
            torch.tensor(_inverse_softplus(0.75), dtype=torch.float32)
        )
        self.raw_miss_streak_weight = nn.Parameter(
            torch.tensor(_inverse_softplus(2.0), dtype=torch.float32)
        )
        self.raw_uncertainty_weight = nn.Parameter(
            torch.tensor(_inverse_softplus(0.75), dtype=torch.float32)
        )
        self.raw_public_risk_weight = nn.Parameter(
            torch.tensor(_inverse_softplus(2.0), dtype=torch.float32)
        )

    @property
    def trust_lcb_weight(self) -> Tensor:
        return self._positive(self.raw_trust_lcb_weight)

    @property
    def fast_response_weight(self) -> Tensor:
        return self._positive(self.raw_fast_response_weight)

    @property
    def miss_streak_weight(self) -> Tensor:
        return self._positive(self.raw_miss_streak_weight)

    @property
    def uncertainty_weight(self) -> Tensor:
        return self._positive(self.raw_uncertainty_weight)

    @property
    def public_risk_weight(self) -> Tensor:
        return self._positive(self.raw_public_risk_weight)

    @staticmethod
    def public_nonresponse_probability(current_nodes: Tensor) -> Tensor:
        # Beta-posterior mean response probability is public feature 2.
        return (1.0 - current_nodes[:, :, 2]).clamp(0.0, 1.0)

    def public_admission_mask(
        self, current_nodes: Tensor, valid_mask: Tensor
    ) -> Tensor:
        miss_steps = current_nodes[:, :, 12] * 8.0
        quarantined = (
            (miss_steps >= float(MISS_STREAK_ADMISSION_STEPS))
            & (current_nodes[:, :, 9] < TRUST_LCB_ADMISSION_FLOOR)
        )
        admitted = valid_mask & ~quarantined
        # Availability-preserving degraded mode: if public quarantine leaves
        # fewer than k candidates, release exactly the deficit with the lowest
        # public posterior risk.  This is pre-action and uses no simulator truth.
        deficient = torch.nonzero(
            admitted.sum(dim=1) < self.committee_size, as_tuple=False
        ).flatten()
        if deficient.numel():
            public_risk = (
                self.public_nonresponse_probability(current_nodes)
                + 0.50 * current_nodes[:, :, 12]
                + 0.50 * (1.0 - current_nodes[:, :, 9])
            )
            admitted = admitted.clone()
            for row_tensor in deficient:
                row = int(row_tensor.item())
                deficit = self.committee_size - int(admitted[row].sum().item())
                candidates = valid_mask[row] & ~admitted[row]
                candidate_ids = torch.nonzero(candidates, as_tuple=False).flatten()
                if candidate_ids.numel() < deficit:
                    raise RuntimeError("public minimum-risk refill cannot reach k")
                release = candidate_ids[
                    torch.argsort(public_risk[row, candidate_ids], stable=True)[:deficit]
                ]
                admitted[row, release] = True
        return admitted

    def forward(
        self,
        current_nodes: Tensor,
        current_global: Tensor,
        node_sequence: Tensor,
        global_sequence: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        admitted = self.public_admission_mask(current_nodes, valid_mask)
        encoded = self.encoder(
            current_nodes, current_global, node_sequence, global_sequence
        )
        learned = self.head(encoded).squeeze(-1)
        trust_mean = current_nodes[:, :, 2]
        trust_lcb = current_nodes[:, :, 9]
        uncertainty_proxy = (trust_mean - trust_lcb).clamp_min(0.0)
        public_risk = self.public_nonresponse_probability(current_nodes)
        score = (
            learned
            + self.node_headroom_weight * current_nodes[:, :, 7]
            + self.cell_headroom_weight * current_nodes[:, :, 8]
            + self.service_age_weight * current_nodes[:, :, 11]
            + self.trust_lcb_weight * trust_lcb
            + self.fast_response_weight * current_nodes[:, :, 10]
            - self.miss_streak_weight * current_nodes[:, :, 12]
            - self.uncertainty_weight * uncertainty_proxy
            - self.public_risk_weight * public_risk
        )
        return score.masked_fill(~admitted, -torch.inf)


class SecurityConstrainedPPO(HardcoreSetAwarePPO):
    """Direct PPO with reward, failure, depletion, and security-risk critics."""

    MODEL_SCHEMA = "qrelay-security-constrained-direct-ppo-4"

    def __init__(self, config: Any):
        super().__init__(config)
        self.actor = PublicSecurityActor(config)
        self.security_cost_critic = HardcoreCritic(config)

    def forward(
        self,
        observation: Tensor,
        observation_sequence: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        observation, observation_sequence = self.contract.normalise(
            observation, observation_sequence
        )
        inputs = self.contract.split(observation, observation_sequence)
        admitted = self.actor.public_admission_mask(inputs[0], inputs[-1])
        return (
            self.actor(*inputs),
            self.reward_critic(*inputs),
            self.failure_cost_critic(*inputs),
            self.depletion_cost_critic(*inputs),
            self.security_cost_critic(*inputs),
            admitted,
        )

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
            logits, reward, failure, depletion, _, valid = self(obs, sequence)
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
        parent = super().architecture_manifest()
        parent.update(
            {
                "schema": self.MODEL_SCHEMA,
                "scientific_parent": "original_paper_direct_full_committee_ppo",
                "public_security_admission": {
                    "trust_lcb_floor": TRUST_LCB_ADMISSION_FLOOR,
                    "consecutive_miss_steps": MISS_STREAK_ADMISSION_STEPS,
                    "below_k_mode": "minimum_public_risk_refill_exact_deficit",
                    "hidden_labels": False,
                },
                "critics": {
                    "independent_reward_failure_depletion_security": True,
                    "security_signal": (
                        "public_beta_posterior_poisson_binomial_tail_probability"
                    ),
                },
                "sampling": {
                    "distribution": "exact_setaware_ordered_plackett_luce",
                    "committee_size": self.committee_size,
                    "temperature": self.temperature,
                    "deterministic_deployment_option": True,
                },
                "fmg_prior": False,
                "residual": False,
                "hidden_inputs": False,
                "detector_oracle": False,
                "post_action_shield": False,
            }
        )
        return parent
