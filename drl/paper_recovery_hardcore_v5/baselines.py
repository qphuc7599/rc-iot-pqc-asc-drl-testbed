"""Prospectively frozen public-information baselines for hardcore v5."""

from __future__ import annotations

import numpy as np

from drl.paper_recovery.config import PaperRecoveryConfig
from drl.paper_recovery_longhorizon_v3.baselines import (
    BaselineSelector,
    RandomSelector,
    StaticValidRefillSelector,
    split_observation,
)


def fmg_scores(node: np.ndarray, global_obs: np.ndarray) -> np.ndarray:
    load = float(global_obs[0])
    queue = float(global_obs[5])
    return (
        1.35 * node[:, 9]
        + 0.95 * node[:, 4]
        + (1.00 + 0.45 * load) * node[:, 5]
        + 0.50 * node[:, 3]
        + 0.40 * node[:, 0]
        + (0.95 + 0.25 * queue) * node[:, 7]
        + 1.05 * node[:, 8]
        + 0.80 * node[:, 10]
        + 0.75 * node[:, 11]
        - 0.90 * node[:, 12]
    )


class FMGSelector(BaselineSelector):
    name = "FMG"

    def scores(self, observation: np.ndarray) -> np.ndarray:
        node, global_obs, _ = split_observation(observation, self.config)
        return fmg_scores(node, global_obs)


class RiskPriorSelector(BaselineSelector):
    name = "Risk-Prior"

    def scores(self, observation: np.ndarray) -> np.ndarray:
        node, global_obs, _ = split_observation(observation, self.config)
        return (
            2.10 * node[:, 9]
            + 1.25 * node[:, 10]
            + 0.85 * node[:, 4]
            + 1.10 * node[:, 7]
            + 1.20 * node[:, 8]
            + 0.90 * node[:, 11]
            - 1.20 * node[:, 12]
            + 0.30 * global_obs[0] * node[:, 5]
        )


class CMABSelector(BaselineSelector):
    name = "CMAB"

    def scores(self, observation: np.ndarray) -> np.ndarray:
        node, _, _ = split_observation(observation, self.config)
        return (
            1.10 * node[:, 2]
            + 0.90 * node[:, 4]
            + 0.65 * node[:, 5]
            + 0.80 * node[:, 7]
            + 0.80 * node[:, 8]
            + 0.55 * node[:, 11]
            - 0.60 * node[:, 12]
        )


class GBASelector(BaselineSelector):
    name = "GBA"

    def scores(self, observation: np.ndarray) -> np.ndarray:
        node, global_obs, mask = split_observation(observation, self.config)
        base = fmg_scores(node, global_obs)
        selected: list[int] = []
        cell_counts = np.zeros(5, dtype=np.int64)
        eligible = set(np.flatnonzero(mask).astype(int).tolist())
        while eligible and len(selected) < self.config.physics.committee_size:
            best = max(
                eligible,
                key=lambda index: (
                    base[index]
                    - 0.55 * cell_counts[int(round(node[index, 6] * 4.0))]
                    + 0.55 * node[index, 7]
                    + 0.60 * node[index, 8]
                    + 0.45 * node[index, 11],
                    -index,
                ),
            )
            selected.append(best)
            cell_counts[int(round(node[best, 6] * 4.0))] += 1
            eligible.remove(best)
        return self._ordered_action(selected, base)


class OCDInspiredSelector(BaselineSelector):
    name = "OCD-inspired"

    def scores(self, observation: np.ndarray) -> np.ndarray:
        node, global_obs, mask = split_observation(observation, self.config)
        base = fmg_scores(node, global_obs)
        vectors = node[:, [0, 3, 4, 5, 6, 7, 8, 10, 11, 12]]
        selected: list[int] = []
        eligible = set(np.flatnonzero(mask).astype(int).tolist())
        while eligible and len(selected) < self.config.physics.committee_size:
            if not selected:
                best = max(eligible, key=lambda index: (base[index], -index))
            else:
                best = max(
                    eligible,
                    key=lambda index: (
                        base[index]
                        + 0.45
                        * float(
                            np.min(
                                np.linalg.norm(
                                    vectors[index] - vectors[selected], axis=1
                                )
                            )
                        ),
                        -index,
                    ),
                )
            selected.append(best)
            eligible.remove(best)
        return self._ordered_action(selected, base)


class SimulatedAnnealingSelector(BaselineSelector):
    name = "SA"

    def scores(self, observation: np.ndarray) -> np.ndarray:
        node, global_obs, mask = split_observation(observation, self.config)
        base = fmg_scores(node, global_obs)
        eligible = np.flatnonzero(mask)
        k = self.config.physics.committee_size
        if len(eligible) < k:
            return base
        current = eligible[np.argsort(-base[eligible], kind="stable")[:k]].copy()

        def utility(committee: np.ndarray) -> float:
            cells = np.rint(node[committee, 6] * 4.0).astype(np.int64)
            concentration = np.square(np.bincount(cells, minlength=5)).sum() / k
            return float(
                base[committee].sum()
                - 0.70 * concentration
                + 0.45 * node[committee, 7].mean() * k
                + 0.50 * node[committee, 8].mean() * k
                + 0.40 * node[committee, 11].mean() * k
            )

        value = utility(current)
        current_set = set(current.astype(int).tolist())
        for iteration in range(160):
            temperature = 0.25 * (0.01 / 0.25) ** (iteration / 159.0)
            position = int(self.rng.integers(0, k))
            outside = np.asarray(
                [item for item in eligible if int(item) not in current_set]
            )
            if not len(outside):
                break
            replacement = int(
                outside[int(self.rng.integers(0, len(outside)))]
            )
            proposal = current.copy()
            removed = int(proposal[position])
            proposal[position] = replacement
            proposal_value = utility(proposal)
            difference = proposal_value - value
            if difference >= 0.0 or self.rng.random() < np.exp(
                difference / temperature
            ):
                current = proposal
                value = proposal_value
                current_set.remove(removed)
                current_set.add(replacement)
        ordered = sorted(
            current.astype(int).tolist(), key=lambda index: (-base[index], index)
        )
        return self._ordered_action(ordered, base)


class HardcoreStaticSelector(StaticValidRefillSelector):
    def reset(self, observation: np.ndarray) -> None:
        node, global_obs, mask = split_observation(observation, self.config)
        base = fmg_scores(node, global_obs)
        order = np.lexsort((np.arange(len(base)), -base))
        self.static_ids = (
            order[mask[order]][: self.config.physics.committee_size]
            .astype(int)
            .tolist()
        )

    def scores(self, observation: np.ndarray) -> np.ndarray:
        node, global_obs, mask = split_observation(observation, self.config)
        base = fmg_scores(node, global_obs)
        selected = [index for index in self.static_ids if mask[index]]
        if len(selected) < self.config.physics.committee_size:
            order = np.lexsort((np.arange(len(base)), -base))
            for index in order:
                index = int(index)
                if mask[index] and index not in selected:
                    selected.append(index)
                    if len(selected) == self.config.physics.committee_size:
                        break
        return self._ordered_action(selected, base)


SELECTORS = {
    selector.name: selector
    for selector in (
        RiskPriorSelector,
        FMGSelector,
        CMABSelector,
        SimulatedAnnealingSelector,
        GBASelector,
        OCDInspiredSelector,
        RandomSelector,
        HardcoreStaticSelector,
    )
}


def make_selector(
    name: str, config: PaperRecoveryConfig, *, seed: int
) -> BaselineSelector:
    if name not in SELECTORS:
        raise KeyError(f"unknown hardcore baseline: {name}")
    return SELECTORS[name](config, seed=seed)


def baseline_manifest() -> dict[str, object]:
    return {
        "schema": "qrelay-reviewer-hardcore-baselines-5",
        "selectors": list(SELECTORS),
        "all_use_same_public_observation": True,
        "hidden_labels": False,
        "all_strong_baselines_receive_delayed_fatigue_cell_and_service_age": True,
        "coefficients_frozen_before_any_v5_baseline_or_ppo_result": True,
    }
