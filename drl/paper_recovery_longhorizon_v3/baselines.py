"""Fair public-information baselines for the controllable cascade."""

from __future__ import annotations

import numpy as np

from drl.paper_recovery.config import PaperRecoveryConfig


def split_observation(
    observation: np.ndarray, config: PaperRecoveryConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(observation, dtype=np.float64)
    boundary = config.physics.num_nodes * config.observations.per_node_features
    node = values[:boundary].reshape(
        config.physics.num_nodes, config.observations.per_node_features
    )
    global_obs = values[boundary:]
    mask = (node[:, 1] > 0.5) & (
        node[:, 0] > config.physics.battery_cutoff_ratio
    )
    return node, global_obs, mask


def fmg_scores(node: np.ndarray, global_obs: np.ndarray) -> np.ndarray:
    """Transparent FMG extended with the two new public headroom signals."""

    load = float(global_obs[0])
    return (
        1.30 * node[:, 9]
        + 0.90 * node[:, 4]
        + (1.10 + 0.40 * load) * node[:, 5]
        + 0.55 * node[:, 3]
        + 0.45 * node[:, 0]
        + 0.65 * node[:, 7]
        + 0.75 * node[:, 8]
        + 0.80 * node[:, 10]
        + 0.40 * node[:, 11]
        - 0.80 * node[:, 12]
    )


class BaselineSelector:
    name = "base"

    def __init__(self, config: PaperRecoveryConfig, *, seed: int):
        self.config = config
        self.rng = np.random.default_rng(int(seed))

    def reset(self, observation: np.ndarray) -> None:
        del observation

    def scores(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _ordered_action(selected: list[int], base: np.ndarray) -> np.ndarray:
        action = np.asarray(base, dtype=np.float64).copy()
        if selected:
            finite = action[np.isfinite(action)]
            anchor = max(float(np.max(finite)) if len(finite) else 0.0, 0.0) + 1_000.0
            for rank, node_id in enumerate(selected):
                action[int(node_id)] = anchor - rank
        return action


class FMGSelector(BaselineSelector):
    name = "FMG"

    def scores(self, observation: np.ndarray) -> np.ndarray:
        node, global_obs, _ = split_observation(observation, self.config)
        return fmg_scores(node, global_obs)


class RiskPriorSelector(BaselineSelector):
    name = "Risk-Prior"

    def scores(self, observation: np.ndarray) -> np.ndarray:
        node, global_obs, _ = split_observation(observation, self.config)
        load = float(global_obs[0])
        return (
            2.10 * node[:, 9]
            + 1.25 * node[:, 10]
            + 0.80 * node[:, 11]
            + 0.75 * node[:, 4]
            + (0.65 + 0.35 * load) * node[:, 5]
            + 0.35 * node[:, 0]
            + 0.90 * node[:, 7]
            + 0.90 * node[:, 8]
            - 1.10 * node[:, 12]
        )


class CMABSelector(BaselineSelector):
    name = "CMAB"

    def scores(self, observation: np.ndarray) -> np.ndarray:
        node, _, _ = split_observation(observation, self.config)
        return (
            1.15 * node[:, 2]
            + 0.90 * node[:, 4]
            + 0.70 * node[:, 5]
            + 0.35 * node[:, 0]
            + 0.65 * node[:, 7]
            + 0.65 * node[:, 8]
            - 0.55 * node[:, 12]
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
                    - 0.32 * cell_counts[int(round(node[index, 6] * 4.0))]
                    + 0.35 * node[index, 7]
                    + 0.40 * node[index, 8],
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
        vectors = node[:, [0, 3, 4, 5, 6, 7, 8, 10, 11]]
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
                        + 0.35 * float(
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
                - 0.45 * concentration
                + 0.30 * node[committee, 7].mean() * k
                + 0.35 * node[committee, 8].mean() * k
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
            replacement = int(outside[int(self.rng.integers(0, len(outside)))])
            proposal = current.copy()
            removed = int(proposal[position])
            proposal[position] = replacement
            proposal_value = utility(proposal)
            difference = proposal_value - value
            if difference >= 0.0 or self.rng.random() < np.exp(difference / temperature):
                current = proposal
                value = proposal_value
                current_set.remove(removed)
                current_set.add(replacement)
        ordered = sorted(current.astype(int).tolist(), key=lambda index: (-base[index], index))
        return self._ordered_action(ordered, base)


class RandomSelector(BaselineSelector):
    name = "Random"

    def scores(self, observation: np.ndarray) -> np.ndarray:
        _, _, mask = split_observation(observation, self.config)
        scores = self.rng.random(self.config.physics.num_nodes)
        scores[~mask] = -np.inf
        return scores


class StaticValidRefillSelector(BaselineSelector):
    name = "Static-with-valid-refill"

    def __init__(self, config: PaperRecoveryConfig, *, seed: int):
        super().__init__(config, seed=seed)
        self.static_ids: list[int] = []

    def reset(self, observation: np.ndarray) -> None:
        node, global_obs, mask = split_observation(observation, self.config)
        base = fmg_scores(node, global_obs)
        order = np.lexsort((np.arange(len(base)), -base))
        self.static_ids = order[mask[order]][: self.config.physics.committee_size].astype(int).tolist()

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
        StaticValidRefillSelector,
    )
}


def make_selector(
    name: str, config: PaperRecoveryConfig, *, seed: int
) -> BaselineSelector:
    if name not in SELECTORS:
        raise KeyError(f"unknown long-horizon baseline: {name}")
    return SELECTORS[name](config, seed=seed)


def baseline_manifest() -> dict[str, object]:
    return {
        "schema": "qrelay-reviewer-longhorizon-baselines-3",
        "selectors": list(SELECTORS),
        "all_use_same_public_observation": True,
        "hidden_labels": False,
        "fmg_includes_public_fatigue_and_cell_headroom": True,
        "gba_and_sa_include_public_cell_diversity": True,
        "coefficients_frozen_before_any_ppo_result": True,
    }
