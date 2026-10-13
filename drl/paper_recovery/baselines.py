"""Eight checkpoint-independent selectors frozen before baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import PaperRecoveryConfig


def split_observation(
    observation: np.ndarray, config: PaperRecoveryConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(observation, dtype=np.float64)
    boundary = config.physics.num_nodes * config.observations.per_node_features
    node = value[:boundary].reshape(
        config.physics.num_nodes, config.observations.per_node_features
    )
    global_obs = value[boundary:]
    mask = (node[:, 1] > 0.5) & (
        node[:, 0] > config.physics.battery_cutoff_ratio
    )
    return node, global_obs, mask


def fmg_scores(node: np.ndarray, global_obs: np.ndarray) -> np.ndarray:
    load = float(global_obs[0])
    return (
        1.30 * node[:, 9]
        + 0.90 * node[:, 4]
        + (1.10 + 0.40 * load) * node[:, 5]
        + 0.55 * node[:, 3]
        + 0.45 * node[:, 0]
        - 0.35 * node[:, 7]
        - 0.20 * node[:, 8]
        + 0.10 * node[:, 6]
        + 0.80 * node[:, 10]
        + 0.40 * node[:, 11]
        - 0.80 * node[:, 12]
    )


class BaselineSelector:
    name = "base"

    def __init__(self, config: PaperRecoveryConfig, *, seed: int):
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def reset(self, observation: np.ndarray) -> None:
        del observation

    def scores(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _ordered_action(self, selected: list[int], base: np.ndarray) -> np.ndarray:
        action = np.asarray(base, dtype=np.float64).copy()
        if selected:
            anchor = max(float(np.nanmax(action[np.isfinite(action)])), 0.0) + 1_000.0
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
            - 1.10 * node[:, 12]
            - 0.40 * node[:, 8]
        )


class CMABSelector(BaselineSelector):
    name = "CMAB"

    def scores(self, observation: np.ndarray) -> np.ndarray:
        node, _, _ = split_observation(observation, self.config)
        # Observable optimistic response utility: posterior means plus a fixed
        # uncertainty bonus; no hidden compromise label is available.
        return (
            1.15 * node[:, 2]
            + 0.90 * node[:, 4]
            + 0.70 * node[:, 5]
            + 0.55 * node[:, 8]
            + 0.35 * node[:, 0]
            - 0.55 * node[:, 12]
            - 0.20 * node[:, 7]
        )


class GBASelector(BaselineSelector):
    name = "GBA"

    def scores(self, observation: np.ndarray) -> np.ndarray:
        node, global_obs, mask = split_observation(observation, self.config)
        base = fmg_scores(node, global_obs)
        selected: list[int] = []
        service_sum = 0.0
        eligible = set(np.flatnonzero(mask).astype(int).tolist())
        while eligible and len(selected) < self.config.physics.committee_size:
            best = max(
                eligible,
                key=lambda index: (
                    base[index]
                    + 0.50 * node[index, 0]
                    + 0.35 * node[index, 3]
                    - 0.45 * (service_sum + node[index, 7]) / (len(selected) + 1),
                    -index,
                ),
            )
            selected.append(best)
            service_sum += float(node[best, 7])
            eligible.remove(best)
        return self._ordered_action(selected, base)


class OCDInspiredSelector(BaselineSelector):
    name = "OCD-inspired"

    def scores(self, observation: np.ndarray) -> np.ndarray:
        node, global_obs, mask = split_observation(observation, self.config)
        base = fmg_scores(node, global_obs)
        selected: list[int] = []
        eligible = set(np.flatnonzero(mask).astype(int).tolist())
        public_vectors = node[:, [0, 3, 4, 5, 10, 11]]
        while eligible and len(selected) < self.config.physics.committee_size:
            if not selected:
                best = max(eligible, key=lambda index: (base[index], -index))
            else:
                distances = np.asarray(
                    [
                        np.min(
                            np.linalg.norm(
                                public_vectors[index] - public_vectors[selected], axis=1
                            )
                        )
                        for index in eligible
                    ]
                )
                candidates = list(eligible)
                best = max(
                    zip(candidates, distances, strict=True),
                    key=lambda pair: (base[pair[0]] + 0.40 * pair[1], -pair[0]),
                )[0]
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
            return float(
                base[committee].sum()
                - 0.30 * np.square(node[committee, 7].mean()) * k
                + 0.20 * np.std(node[committee, 3]) * k
            )

        current_utility = utility(current)
        current_set = set(current.astype(int).tolist())
        for iteration in range(160):
            temperature = 0.25 * (0.01 / 0.25) ** (iteration / 159.0)
            remove_position = int(self.rng.integers(0, k))
            outside = np.asarray([value for value in eligible if int(value) not in current_set])
            if not len(outside):
                break
            replacement = int(outside[int(self.rng.integers(0, len(outside)))])
            candidate = current.copy()
            removed = int(candidate[remove_position])
            candidate[remove_position] = replacement
            candidate_utility = utility(candidate)
            difference = candidate_utility - current_utility
            if difference >= 0.0 or self.rng.random() < np.exp(difference / temperature):
                current = candidate
                current_utility = candidate_utility
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
        retained = [index for index in self.static_ids if mask[index]]
        selected = list(retained)
        if len(selected) < self.config.physics.committee_size:
            order = np.lexsort((np.arange(len(base)), -base))
            for index in order:
                index = int(index)
                if mask[index] and index not in selected:
                    selected.append(index)
                    if len(selected) == self.config.physics.committee_size:
                        break
        return self._ordered_action(selected, base)


SELECTOR_TYPES = (
    RiskPriorSelector,
    FMGSelector,
    CMABSelector,
    SimulatedAnnealingSelector,
    GBASelector,
    OCDInspiredSelector,
    RandomSelector,
    StaticValidRefillSelector,
)


def make_selector(name: str, config: PaperRecoveryConfig, *, seed: int) -> BaselineSelector:
    matching = [selector for selector in SELECTOR_TYPES if selector.name == name]
    if len(matching) != 1:
        raise KeyError(f"unknown baseline selector: {name}")
    return matching[0](config, seed=seed)


def baseline_protocol_manifest(config: PaperRecoveryConfig) -> dict[str, Any]:
    return {
        "schema": "qrelay-paper-recovery-baseline-protocol-1",
        "selectors": [selector.name for selector in SELECTOR_TYPES],
        "observations": list(config.observations.per_node_names)
        + list(config.observations.global_names),
        "hidden_labels": False,
        "same_environment_scenario_and_exogenous_streams": True,
        "coefficients_frozen_before_results": True,
        "post_result_tuning_allowed": False,
        "fmg_formula": {
            "trust_lcb": 1.30,
            "link": 0.90,
            "latency_base": 1.10,
            "latency_load_interaction": 0.40,
            "cpu": 0.55,
            "soc": 0.45,
            "service": -0.35,
            "uncertainty": -0.20,
            "previous_committee": 0.10,
            "fast_response": 0.80,
            "slow_response": 0.40,
            "miss_streak": -0.80,
        },
        "sa_iterations_per_decision": 160,
    }

