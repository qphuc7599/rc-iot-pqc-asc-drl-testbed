"""Pre-registered crossed-seed analysis and grayscale paper-figure generator."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import PaperRecoveryConfig
from .receipts import atomic_write_json, canonical_hash


def _mean_over_environment(values: dict[str, dict[str, float]], metric: str) -> float:
    rows = [float(row[metric]) for row in values.values()]
    if not rows:
        raise ValueError("environment-seed panel is empty")
    return float(np.mean(rows))


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(np.mean(values)))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        exceed += int(statistic >= observed - 1e-15)
        total += 1
    return float(exceed / total)


def mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    """Training-seed t interval; environment seeds are crossed within each seed."""

    values = np.asarray(values, dtype=np.float64)
    if len(values) != 6:
        raise ValueError("primary inference requires exactly six training seeds")
    mean = float(np.mean(values))
    # Registered two-sided 95% t critical value for df=5.
    radius = 2.570581835636314 * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean, mean - radius, mean + radius


def crossed_bootstrap_ci95(
    matrix: np.ndarray,
    *,
    seed: int,
    draws: int = 100_000,
) -> tuple[float, float, float]:
    """Resample training-policy rows and common environment columns."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (6, 20):
        raise ValueError("crossed bootstrap requires a 6x20 paired-effect matrix")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(draws, dtype=np.float64)
    chunk_size = 2_000
    for start in range(0, draws, chunk_size):
        stop = min(start + chunk_size, draws)
        count = stop - start
        rows = rng.integers(0, matrix.shape[0], size=(count, matrix.shape[0]))
        columns = rng.integers(0, matrix.shape[1], size=(count, matrix.shape[1]))
        sampled = matrix[rows[:, :, None], columns[:, None, :]]
        estimates[start:stop] = sampled.mean(axis=(1, 2))
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(np.mean(matrix)), float(low), float(high)


def analyse_primary_results(document: dict[str, Any]) -> dict[str, Any]:
    config = PaperRecoveryConfig()
    policies = document["policies"]
    required = {"PPO", "FMG", "Risk-Prior"}
    if not required.issubset(policies):
        raise ValueError(f"missing primary policies: {sorted(required - set(policies))}")
    training_seeds = [str(value) for value in config.seeds.formal_training]
    effects: dict[str, Any] = {}
    for comparator in ("FMG", "Risk-Prior"):
        tps_differences = []
        failure_differences = []
        energy_relative = []
        security_differences = []
        relative_tps = []
        crossed_rows = []
        ppo_auc_means = []
        comparator_auc_means = []
        for seed in training_seeds:
            if seed not in policies["PPO"]:
                raise ValueError(f"missing PPO training seed {seed}")
            ppo_panel = policies["PPO"][seed]
            baseline_panel = policies[comparator].get(seed) or policies[comparator].get("shared")
            if baseline_panel is None:
                raise ValueError(f"missing {comparator} crossed panel")
            ppo_tps = _mean_over_environment(ppo_panel, "primary_post_event_auc")
            base_tps = _mean_over_environment(baseline_panel, "primary_post_event_auc")
            tps_differences.append(ppo_tps - base_tps)
            relative_tps.append((ppo_tps - base_tps) / max(abs(base_tps), 1e-12))
            ppo_auc_means.append(ppo_tps)
            comparator_auc_means.append(base_tps)
            failure_differences.append(
                _mean_over_environment(ppo_panel, "failure_rate")
                - _mean_over_environment(baseline_panel, "failure_rate")
            )
            ppo_energy = _mean_over_environment(ppo_panel, "mean_energy_drain_wh")
            base_energy = _mean_over_environment(baseline_panel, "mean_energy_drain_wh")
            energy_relative.append((ppo_energy - base_energy) / max(abs(base_energy), 1e-12))
            security_differences.append(
                _mean_over_environment(ppo_panel, "security_budget_exposure_rate")
                - _mean_over_environment(
                    baseline_panel, "security_budget_exposure_rate"
                )
            )
            environment_keys = [str(value) for value in config.seeds.canonical_evaluation]
            crossed_rows.append(
                [
                    float(ppo_panel[key]["primary_post_event_auc"])
                    - float(baseline_panel[key]["primary_post_event_auc"])
                    for key in environment_keys
                ]
            )
        tps_array = np.asarray(tps_differences)
        mean_difference, t_ci_low, t_ci_high = mean_ci95(tps_array)
        _, ci_low, ci_high = crossed_bootstrap_ci95(
            np.asarray(crossed_rows),
            seed=991_001 if comparator == "FMG" else 991_002,
        )
        mean_comparator_auc = float(np.mean(comparator_auc_means))
        row = {
            "training_seed_post_event_auc_differences": tps_differences,
            "mean_post_event_auc_difference": mean_difference,
            "crossed_95_ci": [ci_low, ci_high],
            "crossed_relative_95_ci": [
                ci_low / max(abs(mean_comparator_auc), 1e-12),
                ci_high / max(abs(mean_comparator_auc), 1e-12),
            ],
            "training_seed_t_95_ci": [t_ci_low, t_ci_high],
            "crossed_bootstrap_draws": 100_000,
            "mean_ppo_post_event_auc": float(np.mean(ppo_auc_means)),
            "mean_comparator_post_event_auc": mean_comparator_auc,
            "mean_relative_post_event_auc_effect": float(np.mean(relative_tps)),
            "all_six_effects_positive": bool(np.all(tps_array > 0.0)),
            "exact_sign_flip_p": exact_sign_flip_p(tps_array),
            "mean_failure_rate_difference": float(np.mean(failure_differences)),
            "mean_energy_relative_difference": float(np.mean(energy_relative)),
            "mean_security_exposure_difference": float(np.mean(security_differences)),
        }
        row["superiority_gate"] = {
            "ci_lower_above_zero": bool(ci_low > 0.0),
            "relative_effect_at_least_3_percent": bool(
                row["mean_relative_post_event_auc_effect"] >= 0.03
            ),
            "all_six_positive": row["all_six_effects_positive"],
            "sign_flip_p_at_most_0_03125": bool(row["exact_sign_flip_p"] <= 0.03125),
            "failure_not_worse_by_more_than_1pp": bool(
                row["mean_failure_rate_difference"] <= 0.01
            ),
            "energy_not_worse_by_more_than_5_percent": bool(
                row["mean_energy_relative_difference"] <= 0.05
            ),
            "security_exposure_not_increased": bool(
                row["mean_security_exposure_difference"] <= 0.0
            ),
        }
        row["superiority_gate_pass"] = all(row["superiority_gate"].values())
        effects[comparator] = row
    gate_pass = all(row["superiority_gate_pass"] for row in effects.values())
    wording = "superiority" if gate_pass else "competitive_or_parity_or_negative"
    result = {
        "schema": "qrelay-paper-recovery-primary-analysis-1",
        "synthetic": bool(document.get("synthetic", False)),
        "training_seed_is_primary_unit": True,
        "environment_seeds_are_crossed_factor": True,
        "effects": effects,
        "overall_superiority_gate_pass": gate_pass,
        "permitted_wording": wording,
        "additional_seed_search_allowed": False,
    }
    result["analysis_hash"] = canonical_hash(result)
    return result


def generate_grayscale_figures(
    analysis: dict[str, Any], output_directory: Path, *, label: str
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    comparators = ["FMG", "Risk-Prior"]
    means = [analysis["effects"][name]["mean_relative_post_event_auc_effect"] * 100 for name in comparators]
    fig, axis = plt.subplots(figsize=(5.4, 3.2), constrained_layout=True)
    axis.axhline(3.0, color="0.35", linestyle="--", linewidth=1.0, label="registered 3% gate")
    axis.bar(comparators, means, color=["0.25", "0.65"], edgecolor="black", hatch=["///", "..."])
    axis.set_ylabel("PPO relative post-event AUC effect (%)")
    axis.set_xlabel("Registered comparator")
    axis.set_title(f"{label}: crossed training-seed effects")
    axis.legend(frameon=False)
    axis.grid(axis="y", color="0.85", linewidth=0.6)
    path = output_directory / "primary_effects_grayscale.pdf"
    fig.savefig(path)
    plt.close(fig)
    return [path.as_posix()]


def write_analysis_outputs(
    document: dict[str, Any], output_directory: Path, *, permit_paper_outputs: bool
) -> dict[str, Any]:
    analysis = analyse_primary_results(document)
    if analysis["synthetic"] and permit_paper_outputs:
        raise RuntimeError("synthetic fixture is forbidden from paper output generation")
    atomic_write_json(output_directory / "primary_analysis.json", analysis)
    figures = generate_grayscale_figures(
        analysis,
        output_directory,
        label="SYNTHETIC QA" if analysis["synthetic"] else "Q-Relay",
    )
    result = {
        "status": "PASS",
        "synthetic": analysis["synthetic"],
        "paper_outputs_permitted": bool(permit_paper_outputs and not analysis["synthetic"]),
        "analysis_path": (output_directory / "primary_analysis.json").as_posix(),
        "figure_paths": figures,
    }
    atomic_write_json(output_directory / "generation_receipt.json", result)
    return result


def synthetic_fixture() -> dict[str, Any]:
    config = PaperRecoveryConfig()
    rng = np.random.default_rng(990_001)
    policies: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        "PPO": {},
        "FMG": {"shared": {}},
        "Risk-Prior": {"shared": {}},
    }
    for environment_seed in config.seeds.canonical_evaluation:
        key = str(environment_seed)
        policies["FMG"]["shared"][key] = {
            "primary_post_event_auc": float(500_000 + rng.normal(0, 5_000)),
            "failure_rate": 0.08,
            "mean_energy_drain_wh": 0.010,
            "security_budget_exposure_rate": 0.03,
        }
        policies["Risk-Prior"]["shared"][key] = {
            "primary_post_event_auc": float(490_000 + rng.normal(0, 5_000)),
            "failure_rate": 0.07,
            "mean_energy_drain_wh": 0.0102,
            "security_budget_exposure_rate": 0.025,
        }
    for index, training_seed in enumerate(config.seeds.formal_training):
        panel = {}
        for environment_seed in config.seeds.canonical_evaluation:
            panel[str(environment_seed)] = {
                "primary_post_event_auc": float(530_000 + 2_000 * index + rng.normal(0, 4_000)),
                "failure_rate": 0.075,
                "mean_energy_drain_wh": 0.0103,
                "security_budget_exposure_rate": 0.02,
            }
        policies["PPO"][str(training_seed)] = panel
    return {
        "schema": "qrelay-paper-recovery-evaluation-results-1",
        "synthetic": True,
        "warning": "SYNTHETIC QA ONLY; FORBIDDEN FROM MANUSCRIPT MACROS",
        "policies": policies,
    }
