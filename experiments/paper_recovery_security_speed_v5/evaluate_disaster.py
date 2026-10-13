"""Held-out disaster evaluation for security-speed v5.

The prospectively frozen panel evaluates the fixed final checkpoint in its
deterministic primary deployment and in a three-replica stochastic audit.  All
policy deployments and eight public-information baselines share identical
exogenous environment streams.  Hidden Byzantine identities are logged only
for the offline f=6 audit and never enter an action, reward, mask, or critic.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drl.paper_recovery.policy import TemporalObservationBuffer
from drl.paper_recovery.receipts import (
    atomic_write_json,
    canonical_hash,
    sha256_file,
    utc_now,
)
from drl.paper_recovery_hardcore_v5.baselines import SELECTORS, make_selector
from drl.paper_recovery_hardcore_v5.env import HardcoreCascadeEnv
from drl.paper_recovery_security_speed_v5.trainer import speed_source_manifest
from drl.paper_recovery_security_v4.config import security_v4_config
from drl.paper_recovery_security_v4.policy import SecurityConstrainedPPO
from drl.paper_recovery_security_v4.trainer import public_security_tail_cost


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path(__file__).with_name("disaster_protocol.json")
OUTPUT_ROOT = (
    REPO_ROOT
    / "results/paper_recovery_security_speed_v5/heldout_disaster_seed_657111"
)
OUTPUT_PATH = OUTPUT_ROOT / "evaluation.json"
VALIDATION_PATH = OUTPUT_ROOT / "validation.json"
SUMMARY_PATH = OUTPUT_ROOT / "baseline_comparison.md"
PPO_NAME = "PPO-security-speed-v5"
METHODS = tuple(SELECTORS)
STRONG = ("FMG", "GBA", "SA", "OCD-inspired", "Risk-Prior", "CMAB")
EXOGENOUS_STREAMS = (
    "rng_scenario",
    "rng_compute",
    "rng_link",
    "rng_backoff",
    "rng_churn",
    "rng_load",
    "rng_energy",
    "rng_observation",
    "rng_thermal",
)
MEAN_FIELDS = (
    "mean_tps_pre_0_999",
    "mean_tps_impact_1000_1199",
    "mean_tps_damaged_1200_1499",
    "mean_tps_late_1500_1999",
    "post_event_tps_auc",
    "failure_rate_post_1000_1999",
    "zero_tps_steps_pre_event",
    "zero_tps_steps_post_event",
    "forced_zero_tps_steps",
    "mean_network_reserve_post_event",
    "maximum_backlog_transactions",
    "final_backlog_transactions",
    "mean_active_thermal_nodes_post_event",
    "mean_cell_debt_post_event",
    "expired_transactions_post_event",
    "mean_public_security_tail_post_event",
    "p95_public_security_tail_post_event",
    "maximum_public_security_tail_post_event",
    "mean_actual_security_exposure_post_event",
    "p95_actual_security_exposure_post_event",
    "maximum_security_budget_exposure",
    "actual_security_guard_violation_fraction_post_event",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _exogenous_rng_hash(env: HardcoreCascadeEnv) -> str:
    return canonical_hash(
        {
            "step_before_transition": int(env.step_count),
            "streams": {
                name: getattr(env, name).bit_generator.state
                for name in EXOGENOUS_STREAMS
            },
            "kill_ids": env.kill_ids.tolist(),
            "affected_cells": env.affected_cells.tolist(),
            "aftershock_cells": env.aftershock_cells.tolist(),
        }
    )


def _evaluate_one(
    method: str,
    seed: int,
    config: Any,
    policy: SecurityConstrainedPPO | None,
    method_index: int,
    *,
    stochastic_action_seed: int | None = None,
) -> tuple[dict[str, Any], list[float], str]:
    env = HardcoreCascadeEnv(config)
    observation, reset = env.reset(seed=seed)
    history = TemporalObservationBuffer(config.observations.temporal_delta_window)
    history.reset(observation)
    selector = None
    generator = None
    if method != PPO_NAME:
        selector = make_selector(method, config, seed=seed * 10 + method_index)
        selector.reset(observation)
    elif policy is None:
        raise RuntimeError("PPO evaluation requires a checkpoint")
    elif stochastic_action_seed is not None:
        device = next(policy.parameters()).device
        generator = torch.Generator(device=device)
        generator.manual_seed(int(stochastic_action_seed))

    tps: list[float] = []
    failures: list[float] = []
    backlog: list[float] = []
    reserve: list[float] = []
    exposure: list[int] = []
    public_tail: list[float] = []
    forced: list[bool] = []
    thermal: list[int] = []
    cell_debt: list[float] = []
    expired: list[float] = []
    exogenous: list[str] = []
    closing = Counter()
    for _ in range(config.physics.horizon_steps):
        exogenous.append(_exogenous_rng_hash(env))
        before = observation
        expected: list[int] | None = None
        if method == PPO_NAME:
            assert policy is not None
            action = policy.act(
                observation,
                history.array(),
                deterministic=stochastic_action_seed is None,
                generator=generator,
            )
            scores = action.action_scores
            expected = action.ordered_committee.astype(int).tolist()
        else:
            assert selector is not None
            scores = selector.scores(observation)
        observation, _, terminated, truncated, info = env.step(scores)
        committee = [int(value) for value in info["candidate_committee_ids"]]
        if expected is not None and committee != expected:
            raise RuntimeError("PPO sampled/executed committee drift")
        if not info["candidate_executed_match"]:
            raise RuntimeError(f"{method} candidate/executed parity failure")
        if info["offline_security"]["hidden_labels_used_by_observation_or_reward"]:
            raise RuntimeError("hidden-label contamination")
        public_tail.append(
            float(
                public_security_tail_cost(
                    np.asarray(before)[None, :],
                    np.asarray(committee, dtype=np.int64)[None, :],
                    config,
                )[0]
            )
        )
        tps.append(float(info["served_tps"]))
        failures.append(float(not info["certificate_success"]))
        backlog.append(float(info["long_horizon"]["backlog_transactions"]))
        reserve.append(float(info["long_horizon"]["network_reserve"]))
        exposure.append(
            int(info["offline_security"]["total_compromised_in_committee"])
        )
        forced.append(bool(info["long_horizon"]["forced_zero_tps"]))
        thermal.append(
            int(info["long_horizon"]["active_thermal_unavailable_nodes"])
        )
        cell_debt.append(float(np.mean(info["long_horizon"]["cell_debt"])))
        expired.append(
            float(info["long_horizon"]["expired_transactions_this_step"])
        )
        closing[str(info["closing_condition"])] += 1
        history.append(observation)
        if terminated:
            raise RuntimeError(f"{method} terminated before 2,000 steps")
        if truncated:
            break

    event = int(config.physics.evaluation_event_step)
    if len(tps) != config.physics.horizon_steps:
        raise RuntimeError(f"{method} produced only {len(tps)} transitions")

    def mean(values: list[float] | list[int], start: int, end: int) -> float:
        return float(np.mean(values[start:end]))

    post_public = np.asarray(public_tail[event:], dtype=np.float64)
    post_exposure = np.asarray(exposure[event:], dtype=np.float64)
    summary = {
        "scenario_hash": reset["scenario_record"]["scenario_hash"],
        "transition_count": len(tps),
        "mean_tps_pre_0_999": mean(tps, 0, event),
        "mean_tps_impact_1000_1199": mean(tps, event, event + 200),
        "mean_tps_damaged_1200_1499": mean(tps, event + 200, event + 500),
        "mean_tps_late_1500_1999": mean(tps, event + 500, len(tps)),
        "post_event_tps_auc": float(np.sum(tps[event:])),
        "failure_rate_post_1000_1999": mean(failures, event, len(failures)),
        "zero_tps_steps_pre_event": int(np.sum(np.asarray(tps[:event]) == 0.0)),
        "zero_tps_steps_post_event": int(np.sum(np.asarray(tps[event:]) == 0.0)),
        "forced_zero_tps_steps": int(np.sum(forced)),
        "mean_network_reserve_post_event": mean(reserve, event, len(reserve)),
        "maximum_backlog_transactions": float(max(backlog)),
        "final_backlog_transactions": float(backlog[-1]),
        "mean_active_thermal_nodes_post_event": mean(thermal, event, len(thermal)),
        "mean_cell_debt_post_event": mean(cell_debt, event, len(cell_debt)),
        "expired_transactions_post_event": float(np.sum(expired[event:])),
        "mean_public_security_tail_post_event": float(np.mean(post_public)),
        "p95_public_security_tail_post_event": float(np.percentile(post_public, 95)),
        "maximum_public_security_tail_post_event": float(np.max(post_public)),
        "mean_actual_security_exposure_post_event": float(np.mean(post_exposure)),
        "p95_actual_security_exposure_post_event": float(
            np.percentile(post_exposure, 95)
        ),
        "maximum_security_budget_exposure": int(np.max(post_exposure)),
        "actual_security_guard_violation_fraction_post_event": float(
            np.mean(post_exposure > config.physics.fault_budget)
        ),
        "closing_condition_counts": dict(sorted(closing.items())),
    }
    return summary, tps, canonical_hash(exogenous)


def _mean_fields(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        field: float(np.mean([float(row[field]) for row in rows]))
        for field in MEAN_FIELDS
    }


def _guard(rows: list[dict[str, Any]], fault_budget: int) -> dict[str, Any]:
    global_max = max(int(row["maximum_security_budget_exposure"]) for row in rows)
    violation = float(
        np.mean(
            [row["actual_security_guard_violation_fraction_post_event"] for row in rows]
        )
    )
    return {
        "fault_budget": int(fault_budget),
        "global_max_actual_security_exposure": global_max,
        "post_event_violation_fraction": violation,
        "strict_guard_pass": global_max <= fault_budget and violation == 0.0,
    }


def _paired_interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    critical = 2.093024054
    half = critical * float(np.std(array, ddof=1)) / np.sqrt(len(array))
    return {"mean": mean, "ci95_lower": mean - half, "ci95_upper": mean + half}


def _aggregate_mode(
    by_seed: dict[str, dict[str, Any]], timeline: np.ndarray, seeds: tuple[int, ...]
) -> dict[str, Any]:
    rows = list(by_seed.values())
    return {
        "aggregate": _mean_fields(rows),
        "security_guard": _guard(rows, 6),
        "per_seed": by_seed,
        "mean_tps_timeline": (timeline / len(seeds)).tolist(),
    }


def _compare(
    ppo_by_seed: dict[str, dict[str, Any]],
    ppo_aggregate: dict[str, float],
    baseline_by_method: dict[str, dict[str, dict[str, Any]]],
    baseline_aggregates: dict[str, dict[str, float]],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for method in METHODS:
        late = [
            ppo_by_seed[str(seed)]["mean_tps_late_1500_1999"]
            - baseline_by_method[method][str(seed)]["mean_tps_late_1500_1999"]
            for seed in seeds
        ]
        damaged = [
            ppo_by_seed[str(seed)]["mean_tps_damaged_1200_1499"]
            - baseline_by_method[method][str(seed)]["mean_tps_damaged_1200_1499"]
            for seed in seeds
        ]
        baseline = baseline_aggregates[method]
        comparisons[method] = {
            "late_tps_difference": _paired_interval(late),
            "damaged_tps_difference": _paired_interval(damaged),
            "late_tps_ratio": ppo_aggregate["mean_tps_late_1500_1999"]
            / max(baseline["mean_tps_late_1500_1999"], 1e-12),
            "damaged_tps_ratio": ppo_aggregate["mean_tps_damaged_1200_1499"]
            / max(baseline["mean_tps_damaged_1200_1499"], 1e-12),
            "post_failure_rate_difference": ppo_aggregate[
                "failure_rate_post_1000_1999"
            ]
            - baseline["failure_rate_post_1000_1999"],
        }
    return comparisons


def _load_policy(protocol: dict[str, Any]) -> tuple[SecurityConstrainedPPO, dict[str, Any]]:
    checkpoint_path = REPO_ROOT / protocol["checkpoint"]["path"]
    receipt = _load_json(checkpoint_path.parent / "completion_receipt.json")
    progress = _load_json(checkpoint_path.parent / "progress.json")
    if receipt.get("status") != "complete":
        raise RuntimeError("formal completion receipt is incomplete")
    if receipt.get("updates_completed") != 3_000 or receipt.get(
        "transitions_completed"
    ) != 9_000_000:
        raise RuntimeError("formal completion budget mismatch")
    if progress.get("updates_completed") != 3_000 or progress.get(
        "transitions_completed"
    ) != 9_000_000:
        raise RuntimeError("formal progress budget mismatch")
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != protocol["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint hash drift")
    if checkpoint_hash != receipt.get("checkpoint_sha256"):
        raise RuntimeError("checkpoint/receipt hash mismatch")
    training_config = security_v4_config(training=True)
    source = speed_source_manifest(training_config)
    if source["source_hash"] != protocol["training_source_hash"]:
        raise RuntimeError("training source closure drift")
    if receipt.get("source_hash") != protocol["training_source_hash"]:
        raise RuntimeError("training receipt source mismatch")
    if receipt.get("config_hash") != protocol["config_hash"]:
        raise RuntimeError("training config hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("updates") != 3_000 or checkpoint.get("transitions") != 9_000_000:
        raise RuntimeError("checkpoint budget fields are invalid")
    if checkpoint.get("selected_by_evaluation") is not False:
        raise RuntimeError("checkpoint was selected by evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = SecurityConstrainedPPO(training_config).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    policy.eval()
    return policy, receipt


def _evaluate(protocol: dict[str, Any], policy: SecurityConstrainedPPO) -> dict[str, Any]:
    seeds = tuple(int(value) for value in protocol["evaluation_seeds"])
    replicas = int(protocol["deployment"]["stochastic_replicas_per_environment_seed"])
    config = security_v4_config(training=False)
    if config.physics.evaluation_event_step != 1_000:
        raise RuntimeError("evaluation event-step drift")
    if int(round(config.physics.num_nodes * 0.30)) != 30:
        raise RuntimeError("permanent kill-count drift")

    primary_rows: dict[str, dict[str, Any]] = {}
    primary_timeline = np.zeros(config.physics.horizon_steps, dtype=np.float64)
    stochastic_replicas: dict[str, dict[str, dict[str, Any]]] = {}
    stochastic_seed_means: dict[str, dict[str, float]] = {}
    stochastic_timeline = np.zeros(config.physics.horizon_steps, dtype=np.float64)
    baseline_rows = {method: {} for method in METHODS}
    baseline_timelines = {
        method: np.zeros(config.physics.horizon_steps, dtype=np.float64)
        for method in METHODS
    }
    action_seeds: dict[str, list[int]] = {}
    exogenous_hashes: dict[str, str] = {}

    for seed in seeds:
        hashes: list[str] = []
        primary, timeline, trace_hash = _evaluate_one(
            PPO_NAME, seed, config, policy, 0
        )
        primary_rows[str(seed)] = primary
        primary_timeline += np.asarray(timeline, dtype=np.float64)
        hashes.append(trace_hash)

        replica_rows: dict[str, dict[str, Any]] = {}
        action_seeds[str(seed)] = []
        for replica in range(replicas):
            action_seed = 9_200_000 + 10 * seed + replica
            action_seeds[str(seed)].append(action_seed)
            summary, timeline, trace_hash = _evaluate_one(
                PPO_NAME,
                seed,
                config,
                policy,
                0,
                stochastic_action_seed=action_seed,
            )
            replica_rows[str(replica)] = {"action_seed": action_seed, **summary}
            stochastic_timeline += np.asarray(timeline, dtype=np.float64)
            hashes.append(trace_hash)
        stochastic_replicas[str(seed)] = replica_rows
        stochastic_seed_means[str(seed)] = _mean_fields(list(replica_rows.values()))

        for index, method in enumerate(METHODS):
            summary, timeline, trace_hash = _evaluate_one(
                method, seed, config, None, index
            )
            baseline_rows[method][str(seed)] = summary
            baseline_timelines[method] += np.asarray(timeline, dtype=np.float64)
            hashes.append(trace_hash)
        if len(set(hashes)) != 1:
            raise RuntimeError(f"exogenous pairing failed for seed {seed}")
        exogenous_hashes[str(seed)] = hashes[0]

    primary = _aggregate_mode(primary_rows, primary_timeline, seeds)
    stochastic_rows = [
        row for per_seed in stochastic_replicas.values() for row in per_seed.values()
    ]
    stochastic = {
        "aggregate": _mean_fields(list(stochastic_seed_means.values())),
        "security_guard": _guard(stochastic_rows, config.physics.fault_budget),
        "per_seed_means": stochastic_seed_means,
        "replicas": stochastic_replicas,
        "action_seeds": action_seeds,
        "mean_tps_timeline": (
            stochastic_timeline / (len(seeds) * replicas)
        ).tolist(),
    }
    baseline_aggregates = {
        method: _mean_fields(list(rows.values()))
        for method, rows in baseline_rows.items()
    }
    baseline_guards = {
        method: _guard(list(rows.values()), config.physics.fault_budget)
        for method, rows in baseline_rows.items()
    }
    baselines = {
        "aggregates": baseline_aggregates,
        "security_guards": baseline_guards,
        "per_seed": baseline_rows,
        "mean_tps_timeline": {
            method: (timeline / len(seeds)).tolist()
            for method, timeline in baseline_timelines.items()
        },
    }
    return {
        "seeds": list(seeds),
        "paired_exogenous_hashes": exogenous_hashes,
        "deterministic_primary": primary,
        "stochastic_audit": stochastic,
        "baselines": baselines,
        "comparisons": {
            "deterministic_primary": _compare(
                primary_rows,
                primary["aggregate"],
                baseline_rows,
                baseline_aggregates,
                seeds,
            ),
            "stochastic_audit": _compare(
                stochastic_seed_means,
                stochastic["aggregate"],
                baseline_rows,
                baseline_aggregates,
                seeds,
            ),
        },
    }


def _validate(panel: dict[str, Any]) -> dict[str, Any]:
    primary_rows = list(panel["deterministic_primary"]["per_seed"].values())
    stochastic_rows = [
        row
        for replicas in panel["stochastic_audit"]["replicas"].values()
        for row in replicas.values()
    ]
    baseline_rows = [
        row for rows in panel["baselines"]["per_seed"].values() for row in rows.values()
    ]
    all_rows = primary_rows + stochastic_rows + baseline_rows
    timelines = [
        panel["deterministic_primary"]["mean_tps_timeline"],
        panel["stochastic_audit"]["mean_tps_timeline"],
        *panel["baselines"]["mean_tps_timeline"].values(),
    ]
    checks = {
        "twenty_registered_environment_seeds": len(panel["seeds"]) == 20,
        "one_deterministic_primary_per_seed": len(primary_rows) == 20,
        "three_stochastic_replicas_per_seed": len(stochastic_rows) == 60,
        "all_eight_baselines_present": set(panel["baselines"]["aggregates"])
        == set(METHODS),
        "all_trajectories_exactly_2000_steps": all(
            int(row["transition_count"]) == 2_000 for row in all_rows
        ),
        "all_exogenous_streams_paired": len(panel["paired_exogenous_hashes"]) == 20,
        "all_mean_timelines_exactly_2000_steps": all(
            len(timeline) == 2_000 for timeline in timelines
        ),
        "all_scalar_metrics_finite": all(
            np.isfinite(float(row[field])) for row in all_rows for field in MEAN_FIELDS
        ),
        "all_timeline_values_finite": all(
            np.all(np.isfinite(np.asarray(timeline))) for timeline in timelines
        ),
        "no_forced_zero_tps_injected": all(
            float(row["forced_zero_tps_steps"]) == 0.0 for row in all_rows
        ),
    }
    validation = {
        "schema": "qrelay-security-speed-v5-disaster-validation-1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "paper_evidence_eligible": False,
        "single_training_seed_diagnostic": True,
        "checks": checks,
    }
    validation["receipt_hash"] = canonical_hash(validation)
    return validation


def _source_closure() -> dict[str, Any]:
    paths = (
        REPO_ROOT / "drl/paper_recovery_hardcore_v5/config.py",
        REPO_ROOT / "drl/paper_recovery_hardcore_v5/env.py",
        REPO_ROOT / "drl/paper_recovery_hardcore_v5/baselines.py",
        REPO_ROOT / "drl/paper_recovery_security_v4/config.py",
        REPO_ROOT / "drl/paper_recovery_security_v4/policy.py",
        REPO_ROOT / "drl/paper_recovery_security_v4/trainer.py",
        PROTOCOL_PATH,
        Path(__file__).resolve(),
    )
    files = {path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in paths}
    return {"files": files, "source_hash": canonical_hash(files)}


def _write_summary(panel: dict[str, Any]) -> None:
    primary = panel["deterministic_primary"]["aggregate"]
    stochastic = panel["stochastic_audit"]["aggregate"]
    baselines = panel["baselines"]["aggregates"]
    best = max(baselines, key=lambda name: baselines[name]["mean_tps_late_1500_1999"])
    lines = [
        "# Security-speed v5 held-out disaster comparison",
        "",
        "Single training seed; held-out diagnostic only, not confirmatory paper evidence.",
        "",
        "| Method | Late TPS | Damaged TPS | Post failure | Mean actual exposure | Max exposure | Guard > f rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = {
        "PPO deterministic primary": primary,
        "PPO stochastic audit": stochastic,
        **baselines,
    }
    guards = {
        "PPO deterministic primary": panel["deterministic_primary"]["security_guard"],
        "PPO stochastic audit": panel["stochastic_audit"]["security_guard"],
        **panel["baselines"]["security_guards"],
    }
    for name, row in rows.items():
        guard = guards[name]
        lines.append(
            f"| {name} | {row['mean_tps_late_1500_1999']:.2f} | "
            f"{row['mean_tps_damaged_1200_1499']:.2f} | "
            f"{100*row['failure_rate_post_1000_1999']:.2f}% | "
            f"{row['mean_actual_security_exposure_post_event']:.3f} | "
            f"{guard['global_max_actual_security_exposure']} | "
            f"{100*guard['post_event_violation_fraction']:.4f}% |"
        )
    lines.extend(
        [
            "",
            f"Best late-window baseline: **{best}** "
            f"({baselines[best]['mean_tps_late_1500_1999']:.2f} TPS).",
            "",
            "Strict security guard means no held-out post-event step selected more than "
            "f=6 compromised committee members. Hidden identities were used only for this "
            "offline audit.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if OUTPUT_PATH.exists() or VALIDATION_PATH.exists():
        raise RuntimeError("held-out disaster output exists; rerun forbidden")
    protocol = _load_json(PROTOCOL_PATH)
    expected = (
        "frozen_after_training_completion_before_any_6572xx_evaluation_seed_was_opened"
    )
    if protocol.get("status") != expected:
        raise RuntimeError("held-out disaster protocol is not frozen")
    policy, receipt = _load_policy(protocol)
    panel = _evaluate(protocol, policy)
    validation = _validate(panel)
    if validation["status"] != "PASS":
        raise RuntimeError(f"validation failed: {validation['checks']}")
    source = _source_closure()
    payload = {
        "schema": "qrelay-security-speed-v5-heldout-disaster-evaluation-1",
        "role": protocol["role"],
        "paper_evidence_eligible": False,
        "created_at": utc_now(),
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "training_source_hash": receipt["source_hash"],
        "evaluation_source_hash": source["source_hash"],
        "evaluation_source_files": source["files"],
        "panel": panel,
    }
    payload["receipt_hash"] = canonical_hash(payload)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    atomic_write_json(OUTPUT_PATH, payload)
    atomic_write_json(VALIDATION_PATH, validation)
    _write_summary(panel)
    best = max(
        STRONG,
        key=lambda name: panel["baselines"]["aggregates"][name][
            "mean_tps_late_1500_1999"
        ],
    )
    print(
        json.dumps(
            {
                "status": validation["status"],
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "deterministic_primary": panel["deterministic_primary"]["aggregate"],
                "deterministic_security_guard": panel["deterministic_primary"][
                    "security_guard"
                ],
                "stochastic_audit": panel["stochastic_audit"]["aggregate"],
                "stochastic_security_guard": panel["stochastic_audit"][
                    "security_guard"
                ],
                "best_strong_late_baseline": best,
                "best_strong_late_aggregate": panel["baselines"]["aggregates"][best],
                "output": OUTPUT_PATH.relative_to(REPO_ROOT).as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
