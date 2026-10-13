"""Checkpoint-independent baseline and final-policy evaluation runners."""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .baselines import SELECTOR_TYPES, baseline_protocol_manifest, make_selector
from .config import PaperRecoveryConfig
from .env import PaperRecoveryEnv
from .policy import ResidualPPOPolicy, TemporalObservationBuffer
from .receipts import (
    RESULT_ROOT,
    TrajectoryHasher,
    atomic_write_json,
    canonical_hash,
    recovery_source_manifest,
    sha256_file,
    utc_now,
)
from .validators import validate_transition


def _selector_seed(name: str, environment_seed: int) -> int:
    digest = hashlib.sha256(f"{name}:{environment_seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _profile_config(base: PaperRecoveryConfig, profile: str) -> PaperRecoveryConfig:
    if profile == "canonical":
        return replace(base, training_mode=False, scenario_profile="canonical")
    if profile not in {
        "ood_capability_kill",
        "ood_clustered_rf",
        "ood_bursty_overload",
        "ood_adaptive_byzantine",
        "ood_nakagami",
    }:
        raise ValueError(f"unsupported evaluation profile {profile}")
    return replace(
        base,
        training_mode=False,
        scenario_profile=profile,
        offered_load_mode=("bursty" if profile == "ood_bursty_overload" else "saturated"),
    )


def _summarise_phase(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "transition_count": 0,
            "mean_tps": None,
            "certificate_success_rate": None,
            "failure_rate": None,
            "mean_energy_drain_wh": None,
            "security_budget_exposure_rate": None,
        }
    success = np.asarray([float(row["certificate_success"]) for row in rows])
    return {
        "transition_count": len(rows),
        "mean_tps": float(np.mean([row["served_tps"] for row in rows])),
        "service_auc_tps_steps": float(np.sum([row["served_tps"] for row in rows])),
        "certificate_success_rate": float(np.mean(success)),
        "failure_rate": float(1.0 - np.mean(success)),
        "mean_energy_drain_wh": float(
            np.mean([row["energy"]["drain_total_wh"] for row in rows])
        ),
        "security_budget_exposure_rate": float(
            np.mean(
                [row["offline_security"]["security_budget_exceeded"] for row in rows]
            )
        ),
        "mean_turnover_ratio": float(
            np.mean([row["committee_turnover_ratio"] for row in rows])
        ),
    }


def evaluate_baseline_seed(
    *,
    selector_name: str,
    config: PaperRecoveryConfig,
    environment_seed: int,
    output_directory: Path,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    selector = make_selector(
        selector_name,
        config,
        seed=_selector_seed(selector_name, environment_seed),
    )
    env = PaperRecoveryEnv(config)
    observation, reset = env.reset(seed=int(environment_seed))
    selector.reset(observation)
    trajectory_hasher = TrajectoryHasher(worker_id=None)
    exogenous_hasher = hashlib.sha256()
    compact_rows: list[dict[str, Any]] = []
    closing_reasons: Counter[str] = Counter()
    committee_hashes: set[str] = set()
    underfilled = 0
    trajectory_path = output_directory / f"seed_{environment_seed}_trajectory.jsonl.gz"
    started = time.monotonic()
    with gzip.open(trajectory_path, "wt", encoding="utf-8", compresslevel=6) as stream:
        for _ in range(config.physics.horizon_steps):
            action_scores = selector.scores(observation)
            observation, reward, terminated, truncated, info = env.step(action_scores)
            validate_transition(info)
            trajectory_hasher.update(info, episode_index=0)
            exogenous_hasher.update(info["exogenous_draw_hash"].encode("ascii"))
            exogenous_hasher.update(b"\n")
            closing_reasons[info["closing_condition"]] += 1
            committee_hashes.add(canonical_hash(info["executed_committee_ids"]))
            underfilled += int(info["committee_size_realized"] < config.physics.committee_size)
            compact_rows.append(info)
            stream.write(json.dumps(info, sort_keys=True, separators=(",", ":")) + "\n")
            if terminated and not truncated:
                raise RuntimeError(
                    f"{selector_name} seed {environment_seed} terminated before the fixed horizon"
                )
            if truncated:
                break
    if len(compact_rows) != config.physics.horizon_steps:
        raise RuntimeError("baseline evaluation did not complete the fixed horizon")
    event = config.physics.evaluation_event_step
    summary = {
        "schema": "qrelay-paper-recovery-baseline-seed-result-1",
        "status": "complete",
        "selector": selector_name,
        "profile": config.scenario_profile,
        "environment_seed": int(environment_seed),
        "selector_seed": _selector_seed(selector_name, environment_seed),
        "scenario_hash": reset["scenario_record"]["scenario_hash"],
        "config_hash": canonical_hash(config.to_dict()),
        "transition_count": len(compact_rows),
        "trajectory_hash": trajectory_hasher.receipt()["trajectory_hash"],
        "trajectory_path": trajectory_path.as_posix(),
        "trajectory_file_sha256": sha256_file(trajectory_path),
        "exogenous_trace_hash": exogenous_hasher.hexdigest(),
        "unique_committee_count": len(committee_hashes),
        "underfilled_action_count": underfilled,
        "closing_reason_counts": dict(closing_reasons),
        "overall": _summarise_phase(compact_rows),
        "pre_event": _summarise_phase(compact_rows[:event]),
        "post_event": _summarise_phase(compact_rows[event:]),
        "elapsed_seconds": float(time.monotonic() - started),
    }
    summary["result_hash"] = canonical_hash(summary)
    atomic_write_json(output_directory / f"seed_{environment_seed}_summary.json", summary)
    return summary


def run_baseline_panel(
    *,
    profile: str = "canonical",
    selector_names: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> dict[str, Any]:
    base = PaperRecoveryConfig()
    config = _profile_config(base, profile)
    names = list(selector_names or [value.name for value in SELECTOR_TYPES])
    if seeds is None:
        seeds = (
            base.seeds.canonical_evaluation
            if profile == "canonical"
            else base.seeds.ood_evaluation
        )
    seed_values = tuple(int(value) for value in seeds)
    expected_seed_set = (
        set(base.seeds.canonical_evaluation)
        if profile == "canonical"
        else set(base.seeds.ood_evaluation)
    )
    if not set(seed_values).issubset(expected_seed_set):
        raise RuntimeError("unregistered evaluation seed requested")
    output_root = RESULT_ROOT / "baselines" / profile
    results: list[dict[str, Any]] = []
    for name in names:
        if name not in {value.name for value in SELECTOR_TYPES}:
            raise KeyError(name)
        selector_directory = output_root / name.replace("/", "_").replace(" ", "_")
        for seed in seed_values:
            results.append(
                evaluate_baseline_seed(
                    selector_name=name,
                    config=config,
                    environment_seed=seed,
                    output_directory=selector_directory,
                )
            )

    fairness_mismatches: list[dict[str, Any]] = []
    for seed in seed_values:
        rows = [row for row in results if row["environment_seed"] == seed]
        if len({row["scenario_hash"] for row in rows}) != 1:
            fairness_mismatches.append({"seed": seed, "field": "scenario_hash"})
        if len({row["exogenous_trace_hash"] for row in rows}) != 1:
            fairness_mismatches.append({"seed": seed, "field": "exogenous_trace_hash"})
        for row in rows:
            if row["transition_count"] != config.physics.horizon_steps:
                fairness_mismatches.append(
                    {"seed": seed, "selector": row["selector"], "field": "transition_count"}
                )
            if row["underfilled_action_count"]:
                fairness_mismatches.append(
                    {"seed": seed, "selector": row["selector"], "field": "underfilled_action"}
                )
    panel = {
        "schema": "qrelay-paper-recovery-baseline-panel-1",
        "status": "PASS" if not fairness_mismatches else "FAIL",
        "profile": profile,
        "selectors": names,
        "environment_seeds": list(seed_values),
        "result_count": len(results),
        "expected_transition_count": len(results) * config.physics.horizon_steps,
        "actual_transition_count": sum(row["transition_count"] for row in results),
        "fairness": {
            "same_scenario_per_seed": True,
            "same_exogenous_trace_per_seed": True,
            "mismatch_count": len(fairness_mismatches),
            "mismatches": fairness_mismatches,
        },
        "protocol": baseline_protocol_manifest(config),
        "source_manifest": recovery_source_manifest(config),
        "results": results,
        "generated_at": utc_now(),
    }
    panel["panel_hash"] = canonical_hash(panel)
    atomic_write_json(output_root / "baseline_panel.json", panel)
    if fairness_mismatches:
        raise RuntimeError(f"baseline fairness audit failed: {fairness_mismatches[:10]}")
    return panel


def _policy_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def load_fixed_final_policy(
    *,
    checkpoint_path: Path,
    evaluation_config: PaperRecoveryConfig,
    device: torch.device,
) -> tuple[ResidualPPOPolicy, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("schema") != "qrelay-paper-recovery-checkpoint-1":
        raise RuntimeError("checkpoint schema mismatch")
    if payload.get("run_kind") != "formal":
        raise RuntimeError("development checkpoint is forbidden from paper evaluation")
    if payload.get("checkpoint_selection") != "fixed_budget_final_only":
        raise RuntimeError("checkpoint-selection rule mismatch")
    if payload.get("resumed") or payload.get("early_stopped") or payload.get("selected_by_evaluation"):
        raise RuntimeError("checkpoint violates the fixed-budget protocol")
    if int(payload.get("updates", -1)) != evaluation_config.budgets.formal_updates:
        raise RuntimeError("formal checkpoint update count mismatch")
    if int(payload.get("transitions", -1)) != evaluation_config.budgets.formal_transitions_per_seed:
        raise RuntimeError("formal checkpoint transition count mismatch")
    training_config = replace(
        PaperRecoveryConfig(), training_mode=True, scenario_profile="training_primary"
    )
    current_source = recovery_source_manifest(training_config)
    if payload["source_manifest"]["files"] != current_source["files"]:
        raise RuntimeError("active source files do not match the formal checkpoint")
    policy = ResidualPPOPolicy(evaluation_config).to(device)
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    policy.eval()
    return policy, payload


def evaluate_policy_seed(
    *,
    policy: ResidualPPOPolicy,
    policy_label: str,
    training_seed: int,
    config: PaperRecoveryConfig,
    environment_seed: int,
    output_directory: Path,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    env = PaperRecoveryEnv(config)
    observation, reset = env.reset(seed=int(environment_seed))
    history = TemporalObservationBuffer(config.observations.temporal_delta_window)
    history.reset(observation)
    trajectory_hasher = TrajectoryHasher(worker_id=None)
    exogenous_hasher = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    trajectory_path = output_directory / f"seed_{environment_seed}_trajectory.jsonl.gz"
    with gzip.open(trajectory_path, "wt", encoding="utf-8", compresslevel=6) as stream:
        for _ in range(config.physics.horizon_steps):
            action = policy.act(
                observation,
                history.array(),
                deterministic=True,
            )
            observation, _, terminated, truncated, info = env.step(action.action_scores)
            if info["candidate_committee_ids"] != action.ordered_committee.astype(int).tolist():
                raise RuntimeError("deterministic policy/environment committee mismatch")
            validate_transition(info)
            trajectory_hasher.update(info, episode_index=0)
            exogenous_hasher.update(info["exogenous_draw_hash"].encode("ascii"))
            exogenous_hasher.update(b"\n")
            rows.append(info)
            stream.write(json.dumps(info, sort_keys=True, separators=(",", ":")) + "\n")
            history.append(observation)
            if terminated and not truncated:
                raise RuntimeError("policy evaluation terminated before the registered horizon")
            if truncated:
                break
    if len(rows) != config.physics.horizon_steps:
        raise RuntimeError("policy evaluation trajectory is incomplete")
    event = config.physics.evaluation_event_step
    checkpoint_hash = sha256_file(checkpoint_path) if checkpoint_path else None
    result = {
        "schema": "qrelay-paper-recovery-policy-seed-result-1",
        "status": "complete",
        "policy": policy_label,
        "training_seed": int(training_seed),
        "environment_seed": int(environment_seed),
        "profile": config.scenario_profile,
        "scenario_hash": reset["scenario_record"]["scenario_hash"],
        "config_hash": canonical_hash(config.to_dict()),
        "source_hash": recovery_source_manifest(config)["source_hash"],
        "checkpoint_path": checkpoint_path.as_posix() if checkpoint_path else None,
        "checkpoint_sha256": checkpoint_hash,
        "transition_count": len(rows),
        "trajectory_hash": trajectory_hasher.receipt()["trajectory_hash"],
        "trajectory_path": trajectory_path.as_posix(),
        "trajectory_file_sha256": sha256_file(trajectory_path),
        "exogenous_trace_hash": exogenous_hasher.hexdigest(),
        "overall": _summarise_phase(rows),
        "pre_event": _summarise_phase(rows[:event]),
        "post_event": _summarise_phase(rows[event:]),
    }
    result["result_hash"] = canonical_hash(result)
    atomic_write_json(output_directory / f"seed_{environment_seed}_summary.json", result)
    return result


def run_policy_panel(
    *,
    training_seed: int,
    profile: str,
    checkpoint_path: Path | None,
    update_zero_control: bool = False,
    device: str = "auto",
    seeds: Iterable[int] | None = None,
) -> dict[str, Any]:
    base = PaperRecoveryConfig()
    if int(training_seed) not in base.seeds.formal_training:
        raise RuntimeError("policy panel requires one of the six registered training seeds")
    config = _profile_config(base, profile)
    environment_seeds = tuple(
        int(value)
        for value in (
            seeds
            or (
                base.seeds.canonical_evaluation
                if profile == "canonical"
                else base.seeds.ood_evaluation
            )
        )
    )
    execution_device = _policy_device(device)
    torch.manual_seed(int(training_seed))
    if update_zero_control:
        if checkpoint_path is not None:
            raise RuntimeError("update-0 control must not load a checkpoint")
        policy = ResidualPPOPolicy(config).to(execution_device)
        policy.eval()
        policy_label = "PPO-update-0-FMG-control"
        checkpoint_payload = None
    else:
        if checkpoint_path is None or checkpoint_path.name != "ppo_final.pt":
            raise RuntimeError("formal policy evaluation accepts ppo_final.pt only")
        policy, checkpoint_payload = load_fixed_final_policy(
            checkpoint_path=checkpoint_path,
            evaluation_config=config,
            device=execution_device,
        )
        if int(checkpoint_payload["seed"]) != int(training_seed):
            raise RuntimeError("checkpoint training seed mismatch")
        policy_label = "PPO-final"
    output_root = (
        RESULT_ROOT
        / "policy_evaluation"
        / profile
        / f"training_seed_{training_seed}"
        / ("update_0" if update_zero_control else "final")
    )
    results = [
        evaluate_policy_seed(
            policy=policy,
            policy_label=policy_label,
            training_seed=training_seed,
            config=config,
            environment_seed=seed,
            output_directory=output_root,
            checkpoint_path=checkpoint_path,
        )
        for seed in environment_seeds
    ]
    panel = {
        "schema": "qrelay-paper-recovery-policy-evaluation-panel-1",
        "status": "PASS",
        "policy": policy_label,
        "training_seed": int(training_seed),
        "profile": profile,
        "environment_seeds": list(environment_seeds),
        "update_zero_control": bool(update_zero_control),
        "checkpoint_sha256": (
            sha256_file(checkpoint_path) if checkpoint_path is not None else None
        ),
        "result_count": len(results),
        "transition_count": sum(row["transition_count"] for row in results),
        "results": results,
    }
    panel["panel_hash"] = canonical_hash(panel)
    atomic_write_json(output_root / "policy_panel.json", panel)
    return panel


def sensitivity_configurations(base: PaperRecoveryConfig | None = None) -> list[dict[str, Any]]:
    """Return all pre-registered representative-seed sensitivity contracts."""

    base = base or PaperRecoveryConfig()
    rows: list[dict[str, Any]] = []
    for num_nodes in (75, 100, 150, 200):
        kill_count = int(round(0.30 * num_nodes))
        byzantine_count = int(round(0.20 * num_nodes))
        survivors = num_nodes - kill_count
        surviving_byzantine = int(round(0.20 * survivors))
        killed_byzantine = byzantine_count - surviving_byzantine
        physics = replace(
            base.physics,
            num_nodes=num_nodes,
            permanent_kill_count=kill_count,
            byzantine_count=byzantine_count,
            killed_byzantine_count=killed_byzantine,
            surviving_byzantine_count=surviving_byzantine,
            ood_rf_blackout_count=min(int(round(0.25 * num_nodes)), survivors),
        )
        rows.append(
            {
                "panel": "N",
                "value": num_nodes,
                "config": replace(base, physics=physics, sensitivity_mode=True).to_dict(),
            }
        )
    for committee_size in (15, 21, 27):
        fault_budget = (committee_size - 1) // 3
        quorum = int(np.ceil((committee_size + fault_budget + 1) / 2.0))
        physics = replace(
            base.physics,
            committee_size=committee_size,
            fault_budget=fault_budget,
            quorum=quorum,
        )
        rows.append(
            {
                "panel": "k",
                "value": committee_size,
                "safe_quorum": quorum,
                "config": replace(base, physics=physics, sensitivity_mode=True).to_dict(),
            }
        )
    for kill_fraction in (0.10, 0.20, 0.30, 0.40):
        for byzantine_fraction in (0.05, 0.10, 0.20, 0.30):
            kill_count = int(round(kill_fraction * base.physics.num_nodes))
            byzantine_count = int(round(byzantine_fraction * base.physics.num_nodes))
            killed_byzantine = int(round(byzantine_fraction * kill_count))
            surviving_byzantine = byzantine_count - killed_byzantine
            physics = replace(
                base.physics,
                permanent_kill_count=kill_count,
                byzantine_count=byzantine_count,
                killed_byzantine_count=killed_byzantine,
                surviving_byzantine_count=surviving_byzantine,
                ood_rf_blackout_count=min(
                    base.physics.ood_rf_blackout_count,
                    base.physics.num_nodes - kill_count,
                ),
            )
            rows.append(
                {
                    "panel": "kill_byzantine_grid",
                    "kill_fraction": kill_fraction,
                    "byzantine_fraction": byzantine_fraction,
                    "config": replace(base, physics=physics, sensitivity_mode=True).to_dict(),
                }
            )
    return rows


def _analysis_metric_row(result: dict[str, Any]) -> dict[str, float]:
    post = result["post_event"]
    auc = post.get("service_auc_tps_steps")
    if auc is None:
        auc = float(post["mean_tps"]) * int(post["transition_count"])
    return {
        "primary_post_event_auc": float(auc),
        "failure_rate": float(post["failure_rate"]),
        "mean_energy_drain_wh": float(post["mean_energy_drain_wh"]),
        "security_budget_exposure_rate": float(post["security_budget_exposure_rate"]),
    }


def assemble_primary_evaluation_results(
    *,
    baseline_panel_path: Path,
    policy_panel_paths: Iterable[Path],
    update_zero_panel_paths: Iterable[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Bind frozen baselines, six final policies and update-0 controls."""

    config = PaperRecoveryConfig()
    baseline_panel = json.loads(baseline_panel_path.read_text(encoding="utf-8"))
    baseline_hash = baseline_panel.pop("panel_hash")
    if canonical_hash(baseline_panel) != baseline_hash or baseline_panel["status"] != "PASS":
        raise RuntimeError("baseline panel integrity failure")
    policies: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        "PPO": {},
        "FMG": {"shared": {}},
        "Risk-Prior": {"shared": {}},
    }
    baseline_trajectory_hashes: dict[str, dict[int, str]] = {
        "FMG": {},
        "Risk-Prior": {},
    }
    for result in baseline_panel["results"]:
        if result["selector"] in baseline_trajectory_hashes:
            seed = int(result["environment_seed"])
            policies[result["selector"]]["shared"][str(seed)] = _analysis_metric_row(result)
            baseline_trajectory_hashes[result["selector"]][seed] = result["trajectory_hash"]

    final_panels = []
    for path in policy_panel_paths:
        panel = json.loads(Path(path).read_text(encoding="utf-8"))
        panel_hash = panel.pop("panel_hash")
        if canonical_hash(panel) != panel_hash or panel["policy"] != "PPO-final":
            raise RuntimeError(f"invalid final policy panel {path}")
        seed = int(panel["training_seed"])
        policies["PPO"][str(seed)] = {
            str(result["environment_seed"]): _analysis_metric_row(result)
            for result in panel["results"]
        }
        final_panels.append({"path": Path(path).as_posix(), "panel_hash": panel_hash})
    if set(map(int, policies["PPO"])) != set(config.seeds.formal_training):
        raise RuntimeError("exactly six registered formal policy panels are required")

    update_zero_checks = []
    for path in update_zero_panel_paths:
        panel = json.loads(Path(path).read_text(encoding="utf-8"))
        panel_hash = panel.pop("panel_hash")
        if canonical_hash(panel) != panel_hash or not panel["update_zero_control"]:
            raise RuntimeError(f"invalid update-0 panel {path}")
        mismatches = []
        for result in panel["results"]:
            environment_seed = int(result["environment_seed"])
            if result["trajectory_hash"] != baseline_trajectory_hashes["FMG"][environment_seed]:
                mismatches.append(environment_seed)
        if mismatches:
            raise RuntimeError(
                f"update-0 control does not exactly reproduce FMG for seeds {mismatches}"
            )
        update_zero_checks.append(
            {
                "training_seed": int(panel["training_seed"]),
                "panel_hash": panel_hash,
                "exact_fmg_trajectory_match": True,
            }
        )
    if {row["training_seed"] for row in update_zero_checks} != set(
        config.seeds.formal_training
    ):
        raise RuntimeError("six update-0 control panels are required")
    document = {
        "schema": "qrelay-paper-recovery-evaluation-results-1",
        "synthetic": False,
        "baseline_panel": {
            "path": baseline_panel_path.as_posix(),
            "panel_hash": baseline_hash,
        },
        "final_policy_panels": final_panels,
        "update_zero_controls": update_zero_checks,
        "policies": policies,
    }
    document["results_hash"] = canonical_hash(document)
    atomic_write_json(output_path, document)
    return document
