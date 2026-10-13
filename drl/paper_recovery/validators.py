"""Fail-closed validators written before any paper-recovery result exists."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import PaperRecoveryConfig
from .receipts import REPO_ROOT, canonical_hash, recovery_source_manifest, sha256_file


class ValidationFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFailure(message)


def validate_seed_separation(config: PaperRecoveryConfig) -> dict[str, Any]:
    groups = {
        "development": set(config.seeds.development),
        "formal_training": set(config.seeds.formal_training),
        "formal_replacements": set(config.seeds.formal_replacements),
        "canonical_evaluation": set(config.seeds.canonical_evaluation),
        "ood_evaluation": set(config.seeds.ood_evaluation),
    }
    names = list(groups)
    overlaps: dict[str, list[int]] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            common = sorted(groups[left] & groups[right])
            if common:
                overlaps[f"{left}__{right}"] = common
    _require(not overlaps, f"seed partitions overlap: {overlaps}")
    return {
        "status": "PASS",
        "groups": {name: sorted(values) for name, values in groups.items()},
        "overlaps": overlaps,
    }


def validate_transition(info: Mapping[str, Any], *, tolerance: float = 1e-9) -> None:
    _require(
        info.get("schema") == "qrelay-paper-recovery-transition-1",
        "transition schema drift",
    )
    _require(bool(info["candidate_executed_match"]), "candidate/executed mismatch")
    _require(
        info["candidate_committee_ids"] == info["executed_committee_ids"],
        "execution shield or action rewrite detected",
    )
    reward_sum = sum(float(value) for value in info["reward_components"].values())
    _require(
        math.isclose(reward_sum, float(info["reward_total"]), abs_tol=tolerance),
        "reward decomposition does not recompute",
    )
    energy = info["energy"]
    energy_error = (
        float(energy["pre_total_wh"])
        - float(energy["drain_total_wh"])
        - float(energy["post_total_wh"])
    )
    _require(
        math.isclose(energy_error, float(energy["mass_balance_error_wh"]), abs_tol=tolerance),
        "reported energy mass balance is inconsistent",
    )
    _require(abs(energy_error) <= tolerance, f"energy mass balance error {energy_error}")
    if not bool(info["certificate_success"]):
        _require(float(info["served_tps"]) == 0.0, "positive TPS on failed close")
        _require(
            info["closing_condition"]
            in {
                "fail_closed_fewer_than_k_observable_eligible",
                "fewer_than_q_valid_responses_before_deadline",
                "qth_close_latency_exceeded_deadline",
            },
            "failed close lacks a registered reason",
        )
    _require(
        not bool(info["offline_security"]["hidden_labels_used_by_observation_or_reward"]),
        "hidden security label entered observation or reward",
    )
    expected_hash = canonical_hash(
        {key: value for key, value in info.items() if key != "transition_hash"}
    )
    _require(info["transition_hash"] == expected_hash, "transition hash mismatch")


def validate_transition_stream(
    transitions: Iterable[Mapping[str, Any]],
    *,
    expected_count: int,
) -> dict[str, Any]:
    count = 0
    scenario_hashes: set[str] = set()
    for transition in transitions:
        validate_transition(transition)
        scenario_hashes.add(str(transition["scenario_hash"]))
        count += 1
    _require(count == int(expected_count), f"expected {expected_count} transitions, got {count}")
    return {
        "status": "PASS",
        "transition_count": count,
        "scenario_hash_count": len(scenario_hashes),
    }


def validate_training_receipt(
    path: str | Path,
    *,
    config: PaperRecoveryConfig,
    expected_kind: str,
    expected_seed: int,
) -> dict[str, Any]:
    target = Path(path)
    receipt = json.loads(target.read_text(encoding="utf-8"))
    _require(receipt["status"] == "complete", "training receipt is incomplete")
    _require(receipt["run_kind"] == expected_kind, "run-kind mismatch")
    _require(int(receipt["seed"]) == int(expected_seed), "seed mismatch")
    updates = {
        "smoke": config.budgets.smoke_updates,
        "profiler": config.budgets.profiler_updates,
        "formal": config.budgets.formal_updates,
    }[expected_kind]
    expected_transitions = updates * config.budgets.transitions_per_update
    _require(receipt["updates_completed"] == updates, "update count mismatch")
    _require(
        receipt["transitions_completed"] == expected_transitions,
        "transition count mismatch",
    )
    _require(receipt["checkpoint_rule"] == "fixed_budget_final_only", "checkpoint rule drift")
    _require(not receipt["resumed"], "resume detected")
    _require(not receipt["early_stopped"], "early stop detected")
    _require(not receipt["selected_by_evaluation"], "evaluation-selected checkpoint detected")
    checkpoint = REPO_ROOT / receipt["checkpoint_path"]
    _require(checkpoint.is_file(), "checkpoint file is missing")
    _require(sha256_file(checkpoint) == receipt["checkpoint_sha256"], "checkpoint hash mismatch")
    source = recovery_source_manifest(config)
    _require(source["source_hash"] == receipt["source_hash"], "source hash mismatch")
    workers = receipt["worker_trajectory_receipts"]
    _require(len(workers) == config.budgets.persistent_workers, "worker count mismatch")
    _require(
        sum(int(row["transition_count"]) for row in workers) == expected_transitions,
        "worker trajectory counts are incomplete",
    )
    _require(receipt["worker_runtime"]["all_workers_reaped"], "worker leak detected")
    _require(receipt["actor_updated"], "actor did not update")
    _require(receipt["critic_updated"], "critic did not update")
    _require(receipt["actor_and_critic_updated"], "actor/critic update gate failed")
    return {
        "status": "PASS",
        "receipt_path": target.as_posix(),
        "updates": updates,
        "transitions": expected_transitions,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "trajectory_hash": receipt["trajectory_hash"],
    }
