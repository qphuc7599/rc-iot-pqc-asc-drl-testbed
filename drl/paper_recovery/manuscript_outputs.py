"""Generate TeX macros only from complete, non-synthetic validated PPO evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .receipts import atomic_write_json, canonical_hash, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_PAPER_DIRECTORY = (
    REPO_ROOT
    / "file_project"
    / "paper_file"
    / "_2026__Elsevier_IoT_HoangPhuc"
)


def _load_hashed(path: Path, hash_field: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    stored = document.pop(hash_field)
    if canonical_hash(document) != stored:
        raise RuntimeError(f"hash mismatch in {path}")
    document[hash_field] = stored
    return document


def _format_percent(value: float, *, signed: bool = True) -> str:
    return f"{value * 100:+.2f}\\%" if signed else f"{value * 100:.2f}\\%"


def generate_ppo_tex_macros(
    *,
    evaluation_results_path: Path,
    primary_analysis_path: Path,
    formal_receipt_paths: list[Path],
    output_path: Path | None = None,
) -> dict[str, Any]:
    evaluation = _load_hashed(evaluation_results_path, "results_hash")
    analysis = _load_hashed(primary_analysis_path, "analysis_hash")
    if evaluation.get("synthetic") or analysis.get("synthetic"):
        raise RuntimeError("synthetic evidence is forbidden from manuscript macros")
    if len(formal_receipt_paths) != 6:
        raise RuntimeError("six formal completion receipts are required")
    receipt_hashes = []
    seeds = set()
    for path in formal_receipt_paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "complete"
            or receipt.get("run_kind") != "formal"
            or receipt.get("updates_completed") != 3_000
            or receipt.get("transitions_completed") != 9_000_000
            or receipt.get("checkpoint_rule") != "fixed_budget_final_only"
            or receipt.get("resumed")
            or receipt.get("early_stopped")
            or receipt.get("selected_by_evaluation")
        ):
            raise RuntimeError(f"invalid formal receipt {path}")
        if not receipt.get("actor_updated") or not receipt.get("critic_updated"):
            raise RuntimeError(f"actor/critic update proof missing in {path}")
        seeds.add(int(receipt["seed"]))
        receipt_hashes.append(sha256_file(path))
    if seeds != {630101, 630102, 630103, 630104, 630105, 630106}:
        raise RuntimeError("formal receipt seed set mismatch")

    fmg = analysis["effects"]["FMG"]
    risk = analysis["effects"]["Risk-Prior"]
    mean_auc = fmg["mean_ppo_post_event_auc"]
    failure_delta = max(
        analysis["effects"][name]["mean_failure_rate_difference"]
        for name in ("FMG", "Risk-Prior")
    )
    energy_delta = max(
        analysis["effects"][name]["mean_energy_relative_difference"]
        for name in ("FMG", "Risk-Prior")
    )
    security_delta = max(
        analysis["effects"][name]["mean_security_exposure_difference"]
        for name in ("FMG", "Risk-Prior")
    )

    def contrast(row: dict[str, Any]) -> str:
        effect = row["mean_relative_post_event_auc_effect"]
        low, high = row["crossed_relative_95_ci"]
        return (
            f"{_format_percent(effect)} "
            f"(crossed 95\\% CI {_format_percent(low)} to {_format_percent(high)})"
        )

    wording = (
        "superiority under the registered gates"
        if analysis["overall_superiority_gate_pass"]
        else "competitive/parity/negative wording only; superiority gate not met"
    )
    lines = [
        "% Auto-generated from validated, non-synthetic paper-recovery JSON.",
        "% Any manual edit invalidates the generation receipt.",
        "\\newcommand{\\PPOPending}{validated}",
        "\\newcommand{\\PPOEvidenceStatus}{six cold starts complete and validated}",
        f"\\newcommand{{\\PPOPrimaryAUC}}{{{mean_auc:,.1f} TPS-steps}}",
        f"\\newcommand{{\\PPOVsFMG}}{{{contrast(fmg)}}}",
        f"\\newcommand{{\\PPOVsRiskPrior}}{{{contrast(risk)}}}",
        f"\\newcommand{{\\PPOFailureDelta}}{{{failure_delta * 100:+.2f} percentage points}}",
        f"\\newcommand{{\\PPOEnergyDelta}}{{{_format_percent(energy_delta)}}}",
        f"\\newcommand{{\\PPOSecurityDelta}}{{{security_delta:+.4f}}}",
        f"\\newcommand{{\\PPOClaim}}{{{wording}}}",
    ]
    output_path = output_path or CANONICAL_PAPER_DIRECTORY / "ppo_results_generated.tex"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {
        "schema": "qrelay-paper-recovery-manuscript-macro-receipt-1",
        "status": "PASS",
        "output_path": output_path.relative_to(REPO_ROOT).as_posix(),
        "output_sha256": sha256_file(output_path),
        "evaluation_results_sha256": sha256_file(evaluation_results_path),
        "primary_analysis_sha256": sha256_file(primary_analysis_path),
        "formal_receipt_sha256": receipt_hashes,
        "formal_seeds": sorted(seeds),
        "superiority_gate_pass": analysis["overall_superiority_gate_pass"],
        "permitted_wording": wording,
    }
    receipt["receipt_hash"] = canonical_hash(receipt)
    atomic_write_json(output_path.with_suffix(".receipt.json"), receipt)
    return receipt

