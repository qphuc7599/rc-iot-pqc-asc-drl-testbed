"""Hash-anchored receipts and formal-seed safeguards for paper recovery."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import PaperRecoveryConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "results" / "paper_recovery_ppo_v1"
FORMAL_GATE_PATH = RESULT_ROOT / "pretraining_gate.json"
FORMAL_CLAIMS_PATH = RESULT_ROOT / "formal_seed_claims.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def recovery_source_manifest(
    config: PaperRecoveryConfig,
    *,
    include_tests: bool = True,
) -> dict[str, Any]:
    """Hash every active recovery source plus the frozen response calibration."""

    roots = [REPO_ROOT / "drl" / "paper_recovery"]
    if include_tests:
        roots.append(REPO_ROOT / "tests" / "paper_recovery")
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    calibration = REPO_ROOT / config.physics.calibration_relative_path
    files.append(calibration)
    relative_hashes = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in sorted(set(files))
    }
    manifest = {
        "schema": "qrelay-paper-recovery-source-manifest-1",
        "files": relative_hashes,
        "config_hash": canonical_hash(config.to_dict()),
    }
    manifest["source_hash"] = canonical_hash(manifest)
    return manifest


class TrajectoryHasher:
    """Streaming hash that also enforces a contiguous transition sequence."""

    def __init__(self, *, worker_id: int | None = None):
        self.worker_id = worker_id
        self._digest = hashlib.sha256()
        self.transition_count = 0
        self.episodes = 0
        self._last_episode: int | None = None
        self._expected_step = 0

    def update(self, transition: Mapping[str, Any], *, episode_index: int) -> None:
        step = int(transition["step"])
        episode = int(episode_index)
        if self._last_episode is None or episode != self._last_episode:
            if self._last_episode is not None and episode != self._last_episode + 1:
                raise ValueError("episode indices are not contiguous")
            self._last_episode = episode
            self._expected_step = 0
            self.episodes += 1
        if step != self._expected_step:
            raise ValueError(
                f"non-contiguous trajectory: expected step {self._expected_step}, got {step}"
            )
        record = {
            "worker_id": self.worker_id,
            "episode_index": episode,
            "step": step,
            "transition_hash": transition["transition_hash"],
            "scenario_hash": transition["scenario_hash"],
            "state_hash": transition["state_hash"],
            "candidate_committee_ids": transition["candidate_committee_ids"],
            "executed_committee_ids": transition["executed_committee_ids"],
            "reward_total": transition["reward_total"],
        }
        self._digest.update(canonical_json(record).encode("utf-8"))
        self._digest.update(b"\n")
        self.transition_count += 1
        self._expected_step += 1

    def receipt(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "transition_count": self.transition_count,
            "episode_count": self.episodes,
            "trajectory_hash": self._digest.hexdigest(),
        }


def aggregate_trajectory_hash(worker_receipts: Iterable[Mapping[str, Any]]) -> str:
    ordered = sorted(
        (dict(value) for value in worker_receipts),
        key=lambda row: (-1 if row.get("worker_id") is None else int(row["worker_id"])),
    )
    return canonical_hash(ordered)


def runtime_manifest() -> dict[str, Any]:
    try:
        import numpy
        import torch

        packages = {"numpy": numpy.__version__, "torch": torch.__version__}
        accelerator = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        }
    except ImportError:
        packages = {}
        accelerator = {}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "accelerator": accelerator,
    }


def assert_formal_runtime_authorized() -> dict[str, Any]:
    """Formal seeds are restricted to the registered A100/MPS20 resource."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("formal training requires a PyTorch accelerator runtime") from error
    if not torch.cuda.is_available():
        raise RuntimeError("formal training requires the registered A100/MPS20 CUDA allocation")
    device_name = torch.cuda.get_device_name(0)
    if "A100" not in device_name.upper() and "MPS20" not in device_name.upper():
        raise RuntimeError(
            "formal training is locked to A100/MPS20; detected "
            f"{device_name!r}"
        )
    return {
        "cuda_device": device_name,
        "cuda_device_count": int(torch.cuda.device_count()),
        "registered_resource_class": "A100_or_MPS20",
    }


@dataclass(frozen=True)
class TrainingReceipt:
    schema: str
    status: str
    seed: int
    run_kind: str
    started_at: str
    completed_at: str
    updates_completed: int
    transitions_completed: int
    source_hash: str
    config_hash: str
    checkpoint_path: str
    checkpoint_sha256: str
    trajectory_hash: str
    worker_trajectory_receipts: tuple[dict[str, Any], ...]
    checkpoint_rule: str = "fixed_budget_final_only"
    resumed: bool = False
    early_stopped: bool = False
    selected_by_evaluation: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["receipt_hash"] = canonical_hash(value)
        return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_formal_training_authorized(
    *,
    config: PaperRecoveryConfig,
    seed: int,
    explicit_formal_flag: bool,
) -> dict[str, Any]:
    """Verify the immutable pre-training gate before constructing an environment."""

    seed = int(seed)
    if seed not in config.seeds.formal_training and seed not in config.seeds.formal_replacements:
        raise RuntimeError(f"seed {seed} is not in the registered formal pool")
    if not explicit_formal_flag:
        raise RuntimeError("formal training requires the explicit --formal flag")
    if not FORMAL_GATE_PATH.is_file():
        raise RuntimeError("formal training is locked: pretraining_gate.json does not exist")
    gate = _read_json(FORMAL_GATE_PATH)
    required = {
        "status": "PASS",
        "formal_training_authorized": True,
        "tests": "PASS",
        "equivalence": "PASS",
        "validators": "PASS",
        "baselines": "PASS",
        "smoke": "PASS",
        "profiler": "PASS",
        "protocol_frozen": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": gate.get(key)}
        for key, expected in required.items()
        if gate.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"formal training gate is not satisfied: {mismatches}")
    source = recovery_source_manifest(config)
    if gate.get("source_hash") != source["source_hash"]:
        raise RuntimeError("source changed after the pre-training gate was frozen")
    if gate.get("config_hash") != canonical_hash(config.to_dict()):
        raise RuntimeError("configuration changed after the pre-training gate was frozen")
    return gate


def claim_formal_seed_before_environment(
    *,
    config: PaperRecoveryConfig,
    seed: int,
    explicit_formal_flag: bool,
) -> dict[str, Any]:
    """Claim a formal seed exactly once; called before environment construction."""

    gate = assert_formal_training_authorized(
        config=config, seed=seed, explicit_formal_flag=explicit_formal_flag
    )
    document = (
        _read_json(FORMAL_CLAIMS_PATH)
        if FORMAL_CLAIMS_PATH.is_file()
        else {"schema": "qrelay-paper-recovery-formal-seed-claims-1", "claims": []}
    )
    active = [row for row in document["claims"] if row["status"] == "claimed"]
    if active:
        raise RuntimeError(
            f"formal seeds must run sequentially; active claim is {active[0]['seed']}"
        )
    for existing in document["claims"]:
        if int(existing["seed"]) == int(seed):
            raise RuntimeError(
                f"formal seed {seed} was already claimed with status {existing['status']}"
            )
    claimed_seeds = {int(row["seed"]) for row in document["claims"]}
    next_primary = next(
        (value for value in config.seeds.formal_training if value not in claimed_seeds),
        None,
    )
    if next_primary is not None and seed != next_primary:
        raise RuntimeError(f"formal primary seeds must be claimed in order; next is {next_primary}")
    if next_primary is None and seed in config.seeds.formal_replacements:
        failed_count = sum(row["status"] == "spent_failed" for row in document["claims"])
        completed_count = sum(row["status"] == "completed" for row in document["claims"])
        if failed_count == 0 or completed_count >= len(config.seeds.formal_training):
            raise RuntimeError("no replacement seed is currently required")
        next_replacement = next(
            (value for value in config.seeds.formal_replacements if value not in claimed_seeds),
            None,
        )
        if seed != next_replacement:
            raise RuntimeError(
                f"formal replacement seeds must be claimed in order; next is {next_replacement}"
            )
    claim = {
        "seed": int(seed),
        "status": "claimed",
        "claimed_at": utc_now(),
        "gate_hash": canonical_hash(gate),
        "source_hash": gate["source_hash"],
        "config_hash": gate["config_hash"],
        "claimed_before_environment_construction": True,
    }
    document["claims"].append(claim)
    document["document_hash"] = canonical_hash(
        {key: value for key, value in document.items() if key != "document_hash"}
    )
    atomic_write_json(FORMAL_CLAIMS_PATH, document)
    return claim


def update_formal_seed_status(seed: int, *, status: str, receipt_path: str | None = None) -> None:
    if status not in {"completed", "spent_failed"}:
        raise ValueError("formal seed status must be completed or spent_failed")
    if not FORMAL_CLAIMS_PATH.is_file():
        raise RuntimeError("formal seed claims ledger does not exist")
    document = _read_json(FORMAL_CLAIMS_PATH)
    matching = [row for row in document["claims"] if int(row["seed"]) == int(seed)]
    if len(matching) != 1 or matching[0]["status"] != "claimed":
        raise RuntimeError("seed has no unique active claim")
    matching[0]["status"] = status
    matching[0]["closed_at"] = utc_now()
    matching[0]["receipt_path"] = receipt_path
    document["document_hash"] = canonical_hash(
        {key: value for key, value in document.items() if key != "document_hash"}
    )
    atomic_write_json(FORMAL_CLAIMS_PATH, document)
