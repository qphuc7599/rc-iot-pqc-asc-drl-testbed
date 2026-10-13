"""Prepare or explicitly execute one of the three separately scheduled sessions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from drl.paper_recovery.receipts import atomic_write_json, canonical_hash, sha256_file, utc_now
from drl.paper_recovery_minor_ablation_v1.trainer import (
    CAMPAIGN_PLAN_PATH,
    RESULT_ROOT,
    REPO_ROOT,
    ablation_source_manifest,
    load_campaign_plan,
    load_protocol,
)
from drl.paper_recovery_security_v4.config import security_v4_config


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _completed(root: Path, *, arm: str, seed: int) -> dict[str, Any]:
    receipt = _json(root / "completion_receipt.json")
    for key, expected in {
        "status": "complete",
        "updates_completed": 3000,
        "transitions_completed": 9000000,
        "config_hash": load_protocol()["frozen_config_hash"],
    }.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"invalid completed seed {seed}: {key}")
    if seed != 657111:
        if receipt.get("arm") != arm or receipt.get("seed") != seed:
            raise RuntimeError("completed seed arm/seed drift")
        if receipt.get("all_updates_finite") is not True:
            raise RuntimeError("completed seed has nonfinite updates")
        if receipt.get("all_required_modules_updated") is not True:
            raise RuntimeError("completed seed has unupdated modules")
        if receipt.get("worker_runtime", {}).get("all_workers_reaped") is not True:
            raise RuntimeError("completed seed worker shutdown failed")
    if sha256_file(root / "ppo_final.pt") != receipt.get("checkpoint_sha256"):
        raise RuntimeError("completed checkpoint hash mismatch")
    if seed == 657111 and receipt["checkpoint_sha256"] != "eabcd94a6c1a86c67a78241f894c3e9580905f04f71f8518bb32a1a45d40467b":
        raise RuntimeError("frozen parent checkpoint hash drift")
    return receipt


def preflight(session: int) -> dict[str, Any]:
    plan = load_campaign_plan()
    selected = plan["sessions"][str(session)]
    arm = selected["arm"]
    config = security_v4_config(training=True)
    if canonical_hash(config.to_dict()) != load_protocol()["frozen_config_hash"]:
        raise RuntimeError("scientific training config drift")
    source = ablation_source_manifest(config, arm=arm)
    rows: list[dict[str, Any]] = []
    for seed in selected["reuse_completed_seeds"]:
        root = REPO_ROOT / f"results/paper_recovery_security_speed_v5/seed_{seed}"
        receipt = _completed(root, arm=arm, seed=seed)
        rows.append({"seed": seed, "state": "reuse_completed", "checkpoint_sha256": receipt["checkpoint_sha256"]})
    for seed in selected["new_training_seeds"]:
        root = RESULT_ROOT / "formal_3000" / arm / f"seed_{seed}"
        claim = RESULT_ROOT / "claims" / f"{arm}_seed_{seed}_formal_3000.json"
        if (root / "completion_receipt.json").exists():
            receipt = _completed(root, arm=arm, seed=seed)
            rows.append({"seed": seed, "state": "already_completed", "checkpoint_sha256": receipt["checkpoint_sha256"]})
        elif root.exists() or claim.exists():
            raise RuntimeError(f"seed {seed} already started without completion; no resume or rerun allowed")
        else:
            rows.append({"seed": seed, "state": "ready_unclaimed"})
    return {
        "schema": "qrelay-minor-revision-session-preflight-1",
        "session": session,
        "arm": arm,
        "campaign_plan_sha256": sha256_file(CAMPAIGN_PLAN_PATH),
        "source_hash": source["source_hash"],
        "updates_per_seed": 3000,
        "transitions_per_seed": 9000000,
        "estimated_training_hours": selected["estimated_training_hours"],
        "seeds": rows,
        "next_session_automatically_started": False,
    }


def _sync(workspace: Path, arm: str, seed: int) -> None:
    root = RESULT_ROOT / "formal_3000" / arm / f"seed_{seed}"
    destination = workspace / root.relative_to(REPO_ROOT)
    shutil.copytree(root, destination, dirs_exist_ok=True)
    claims = RESULT_ROOT / "claims"
    target = workspace / claims.relative_to(REPO_ROOT)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(claims / f"{arm}_seed_{seed}_formal_3000.json", target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--execute", action="store_true", help="Explicitly start training; otherwise only inspect preparation.")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--sync-workspace", type=Path)
    args = parser.parse_args()
    snapshot = preflight(args.session)
    print(json.dumps(snapshot, indent=2), flush=True)
    if not args.execute:
        return
    if os.name != "posix" or str(REPO_ROOT).startswith("/mnt/"):
        raise RuntimeError("execute only from the native WSL runtime")
    log_root = RESULT_ROOT / "session_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    receipt_path = log_root / f"session_{args.session}_receipt.json"
    if receipt_path.exists():
        raise RuntimeError("session receipt already exists; inspect it before a new launch")
    session_receipt = {
        **snapshot,
        "status": "running",
        "started_at": utc_now(),
        "completed_new_seeds": [],
    }
    atomic_write_json(receipt_path, session_receipt)
    try:
        for item in snapshot["seeds"]:
            if item["state"] != "ready_unclaimed":
                continue
            arm = snapshot["arm"]
            seed = item["seed"]
            prefix = log_root / f"session_{args.session}_{arm}_seed_{seed}"
            environment = os.environ.copy()
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                environment[name] = "1"
            environment["PYTHONUNBUFFERED"] = "1"
            with Path(str(prefix) + ".stdout").open("x", encoding="utf-8") as stdout, Path(str(prefix) + ".stderr").open("x", encoding="utf-8") as stderr:
                result = subprocess.run(
                    [sys.executable, "-u", "-m", "experiments.paper_recovery_minor_ablation_v1.run_training", "--arm", arm, "--seed", str(seed), "--run-kind", "formal_3000", "--device", args.device],
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(f"seed {seed} failed with exit code {result.returncode}; inspect preserved logs")
            _completed(RESULT_ROOT / "formal_3000" / arm / f"seed_{seed}", arm=arm, seed=seed)
            if args.sync_workspace:
                _sync(args.sync_workspace, arm, seed)
            session_receipt["completed_new_seeds"].append(seed)
            atomic_write_json(receipt_path, session_receipt)
        session_receipt["status"] = "complete"
        session_receipt["completed_at"] = utc_now()
    except BaseException as error:
        session_receipt["status"] = "failed_or_interrupted"
        session_receipt["error"] = str(error)
        raise
    finally:
        atomic_write_json(receipt_path, session_receipt)
        if args.sync_workspace:
            target = args.sync_workspace / receipt_path.relative_to(REPO_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(receipt_path, target)


if __name__ == "__main__":
    main()
