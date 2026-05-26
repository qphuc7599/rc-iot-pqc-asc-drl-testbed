#!/usr/bin/env bash
#
# Reproducible DRL runner for the Q-Relay paper.
#
# Usage:
#   bash run-drl-experiments.sh smoke
#   bash run-drl-experiments.sh train-main [episodes] [steps] [workers]
#   bash run-drl-experiments.sh train-gumbel [episodes] [steps] [workers]
#   bash run-drl-experiments.sh disaster-main
#   bash run-drl-experiments.sh disaster-gumbel
#   bash run-drl-experiments.sh sensitivity-main
#   bash run-drl-experiments.sh sensitivity-gumbel
#   bash run-drl-experiments.sh variable-n-gumbel
#
# Notes:
#   - "gumbel" is the paper-reported policy: masked Gumbel-Top-k actor
#     + pooled critic + invalid-action masking.
#   - "main" is the earlier masked-Gaussian pooled checkpoint kept for
#     backwards-compatible ablation/reproduction only.
#   - Large train runs are CPU-intensive; avoid running them alongside NS-3.

set -euo pipefail

MODE="${1:-smoke}"
EPISODES="${2:-3000}"
STEPS="${3:-3000}"
WORKERS="${4:-11}"

PYTHON="${PYTHON:-}"
choose_python() {
    local candidates=()
    if [ -n "$PYTHON" ]; then
        candidates+=("$PYTHON")
    fi
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        candidates+=("$VIRTUAL_ENV/bin/python")
    fi
    # Prefer the active PATH environment before the repo CPU-only fallback. This
    # lets a CUDA-enabled conda/venv stay in control when it is activated.
    candidates+=(python python3 ".venv-wsl/bin/python")
    local cand
    for cand in "${candidates[@]}"; do
        command -v "$cand" >/dev/null 2>&1 || continue
        if "$cand" - <<'PY' >/dev/null 2>&1
import numpy  # noqa: F401
import torch  # noqa: F401
import gymnasium  # noqa: F401
PY
        then
            PYTHON="$cand"
            return 0
        fi
    done
    echo "[ERROR] No Python environment with numpy, torch, and gymnasium was found." >&2
    echo "        Install dependencies, or run from the environment where drl/ppo_agent.py already works." >&2
    echo "        Example: python -m pip install numpy torch gymnasium" >&2
    return 1
}

choose_python

MAIN_DIR="${MAIN_DIR:-drl/models_robust_pooled}"
GUMBEL_DIR="${GUMBEL_DIR:-drl/models_robust_gumbel_pooled}"
RESULTS_DIR="${RESULTS_DIR:-results/drl_pooled}"
GUMBEL_RESULTS_DIR="${GUMBEL_RESULTS_DIR:-results/drl_gumbel_pooled}"
SEEDS="${SEEDS:-42,123,456,789,1024,2048,3333,4096,5555,7777}"
VARIABLE_NODES="${VARIABLE_NODES:-75 150 200}"

common_train=(
    drl/ppo_agent.py
    --mode train
    --sim
    --episodes "$EPISODES"
    --steps "$STEPS"
    --workers "$WORKERS"
    --seed 2026
    --train-profile robust
    --critic-mode pooled
)

common_eval=(
    drl/ppo_agent.py
    --sim
    --nodes 100
    --committee 21
    --critic-mode pooled
    --total-steps 2000
    --disaster-step 1000
    --eval-seeds "$SEEDS"
    --results-dir "$RESULTS_DIR"
)

case "$MODE" in
    smoke)
        "$PYTHON" drl/ppo_agent.py --mode train --sim \
            --episodes 1 --steps 16 --workers 1 --seed 7 \
            --save-dir tmp/ppo_pooled_smoke \
            --actor-mode gaussian_masked \
            --critic-mode pooled
        ;;

    train-main)
        "$PYTHON" "${common_train[@]}" \
            --save-dir "$MAIN_DIR" \
            --actor-mode gaussian_masked
        ;;

    train-gumbel)
        "$PYTHON" "${common_train[@]}" \
            --save-dir "$GUMBEL_DIR" \
            --actor-mode gumbel_topk \
            --gumbel-temperature 0.7
        ;;

    disaster-main)
        "$PYTHON" "${common_eval[@]}" \
            --mode disaster \
            --model "$MAIN_DIR/ppo_best.pt" \
            --save-dir "$MAIN_DIR" \
            --actor-mode gaussian_masked \
            --strategies all
        ;;

    disaster-gumbel)
        "$PYTHON" "${common_eval[@]}" \
            --mode disaster \
            --model "$GUMBEL_DIR/ppo_best.pt" \
            --save-dir "$GUMBEL_DIR" \
            --results-dir "$GUMBEL_RESULTS_DIR" \
            --actor-mode gumbel_topk \
            --gumbel-temperature 0.7 \
            --strategies all
        ;;

    sensitivity-main)
        "$PYTHON" "${common_eval[@]}" \
            --mode sensitivity \
            --model "$MAIN_DIR/ppo_best.pt" \
            --save-dir "$MAIN_DIR" \
            --actor-mode gaussian_masked \
            --strategies ppo,gba \
            --kill-ratios 0.10,0.20,0.30,0.40 \
            --byz-ratios 0.00,0.10,0.20,0.30
        ;;

    sensitivity-gumbel)
        "$PYTHON" "${common_eval[@]}" \
            --mode sensitivity \
            --model "$GUMBEL_DIR/ppo_best.pt" \
            --save-dir "$GUMBEL_DIR" \
            --results-dir "$GUMBEL_RESULTS_DIR" \
            --actor-mode gumbel_topk \
            --gumbel-temperature 0.7 \
            --strategies ppo,gba \
            --kill-ratios 0.10,0.20,0.30,0.40 \
            --byz-ratios 0.00,0.10,0.20,0.30
        ;;

    variable-n-gumbel)
        for n in $VARIABLE_NODES; do
            echo
            echo "================================================================"
            echo "  Gumbel zero-shot variable-N disaster evaluation: N=$n"
            echo "  Results: results/drl_gumbel_pooled_N${n}"
            echo "================================================================"
            "$PYTHON" drl/ppo_agent.py \
                --mode disaster \
                --sim \
                --nodes "$n" \
                --committee 21 \
                --model "$GUMBEL_DIR/ppo_best.pt" \
                --save-dir "$GUMBEL_DIR" \
                --actor-mode gumbel_topk \
                --gumbel-temperature 0.7 \
                --critic-mode pooled \
                --total-steps 2000 \
                --disaster-step 1000 \
                --eval-seeds "$SEEDS" \
                --strategies ppo,gba \
                --results-dir "results/drl_gumbel_pooled_N${n}"
        done
        ;;

    *)
        echo "[ERROR] Unknown mode: $MODE" >&2
        exit 1
        ;;
esac
