#!/usr/bin/env bash
# Full TruthfulQA sweep across all 11 ported baselines.
# Each method runs as a fresh subprocess so VRAM is released between methods.
#
# Env overrides:
#   MODEL, DATA_MODEL_NAME, LAYER, SEED, BATCH, DTYPE, JUDGE_DTYPE
#
# Examples:
#   ./scripts/run_full_sweep.sh
#   MODEL=meta-llama/Llama-3.1-8B LAYER=13 SEED=42 ./scripts/run_full_sweep.sh
set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.1-8B}
DATA_MODEL_NAME=${DATA_MODEL_NAME:-Llama3.1-8B-Base}
LAYER=${LAYER:-13}
SEED=${SEED:-42}
BATCH=${BATCH:-8}
DTYPE=${DTYPE:-float32}
JUDGE_DTYPE=${JUDGE_DTYPE:-auto}


REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/.venv/bin/python"

# method:T pairs (T values mirror source repo's confs/steer/*.yaml defaults)
METHODS=(
    "repe:1.0"
    "caa:5.0"
    "iti:1.0"
    "mimic:1.0"
    "lin_act:1.0"
    "sphere_steer:1.0"
    "cobras:0.6"
    "ode_steer:5.0"
    "rff_ode_steer:5.0"
    "step_ode_steer:5.0"
    "rff_step_ode_steer:5.0"
)

cd "${REPO_ROOT}"
for entry in "${METHODS[@]}"; do
    method="${entry%%:*}"
    T="${entry##*:}"
    echo
    echo "=============================================="
    echo "  ${method} (T=${T})"
    echo "=============================================="
    "${PY}" scripts/reproduce_truthfulqa.py \
        --model "${MODEL}" \
        --data-model-name "${DATA_MODEL_NAME}" \
        --layer "${LAYER}" \
        --method "${method}" --T "${T}" \
        --seed "${SEED}" --batch-size "${BATCH}" \
        --dtype "${DTYPE}" --judge-dtype "${JUDGE_DTYPE}"
done

echo
echo "Sweep complete. Aggregated results: results/eval/${DATA_MODEL_NAME}/l${LAYER}-TruthfulQA-seed${SEED}.csv"
