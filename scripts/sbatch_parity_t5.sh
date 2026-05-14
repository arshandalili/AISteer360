#!/usr/bin/env bash
#SBATCH --account=icdsai_crch_llmspring26
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=00:45:00
#SBATCH --job-name=parity-t5
#SBATCH --output=logs/parity-t5-%j.out
#SBATCH --error=logs/parity-t5-%j.err
#SBATCH --gres=gpu:a100:2
set -euo pipefail

export HF_HOME="/scratch/sbd5760/.cache"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

"${REPO_ROOT}/.venv/bin/python" -u "${REPO_ROOT}/scripts/parity_check.py" \
    --model meta-llama/Llama-3.1-8B \
    --data-model-name Llama3.1-8B-Base \
    --layer 13 --num-prompts 30 --seed 42 --T 5.0 --dtype float32
