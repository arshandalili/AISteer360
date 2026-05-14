#!/usr/bin/env bash
#SBATCH --account=icdsai_crch_llmspring26
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=00:20:00
#SBATCH --job-name=trace
#SBATCH --output=logs/trace-%j.out
#SBATCH --error=logs/trace-%j.err
#SBATCH --gres=gpu:a100:2
set -euo pipefail

export HF_HOME="/scratch/sbd5760/.cache"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

echo "[sbatch] job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?}"
"${REPO_ROOT}/.venv/bin/python" -u "${REPO_ROOT}/scripts/runtime_trace.py"
