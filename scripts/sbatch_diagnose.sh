#!/usr/bin/env bash
#SBATCH --account=icdsai_crch_llmspring26
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=00:30:00
#SBATCH --job-name=diag
#SBATCH --output=logs/diag-%j.out
#SBATCH --error=logs/diag-%j.err
#SBATCH --gres=gpu:a100:1
set -euo pipefail

export HF_HOME="/scratch/sbd5760/.cache"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

echo "[sbatch] job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?}"
"${REPO_ROOT}/.venv/bin/python" -u "${REPO_ROOT}/scripts/diagnose_ppl.py"
