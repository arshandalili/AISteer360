#!/usr/bin/env bash
#SBATCH --account=icdsai_crch_llmspring26
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --job-name=sweep
#SBATCH --output=logs/sweep-%j.out
#SBATCH --error=logs/sweep-%j.err
#SBATCH --gres=gpu:a100:2
#
# Submit:   sbatch scripts/sbatch_full_sweep.sh
# Override: MODEL=... LAYER=... sbatch scripts/sbatch_full_sweep.sh
set -euo pipefail

export HF_HOME="/scratch/sbd5760/.cache"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# SLURM_SUBMIT_DIR = directory where `sbatch` was invoked (the repo root).
# BASH_SOURCE points to /var/spool/slurm/... under sbatch, so don't use it.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

echo "[sbatch] job=${SLURM_JOB_ID:-local} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

bash "${REPO_ROOT}/scripts/run_full_sweep.sh"
