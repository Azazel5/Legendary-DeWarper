#!/bin/bash
#SBATCH --job-name=p_a_d2_a100
#SBATCH --time=10:00:00
#
# Replace with your site's A100 partition / account / qos:
#SBATCH --partition=REPLACE_A100_PARTITION
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-phase-a-a100-%j.out
#SBATCH --error=slurm-phase-a-a100-%j.err

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Optional: activate your environment
# source "${ROOT}/.venv/bin/activate"

CONFIG="${CONFIG:-configs/phase_a/baseline_l1_uv_tv.yaml}"
export BATCH_SIZE="${BATCH_SIZE:-4}"

echo "=== Slurm A100 Phase A ===" | tee /dev/stderr
echo "ROOT=${ROOT}" | tee /dev/stderr
echo "CONFIG=${CONFIG}" | tee /dev/stderr
echo "BATCH_SIZE=${BATCH_SIZE}" | tee /dev/stderr
echo "RESUME=${RESUME:-<none>}" | tee /dev/stderr
echo "Host: $(hostname), CUDA: ${CUDA_VISIBLE_DEVICES:-}" | tee /dev/stderr
date -u

python -m src.train \
  --config "${CONFIG}" \
  --batch-size "${BATCH_SIZE}" \
  ${RESUME:+--resume "${RESUME}"}

date -u
echo "=== done ===" | tee /dev/stderr
