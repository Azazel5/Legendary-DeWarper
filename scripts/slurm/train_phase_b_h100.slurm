#!/bin/bash
#SBATCH --job-name=p_b_d2_h100
#SBATCH --time=10:00:00
#
# H100 is available in the gpu partition on this cluster.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=slurm_logs/slurm-phase-b-h100-%j.out
#SBATCH --error=slurm_logs/slurm-phase-b-h100-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT}"
mkdir -p "${ROOT}/slurm_logs"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

module purge
module load pytorch/2.8.0-cuda12.9-cudnn9
PYTHON_CMD="python"

CONFIG="${CONFIG:-configs/phase_b/baseline_unet_l1_uv_tv.yaml}"
export BATCH_SIZE="${BATCH_SIZE:-4}"

echo "=== Slurm Phase B H100 ===" | tee /dev/stderr
echo "ROOT=${ROOT}" | tee /dev/stderr
echo "CONFIG=${CONFIG}" | tee /dev/stderr
echo "BATCH_SIZE=${BATCH_SIZE}" | tee /dev/stderr
echo "RESUME=${RESUME:-<none>}" | tee /dev/stderr
echo "PYTHON_CMD=${PYTHON_CMD}" | tee /dev/stderr
echo "Host: $(hostname), CUDA: ${CUDA_VISIBLE_DEVICES:-}" | tee /dev/stderr
date -u

"${PYTHON_CMD}" -m src.train \
  --config "${CONFIG}" \
  --batch-size "${BATCH_SIZE}" \
  ${RESUME:+--resume "${RESUME}"}

date -u
echo "=== done ===" | tee /dev/stderr