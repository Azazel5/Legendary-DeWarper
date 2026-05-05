#!/bin/bash
#SBATCH --job-name=p_c_d2_a100
#SBATCH --time=10:00:00
#
# A100-80G is available in the gpu partition on this cluster.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/slurm-phase-c-a100-%j.out
#SBATCH --error=slurm_logs/slurm-phase-c-a100-%j.err

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
export SCRATCH_CHECKPOINT_DIR="/tmp/$USER/legendary_dewarper_checkpoints"

CONFIG="${CONFIG:-configs/phase_c/baseline_l1_uv_tv_perceptual.yaml}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
REMOTE_PHASE_DIR="Phase C"

RUN_VERSION=""
if [[ -z "${RESUME:-}" ]]; then
  RUN_VERSION="${RUN_VERSION:-$(date -u +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-manual}}"
  ACTIVE_RUN_VERSION="${RUN_VERSION}"
else
  ACTIVE_RUN_VERSION="$(basename "$(dirname "$(dirname "${RESUME}")")")"
fi

echo "=== Slurm Phase C A100 ===" | tee /dev/stderr
echo "ROOT=${ROOT}" | tee /dev/stderr
echo "CONFIG=${CONFIG}" | tee /dev/stderr
echo "BATCH_SIZE=${BATCH_SIZE}" | tee /dev/stderr
echo "RESUME=${RESUME:-<none>}" | tee /dev/stderr
echo "RUN_VERSION=${RUN_VERSION:-<auto>}" | tee /dev/stderr
echo "ACTIVE_RUN_VERSION=${ACTIVE_RUN_VERSION}" | tee /dev/stderr
echo "PYTHON_CMD=${PYTHON_CMD}" | tee /dev/stderr
echo "Host: $(hostname), CUDA: ${CUDA_VISIBLE_DEVICES:-}" | tee /dev/stderr
date -u

train_cmd=("${PYTHON_CMD}" -m src.train --config "${CONFIG}" --batch-size "${BATCH_SIZE}")
if [[ -n "${RUN_VERSION}" ]]; then
  train_cmd+=(--run-version "${RUN_VERSION}")
fi
if [[ -n "${RESUME:-}" ]]; then
  train_cmd+=(--resume "${RESUME}")
fi
"${train_cmd[@]}"

RCLONE_REMOTE="${RCLONE_REMOTE:-shu}"
RCLONE_REMOTE_ROOT="${RCLONE_REMOTE_ROOT:-Legendary-DeWarper}"
SRC_CKPT_DIR="${SCRATCH_CHECKPOINT_DIR}/${ACTIVE_RUN_VERSION}/checkpoints"
DEST_CKPT_DIR="${RCLONE_REMOTE}:${RCLONE_REMOTE_ROOT}/${REMOTE_PHASE_DIR}/${ACTIVE_RUN_VERSION}/checkpoints"

if command -v rclone >/dev/null 2>&1; then
  if [[ -d "${SRC_CKPT_DIR}" ]]; then
    echo "[GDRIVE_UPLOAD] START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee /dev/stderr
    echo "[GDRIVE_UPLOAD] SRC=${SRC_CKPT_DIR}" | tee /dev/stderr
    echo "[GDRIVE_UPLOAD] DEST=${DEST_CKPT_DIR}" | tee /dev/stderr
    rclone_cmd=(
      rclone copy --progress
      --include "epoch_*.pt" --include "last.pt" --include "best.pt" --exclude "*"
      "${SRC_CKPT_DIR}" "${DEST_CKPT_DIR}"
    )
    printf "[GDRIVE_UPLOAD] CMD:" | tee /dev/stderr
    printf " %q" "${rclone_cmd[@]}" | tee /dev/stderr
    printf "\n" | tee /dev/stderr
    if ! "${rclone_cmd[@]}"; then
      echo "WARNING: rclone upload failed; checkpoints remain on scratch at ${SRC_CKPT_DIR}" | tee /dev/stderr
    else
      echo "[GDRIVE_UPLOAD] SUCCESS $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee /dev/stderr
      echo "Remote checkpoints uploaded:" | tee /dev/stderr
      rclone ls "${DEST_CKPT_DIR}" | tee /dev/stderr || true
    fi
  else
    echo "WARNING: no scratch checkpoints found at ${SRC_CKPT_DIR}" | tee /dev/stderr
  fi
else
  echo "WARNING: rclone not found; skipping checkpoint upload" | tee /dev/stderr
fi

date -u
echo "=== done ===" | tee /dev/stderr
