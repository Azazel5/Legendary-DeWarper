#!/bin/bash
#SBATCH --job-name=eval_ckpt_a100
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/evaluate/slurm-eval-%j.out
#SBATCH --error=slurm_logs/evaluate/slurm-eval-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT}"
mkdir -p "${ROOT}/slurm_logs/evaluate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

module purge
module load pytorch/2.8.0-cuda12.9-cudnn9
PYTHON_CMD="python"

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch $0 <checkpoint_path> [extra evaluate.py args via EVAL_ARGS env]" >&2
  exit 2
fi

CKPT_PATH="$1"

echo "=== Slurm Evaluate A100 ===" | tee /dev/stderr
echo "ROOT=${ROOT}" | tee /dev/stderr
echo "CKPT_PATH=${CKPT_PATH}" | tee /dev/stderr
echo "EVAL_ARGS=${EVAL_ARGS:-<none>}" | tee /dev/stderr
echo "Host: $(hostname), CUDA: ${CUDA_VISIBLE_DEVICES:-}" | tee /dev/stderr
date -u

# Install missing plotting dependency on-demand for this module environment.
if ! "${PYTHON_CMD}" -c "import matplotlib" >/dev/null 2>&1; then
  echo "matplotlib missing; installing into user site-packages" | tee /dev/stderr
  "${PYTHON_CMD}" -m pip install --user -r requirements.cluster.txt matplotlib
fi

eval_cmd=("${PYTHON_CMD}" evaluate.py --checkpoint "${CKPT_PATH}")
if [[ -n "${EVAL_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${EVAL_ARGS} )
  eval_cmd+=("${extra_args[@]}")
fi

printf "Running:" | tee /dev/stderr
printf " %q" "${eval_cmd[@]}" | tee /dev/stderr
printf "\n" | tee /dev/stderr

"${eval_cmd[@]}"

date -u
echo "=== done ===" | tee /dev/stderr