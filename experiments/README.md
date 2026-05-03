# Experiments layout

Runs are organized so loss-function experiments stay isolated and reproducible.

```text
experiments/
  phase_a/
    <loss_slug>/              # e.g. baseline_l1_uv_tv, ssim_perceptual (future)
      runs/
        <run_version>/        # e.g. 20260202_143022_seed42
          config_resolved.yaml
          checkpoints/
            epoch_000.pt , epoch_001.pt , ...
            last.pt          # resume here after wall-time kill
            best.pt          # best val_ssim_masked (primary)
          logs/
            train.log        # optional mirror if Slurm logs elsewhere
            metrics.jsonl    # one JSON object per epoch
```

- **`loss_slug`**: Fixed loss recipe identifier (not swept within one folder).
- **`run_version`**: Unique per job — timestamp + seed; never overwrite past runs.
- **Resume**: Continue training with `--resume .../checkpoints/last.pt` using the **same** `run_version` directory so checkpoints append.

Phase B/C can add `phase_b/`, etc., using the same pattern.

## Submitting Slurm jobs (cluster)

1. Edit [`scripts/slurm/train_phase_a_h100.slurm`](scripts/slurm/train_phase_a_h100.slurm) or [`scripts/slurm/train_phase_a_a100.slurm`](scripts/slurm/train_phase_a_a100.slurm): replace `REPLACE_H100_PARTITION` / `REPLACE_A100_PARTITION` and `REPLACE_ACCOUNT` with your site’s partition and Slurm account (add `#SBATCH --qos=...` if required).
2. From the **project root** (where `src/` and `configs/` live), activate your venv and install deps: `pip install -r requirements.txt`.
3. Submit: `sbatch scripts/slurm/train_phase_a_h100.slurm` (or `_a100_`).
4. Monitor: `squeue -u "$USER"` and `tail -f slurm-phase-a-h100-<jobid>.out` (log file names match the `#SBATCH --output=` lines in the script).
5. Resume after wall time: re-submit the same script with `RESUME=/abs/path/to/.../checkpoints/last.pt sbatch ...` (see script `echo RESUME=...`).

HF weights download on first run—ensure the compute node has network access or cache models beforehand.
