# Legendary-DeWarper

**Tufts Computer Vision — Final Project**

Learn a **geometric** dewarping model: from a **warped document RGB** (with background) predict a **UV / sampling field**, warp with differentiable `grid_sample`, and match the **flat ground truth** texture. **Focus on geometry, not lighting** — the README in the course materials and in `dataset_loader.py` explain why photometric MSE is misleading.

---

## Repository map

| Area | Role |
|------|------|
| [`dataset_loader.py`](dataset_loader.py) | `DocumentDataset`, `get_dataloaders`, ImageNet-normalized RGB/GT, optional **UV** (`[0,1]`, 2ch) and **`uv_mask`**, `UVReconstructionLoss`, `create_base_grid`, `MaskedL1Loss`, `SSIMLoss`, **`VGGPerceptualLoss`** (Phase C+) |
| [`uv_dewarp.py`](uv_dewarp.py) | **Upper bound:** forward-warp RGB with **GT UV** (bilinear splat + hole fill). Baseline for “what perfect UV would look like” on a sample. |
| [`src/models/dinov2_dewarp.py`](src/models/dinov2_dewarp.py) | **Phase A model:** `facebook/dinov2-large` → conv decoder → **flow** (for `grid_sample`) + **UV** head; outputs dewarped tensor. |
| [`src/models/phase_b_unet_dewarp.py`](src/models/phase_b_unet_dewarp.py) | **Phase B baseline:** same `facebook/dinov2-large` encoder, but a **U-Net style decoder** with internal skip connections; still predicts **flow** + **UV**. |
| [`src/train.py`](src/train.py) | Training loop: AMP, checkpointing, resume, `metrics.jsonl`, Slurm-friendly logging. |
| [`src/metrics.py`](src/metrics.py) | Denorm to `[0,1]`; **SSIM / MS-SSIM / PSNR** (full + masked); **UV L1** (full + masked on foreground). |
| [`src/config.py`](src/config.py) | YAML + CLI overrides. |
| [`configs/phase_a/baseline_l1_uv_tv.yaml`](configs/phase_a/baseline_l1_uv_tv.yaml) | Default Phase A hyperparameters. |
| [`scripts/eval_upper_bound_sample.py`](scripts/eval_upper_bound_sample.py) | Batches: **GT-UV dewarp** vs flat GT (ceiling-style diagnostic, no trained model). |
| [`scripts/slurm/`](scripts/slurm/) | H100 / A100 **10h** Slurm jobs for Phase A and Phase B. |
| [`experiments/`](experiments/) | Per-run outputs; see [`experiments/README.md`](experiments/README.md) for directory convention. |
| [`requirements.txt`](requirements.txt) | Python dependencies. |

**Data** (not in git): place under `renders/synthetic_data_pitch_sweep/` with `rgb/`, `ground_truth/`, `uv/`, etc., as in the course handout.

---

## Phase A (implemented)

### What we built

- **Backbone:** [DINOv2 Large](https://huggingface.co/facebook/dinov2-large) via Hugging Face `Dinov2Model` (self-supervised ViT features; not document-specific, strong transfer).
- **Heads:** Convolutional decoder upsamples patch features to full resolution; **2ch flow** (residual on `[-1,1]` sampling grid) + **2ch UV** in `[0,1]` (supervised against dataset UV).
- **Warp:** `torch.nn.functional.grid_sample` on the **input warped RGB** (ImageNet-normalized) to produce **dewarped** features in the same space as `ground_truth`.
- **Loss (one fixed recipe, `loss_slug` = `baseline_l1_uv_tv`):** `UVReconstructionLoss` with `loss_type: l1` — **masked L1** on dewarped vs GT, **L1** on predicted vs GT UV, **TV** on the flow (smoothness). SSIM/MS-SSIM are **metrics only** in Phase A, not the training loss.
- **Training:** AdamW, gradient clip, **BF16** autocast on supported GPUs (else FP16 + `GradScaler` when needed). Dataloaders use `use_uv=True` and **`uv_mask`** for masked losses/metrics.
- **Checkpoints (every epoch):** `checkpoints/epoch_####.pt`, `last.pt` (resume), `best.pt` (highest **val_ssim_masked** by default). Atomic save to reduce corruption on wall-time kill. Run folder also gets `config_resolved.yaml` and `logs/metrics.jsonl` (one JSON object per epoch).
- **Slurm:** [`scripts/slurm/train_phase_a_h100.slurm`](scripts/slurm/train_phase_a_h100.slurm) and [`scripts/slurm/train_phase_a_a100.slurm`](scripts/slurm/train_phase_a_a100.slurm) — **10:00:00** wall time; replace `REPLACE_*` partition/account before submitting.

## Phase B (implemented baseline)

Phase B keeps the same pretrained `dinov2-large` encoder and the same reconstruction / UV / TV losses and metrics, but swaps in a **U-Net style decoder** with explicit skip connections inside the decoder path.

- **Model:** [`src/models/phase_b_unet_dewarp.py`](src/models/phase_b_unet_dewarp.py)
- **Config:** [`configs/phase_b/baseline_unet_l1_uv_tv.yaml`](configs/phase_b/baseline_unet_l1_uv_tv.yaml)
- **Slurm:** [`scripts/slurm/train_phase_b_h100.slurm`](scripts/slurm/train_phase_b_h100.slurm) and [`scripts/slurm/train_phase_b_a100.slurm`](scripts/slurm/train_phase_b_a100.slurm)

Phase B uses the same `uv_mask` handling and the same evaluation script, so you can compare Phase A and Phase B with identical metrics and the same upper-bound diagnostic.

## Phase C (implemented)

Phase C introduces a **perceptual loss** (VGG-based feature distance) and an **optional refinement head** (small CNN) for post-dewarping correction. Two variants explore different decoders under identical loss conditions.

### Phase C Variants

**V1: Dinov2-based (Phase A decoder + perceptual loss)**
- **Model:** `facebook/dinov2-large` encoder + convolutional decoder (Phase A style)
- **Loss:** [`UVReconstructionLoss`](dataset_loader.py) with **`perceptual_weight: 0.1`** (`VGGPerceptualLoss` on frozen VGG16 features) + masked L1 on UV + TV on flow
- **Config:** [`configs/phase_c/baseline_l1_uv_tv_perceptual.yaml`](configs/phase_c/baseline_l1_uv_tv_perceptual.yaml) (`phase: phase_c`)
- **Slurm:** [`scripts/slurm/train_phase_c_a100.slurm`](scripts/slurm/train_phase_c_a100.slurm)

**V2: UNet-based (Phase B decoder + perceptual loss)**
- **Model:** `facebook/dinov2-large` encoder + U-Net decoder with skip connections (Phase B style)
- **Loss:** Same [`UVReconstructionLoss`](dataset_loader.py) as V1 with **`perceptual_weight: 0.1`**
- **Config:** [`configs/phase_c/baseline_unet_l1_uv_tv_perceptual.yaml`](configs/phase_c/baseline_unet_l1_uv_tv_perceptual.yaml) (`phase: phase_b`)
- **Slurm:** [`scripts/slurm/train_phase_c_a100.slurm`](scripts/slurm/train_phase_c_a100.slurm)

### Phase C Configuration Knobs

- **`perceptual_weight`** (default: `0.1`) — controls strength of VGG perceptual loss term; set to `0.0` to disable
- **`use_refinement_head`** (default: `false`) — enables optional post-dewarping CNN refinement (takes concat of dewarped + original RGB, predicts per-pixel residual)
- **`refinement_channels`** (default: `[64, 32]`) — hidden layer sizes in refinement CNN (only used if `use_refinement_head: true`)

### Phase C Training Results

- **Job 719821 (A100, V1 Dinov2):** 100 epochs, run version `20260505_021622_job719821`
  - Best val_ssim_masked: **0.7719** (epoch 96)
  - Final metrics: PSNR 6.81 dB (full), 18.66 dB (masked); MS-SSIM masked 0.7519
  - Checkpoints: `epoch_0000.pt`, `epoch_0050.pt`, `best.pt`, `last.pt` → **13.7 GiB**

- **Job 719822 (A100, V2 UNet):** 100 epochs, run version `20260505_022425_job719822`
  - All checkpoints: `epoch_0000.pt`, `epoch_0050.pt`, `best.pt`, `last.pt` → **13.8 GiB**

**Cross-Phase Comparison (A vs B vs C):**
Phases A, B, and C enable direct comparison of:
- **Architecture** (Phase A conv decoder vs Phase B U-Net vs Phase C both with perceptual loss)
- **Loss functions** (Phase A/B: L1+TV | Phase C: L1+TV+VGG Perceptual)
- **Convergence** (epochs to plateau, final metrics)
- **Decoder impact** (same perceptual loss, different decoders: Phase C V1 vs V2)

### Run training locally

From the **project root** (directory containing `src/`):

```bash
pip install -r requirements.txt
python -m src.train --config configs/phase_a/baseline_l1_uv_tv.yaml
# or
python -m src.train --config configs/phase_b/baseline_unet_l1_uv_tv.yaml
```

Optional CLI overrides: `--batch-size`, `--epochs`, `--resume`, `--output-root`, `--data-dir`, etc. See [`src/config.py`](src/config.py).

### Run on an HPC (Slurm)

1. Copy/sync this repo + dataset to the cluster.
2. Install deps in a venv (uncomment `source .../activate` in the Slurm script if needed).
3. Edit the chosen Slurm file: set **`#SBATCH --partition`**, **`--account`**, and optionally **QoS**.
4. From project root:

   ```bash
   sbatch scripts/slurm/train_phase_a_h100.slurm
   ```

   For A100 (e.g. smaller batch): `sbatch scripts/slurm/train_phase_a_a100.slurm`

   Phase B equivalents: `sbatch scripts/slurm/train_phase_b_h100.slurm` or `sbatch scripts/slurm/train_phase_b_a100.slurm`

5. **Resume** after timeout:

   ```bash
   RESUME=/abs/path/to/experiments/phase_a/baseline_l1_uv_tv/runs/<run>/checkpoints/last.pt sbatch scripts/slurm/train_phase_a_h100.slurm
   ```

First run **downloads** `dinov2-large` from Hugging Face — ensure nodes have **network** or set `HF_HOME` / cache on shared storage.

### What to expect in outputs

Under `experiments/phase_a/baseline_l1_uv_tv/runs/<run_version>/`:

- **`config_resolved.yaml`** — effective config.
- **`checkpoints/`** — `epoch_*.pt`, `last.pt`, `best.pt`.
- **`logs/metrics.jsonl`** — train loss + val SSIM / MS-SSIM / PSNR / UV errors per epoch.

Slurm writes **`slurm-phase-a-h100-<jobid>.out`** (and `.err`) in the **submission working directory** (see `#SBATCH --output` in the script).

### Upper-bound diagnostic

```bash
python scripts/eval_upper_bound_sample.py --num-batches 2 --batch-size 4
```

Uses GT UV + [`uv_dewarp.dewarp_with_uv`](uv_dewarp.py) vs flat GT — **not** the neural net. Useful to gauge how tight the geometric ceiling is for UV-based dewarp on your split.

---

## GDrive Checkpoint Backup with rclone

All phase checkpoints (A, B, C) are automatically uploaded to **Google Drive** via `rclone` in the Slurm epilogue. This ensures training artifacts are safely backed up to persistent remote storage.

### Infrastructure

- **rclone remote:** Configured as `shu` (Google Drive)
- **Root folder on Drive:** `Legendary-DeWarper/`
- **Phase subfolders:** `Phase A/`, `Phase B/`, `Phase C/`, `Phase D/`, etc.
- **Per-run structure:** `<Phase>//<run_version>/checkpoints/` containing `epoch_*.pt`, `last.pt`, `best.pt`
- **Local scratch:** Checkpoints written to `/tmp/$USER/legendary_dewarper_checkpoints/<run_version>/checkpoints/` during training; rclone copies only `epoch_*.pt`, `last.pt`, and `best.pt` (not all intermediate files).

### Slurm Integration

Each Slurm script (Phase A/B/C/D) includes a **post-job epilogue** that:

```bash
[GDRIVE_UPLOAD] START <ISO_timestamp>
[GDRIVE_UPLOAD] SRC=<local_checkpoint_dir>
[GDRIVE_UPLOAD] DEST=shu:Legendary-DeWarper/Phase <N>/<run_version>/checkpoints
[GDRIVE_UPLOAD] CMD: rclone copy --progress --include epoch_\*.pt --include last.pt --include best.pt --exclude \* <src> <dest>
Transferred:  <bytes> / <total_bytes>, <percentage>, <rate>, ETA <time>
...
[GDRIVE_UPLOAD] SUCCESS <ISO_timestamp>
```

This ensures uploads are **auditable** in Slurm output logs with explicit timing and rclone progress metrics.

### Known Issue: GDrive UI Lag

**Bug discovered:** Files successfully copied to Google Drive via rclone **do not immediately appear in the GDrive web UI**, even after refreshing. However, the files **exist on the remote** and are **fully accessible** via rclone.

**Workaround:** Always trust **rclone terminal commands** over the UI. The remote filesystem is authoritative.

### Verification Commands

```bash
# List all Phase C runs
rclone ls shu:Legendary-DeWarper/Phase\ C/

# Verify specific run checkpoints exist
rclone ls shu:Legendary-DeWarper/Phase\ C/20260505_021622_job719821/checkpoints/

# Download a checkpoint to verify accessibility
rclone copyto shu:Legendary-DeWarper/Phase\ C/20260505_021622_job719821/checkpoints/best.pt ./local_best.pt

# Check total size of a phase
rclone du -s shu:Legendary-DeWarper/Phase\ C/
```

### Phase A/B/C Backup Status

✓ **Phase A:** All runs uploaded to `shu:Legendary-DeWarper/Phase A/`  
✓ **Phase B:** All runs uploaded to `shu:Legendary-DeWarper/Phase B/`  
✓ **Phase C:** All runs uploaded to `shu:Legendary-DeWarper/Phase C/`
  - Job 719821 (20260505_021622): 4 files, ~13.7 GiB
  - Job 719822 (20260505_022425): 4 files, ~13.8 GiB

---

## Planned work (roadmap)

**Completed phases:** Phase A ✓, Phase B ✓, Phase C ✓

These items reflect **Phase D and beyond**:

| Direction | Notes |
|-----------|--------|
| **Phase D** | Next training phase with refined hyperparameters or loss formulation (decided after A/B/C analysis). Analyze convergence curves, best metrics, and failure modes across phases to inform design. |
| **Cross-phase analysis** | Extract metrics (val_ssim_masked, val_psnr_masked, final loss, best epoch) from all phase logs; generate comparison tables and convergence plots (Phase A vs B vs C). |
| **Loss experiments** | Add folders under `experiments/phase_a/<new_loss_slug>/` with new YAMLs — e.g. **SSIM** or **perceptual (VGG)** as training objectives, **masked** where applicable. Compare against `baseline_l1_uv_tv` on the same val split. |
| **Augmentation** | Start with **photometric** aug on **RGB only** (jitter, blur). **Geometric** aug only if every modality (including UV / grid) transforms consistently — easy to break correspondence. |
| **Architecture refinements** | Optional **refinement** branch after `grid_sample`; coarse-to-fine or DocTr-style **learned upsampling** for the field; optional **depth** auxiliary if used. |
| **Metrics / reporting** | MS-SSIM, masked metrics, UV error already logged; extend with **OCR CER**, **line straightness**, or **Local Distortion** if required by the writeup. Side-by-side figures: model vs GT vs `uv_dewarp` upper bound. |
| **Pretrained / scale** | Phase A uses **DINOv2-L**; ablations (other sizes, frozen backbone epochs) can be toggled in YAML. |
| **Large ablation grids** | Explicitly **deprioritized** — prefer a small number of clear comparisons over exhaustive sweeps. |

Paper-inspired ideas to cite or borrow incrementally: **UVDoc** (grid/dual task), **DocTr** (transformer + learned decode), **DewarpNet** (3D→2D story), **DocGeoNet** (extra geometric cues — needs labels you may not have).

---

## Docs and assignments

- Course **PDF** and original long-form README text may live outside this file; keep submission requirements (report, weights, figures) aligned with the **current** course handout.
- Additional experiment-layout detail: [`experiments/README.md`](experiments/README.md).

---

## License

See [`LICENSE`](LICENSE).
