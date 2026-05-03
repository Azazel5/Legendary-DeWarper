# Legendary-DeWarper

**Tufts Computer Vision — Final Project**

Learn a **geometric** dewarping model: from a **warped document RGB** (with background) predict a **UV / sampling field**, warp with differentiable `grid_sample`, and match the **flat ground truth** texture. **Focus on geometry, not lighting** — the README in the course materials and in `dataset_loader.py` explain why photometric MSE is misleading.

---

## Repository map

| Area | Role |
|------|------|
| [`dataset_loader.py`](dataset_loader.py) | `DocumentDataset`, `get_dataloaders`, ImageNet-normalized RGB/GT, optional **UV** (`[0,1]`, 2ch) and **`uv_mask`**, `UVReconstructionLoss`, `create_base_grid`, `MaskedL1Loss`, `SSIMLoss` |
| [`uv_dewarp.py`](uv_dewarp.py) | **Upper bound:** forward-warp RGB with **GT UV** (bilinear splat + hole fill). Baseline for “what perfect UV would look like” on a sample. |
| [`src/models/dinov2_dewarp.py`](src/models/dinov2_dewarp.py) | **Phase A model:** `facebook/dinov2-large` → conv decoder → **flow** (for `grid_sample`) + **UV** head; outputs dewarped tensor. |
| [`src/train.py`](src/train.py) | Training loop: AMP, checkpointing, resume, `metrics.jsonl`, Slurm-friendly logging. |
| [`src/metrics.py`](src/metrics.py) | Denorm to `[0,1]`; **SSIM / MS-SSIM / PSNR** (full + masked); **UV L1** (full + masked on foreground). |
| [`src/config.py`](src/config.py) | YAML + CLI overrides. |
| [`configs/phase_a/baseline_l1_uv_tv.yaml`](configs/phase_a/baseline_l1_uv_tv.yaml) | Default Phase A hyperparameters. |
| [`scripts/eval_upper_bound_sample.py`](scripts/eval_upper_bound_sample.py) | Batches: **GT-UV dewarp** vs flat GT (ceiling-style diagnostic, no trained model). |
| [`scripts/slurm/`](scripts/slurm/) | H100 / A100 **10h** Slurm jobs. |
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

### Run training locally

From the **project root** (directory containing `src/`):

```bash
pip install -r requirements.txt
python -m src.train --config configs/phase_a/baseline_l1_uv_tv.yaml
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

## Planned work (roadmap)

These items are **not** all implemented; they reflect the project direction and folder layout we set up for future runs.

| Direction | Notes |
|-----------|--------|
| **Loss experiments** | Add folders under `experiments/phase_a/<new_loss_slug>/` with new YAMLs — e.g. **SSIM** or **perceptual (VGG)** as training objectives, **masked** where applicable. Compare against `baseline_l1_uv_tv` on the same val split. |
| **Augmentation** | Start with **photometric** aug on **RGB only** (jitter, blur). **Geometric** aug only if every modality (including UV / grid) transforms consistently — easy to break correspondence. |
| **Architecture** | Optional **refinement** branch after `grid_sample`; coarse-to-fine or DocTr-style **learned upsampling** for the field; optional **depth** auxiliary if used. |
| **Metrics / reporting** | MS-SSIM, masked metrics, UV error already logged; extend with **OCR CER**, **line straightness**, or **Local Distortion** if required by the writeup. Side-by-side figures: model vs GT vs `uv_dewarp` upper bound. |
| **Pretrained / scale** | Phase A uses **DINOv2-L**; ablations (other sizes, frozen backbone epochs) can be toggled in YAML. |
| **Large ablation grids (“Phase E”)** | Explicitly **deprioritized** — prefer a small number of clear comparisons over exhaustive sweeps. |

Paper-inspired ideas to cite or borrow incrementally: **UVDoc** (grid/dual task), **DocTr** (transformer + learned decode), **DewarpNet** (3D→2D story), **DocGeoNet** (extra geometric cues — needs labels you may not have).

---

## Docs and assignments

- Course **PDF** and original long-form README text may live outside this file; keep submission requirements (report, weights, figures) aligned with the **current** course handout.
- Additional experiment-layout detail: [`experiments/README.md`](experiments/README.md).

---

## License

See [`LICENSE`](LICENSE).
