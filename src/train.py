"""
Phase A training: DINOv2 dewarp + checkpoint / resume + metrics.jsonl.
Run from project root: python -m src.train --config configs/phase_a/baseline_l1_uv_tv.yaml
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loader import UVReconstructionLoss, get_dataloaders

from src.config import build_config, parse_args, save_resolved_config
from src.metrics import compute_metrics_package
from src.models.dinov2_dewarp import Dinov2DewarpNet


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _try_cuda_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    dev = torch.cuda.get_device_properties(0)
    # BF16 tensor cores on Ampere+
    major = torch.cuda.get_device_capability()[0]
    return major >= 8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_run_dir(cfg: Dict[str, Any], resume_path: Optional[str]) -> Path:
    base = Path(cfg.get("output_root") or "experiments")
    phase = cfg["phase"]
    loss_slug = cfg["loss_slug"]
    if resume_path:
        ckpt = Path(resume_path).resolve()
        if ckpt.parent.name != "checkpoints":
            raise ValueError(
                f"--resume must be path like .../runs/<version>/checkpoints/last.pt; got {ckpt}"
            )
        return ckpt.parent.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ver = cfg.get("run_version") or f"{ts}_seed{cfg['seed']}_{_git_sha()}"
    return base / phase / loss_slug / "runs" / ver


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.pt")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: Optional[GradScaler]) -> Dict[str, Any]:
    kw = {"map_location": "cpu"}
    try:
        ckpt = torch.load(path, **kw, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, **kw)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: UVReconstructionLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    amp_dtype: torch.dtype,
    grad_clip: float,
    epoch: int,
    use_cuda_amp: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n = 0
    tic = time.time()

    for batch_idx, batch in enumerate(loader):
        rgb = batch["rgb"].to(device, non_blocking=True)
        gt = batch["ground_truth"].to(device, non_blocking=True)
        uv_gt = batch["uv"].to(device, non_blocking=True)
        mask = batch["uv_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_cuda_amp:
            with autocast(dtype=amp_dtype):
                out = model(rgb)
        else:
            out = model(rgb)
        dewarped = out["dewarped"]
        pred_uv = out["pred_uv"]
        flow = out["flow"]
        losses = criterion(
            pred_image=dewarped,
            target_image=gt,
            pred_uv=pred_uv,
            target_uv=uv_gt,
            flow=flow,
            mask=mask,
        )
        loss = losses["total"]

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item())
        n += 1

        if batch_idx % 50 == 0:
            elapsed = time.time() - tic
            print(
                f"  train epoch {epoch} batch {batch_idx}/{len(loader)} "
                f"loss={loss.item():.5f} elapsed_batch_loop={elapsed:.1f}s",
                flush=True,
            )
            tic = time.time()

    return {"train_loss": total_loss / max(n, 1)}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_cuda_amp: bool,
) -> Dict[str, float]:
    model.eval()
    agg: Dict[str, float] = {}
    count = 0

    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        gt = batch["ground_truth"].to(device, non_blocking=True)
        uv_gt = batch["uv"].to(device, non_blocking=True)
        mask = batch["uv_mask"].to(device, non_blocking=True)

        if use_cuda_amp:
            with autocast(dtype=amp_dtype):
                out = model(rgb)
        else:
            out = model(rgb)
        dewarped = out["dewarped"]
        pred_uv = out["pred_uv"]

        m = compute_metrics_package(dewarped, gt, mask, pred_uv=pred_uv, gt_uv=uv_gt)
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        count += 1

    for k in list(agg.keys()):
        agg[k] /= max(count, 1)
    return agg


def apply_backbone_freeze(model: Dinov2DewarpNet, freeze: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = not freeze


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)

    resume_path = cfg.get("resume") or args.resume
    run_dir = resolve_run_dir(cfg, resume_path)
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    save_resolved_config(cfg, run_dir / "config_resolved.yaml")

    device_s = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    use_bf16 = cfg.get("amp_dtype", "bfloat16").lower() == "bfloat16" and _try_cuda_bf16()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("=" * 72, flush=True)
    print("Phase A — DINOv2 dewarp training", flush=True)
    print(f"  host     : {platform.node()}", flush=True)
    print(f"  cwd      : {os.getcwd()}", flush=True)
    print(f"  project  : {PROJECT_ROOT}", flush=True)
    print(f"  device   : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})", flush=True)
    print(f"  torch    : {torch.__version__}, cuda: {torch.version.cuda}", flush=True)
    print(f"  git sha  : {_git_sha()}", flush=True)
    print(f"  run_dir  : {run_dir}", flush=True)
    print(f"  resume   : {resume_path or 'fresh'}", flush=True)
    print(f"  AMP      : {cfg.get('amp', True)} dtype={amp_dtype}", flush=True)
    print("=" * 72, flush=True)

    set_seed(int(cfg["seed"]))
    img_size = tuple(cfg["img_size"])
    if len(img_size) == 1:
        img_size = (img_size[0], img_size[0])

    train_loader, val_loader = get_dataloaders(
        data_dir=str(PROJECT_ROOT / cfg["data_dir"]) if not Path(cfg["data_dir"]).is_absolute() else cfg["data_dir"],
        batch_size=int(cfg["batch_size"]),
        train_split=float(cfg["train_split"]),
        use_uv=True,
        use_border=False,
        img_size=img_size,
        num_workers=int(cfg["num_workers"]),
    )
    print(f"  samples  : train={len(train_loader.dataset)} val={len(val_loader.dataset)}", flush=True)

    freeze_epochs = int(cfg.get("freeze_backbone_epochs", 0))
    model = Dinov2DewarpNet(
        model_id=cfg["model_id"],
        img_size=img_size,
        freeze_backbone=freeze_epochs > 0,
    ).to(device)

    groups = model.trainable_parameter_groups()
    print(f"  params   : encoder_trainable={groups['encoder_trainable']:,} decoder_trainable={groups['decoder_trainable']:,}", flush=True)

    criterion = UVReconstructionLoss(
        reconstruction_weight=float(cfg["reconstruction_weight"]),
        uv_weight=float(cfg["uv_weight"]),
        smoothness_weight=float(cfg["smoothness_weight"]),
        use_mask=True,
        loss_type="l1",
    )

    lr = float(cfg["lr"])
    wd = float(cfg["weight_decay"])
    if cfg.get("optimizer", "adamw").lower() == "adamw":
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)

    use_cuda_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=bool(use_cuda_amp and amp_dtype == torch.float16))

    start_epoch = 0
    best_metric = float("-inf")
    primary = cfg.get("primary_metric", "val_ssim_masked")

    if resume_path:
        ckpt = load_checkpoint(
            Path(resume_path),
            model,
            optimizer,
            scaler if (scaler is not None and scaler.is_enabled()) else None,
        )
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_metric = float(ckpt.get("best_metric", float("-inf")))
        print(f"  resumed  : start_epoch={start_epoch} best_{primary}={best_metric}", flush=True)

    epochs = int(cfg["epochs"])
    grad_clip = float(cfg.get("grad_clip_norm", 1.0))

    metrics_jsonl = log_dir / "metrics.jsonl"

    for epoch in range(start_epoch, epochs):
        epoch_tic = time.time()
        if freeze_epochs > 0:
            apply_backbone_freeze(model, epoch < freeze_epochs)

        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler if (scaler is not None and scaler.is_enabled()) else None,
            device,
            amp_dtype,
            grad_clip,
            epoch,
            use_cuda_amp,
        )
        val_metrics = validate(model, val_loader, device, amp_dtype, use_cuda_amp)

        primary_key = primary[4:] if primary.startswith("val_") else primary
        pm = val_metrics.get(primary_key)
        if pm is None or (isinstance(pm, float) and math.isnan(pm)):
            pm = val_metrics.get("ssim_masked", val_metrics["ssim_full"])
        # Higher is better for SSIM / PSNR used here
        is_better = pm > best_metric
        if is_better:
            best_metric = pm

        row = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "val_primary": pm,
            "best_primary": best_metric,
            "epoch_time_s": time.time() - epoch_tic,
        }
        with open(metrics_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"epoch {epoch:04d} | train_loss={train_stats['train_loss']:.5f} | "
            f"val_ssim_full={val_metrics['ssim_full']:.4f} val_ssim_masked={val_metrics.get('ssim_masked', float('nan')):.4f} | "
            f"val_ms_ssim_full={val_metrics['ms_ssim_full']:.4f} val_ms_ssim_masked={val_metrics.get('ms_ssim_masked', float('nan')):.4f} | "
            f"val_psnr_full={val_metrics['psnr_full']:.2f} val_psnr_masked={val_metrics.get('psnr_masked', float('nan')):.2f} | "
            f"uv_l1={val_metrics.get('uv_l1_masked', val_metrics.get('uv_l1_full', float('nan'))):.5f} | "
            f"primary={pm:.4f} best={best_metric:.4f} | {row['epoch_time_s']:.1f}s",
            flush=True,
        )

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "best_metric": best_metric,
            "primary_metric_name": primary,
            "cfg": cfg,
        }

        ep_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
        atomic_torch_save(payload, ep_path)
        sz_mb = ep_path.stat().st_size / (1024 * 1024)
        print(f"  saved checkpoint epoch_{epoch:04d}.pt ({sz_mb:.1f} MB)", flush=True)

        atomic_torch_save(payload, ckpt_dir / "last.pt")
        print(f"  saved last.pt ({ckpt_dir / 'last.pt'})", flush=True)

        if is_better:
            atomic_torch_save(payload, ckpt_dir / "best.pt")
            print(f"  saved best.pt (primary={primary} value={pm:.6f})", flush=True)

    print("Training finished.", flush=True)


if __name__ == "__main__":
    main()
