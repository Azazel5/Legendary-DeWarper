#!/usr/bin/env python3
"""Evaluate a trained dewarping checkpoint on the deterministic validation split.

Outputs:
- run_dir/eval/panels/*.png            (Input, Pred UV, Rectified-by-UV, GT)
- run_dir/eval/eval_metrics.json       (aggregate metrics)
- run_dir/eval/per_sample_metrics.jsonl
- run_dir/flow_diagrams/*.png          (flow visualizations; configurable)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loader import get_dataloaders
from src.metrics import compute_metrics_package, denormalize_imagenet
from src.models.dinov2_dewarp import Dinov2DewarpNet
from src.models.phase_b_unet_dewarp import Dinov2UNetDewarpNet
from uv_dewarp import dewarp_with_uv


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    kw = {"map_location": "cpu"}
    try:
        return torch.load(path, **kw, weights_only=False)
    except TypeError:
        return torch.load(path, **kw)


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    img_size = tuple(cfg.get("img_size", [518, 518]))
    if len(img_size) == 1:
        img_size = (img_size[0], img_size[0])

    phase = str(cfg.get("phase", "phase_a"))
    if phase == "phase_b":
        model = Dinov2UNetDewarpNet(
            model_id=cfg.get("model_id", "facebook/dinov2-large"),
            img_size=img_size,
            freeze_backbone=False,
            decoder_channels=tuple(cfg.get("decoder_channels", [128, 192, 256])),
            flow_scale=float(cfg.get("flow_scale", 0.35)),
            use_refinement_head=bool(cfg.get("use_refinement_head", False)),
            refinement_channels=tuple(cfg.get("refinement_channels", [64, 32])),
        )
    else:
        model = Dinov2DewarpNet(
            model_id=cfg.get("model_id", "facebook/dinov2-large"),
            img_size=img_size,
            freeze_backbone=False,
        )
    return model


def tensor_to_rgb01(x: torch.Tensor) -> np.ndarray:
    """Input [3,H,W] ImageNet-normalized tensor -> [H,W,3] float in [0,1]."""
    x01 = denormalize_imagenet(x.unsqueeze(0))[0]
    return x01.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def uv_to_rgb01(uv: torch.Tensor) -> np.ndarray:
    """UV [2,H,W] -> RGB visualization [H,W,3] with B=0."""
    h, w = uv.shape[-2:]
    out = torch.zeros(3, h, w, dtype=uv.dtype, device=uv.device)
    out[:2] = uv.clamp(0.0, 1.0)
    return out.detach().cpu().permute(1, 2, 0).numpy()


def flow_to_rgb01(flow: torch.Tensor) -> np.ndarray:
    """Flow [2,H,W] -> RGB visualization using angle as hue, magnitude as value."""
    fx = flow[0].detach().cpu().numpy()
    fy = flow[1].detach().cpu().numpy()
    mag = np.sqrt(fx * fx + fy * fy)
    ang = np.arctan2(fy, fx)

    hue = (ang + np.pi) / (2.0 * np.pi)
    sat = np.ones_like(hue, dtype=np.float32)
    vmax = np.percentile(mag, 99) + 1e-8
    val = np.clip(mag / vmax, 0.0, 1.0)

    hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)
    rgb = plt.cm.hsv(hsv[..., 0])[..., :3]
    rgb = rgb * hsv[..., 2:3]
    return np.clip(rgb, 0.0, 1.0)


def save_panel(path: Path, inp: np.ndarray, uv_rgb: np.ndarray, rectified: np.ndarray, gt: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(inp)
    axes[0].set_title("Input")
    axes[1].imshow(uv_rgb)
    axes[1].set_title("Pred UV")
    axes[2].imshow(rectified)
    axes[2].set_title("Rectified (uv_dewarp)")
    axes[3].imshow(gt)
    axes[3].set_title("Ground Truth")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate dewarping model and save visual outputs")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (best.pt recommended)")
    p.add_argument("--config", type=str, default=None, help="Optional config yaml override")
    p.add_argument("--data-dir", type=str, default=None, help="Optional dataset path override")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-panels", type=int, default=32, help="Max number of per-sample panel PNGs")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu")
    p.add_argument("--out-dir", type=str, default=None, help="Default is <run_dir>/eval")
    p.add_argument("--flow-dir", type=str, default=None, help="Default is <run_dir>/flow_diagrams")
    p.add_argument("--no-flip-v", action="store_true", help="Disable V-axis flip in uv_dewarp")
    return p.parse_args()


def merge_cfg(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not override:
        return base
    out = dict(base)
    out.update(override)
    return out


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    run_dir = ckpt_path.parent.parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "eval"
    flow_dir = Path(args.flow_dir).resolve() if args.flow_dir else run_dir / "flow_diagrams"
    panel_dir = out_dir / "panels"

    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    flow_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(ckpt_path)
    ckpt_cfg = ckpt.get("cfg", {})
    file_cfg = load_yaml(Path(args.config)) if args.config else None
    cfg = merge_cfg(ckpt_cfg, file_cfg)

    data_dir_cfg = args.data_dir or cfg.get("data_dir", "renders/synthetic_data_pitch_sweep")
    data_dir = PROJECT_ROOT / data_dir_cfg if not Path(data_dir_cfg).is_absolute() else Path(data_dir_cfg)

    seed = int(cfg.get("seed", 42))
    train_split = float(cfg.get("train_split", 0.8))
    img_size = tuple(cfg.get("img_size", [518, 518]))
    if len(img_size) == 1:
        img_size = (img_size[0], img_size[0])

    _, val_loader = get_dataloaders(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        train_split=train_split,
        use_uv=True,
        use_border=False,
        img_size=img_size,
        num_workers=args.num_workers,
        shuffle=False,
        random_seed=seed,
    )

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)
    model = model.to(device)
    model.eval()

    aggregate: Dict[str, float] = {}
    num_batches = 0
    per_sample_rows = []
    saved_panels = 0
    seen_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            gt = batch["ground_truth"].to(device, non_blocking=True)
            gt_uv = batch["uv"].to(device, non_blocking=True)
            uv_mask = batch["uv_mask"].to(device, non_blocking=True)
            filenames = batch["filename"]

            out = model(rgb)
            pred_uv = out["pred_uv"]
            dewarped = out["dewarped"]
            flow = out.get("flow")

            batch_metrics = compute_metrics_package(
                dewarped,
                gt,
                uv_mask,
                pred_uv=pred_uv,
                gt_uv=gt_uv,
            )
            for k, v in batch_metrics.items():
                aggregate[k] = aggregate.get(k, 0.0) + float(v)
            num_batches += 1

            bsz = rgb.shape[0]
            for i in range(bsz):
                seen_samples += 1
                rgb01 = tensor_to_rgb01(rgb[i].cpu())
                gt01 = tensor_to_rgb01(gt[i].cpu())
                uv_rgb = uv_to_rgb01(pred_uv[i].cpu())
                mask_np = (uv_mask[i, 0].detach().cpu().numpy() > 0.5)

                pred_uv_np = pred_uv[i].detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)
                rgb_u8 = (rgb01 * 255.0).astype(np.uint8)
                rect_u8 = dewarp_with_uv(
                    rgb_u8,
                    pred_uv_np,
                    out_size=img_size[0],
                    mask=mask_np,
                    flip_v=not bool(args.no_flip_v),
                )
                rect01 = rect_u8.astype(np.float32) / 255.0

                fname = str(filenames[i])

                # Per-sample metrics (computed from model's dewarped tensor to stay
                # consistent with training-time metrics implementation).
                sample_metrics = compute_metrics_package(
                    dewarped[i : i + 1],
                    gt[i : i + 1],
                    uv_mask[i : i + 1],
                    pred_uv=pred_uv[i : i + 1],
                    gt_uv=gt_uv[i : i + 1],
                )
                row = {"filename": fname, **sample_metrics}
                per_sample_rows.append(row)

                if saved_panels < args.max_panels:
                    panel_path = panel_dir / f"{saved_panels:04d}_{fname}.png"
                    save_panel(panel_path, rgb01, uv_rgb, rect01, gt01, title=fname)
                    saved_panels += 1

                if flow is not None and flow.shape[1] == 2:
                    flow_rgb = flow_to_rgb01(flow[i].cpu())
                    flow_path = flow_dir / f"{fname}.png"
                    plt.imsave(flow_path, flow_rgb)

    if num_batches == 0:
        raise RuntimeError("Validation loader produced zero batches")

    metrics_out: Dict[str, Any] = {}
    for k, v in aggregate.items():
        metrics_out[k] = float(v) / float(num_batches)

    metrics_out.update(
        {
            "num_val_batches": int(num_batches),
            "num_val_samples": int(seen_samples),
            "saved_panels": int(saved_panels),
            "checkpoint": str(ckpt_path),
            "train_split": train_split,
            "seed": seed,
            "flow_dir": str(flow_dir),
            "eval_dir": str(out_dir),
        }
    )

    with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    with open(out_dir / "per_sample_metrics.jsonl", "w", encoding="utf-8") as f:
        for row in per_sample_rows:
            f.write(json.dumps(row) + "\n")

    print("Evaluation complete")
    print(f"checkpoint: {ckpt_path}")
    print(f"eval_dir:   {out_dir}")
    print(f"flow_dir:   {flow_dir}")
    for k in sorted(metrics_out.keys()):
        v = metrics_out[k]
        if isinstance(v, float) and not math.isnan(v):
            print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
