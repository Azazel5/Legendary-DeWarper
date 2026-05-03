#!/usr/bin/env python3
"""
Compare GT UV forward-warp upper bound (uv_dewarp) vs flat GT using same metrics as training.

Does not load the neural model — estimates the ceiling imposed by GT UV + bilinear splat.

Usage (from project root):
  python scripts/eval_upper_bound_sample.py --num-batches 2 --batch-size 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loader import get_dataloaders
from src.metrics import compute_metrics_package, denormalize_imagenet
from uv_dewarp import dewarp_with_uv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="renders/synthetic_data_pitch_sweep")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-batches", type=int, default=2)
    ap.add_argument("--img-size", type=int, nargs=2, default=[518, 518])
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    img_size = tuple(args.img_size)

    _, val_loader = get_dataloaders(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        train_split=0.8,
        use_uv=True,
        img_size=img_size,
        shuffle=False,
    )

    agg = {}
    count = 0
    for bi, batch in enumerate(val_loader):
        if bi >= args.num_batches:
            break
        rgb = batch["rgb"]
        gt = batch["ground_truth"]
        uv = batch["uv"]
        mask = batch["uv_mask"]

        bsz = rgb.shape[0]
        dewarp_tensors = []
        for i in range(bsz):
            # Denormalize RGB to uint8 for dewarp_with_uv (expects H,W,3 uint8)
            r01 = denormalize_imagenet(rgb[i : i + 1])[0].cpu().numpy().transpose(1, 2, 0)
            r_u8 = (np.clip(r01, 0, 1) * 255.0).astype(np.uint8)
            u = uv[i, 0].cpu().numpy()
            v = uv[i, 1].cpu().numpy()
            uv_hw = np.stack([u, v], axis=-1)
            m = mask[i, 0].cpu().numpy().astype(bool)
            out_u8 = dewarp_with_uv(r_u8, uv_hw, out_size=img_size[0], mask=m, flip_v=True)
            # Back to ImageNet-normalized tensor like GT
            t = torch.from_numpy(out_u8).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            t_norm = (t - mean) / std
            dewarp_tensors.append(t_norm)

        dew = torch.stack(dewarp_tensors, dim=0)
        m = compute_metrics_package(dew, gt, mask, pred_uv=None, gt_uv=None)
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        count += 1

    for k in list(agg.keys()):
        agg[k] /= max(count, 1)

    print("GT-UV forward dewarp vs flat GT (upper-bound diagnostic, mean over batches):", flush=True)
    for k in sorted(agg.keys()):
        print(f"  {k}: {agg[k]:.6f}", flush=True)


if __name__ == "__main__":
    main()
