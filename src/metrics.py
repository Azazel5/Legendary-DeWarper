"""Image metrics in [0,1] space after ImageNet denormalization."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def imagenet_to_device(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    m = IMAGENET_MEAN.to(device)
    s = IMAGENET_STD.to(device)
    return m, s


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """x: [B,3,H,W] normalized -> [0,1] approximate (clamped)."""
    device = x.device
    m, s = imagenet_to_device(device)
    out = x * s + m
    return out.clamp(0.0, 1.0)


def psnr_batch(pred01: torch.Tensor, target01: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scalar mean PSNR over batch; images in [0,1]."""
    mse = F.mse_loss(pred01, target01, reduction="mean")
    if mse.item() < eps:
        return torch.tensor(100.0, device=pred01.device)
    return 10.0 * torch.log10(torch.tensor(1.0, device=pred01.device) / mse)


def psnr_masked_batch(
    pred01: torch.Tensor,
    target01: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """mask [B,1,H,W] in {0,1}."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    m = mask.expand_as(pred01)
    diff = (pred01 - target01) ** 2 * m
    denom = m.sum() * pred01.shape[1]
    denom = torch.clamp(denom, min=1.0)
    mse = diff.sum() / denom
    if mse.item() < eps:
        return torch.tensor(100.0, device=pred01.device)
    return 10.0 * torch.log10(torch.tensor(1.0, device=pred01.device) / mse)


def ssim_ms_ssim_batch(
    pred01: torch.Tensor,
    target01: torch.Tensor,
    data_range: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns mean SSIM and mean MS-SSIM over batch (requires pytorch_msssim)."""

    # pytorch_msssim expects NCHW
    s = ssim(pred01, target01, data_range=data_range, size_average=True)
    ms = ms_ssim(pred01, target01, data_range=data_range, size_average=True)
    return s, ms


def ssim_ms_ssim_masked_batch(
    pred01: torch.Tensor,
    target01: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0,
    bg_value: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply same soft masking: outside mask set channels to bg_value for both pred and target.
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    m = mask.expand_as(pred01)
    bg = torch.full_like(pred01, bg_value)
    p = pred01 * m + bg * (1.0 - m)
    t = target01 * m + bg * (1.0 - m)
    return ssim_ms_ssim_batch(p, t, data_range=data_range)


def uv_l1_masked(
    pred_uv: torch.Tensor,
    target_uv: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """pred/target [B,2,H,W], mask [B,1,H,W]."""
    diff = (pred_uv - target_uv).abs()
    if mask is None:
        return diff.mean()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    m = mask.expand_as(diff)
    denom = m.sum() * diff.shape[1]
    denom = torch.clamp(denom, min=1.0)
    return (diff * m).sum() / denom


def compute_metrics_package(
    pred_norm: torch.Tensor,
    gt_norm: torch.Tensor,
    mask: Optional[torch.Tensor],
    pred_uv: Optional[torch.Tensor] = None,
    gt_uv: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """All scalars on CPU as Python floats."""
    pred01 = denormalize_imagenet(pred_norm)
    gt01 = denormalize_imagenet(gt_norm)

    out: Dict[str, float] = {}
    out["psnr_full"] = float(psnr_batch(pred01, gt01).item())

    ssim_f, mssim_f = ssim_ms_ssim_batch(pred01, gt01)
    out["ssim_full"] = float(ssim_f.item())
    out["ms_ssim_full"] = float(mssim_f.item())

    if mask is not None:
        out["psnr_masked"] = float(psnr_masked_batch(pred01, gt01, mask).item())
        ssim_m, mssim_m = ssim_ms_ssim_masked_batch(pred01, gt01, mask)
        out["ssim_masked"] = float(ssim_m.item())
        out["ms_ssim_masked"] = float(mssim_m.item())
    else:
        out["psnr_masked"] = float("nan")
        out["ssim_masked"] = float("nan")
        out["ms_ssim_masked"] = float("nan")

    if pred_uv is not None and gt_uv is not None:
        out["uv_l1_full"] = float(torch.abs(pred_uv - gt_uv).mean().item())
        if mask is not None:
            out["uv_l1_masked"] = float(uv_l1_masked(pred_uv, gt_uv, mask).item())
        else:
            out["uv_l1_masked"] = float("nan")

    return out
