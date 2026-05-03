"""
DINOv2 encoder + convolutional decoder predicting sampling-grid residual + UV map.

Warp: dewarped = grid_sample(distorted_rgb, base_grid + flow_residual_permuted)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Dinov2Model


def _tokens_to_map(tokens: torch.Tensor, batch_size: int, dim: int) -> Tuple[torch.Tensor, int, int]:
    """tokens: [B, N, C] patch tokens only -> [B, C, Hp, Wp]."""
    n = tokens.shape[1]
    g = int(n ** 0.5)
    if g * g != n:
        raise ValueError(f"Patch count {n} is not a square grid")
    x = tokens.transpose(1, 2).reshape(batch_size, dim, g, g)
    return x, g, g


class DecoderHead(nn.Module):
    """Upsample patch grid to full resolution and predict flow + UV."""

    def __init__(
        self,
        in_dim: int,
        out_hw: Tuple[int, int],
        decoder_channels: Tuple[int, ...] = (256, 128, 64),
        flow_scale: float = 0.35,
    ):
        super().__init__()
        self.out_h, self.out_w = out_hw
        self.flow_scale = flow_scale

        c1, c2, c3 = decoder_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.flow_head = nn.Conv2d(c3, 2, kernel_size=3, padding=1)
        self.uv_head = nn.Conv2d(c3, 2, kernel_size=3, padding=1)

    def forward(self, patch_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        patch_map [B, C, Hp, Wp] -> flow [B,2,H,W], uv [B,2,H,W]
        """
        x = self.stem(patch_map)
        x = F.interpolate(x, size=(self.out_h, self.out_w), mode="bilinear", align_corners=True)
        x = self.refine(x)
        flow = torch.tanh(self.flow_head(x)) * self.flow_scale
        uv = torch.sigmoid(self.uv_head(x))
        return flow, uv


class Dinov2DewarpNet(nn.Module):
    def __init__(
        self,
        model_id: str = "facebook/dinov2-large",
        img_size: Tuple[int, int] = (518, 518),
        freeze_backbone: bool = False,
        decoder_channels: Tuple[int, ...] = (256, 128, 64),
        flow_scale: float = 0.35,
    ):
        super().__init__()
        self.img_h, self.img_w = img_size
        self.backbone = Dinov2Model.from_pretrained(model_id)
        hidden = self.backbone.config.hidden_size
        self.decoder = DecoderHead(
            in_dim=hidden,
            out_hw=img_size,
            decoder_channels=decoder_channels,
            flow_scale=flow_scale,
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: [B, 3, H, W] ImageNet-normalized (same as dataset / HF processor).

        Returns:
            dewarped [B,3,H,W], pred_uv [B,2,H,W], flow [B,2,H,W] (same resolution as grid residual),
            grid [B,H,W,2] sampling grid passed to grid_sample.
        """
        b, _, h, w = pixel_values.shape
        if h != self.img_h or w != self.img_w:
            raise ValueError(f"pixel_values spatial size {(h, w)} != model {(self.img_h, self.img_w)}")

        out = self.backbone(pixel_values=pixel_values)
        tokens = out.last_hidden_state[:, 1:, :]
        patch_map, _, _ = _tokens_to_map(tokens, b, tokens.shape[-1])
        flow_chw, pred_uv = self.decoder(patch_map)

        base = self._base_grid(b, h, w, pixel_values.device)
        grid = base + flow_chw.permute(0, 2, 3, 1)

        dewarped = F.grid_sample(
            pixel_values,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return {
            "dewarped": dewarped,
            "pred_uv": pred_uv,
            "flow": flow_chw,
            "grid": grid,
        }

    def _base_grid(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def trainable_parameter_groups(self) -> Dict[str, int]:
        enc = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        dec = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        return {"encoder_trainable": enc, "decoder_trainable": dec}
