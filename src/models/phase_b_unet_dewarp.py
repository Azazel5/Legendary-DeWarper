"""
Phase B DINOv2 dewarp model with a U-Net style decoder.

This keeps the pretrained DINOv2 encoder from Phase A, but replaces the
plain upsample/refine decoder with a decoder that has explicit skip
connections within the decoding path.
"""

from __future__ import annotations

from typing import Dict, Tuple

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


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1, groups: int = 8):
        super().__init__()
        gn_groups = max(1, min(groups, out_ch))
        while out_ch % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(gn_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDecoderHead(nn.Module):
    """U-Net style decoder with internal skip connections."""

    def __init__(
        self,
        in_dim: int,
        out_hw: Tuple[int, int],
        decoder_channels: Tuple[int, int, int] = (128, 192, 256),
        flow_scale: float = 0.35,
    ):
        super().__init__()
        self.out_h, self.out_w = out_hw
        self.flow_scale = flow_scale

        c0, c1, c2 = decoder_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, c0, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if c0 % 8 == 0 else 1, c0),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            ConvGNAct(c0, c1, stride=2),
            ConvGNAct(c1, c1),
        )
        self.down2 = nn.Sequential(
            ConvGNAct(c1, c2, stride=2),
            ConvGNAct(c2, c2),
        )
        self.bottleneck = nn.Sequential(
            ConvGNAct(c2, c2),
            ConvGNAct(c2, c2),
        )
        self.up2 = nn.Sequential(
            ConvGNAct(c2 + c1, c1),
            ConvGNAct(c1, c1),
        )
        self.up1 = nn.Sequential(
            ConvGNAct(c1 + c0, c0),
            ConvGNAct(c0, c0),
        )
        self.post_full = nn.Sequential(
            ConvGNAct(c0, c0),
            ConvGNAct(c0, c0),
        )
        self.flow_head = nn.Conv2d(c0, 2, kernel_size=3, padding=1)
        self.uv_head = nn.Conv2d(c0, 2, kernel_size=3, padding=1)

    def forward(self, patch_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        patch_map [B, C, Hp, Wp] -> flow [B,2,H,W], uv [B,2,H,W]
        """
        stem = self.stem(patch_map)
        down1 = self.down1(stem)
        down2 = self.down2(down1)
        bottleneck = self.bottleneck(down2)

        up2 = F.interpolate(bottleneck, size=down1.shape[-2:], mode="bilinear", align_corners=True)
        up2 = torch.cat([up2, down1], dim=1)
        up2 = self.up2(up2)

        up1 = F.interpolate(up2, size=stem.shape[-2:], mode="bilinear", align_corners=True)
        up1 = torch.cat([up1, stem], dim=1)
        up1 = self.up1(up1)

        x = F.interpolate(up1, size=(self.out_h, self.out_w), mode="bilinear", align_corners=True)
        x = self.post_full(x)

        flow = torch.tanh(self.flow_head(x)) * self.flow_scale
        uv = torch.sigmoid(self.uv_head(x))
        return flow, uv


class Dinov2UNetDewarpNet(nn.Module):
    def __init__(
        self,
        model_id: str = "facebook/dinov2-large",
        img_size: Tuple[int, int] = (518, 518),
        freeze_backbone: bool = False,
        decoder_channels: Tuple[int, int, int] = (128, 192, 256),
        flow_scale: float = 0.35,
        use_refinement_head: bool = False,
        refinement_channels: Tuple[int, ...] = (64, 32),
    ):
        super().__init__()
        self.img_h, self.img_w = img_size
        self.backbone = Dinov2Model.from_pretrained(model_id)
        hidden = self.backbone.config.hidden_size
        self.decoder = UNetDecoderHead(
            in_dim=hidden,
            out_hw=img_size,
            decoder_channels=decoder_channels,
            flow_scale=flow_scale,
        )
        self.use_refinement_head = bool(use_refinement_head)
        if self.use_refinement_head:
            rc = tuple(refinement_channels)
            self.refinement = nn.Sequential(
                nn.Conv2d(6, rc[0], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(rc[0], rc[1], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(rc[1], 3, kernel_size=3, padding=1),
            )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
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

        if self.use_refinement_head:
            inp = torch.cat([dewarped, pixel_values], dim=1)
            residual = self.refinement(inp)
            dewarped = (dewarped + residual).clamp(-1.0, 1.0)

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
        ref = sum(p.numel() for p in getattr(self, "refinement", nn.Module()).parameters() if p.requires_grad) if self.use_refinement_head else 0
        return {"encoder_trainable": enc, "decoder_trainable": dec, "refinement_trainable": ref}