"""YAML + CLI configuration for Phase A training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase A/B DINOv2 document dewarp training")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--output-root", type=str, default=None, help="Root for experiments/<phase>/... (default: experiments/)")
    p.add_argument("--loss-slug", type=str, default=None)
    p.add_argument("--run-version", type=str, default=None, help="Subfolder under runs/; auto if omitted")
    p.add_argument("--resume", type=str, default=None, help="Path to last.pt or epoch checkpoint")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.config:
        cfg_path = Path(args.config)
        cfg = load_yaml(cfg_path)
    else:
        cfg = {}

    overrides = {
        "data_dir": args.data_dir,
        "output_root": args.output_root,
        "loss_slug": args.loss_slug,
        "run_version": args.run_version,
        "resume": args.resume,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "model_id": args.model_id,
        "device": args.device,
    }
    cfg = merge_dict(cfg, {k: v for k, v in overrides.items() if v is not None})

    # Defaults if still missing
    defaults = {
        "phase": "phase_a",
        "loss_slug": "baseline_l1_uv_tv",
        "data_dir": "renders/synthetic_data_pitch_sweep",
        "train_split": 0.8,
        "batch_size": 4,
        "num_workers": 4,
        "img_size": [518, 518],
        "model_id": "facebook/dinov2-large",
        "freeze_backbone_epochs": 0,
        "epochs": 100,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "optimizer": "adamw",
        "grad_clip_norm": 1.0,
        "reconstruction_weight": 1.0,
        "uv_weight": 0.5,
        "smoothness_weight": 0.01,
        "amp": True,
        "amp_dtype": "bfloat16",
        "seed": 42,
        "primary_metric": "val_ssim_masked",
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    return cfg


def save_resolved_config(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
