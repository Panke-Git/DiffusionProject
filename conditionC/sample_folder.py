# coding=utf-8
"""
    @Project: 
    @Author: PyCharm
    @FileName： sample_folder.py
    @Date：2025/12/25 16:14
    @Email: None
"""

import os
import argparse
import torch

from utils.misc import load_yaml, ensure_dir
from models.unet_ddpm import UNetModel, UNetConfig
from diffusion.gaussian_diffusion import GaussianDiffusion, DiffusionConfig
from utils.ema import EMA, EMAConfig
from train import generate_folder  # reuse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/uie_ddpm.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = int(cfg["data"]["image_size"])

    unet_cfg = UNetConfig(
        in_channels=int(cfg["model"]["in_channels"]),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        channel_mult=list(cfg["model"]["channel_mult"]),
        num_res_blocks=int(cfg["model"]["num_res_blocks"]),
        attn_resolutions=list(cfg["model"]["attn_resolutions"]),
        dropout=float(cfg["model"]["dropout"]),
        num_heads=int(cfg["model"]["num_heads"]),
        use_scale_shift_norm=bool(cfg["model"]["use_scale_shift_norm"]),
        image_size=image_size,
    )
    model = UNetModel(unet_cfg).to(device)

    diff_cfg = DiffusionConfig(
        timesteps=int(cfg["diffusion"]["timesteps"]),
        beta_schedule=str(cfg["diffusion"]["beta_schedule"]),
        objective=str(cfg["diffusion"]["objective"]),
        p2_loss_weight_gamma=float(cfg["diffusion"].get("p2_loss_weight_gamma", 0.0)),
        p2_loss_weight_k=float(cfg["diffusion"].get("p2_loss_weight_k", 1.0)),
    )
    diffusion = GaussianDiffusion(diff_cfg, device=device)

    ema = None
    if bool(cfg["ema"]["enable"]):
        ema_cfg = EMAConfig(
            enable=True,
            decay=float(cfg["ema"]["decay"]),
            update_every=int(cfg["ema"]["update_every"]),
            warmup_steps=int(cfg["ema"]["warmup_steps"]),
        )
        ema = EMA(model, ema_cfg)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])

    model_for_sample = ema.ema_model if ema is not None else model
    ensure_dir(args.out_dir)
    generate_folder(cfg, model_for_sample, diffusion, args.input_dir, args.out_dir, device, tag="manual")
    print(f"[Done] Saved enhanced images to: {args.out_dir}")

if __name__ == "__main__":
    main()