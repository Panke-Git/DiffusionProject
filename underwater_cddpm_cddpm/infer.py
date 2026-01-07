from __future__ import annotations
import os
import argparse
import yaml
from typing import Dict, Any, List
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from models.unet import UNetModel
from diffusion.gaussian_diffusion import GaussianDiffusion, DiffusionConfig
from utils.misc import get_device, ensure_dir
from utils.ema import EMA
from utils.image import save_tensor_image


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: Dict[str, Any]) -> UNetModel:
    m = cfg["model"]
    d = cfg["data"]
    return UNetModel(
        image_size=d["image_size"],
        in_channels=m["in_channels"],
        cond_channels=m["cond_channels"],
        model_channels=m["model_channels"],
        out_channels=m["out_channels"],
        num_res_blocks=m["num_res_blocks"],
        attention_resolutions=m["attention_resolutions"],
        dropout=m.get("dropout", 0.0),
        channel_mult=m.get("channel_mult", [1, 2, 4, 8]),
        num_heads=m.get("num_heads", 4),
        use_checkpoint=m.get("use_checkpoint", False),
        use_scale_shift_norm=m.get("use_scale_shift_norm", True),
    )


def build_diffusion(cfg: Dict[str, Any]) -> GaussianDiffusion:
    dc = cfg["diffusion"]
    dcfg = DiffusionConfig(
        timesteps=int(dc["timesteps"]),
        beta_schedule=str(dc.get("beta_schedule", "linear")),
        beta_start=float(dc.get("beta_start", 1e-4)),
        beta_end=float(dc.get("beta_end", 2e-2)),
        clip_denoised=bool(dc.get("clip_denoised", True)),
        loss_type=str(dc.get("loss_type", "eps")),
    )
    return GaussianDiffusion(dcfg)


@torch.no_grad()
def sample(
    diffusion: GaussianDiffusion,
    model: torch.nn.Module,
    cond: torch.Tensor,
    sampler: str = "ddim",
    sample_steps: int = 50,
    eta: float = 0.0,
):
    device = cond.device
    b, c, h, w = cond.shape
    shape = (b, 3, h, w)
    sampler = sampler.lower()
    if sampler == "ddpm":
        return diffusion.p_sample_loop(model, shape=shape, cond=cond, device=device)
    elif sampler == "ddim":
        return diffusion.ddim_sample_loop(model, shape=shape, cond=cond, device=device, steps=sample_steps, eta=eta)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")


def list_images(path: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    if os.path.isfile(path):
        return [path]
    names = []
    for fn in os.listdir(path):
        if fn.lower().endswith(exts):
            names.append(os.path.join(path, fn))
    names.sort()
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="image file or folder")
    parser.add_argument("--output", type=str, required=True, help="output folder")
    parser.add_argument("--sampler", type=str, default=None, help="override: ddpm|ddim")
    parser.add_argument("--steps", type=int, default=None, help="override ddim steps")
    parser.add_argument("--eta", type=float, default=None, help="override ddim eta")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    print(f"Using device: {device}")

    model = build_model(cfg).to(device)
    diffusion = build_diffusion(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    # If EMA exists, prefer EMA weights (common in diffusion)
    ema = None
    if ckpt.get("ema") is not None:
        ema = EMA(model, decay=0.9999)
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow(model)

    model.eval()

    dcfg = cfg["data"]
    image_size = int(dcfg["image_size"])

    sampler = args.sampler or cfg.get("sampling_after_train", {}).get("sampler", cfg.get("validation", {}).get("sampler", "ddim"))
    steps = args.steps if args.steps is not None else int(cfg.get("sampling_after_train", {}).get("sample_steps", cfg.get("validation", {}).get("sample_steps", 50)))
    eta = args.eta if args.eta is not None else float(cfg.get("sampling_after_train", {}).get("eta", cfg.get("validation", {}).get("eta", 0.0)))

    in_list = list_images(args.input)
    ensure_dir(args.output)

    for p in tqdm(in_list, desc="Infer"):
        img = Image.open(p).convert("RGB")
        img = img.resize((image_size, image_size), resample=Image.BICUBIC)
        cond = TF.to_tensor(img) * 2.0 - 1.0
        cond = cond.unsqueeze(0).to(device)

        pred = sample(diffusion, model, cond, sampler=sampler, sample_steps=steps, eta=eta)
        pred = pred[0].detach().cpu()

        name = os.path.basename(p)
        save_tensor_image(pred, os.path.join(args.output, name))

    print(f"Done. Saved to: {args.output}")


if __name__ == "__main__":
    main()
