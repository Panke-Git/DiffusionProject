"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: validation_random_cond.py
    @Time: 2025/12/21 00:58
    @Email: None
"""


import random
from pathlib import Path
import torch
from PIL import Image
from .train_utils import *

def _set_all_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _to_uint8_img(x_chw: torch.Tensor) -> Image.Image:
    x = ((x_chw + 1) / 2).clamp(0, 1)
    x = (x * 255.0).round().to(torch.uint8)
    arr = x.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)

def _save_2x2(condA, outA, condB, outB, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    im1 = _to_uint8_img(condA)
    im2 = _to_uint8_img(outA)
    im3 = _to_uint8_img(condB)
    im4 = _to_uint8_img(outB)

    W, H = im1.size
    canvas = Image.new("RGB", (W * 2, H * 2))
    canvas.paste(im1, (0, 0))
    canvas.paste(im2, (W, 0))
    canvas.paste(im3, (0, H))
    canvas.paste(im4, (W, H))
    canvas.save(save_path)

@torch.no_grad()
def cond_effect_experiment(
    cfg,
    device,
    ckpt_path,
    out_dir,
    idx_a: int = None,
    idx_b: int = None,
    seed: int = 42,
    use_ema: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, diffusion = build_model_and_diffusion(cfg, device)
    model.to(device).eval()
    load_checkpoint(model, ckpt_path, device=device, optimizer=None, use_ema=use_ema, strict=True)

    _, val_loader = build_dataloaders(cfg)
    ds = val_loader.dataset
    n = len(ds)

    if idx_a is None or idx_b is None:
        idx_a, idx_b = random.sample(range(n), 2)

    a = ds[idx_a]
    b = ds[idx_b]
    condA = a["input"].unsqueeze(0).to(device)  # (1,3,H,W)
    condB = b["input"].unsqueeze(0).to(device)

    _set_all_seeds(seed)
    outA = diffusion.sample(model, condA).clamp(-1, 1)  # (1,3,H,W)

    _set_all_seeds(seed)
    outB = diffusion.sample(model, condB).clamp(-1, 1)

    mad = (outA - outB).abs().mean().item()
    print(f"[CondEffect] idx_a={idx_a}, idx_b={idx_b}, seed={seed}, mean_abs_diff(outA,outB)={mad:.6f}")

    save_path = out_dir / f"cond_effect_seed{seed}_a{idx_a}_b{idx_b}.png"
    _save_2x2(
        condA[0].cpu(), outA[0].cpu(),
        condB[0].cpu(), outB[0].cpu(),
        save_path
    )
    print(f"[CondEffect] saved: {save_path}")


