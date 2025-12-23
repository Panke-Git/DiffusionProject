# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
# [MOD] Turned into a runnable underwater RGB conditional DDPM trainer.
# [MOD] Removed torchvision dependency (save_image / transforms) to avoid torch/torchvision binary mismatch issues.

import sys  # [MOD] make local imports work when running from elsewhere
from pathlib import Path as _Path  # [MOD]
_THIS_DIR = _Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import os
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets import UnderwaterPairDataset  # [MOD]
from Model_condition import UNet            # [MOD]
from Diffusion_condition import GaussianDiffusionTrainer_cond, GaussianDiffusionSampler_cond  # [MOD]


def set_seed(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denorm_to_01(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
    return (x.clamp(-1, 1) + 1.0) * 0.5


def tensor_to_pil(x_01: torch.Tensor) -> Image.Image:
    """
    x_01: (3,H,W) in [0,1]
    """
    x = (x_01.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def save_triplet(cond, pred, gt, path: Path):
    """
    Save a concatenated image: input | pred | gt
    cond/pred/gt: (3,H,W) in [-1,1]
    """
    c = tensor_to_pil(denorm_to_01(cond))
    p = tensor_to_pil(denorm_to_01(pred))
    g = tensor_to_pil(denorm_to_01(gt))

    W, H = c.size
    canvas = Image.new("RGB", (W * 3, H))
    canvas.paste(c, (0, 0))
    canvas.paste(p, (W, 0))
    canvas.paste(g, (W * 2, 0))
    canvas.save(path)


def psnr(pred_01: torch.Tensor, target_01: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    mse = torch.mean((pred_01 - target_01) ** 2, dim=(1, 2, 3)).clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)


def _gaussian_window(window_size: int, sigma: float, device):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def ssim(pred_01: torch.Tensor, target_01: torch.Tensor, window_size: int = 11, sigma: float = 1.5, eps: float = 1e-12) -> torch.Tensor:
    """
    Lightweight SSIM (per-image) for RGB, averaged over channels.
    pred_01/target_01 in [0,1]
    """
    device = pred_01.device
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    g = _gaussian_window(window_size, sigma, device)
    kx = g.view(1, 1, 1, -1)
    ky = g.view(1, 1, -1, 1)

    def blur(x):
        B, C, H, W = x.shape
        x = F.pad(x, (window_size // 2, window_size // 2, window_size // 2, window_size // 2), mode="reflect")
        x = F.conv2d(x, kx.expand(C, 1, 1, window_size), groups=C)
        x = F.conv2d(x, ky.expand(C, 1, window_size, 1), groups=C)
        return x

    mu_x = blur(pred_01)
    mu_y = blur(target_01)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = blur(pred_01 * pred_01) - mu_x2
    sigma_y2 = blur(target_01 * target_01) - mu_y2
    sigma_xy = blur(pred_01 * target_01) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + eps)
    return ssim_map.mean(dim=(1, 2, 3))


@torch.no_grad()
def run_validation(
    sampler: GaussianDiffusionSampler_cond,
    val_loader: DataLoader,
    device,
    max_batches: int = 2,
) -> Dict[str, float]:
    sampler.model.eval()
    n = 0
    psnr_sum = 0.0
    ssim_sum = 0.0

    for bi, batch in enumerate(val_loader):
        if bi >= max_batches:
            break
        cond = batch["cond"].to(device)
        gt = batch["gt"].to(device)

        pred = sampler(cond)  # [-1,1]
        pred_01 = denorm_to_01(pred)
        gt_01 = denorm_to_01(gt)

        psnr_b = psnr(pred_01, gt_01).mean().item()
        ssim_b = ssim(pred_01, gt_01).mean().item()

        bs = cond.shape[0]
        n += bs
        psnr_sum += psnr_b * bs
        ssim_sum += ssim_b * bs

    if n == 0:
        return {"psnr": 0.0, "ssim": 0.0}

    return {"psnr": psnr_sum / n, "ssim": ssim_sum / n}


@torch.no_grad()
def save_preview_images(
    sampler: GaussianDiffusionSampler_cond,
    val_loader: DataLoader,
    out_dir: Path,
    device,
    num_images: int = 8,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler.model.eval()

    saved = 0
    for batch in val_loader:
        cond = batch["cond"].to(device)
        gt = batch["gt"].to(device)
        names = batch["name"]

        pred = sampler(cond)  # [-1,1]

        for i in range(cond.shape[0]):
            if saved >= num_images:
                return
            save_triplet(cond[i], pred[i], gt[i], out_dir / f"{names[i]}_input_pred_gt.png")
            saved += 1


def main():
    parser = argparse.ArgumentParser()
    # [MOD] dataset args for your directory structure
    parser.add_argument("--data_root", type=str, default="/public/home/hnust15874739861/pro/publicdata/LSUI19", help="Root contains Train/Val folders.")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    # diffusion / model
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    # train
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./runs/cond_ddpm_uie")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--val_max_batches", type=int, default=2)  # [MOD] limit val sampling cost
    parser.add_argument("--preview_num", type=int, default=8)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    img_dir = save_dir / "previews"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # datasets
    root = Path(args.data_root)
    train_ds = UnderwaterPairDataset(
        input_dir=str(root / "Train" / "input"),
        gt_dir=str(root / "Train" / "GT"),
        image_size=args.image_size,
        is_train=True,
    )
    val_ds = UnderwaterPairDataset(
        input_dir=str(root / "Val" / "input"),
        gt_dir=str(root / "Val" / "GT"),
        image_size=args.image_size,
        is_train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=min(args.batch_size, 8), shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # model + diffusion
    net_model = UNet(
        T=args.timesteps,
        ch=args.ch,
        ch_mult=(1, 2, 2, 2),
        attn=(3,),
        num_res_blocks=2,
        dropout=args.dropout,
        in_channels=6,   # [MOD]
        out_channels=3,  # [MOD]
    ).to(device)

    trainer = GaussianDiffusionTrainer_cond(net_model, args.beta_1, args.beta_T, args.timesteps).to(device)
    sampler = GaussianDiffusionSampler_cond(net_model, args.beta_1, args.beta_T, args.timesteps).to(device)

    optimizer = optim.AdamW(net_model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = {"loss": float("inf"), "psnr": -1.0, "ssim": -1.0}

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        net_model.train()
        running = 0.0

        for it, batch in enumerate(train_loader, start=1):
            cond = batch["cond"].to(device)
            gt = batch["gt"].to(device)

            loss = trainer(gt, cond)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                avg = running / it
                print(f"[Epoch {epoch:03d}] step={global_step} loss={avg:.6f}")

        # save last epoch ckpt
        torch.save(net_model.state_dict(), ckpt_dir / f"ckpt_epoch_{epoch:03d}.pt")

        # validation
        if (epoch % args.val_every) == 0:
            metrics = run_validation(sampler, val_loader, device, max_batches=args.val_max_batches)
            print(f"[Val] epoch={epoch:03d} PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.4f}")

            # [MOD] best checkpoints
            epoch_loss = running / max(1, len(train_loader))
            if epoch_loss < best["loss"]:
                best["loss"] = epoch_loss
                torch.save(net_model.state_dict(), ckpt_dir / "best_loss.pt")
                print(f"[SAVE] best_loss.pt (loss={epoch_loss:.6f})")

            if metrics["psnr"] > best["psnr"]:
                best["psnr"] = metrics["psnr"]
                torch.save(net_model.state_dict(), ckpt_dir / "best_psnr.pt")
                print(f"[SAVE] best_psnr.pt (psnr={metrics['psnr']:.4f})")

            if metrics["ssim"] > best["ssim"]:
                best["ssim"] = metrics["ssim"]
                torch.save(net_model.state_dict(), ckpt_dir / "best_ssim.pt")
                print(f"[SAVE] best_ssim.pt (ssim={metrics['ssim']:.4f})")

            # [MOD] preview images each val
            save_preview_images(sampler, val_loader, img_dir / f"epoch_{epoch:03d}", device, num_images=args.preview_num)

    # [MOD] After training, generate previews using best models
    best_out = save_dir / "best_results"
    best_out.mkdir(parents=True, exist_ok=True)
    for tag in ["best_psnr", "best_ssim", "best_loss"]:
        p = ckpt_dir / f"{tag}.pt"
        if p.exists():
            net_model.load_state_dict(torch.load(p, map_location=device))
            save_preview_images(sampler, val_loader, best_out / tag, device, num_images=max(args.preview_num, 10))
            print(f"[DONE] Saved sample results for {tag} -> {best_out / tag}")
        else:
            print(f"[WARN] {p} not found, skip.")


if __name__ == "__main__":
    main()