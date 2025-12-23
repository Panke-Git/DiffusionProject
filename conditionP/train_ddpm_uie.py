"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: train_ddpm_uie.py.py
    @Time: 2025/12/23 23:32
    @Email: None
"""
import os
import time
import yaml
import random
import argparse

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from dataset_uie import UnderwaterPairDataset
from eps_unet import EpsUNet
from diffusion_core_full import GaussianDiffusionCore
from metrics import psnr, ssim


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_ckpt(path, model, optim, epoch, step, best):
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "step": step,
        "best": best
    }, path)


@torch.no_grad()
def validate(diffusion, model, loader, cfg, device, out_dir, epoch):
    diffusion.eval()
    model.eval()

    max_batches = int(cfg["train"]["val_max_batches"])
    steps = int(cfg["train"]["sample_steps"])
    eta = float(cfg["train"]["eta"])

    psnr_list = []
    ssim_list = []

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break

        inp = batch["input"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        extra = torch.ones((inp.shape[0], 1, inp.shape[2], inp.shape[3]), device=device, dtype=inp.dtype)

        pred = diffusion.sample_loop_ddim(
            shape=gt.shape,
            cond=inp,
            extra_cond=extra,
            steps=steps,
            eta=eta,
            return_all=False
        )

        psnr_list.append(psnr(pred, gt))
        ssim_list.append(ssim(pred, gt))

        if bi == 0:
            grid = torch.cat([inp, pred, gt], dim=0)
            save_path = os.path.join(out_dir, "val_ep%03d.png" % epoch)
            save_image((grid + 1) * 0.5, save_path, nrow=inp.shape[0])

    psnr_m = torch.cat(psnr_list).mean().item() if len(psnr_list) else 0.0
    ssim_m = torch.cat(ssim_list).mean().item() if len(ssim_list) else 0.0

    diffusion.train()
    model.train()
    return psnr_m, ssim_m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ddpm_uie.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 1234)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    exp_dir = os.path.join(cfg["train"]["save_dir"], cfg["exp_name"])
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    img_dir = os.path.join(exp_dir, "images")
    ensure_dir(ckpt_dir)
    ensure_dir(img_dir)

    train_ds = UnderwaterPairDataset(
        cfg["data"]["train_input_dir"],
        cfg["data"]["train_gt_dir"],
        img_size=cfg["data"]["img_size"],
        augment=cfg["data"]["augment"]
    )
    val_ds = UnderwaterPairDataset(
        cfg["data"]["val_input_dir"],
        cfg["data"]["val_gt_dir"],
        img_size=cfg["data"]["img_size"],
        augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(int(cfg["train"]["batch_size"]), 4),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=True,
        drop_last=False
    )

    cond_ch = int(cfg["model"]["cond_channels"])
    img_ch = int(cfg["model"]["img_channels"])
    in_ch = cond_ch + img_ch

    model = EpsUNet(
        in_ch=in_ch,
        out_ch=img_ch,
        base_ch=int(cfg["model"]["base_channels"]),
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=int(cfg["model"]["num_res_blocks"]),
        attn_resolutions=tuple(cfg["model"]["attn_resolutions"]),
        dropout=float(cfg["model"]["dropout"]),
        img_size=int(cfg["data"]["img_size"])
    ).to(device)

    diffusion = GaussianDiffusionCore(
        eps_model=model,
        timesteps=int(cfg["diffusion"]["timesteps"]),
        beta_schedule=str(cfg["diffusion"]["beta_schedule"]),
        linear_start=float(cfg["diffusion"]["linear_start"]),
        linear_end=float(cfg["diffusion"]["linear_end"]),
        loss_type=str(cfg["diffusion"]["loss_type"]),
        clip_denoised=bool(cfg["diffusion"]["clip_denoised"])
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"])
    )

    use_amp = bool(cfg["train"]["amp"]) and (device == "cuda")
    scaler = GradScaler(enabled=use_amp)

    best = {"loss": 1e9, "psnr": -1.0, "ssim": -1.0}
    step = 0

    print("[INFO] device=%s train=%d val=%d" % (device, len(train_ds), len(val_ds)))
    print("[INFO] exp_dir=%s" % exp_dir)

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        diffusion.train()

        t0 = time.time()
        running = 0.0

        for batch in train_loader:
            inp = batch["input"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            extra = torch.ones((inp.shape[0], 1, inp.shape[2], inp.shape[3]), device=device, dtype=inp.dtype)

            with autocast(enabled=use_amp):
                loss = diffusion(x_start=gt, cond=inp, extra_cond=extra)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            grad_clip = float(cfg["train"]["grad_clip"])
            if grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optim)
            scaler.update()

            step += 1
            running += loss.item()

            if step % int(cfg["train"]["log_interval"]) == 0:
                avg = running / int(cfg["train"]["log_interval"])
                running = 0.0
                print("[E%03d][S%07d] loss=%.4f" % (epoch, step, avg))
                if avg < best["loss"]:
                    best["loss"] = avg
                    save_ckpt(os.path.join(ckpt_dir, "best_loss.pt"), model, optim, epoch, step, best)

        if epoch % int(cfg["train"]["val_interval"]) == 0:
            v_psnr, v_ssim = validate(diffusion, model, val_loader, cfg, device, img_dir, epoch)
            print("[VAL][E%03d] PSNR=%.3f SSIM=%.4f time=%.1fs" % (epoch, v_psnr, v_ssim, time.time() - t0))

            if v_psnr > best["psnr"]:
                best["psnr"] = v_psnr
                save_ckpt(os.path.join(ckpt_dir, "best_psnr.pt"), model, optim, epoch, step, best)
            if v_ssim > best["ssim"]:
                best["ssim"] = v_ssim
                save_ckpt(os.path.join(ckpt_dir, "best_ssim.pt"), model, optim, epoch, step, best)

        save_ckpt(os.path.join(ckpt_dir, "last.pt"), model, optim, epoch, step, best)

    print("[DONE] Best:", best)


if __name__ == "__main__":
    main()

