from __future__ import annotations
import os
import argparse
import time
import math
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.paired_dataset import PairedImageDataset
from models.unet import UNetModel
from diffusion.gaussian_diffusion import GaussianDiffusion, DiffusionConfig
from utils.misc import set_seed, get_device, ensure_dir
from utils.ema import EMA
from utils.metrics import psnr as psnr_fn, ssim as ssim_fn
from utils.image import denorm_to_0_1, save_tensor_image


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(path: str, model: nn.Module, ema: Optional[EMA], cfg: Dict[str, Any], meta: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    ckpt = {
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "config": cfg,
        "meta": meta,
    }
    torch.save(ckpt, path)


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
def sample_images(
    diffusion: GaussianDiffusion,
    model: nn.Module,
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


@torch.no_grad()
def validate(
    cfg: Dict[str, Any],
    diffusion: GaussianDiffusion,
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    out_dir: str,
) -> Dict[str, float]:
    model.eval()

    # 1) Val loss: noise prediction MSE (fast)
    losses = []
    for batch in tqdm(val_loader, desc=f"ValLoss E{epoch}", leave=False):
        cond = batch["cond"].to(device)
        gt = batch["gt"].to(device)
        b = gt.shape[0]
        t = torch.randint(0, diffusion.cfg.timesteps, (b,), device=device).long()
        loss = diffusion.training_losses(model, gt, cond, t)
        losses.append(loss.item())
    val_loss = float(sum(losses) / max(1, len(losses)))

    # 2) PSNR/SSIM: requires sampling (can be slow)
    vcfg = cfg["validation"]
    num_samples = int(vcfg.get("num_samples", 50))
    sampler = str(vcfg.get("sampler", "ddim"))
    sample_steps = int(vcfg.get("sample_steps", 50))
    eta = float(vcfg.get("eta", 0.0))

    psnrs = []
    ssims = []

    # iterate again, but only up to num_samples images if set
    processed = 0
    save_visuals = bool(vcfg.get("save_visuals", True))
    visuals_max = int(vcfg.get("visuals_max", 8))
    visuals_saved = 0
    vis_dir = os.path.join(out_dir, "val_visuals", f"epoch_{epoch:04d}")
    if save_visuals:
        ensure_dir(vis_dir)

    for batch in tqdm(val_loader, desc=f"ValSample E{epoch}", leave=False):
        cond = batch["cond"].to(device)
        gt = batch["gt"].to(device)
        names = batch["name"]
        b = gt.shape[0]

        # limit number of images for sampling metrics
        if num_samples > 0 and processed >= num_samples:
            break

        # If batch would exceed, truncate
        if num_samples > 0 and processed + b > num_samples:
            keep = num_samples - processed
            cond = cond[:keep]
            gt = gt[:keep]
            names = names[:keep]
            b = keep

        pred = sample_images(diffusion, model, cond, sampler=sampler, sample_steps=sample_steps, eta=eta)

        pred_01 = denorm_to_0_1(pred)
        gt_01 = denorm_to_0_1(gt)

        psnr_b = psnr_fn(pred_01, gt_01)  # (B,)
        ssim_b = ssim_fn(pred_01, gt_01)  # (B,)
        psnrs.extend(psnr_b.detach().cpu().tolist())
        ssims.extend(ssim_b.detach().cpu().tolist())

        if save_visuals and visuals_saved < visuals_max:
            for i in range(b):
                if visuals_saved >= visuals_max:
                    break
                name = names[i]
                save_tensor_image(denorm_to_0_1(cond[i]), os.path.join(vis_dir, f"{visuals_saved:03d}_{name}_cond.png"))
                save_tensor_image(denorm_to_0_1(pred[i]), os.path.join(vis_dir, f"{visuals_saved:03d}_{name}_pred.png"))
                save_tensor_image(denorm_to_0_1(gt[i]), os.path.join(vis_dir, f"{visuals_saved:03d}_{name}_gt.png"))
                visuals_saved += 1

        processed += b

    val_psnr = float(sum(psnrs) / max(1, len(psnrs))) if psnrs else float("nan")
    val_ssim = float(sum(ssims) / max(1, len(ssims))) if ssims else float("nan")

    return {"val_loss": val_loss, "val_psnr": val_psnr, "val_ssim": val_ssim}


@torch.no_grad()
def generate_split(
    cfg: Dict[str, Any],
    diffusion: GaussianDiffusion,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: str,
    sampler: str,
    sample_steps: int,
    eta: float,
):
    model.eval()
    ensure_dir(out_dir)
    for batch in tqdm(loader, desc=f"Generate -> {out_dir}"):
        cond = batch["cond"].to(device)
        names = batch["name"]
        pred = sample_images(diffusion, model, cond, sampler=sampler, sample_steps=sample_steps, eta=eta)
        for i in range(pred.shape[0]):
            save_tensor_image(pred[i], os.path.join(out_dir, names[i]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='/public/home/hnust15874739861/pro/DiffusionProject/underwater_cddpm_cddpm_2/configs/train.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = get_device()
    print(f"Using device: {device}")

    out_dir = cfg["paths"]["out_dir"]
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "checkpoints"))

    # datasets
    dcfg = cfg["data"]
    train_ds = PairedImageDataset(
        cfg["paths"]["train_input"],
        cfg["paths"]["train_gt"],
        image_size=dcfg["image_size"],
        random_crop=dcfg.get("random_crop", True),
        random_flip=dcfg.get("random_flip", True),
    )
    val_ds = PairedImageDataset(
        cfg["paths"]["val_input"],
        cfg["paths"]["val_gt"],
        image_size=dcfg["image_size"],
        random_crop=False,
        random_flip=False,
    )

    tcfg = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        num_workers=int(dcfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        num_workers=int(dcfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )

    # model + diffusion
    model = build_model(cfg).to(device)
    diffusion = build_diffusion(cfg).to(device)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    # AMP
    use_amp = bool(tcfg.get("mixed_precision", True)) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # EMA
    ema_cfg = tcfg.get("ema", {})
    ema_enabled = bool(ema_cfg.get("enabled", True))
    ema = EMA(model, decay=float(ema_cfg.get("decay", 0.9999))) if ema_enabled else None
    ema_update_after_step = int(ema_cfg.get("update_after_step", 0))
    ema_update_every = int(ema_cfg.get("update_every", 1))

    # best trackers
    best_ssim = -1e9
    best_psnr = -1e9
    best_loss = 1e9

    global_step = 0
    epochs = int(tcfg["epochs"])
    grad_clip = float(tcfg.get("grad_clip", 1.0))

    log_path = os.path.join(out_dir, "train_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n==== New Run ====: {time.ctime()} | seed={seed}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train E{epoch}/{epochs}")
        running_loss = 0.0

        for batch in pbar:
            cond = batch["cond"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            b = gt.shape[0]

            t = torch.randint(0, diffusion.cfg.timesteps, (b,), device=device).long()

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = diffusion.training_losses(model, gt, cond, t)

            scaler.scale(loss).backward()

            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # EMA update
            if ema is not None and global_step >= ema_update_after_step and (global_step % ema_update_every == 0):
                ema.update(model)

            global_step += 1
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), avg=running_loss / max(1, (global_step)))
        train_loss_epoch = float(running_loss / max(1, global_step))
        # ===== Validation =====
        vcfg = cfg["validation"]
        interval_epochs = int(vcfg.get("interval_epochs", 1))
        do_val = (epoch % interval_epochs == 0) or (epoch == epochs)

        if do_val:
            # Use EMA weights for validation if available
            if ema is not None:
                ema.apply_shadow(model)
            metrics = validate(cfg, diffusion, model, val_loader, device, epoch, out_dir)
            if ema is not None:
                ema.restore(model)

            val_loss = metrics["val_loss"]
            val_psnr = metrics["val_psnr"]
            val_ssim = metrics["val_ssim"]

            log_line = (
                f"Epoch {epoch}: "
                f"train_loss={train_loss_epoch:.6f}, "
                f"val_loss={val_loss:.6f}, "
                f"val_psnr={val_psnr:.4f}, "
                f"val_ssim={val_ssim:.4f}\n"
            )
            print(log_line.strip())
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_line)

            # save best-loss checkpoint (lower is better)
            if val_loss < best_loss:
                best_loss = val_loss
                path = os.path.join(out_dir, "checkpoints", "best_loss.pt")
                save_checkpoint(
                    path,
                    model,
                    ema,
                    cfg,
                    meta={"epoch": epoch, "global_step": global_step, "best_loss": best_loss, "best_psnr": best_psnr, "best_ssim": best_ssim},
                )
                print(f"[CKPT] Saved best_loss -> {path}")

            # save best-psnr checkpoint (higher is better)
            if not math.isnan(val_psnr) and val_psnr > best_psnr:
                best_psnr = val_psnr
                path = os.path.join(out_dir, "checkpoints", "best_psnr.pt")
                save_checkpoint(
                    path,
                    model,
                    ema,
                    cfg,
                    meta={"epoch": epoch, "global_step": global_step, "best_loss": best_loss, "best_psnr": best_psnr, "best_ssim": best_ssim},
                )
                print(f"[CKPT] Saved best_psnr -> {path}")

            # save best-ssim checkpoint (higher is better)
            if not math.isnan(val_ssim) and val_ssim > best_ssim:
                best_ssim = val_ssim
                path = os.path.join(out_dir, "checkpoints", "best_ssim.pt")
                save_checkpoint(
                    path,
                    model,
                    ema,
                    cfg,
                    meta={"epoch": epoch, "global_step": global_step, "best_loss": best_loss, "best_psnr": best_psnr, "best_ssim": best_ssim},
                )
                print(f"[CKPT] Saved best_ssim -> {path}")

    # ===== After training: generate outputs using the 3 best checkpoints =====
    sacfg = cfg.get("sampling_after_train", {})
    if bool(sacfg.get("enabled", True)):
        split = str(sacfg.get("split", "val")).lower()
        sampler = str(sacfg.get("sampler", "ddim"))
        sample_steps = int(sacfg.get("sample_steps", 50))
        eta = float(sacfg.get("eta", 0.0))

        gen_ds = val_ds if split == "val" else train_ds
        gen_loader = DataLoader(
            gen_ds,
            batch_size=int(tcfg["batch_size"]),
            shuffle=False,
            num_workers=int(dcfg.get("num_workers", 4)),
            pin_memory=True,
            drop_last=False,
        )

        ckpt_dir = os.path.join(out_dir, "checkpoints")
        ckpts = [
            ("best_ssim", os.path.join(ckpt_dir, "best_ssim.pt")),
            ("best_psnr", os.path.join(ckpt_dir, "best_psnr.pt")),
            ("best_loss", os.path.join(ckpt_dir, "best_loss.pt")),
        ]
        for tag, ckpt_path in ckpts:
            if not os.path.exists(ckpt_path):
                print(f"[WARN] Missing checkpoint: {ckpt_path}")
                continue

            print(f"\n[GEN] Loading {tag}: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")

            model.load_state_dict(ckpt["model"], strict=True)
            # if EMA exists in checkpoint, prefer EMA weights for generation
            if ckpt.get("ema") is not None and ema is not None:
                ema.load_state_dict(ckpt["ema"])
                ema.apply_shadow(model)

            model.to(device)
            out_gen = os.path.join(out_dir, "generated", tag)
            generate_split(cfg, diffusion, model, gen_loader, device, out_gen, sampler, sample_steps, eta)

            if ckpt.get("ema") is not None and ema is not None:
                ema.restore(model)

        print(f"\nDone. Generated images are under: {os.path.join(out_dir, 'generated')}")
    else:
        print("Sampling after train is disabled.")


if __name__ == "__main__":
    main()
