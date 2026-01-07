from __future__ import annotations

import os
import argparse
import time
import math
import json
import random
import pytz
from datetime import datetime

import yaml
from typing import Dict, Any, Tuple, Optional, List

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
from utils.image import denorm_to_0_1
from utils.grid import make_triplet_grid, save_pil
from utils.plot import plot_train_val_loss


def compute_sum_metric(psnr: float, ssim: float, val_loss: float) -> float:
    """
    你示例里的 SUM=120.056... 看起来不是简单 psnr+ssim。
    我这里给一个通用方案：你可以按自己想要的定义改。
    常见定义：
      - sum = psnr + 100*ssim
      - sum = psnr + 100*ssim - 10*val_loss
    下面用：psnr + 100*ssim（和你示例量级更接近）
    """
    return float(psnr + 100.0 * ssim)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(
    path: str,
    model: nn.Module,
    ema: Optional[EMA],
    cfg: Dict[str, Any],
    meta: Dict[str, Any],
    best_records: Optional[Dict[str, Any]] = None,
):
    """Save checkpoint.

    Notes:
    - If EMA is enabled, we save EMA-shadow weights into ckpt["model"] to ensure inference quality.
      Raw (non-EMA) weights are additionally stored in ckpt["model_raw"].
    - best_records is stored both as dict and as a pretty JSON string for readability.
    """
    ensure_dir(os.path.dirname(path))

    model_raw = model.state_dict()
    model_to_save = model_raw

    if ema is not None:
        try:
            ema.apply_shadow(model)
            model_to_save = model.state_dict()
        finally:
            ema.restore(model)

    ckpt = {
        "model": model_to_save,
        "model_raw": model_raw,
        "ema": ema.state_dict() if ema is not None else None,
        "config": cfg,
        "meta": meta,
        "best_records": best_records,
        "best_records_json": json.dumps(best_records, ensure_ascii=False, indent=2) if best_records is not None else None,
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
    sample_steps: int = 200,
    eta: float = 0.0,
):
    device = cond.device
    b, _, h, w = cond.shape
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
) -> Dict[str, float]:
    model.eval()

    # 1) val_loss
    losses = []
    for batch in tqdm(val_loader, desc=f"ValLoss E{epoch}", leave=False):
        cond = batch["cond"].to(device)
        gt = batch["gt"].to(device)
        b = gt.shape[0]
        t = torch.randint(0, diffusion.cfg.timesteps, (b,), device=device).long()
        loss = diffusion.training_losses(model, gt, cond, t)
        losses.append(loss.item())
    val_loss = float(sum(losses) / max(1, len(losses)))

    # 2) PSNR/SSIM（不保存 val_visuals）
    vcfg = cfg["validation"]
    num_samples = int(vcfg.get("num_samples", 50))
    sampler = str(vcfg.get("sampler", "ddim"))
    sample_steps = int(vcfg.get("sample_steps", 200))
    eta = float(vcfg.get("eta", 0.0))

    psnrs, ssims = [], []
    processed = 0

    # 让每个 epoch 的评估更稳定
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    for batch in tqdm(val_loader, desc=f"ValSample E{epoch}", leave=False):
        cond = batch["cond"].to(device)
        gt = batch["gt"].to(device)
        b = gt.shape[0]

        if num_samples > 0 and processed >= num_samples:
            break

        pred = sample_images(diffusion, model, cond, sampler=sampler, sample_steps=sample_steps, eta=eta)

        pred_01 = denorm_to_0_1(pred)
        gt_01 = denorm_to_0_1(gt)
        psnrs.extend(psnr_fn(pred_01, gt_01).detach().cpu().tolist())
        ssims.extend(ssim_fn(pred_01, gt_01).detach().cpu().tolist())

        processed += b

    val_psnr = float(sum(psnrs) / max(1, len(psnrs))) if psnrs else float("nan")
    val_ssim = float(sum(ssims) / max(1, len(ssims))) if ssims else float("nan")

    return {"val_loss": val_loss, "val_psnr": val_psnr, "val_ssim": val_ssim}


def _timestamp() -> str:
    beijing_tz = pytz.timezone("Asia/Shanghai")
    now_local = datetime.now(beijing_tz)
    return now_local.strftime("%Y%m%d_%H%M%S")


@torch.no_grad()
def make_final_grids(
    cfg: Dict[str, Any],
    diffusion: GaussianDiffusion,
    model: nn.Module,
    ema: Optional[EMA],
    val_ds: PairedImageDataset,
    device: torch.device,
    run_dir: str,
):
    sacfg = cfg.get("sampling_after_train", {})
    if not bool(sacfg.get("enabled", True)):
        return

    sampler = str(sacfg.get("sampler", "ddim"))
    sample_steps = int(sacfg.get("sample_steps", 200))
    eta = float(sacfg.get("eta", 0.0))
    num_pick = int(sacfg.get("num_pick", 10))

    seed = int(cfg.get("seed", cfg.get("project", {}).get("seed", 42)))
    rng = random.Random(seed + 999)
    idxs = list(range(len(val_ds)))
    rng.shuffle(idxs)
    idxs = idxs[: min(num_pick, len(idxs))]

    inputs, gts = [], []
    for i in idxs:
        item = val_ds[i]
        inputs.append(item["cond"])
        gts.append(item["gt"])

    cond_batch = torch.stack(inputs, dim=0).to(device)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    grid_dir = os.path.join(run_dir, "grids")
    ensure_dir(grid_dir)

    ckpts = [
        ("best_ssim", os.path.join(ckpt_dir, "best_ssim.pt")),
        ("best_psnr", os.path.join(ckpt_dir, "best_psnr.pt")),
        ("best_loss", os.path.join(ckpt_dir, "best_loss.pt")),
    ]

    for tag, ckpt_path in ckpts:
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Missing checkpoint for grid: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)

        # ✅ 优先用 ckpt 里保存的 EMA 权重生成
        if ckpt.get("ema") is not None and ema is not None:
            ema.load_state_dict(ckpt["ema"])
            ema.apply_shadow(model)

        pred = sample_images(diffusion, model, cond_batch, sampler=sampler, sample_steps=sample_steps, eta=eta)

        if ckpt.get("ema") is not None and ema is not None:
            ema.restore(model)

        preds = [pred[i].detach().cpu() for i in range(pred.shape[0])]
        grid = make_triplet_grid(
            inputs=[t.cpu() for t in inputs],
            gts=[t.cpu() for t in gts],
            preds=preds,
        )

        meta = ckpt.get("meta", {})
        vpsnr = meta.get("val_psnr", None)
        vssim = meta.get("val_ssim", None)
        vloss = meta.get("val_loss", None)

        def _fmt(x, nd=4):
            if x is None:
                return "NA"
            try:
                return f"{float(x):.{nd}f}"
            except Exception:
                return "NA"

        fname = (
            f"grid_{tag}_psnr{_fmt(vpsnr,2)}_ssim{_fmt(vssim,4)}_loss{_fmt(vloss,6)}_"
            f"{sampler}{sample_steps}.png"
        )
        save_pil(grid, os.path.join(grid_dir, fname))
        print(f"[GRID] Saved: {os.path.join(grid_dir, fname)}")


def main():
    parser = argparse.ArgumentParser()
    # 获取config参数以及版本号；
    parser.add_argument("--config", type=str, default='/public/home/hnust15874739861/pro/DiffusionProject/underwater_cddpm_cddpm/configs/train.yaml')
    parser.add_argument("--version", type=str, default=None, help="Override project.version")
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = int(cfg.get("seed", cfg.get("project", {}).get("seed", 42)))
    set_seed(seed)

    device = get_device()
    print(f"Using device: {device}")

    # ===== Run dir: ./runs/{version}/{timestamp} =====
    # 获取关于project的一些信息
    proj = cfg.get("project", {})
    run_root = str(proj.get("run_root", "runs"))
    version = args.version or str(proj.get("version", "V01"))
    out_dir = os.path.join(run_root, version, _timestamp())
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "checkpoints"))

    # 保存本次训练用的配置
    with open(os.path.join(out_dir, "train.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    # 更新 cfg.paths.out_dir（内部用）
    if "paths" in cfg:
        cfg["paths"]["out_dir"] = out_dir
    else:
        cfg["paths"] = {"out_dir": out_dir}

    dcfg = cfg["data"]

    train_ds = PairedImageDataset(
        cfg["paths"]["train_input"],
        cfg["paths"]["train_gt"],
        image_size=dcfg["image_size"],
        random_crop=dcfg.get("random_crop", True),
        random_flip=dcfg.get("random_flip", True),
        resize_only=False,
    )

    # ✅ Val：直接 resize（不 crop）
    val_random_crop = bool(dcfg.get("val_random_crop", False))
    val_ds = PairedImageDataset(
        cfg["paths"]["val_input"],
        cfg["paths"]["val_gt"],
        image_size=dcfg["image_size"],
        random_crop=val_random_crop,
        random_flip=False,
        resize_only=not val_random_crop,
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

    # ✅ Val 用 batch_size=1：评估更稳定，也更符合“随机挑 10 张”的逻辑
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        num_workers=int(dcfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(cfg).to(device)
    diffusion = build_diffusion(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    use_amp = bool(tcfg.get("mixed_precision", True)) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ema_cfg = tcfg.get("ema", {})
    ema_enabled = bool(ema_cfg.get("enabled", True))
    ema = EMA(model, decay=float(ema_cfg.get("decay", 0.9999))) if ema_enabled else None
    ema_update_after_step = int(ema_cfg.get("update_after_step", 0))
    ema_update_every = int(ema_cfg.get("update_every", 1))

    best_ssim, best_psnr, best_loss = -1e9, -1e9, 1e9

    # ===== Best-records tracking (saved into checkpoints as JSON-like dict) =====
    best_records: Dict[str, Any] = {
        "PSNR": {"top_psnr": -1e9, "top_psnr_data": None},
        "SSIM": {"top_ssim": -1e9, "top_ssim_data": None},
        "SUM":  {"top_sum":  -1e9, "top_sum_data":  None},
        "LOSS": {"top_loss":  1e9, "top_loss_data": None},
    }
    best_records_path = os.path.join(out_dir, "checkpoints", "best_records.json")
    global_step = 0
    epochs = int(tcfg["epochs"])
    grad_clip = float(tcfg.get("grad_clip", 1.0))

    log_path = os.path.join(out_dir, "train_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n==== New Run ====: {time.ctime()} | seed={seed}\n")

    history: List[Dict[str, Any]] = []
    history_path = os.path.join(out_dir, "history.json")

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train E{epoch}/{epochs}")
        running_loss, n_batches = 0.0, 0

        for batch in pbar:
            cond = batch["cond"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            b = gt.shape[0]

            t = torch.randint(0, diffusion.cfg.timesteps, (b,), device=device).long()

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = diffusion.training_losses(model, gt, cond, t)

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if ema is not None and global_step >= ema_update_after_step and (global_step % ema_update_every == 0):
                ema.update(model)

            global_step += 1
            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=loss.item(), avg=running_loss / max(1, n_batches))

        train_loss_epoch = float(running_loss / max(1, n_batches))

        # Val
        vcfg = cfg["validation"]
        interval_epochs = int(vcfg.get("interval_epochs", 1))
        do_val = (epoch % interval_epochs == 0) or (epoch == epochs)

        if do_val:
            if ema is not None:
                ema.apply_shadow(model)
            metrics = validate(cfg, diffusion, model, val_loader, device, epoch)
            if ema is not None:
                ema.restore(model)

            val_loss = metrics["val_loss"]
            val_psnr = metrics["val_psnr"]
            val_ssim = metrics["val_ssim"]

            log_line = f"Epoch {epoch}: val_loss={val_loss:.6f}, val_psnr={val_psnr:.4f}, val_ssim={val_ssim:.4f}\n"
            print(log_line.strip())
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_line)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss_epoch,
                    "val_loss": val_loss,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
            )
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            plot_train_val_loss(history, os.path.join(out_dir, "loss_curve.png"))

            # ===== Best-records (PSNR/SSIM/SUM/LOSS) + save best checkpoints (only 3 pt) =====
            sum_score = compute_sum_metric(val_psnr, val_ssim, val_loss)
            epoch_data = {
                "epoch": epoch,
                "train_loss": train_loss_epoch,
                "val_loss": val_loss,
                "psnr": val_psnr,
                "ssim": val_ssim,
                "sum": sum_score,
            }

            # Update SUM record (no dedicated checkpoint to keep "only 3 pt" policy)
            if not math.isnan(sum_score) and sum_score > float(best_records["SUM"]["top_sum"]):
                best_records["SUM"]["top_sum"] = float(sum_score)
                best_records["SUM"]["top_sum_data"] = dict(epoch_data)

            # Persist best_records snapshot as json for quick view (optional but useful)
            with open(best_records_path, "w", encoding="utf-8") as f:
                json.dump(best_records, f, ensure_ascii=False, indent=2)

            # LOSS (best_loss.pt)
            if val_loss < best_loss:
                best_loss = val_loss
                best_records["LOSS"]["top_loss"] = float(val_loss)
                best_records["LOSS"]["top_loss_data"] = dict(epoch_data)

                with open(best_records_path, "w", encoding="utf-8") as f:
                    json.dump(best_records, f, ensure_ascii=False, indent=2)

                save_checkpoint(
                    os.path.join(out_dir, "checkpoints", "best_loss.pt"),
                    model,
                    ema,
                    cfg,
                    meta={
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_loss": best_loss,
                        "best_psnr": best_psnr,
                        "best_ssim": best_ssim,
                        "best_sum": float(best_records["SUM"]["top_sum"]),
                        "val_loss": val_loss,
                        "val_psnr": val_psnr,
                        "val_ssim": val_ssim,
                        "val_sum": sum_score,
                        "sampler": str(cfg.get("validation", {}).get("sampler", "ddim")),
                        "sample_steps": int(cfg.get("validation", {}).get("sample_steps", 200)),
                    },
                    best_records=best_records,
                )

            # PSNR (best_psnr.pt)
            if not math.isnan(val_psnr) and val_psnr > best_psnr:
                best_psnr = val_psnr
                best_records["PSNR"]["top_psnr"] = float(val_psnr)
                best_records["PSNR"]["top_psnr_data"] = dict(epoch_data)

                with open(best_records_path, "w", encoding="utf-8") as f:
                    json.dump(best_records, f, ensure_ascii=False, indent=2)

                save_checkpoint(
                    os.path.join(out_dir, "checkpoints", "best_psnr.pt"),
                    model,
                    ema,
                    cfg,
                    meta={
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_loss": best_loss,
                        "best_psnr": best_psnr,
                        "best_ssim": best_ssim,
                        "best_sum": float(best_records["SUM"]["top_sum"]),
                        "val_loss": val_loss,
                        "val_psnr": val_psnr,
                        "val_ssim": val_ssim,
                        "val_sum": sum_score,
                        "sampler": str(cfg.get("validation", {}).get("sampler", "ddim")),
                        "sample_steps": int(cfg.get("validation", {}).get("sample_steps", 200)),
                    },
                    best_records=best_records,
                )

            # SSIM (best_ssim.pt)
            if not math.isnan(val_ssim) and val_ssim > best_ssim:
                best_ssim = val_ssim
                best_records["SSIM"]["top_ssim"] = float(val_ssim)
                best_records["SSIM"]["top_ssim_data"] = dict(epoch_data)

                with open(best_records_path, "w", encoding="utf-8") as f:
                    json.dump(best_records, f, ensure_ascii=False, indent=2)

                save_checkpoint(
                    os.path.join(out_dir, "checkpoints", "best_ssim.pt"),
                    model,
                    ema,
                    cfg,
                    meta={
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_loss": best_loss,
                        "best_psnr": best_psnr,
                        "best_ssim": best_ssim,
                        "best_sum": float(best_records["SUM"]["top_sum"]),
                        "val_loss": val_loss,
                        "val_psnr": val_psnr,
                        "val_ssim": val_ssim,
                        "val_sum": sum_score,
                        "sampler": str(cfg.get("validation", {}).get("sampler", "ddim")),
                        "sample_steps": int(cfg.get("validation", {}).get("sample_steps", 200)),
                    },
                    best_records=best_records,
                )
        else:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss_epoch,
                    "val_loss": float("nan"),
                    "val_psnr": float("nan"),
                    "val_ssim": float("nan"),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
            )
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            plot_train_val_loss(history, os.path.join(out_dir, "loss_curve.png"))

    # Ensure best_records is embedded into all best checkpoints (even if SUM updated on an epoch without saving a ckpt)
    for _ck in ["best_loss.pt", "best_psnr.pt", "best_ssim.pt"]:
        _p = os.path.join(out_dir, "checkpoints", _ck)
        if os.path.exists(_p):
            _obj = torch.load(_p, map_location="cpu")
            _obj["best_records"] = best_records
            _obj["best_records_json"] = json.dumps(best_records, ensure_ascii=False, indent=2)
            torch.save(_obj, _p)

    # 训练结束：三个 best ckpt 各出一张 10×3 大图
    make_final_grids(cfg, diffusion, model, ema, val_ds, device, out_dir)
    print(f"\nDone. Run dir: {out_dir}")


if __name__ == "__main__":
    main()
