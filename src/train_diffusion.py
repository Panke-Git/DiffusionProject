"""
    @Project: UnderwaterImageEnhanced-Diffusion
    @Author: ChatGPT (adapted to user's style)
    @FileName: train_diffusion.py
    @Time: 2025/11/09
    @Email: None
"""
from __future__ import annotations
import argparse
import math
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.config import Config
from data.paired_dataset import PairedFolder
from diffusion.scheduler import Diffusion
from models.diffusion_unet import UNet
from utils.ema import EMA
from utils.metrics import psnr as psnr_metric, ssim as ssim_metric, to01
from utils.record_utils import make_train_path, save_train_config
from utils.train_utils import seed_everything

def build_dataloaders(cfg):
    train_ds = PairedFolder(cfg.PROJECT.TRAIN_DIR, 'Train', cfg.DATASET.IMG_H, cfg.DATASET.INPUT, cfg.DATASET.TARGET, augment=True)
    val_ds   = PairedFolder(cfg.PROJECT.VAL_DIR, 'Val',   cfg.DATASET.IMG_H, cfg.DATASET.INPUT, cfg.DATASET.TARGET, augment=False)
    train_dl = DataLoader(train_ds, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=max(1, cfg.TRAIN.BATCH_SIZE//2), shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    return train_dl, val_dl

@torch.no_grad()
def validate(model, ema, diffusion, val_dl, device, cfg, record_path, step, writer):
    # 复制 EMA 权重
    ema_model = UNet(image_size=cfg.MODEL.IMAGE_SIZE, in_channels=6, out_channels=3,
                     base_channels=cfg.MODEL.BASE_CHANNELS, channel_mults=tuple(cfg.MODEL.CHANNEL_MULTS),
                     num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, attn_resolutions=tuple(cfg.MODEL.ATTN_RES),
                     time_pos_dim=cfg.MODEL.TIME_POS_DIM, time_dim=cfg.MODEL.TIME_DIM,
                     dropout=cfg.MODEL.DROPOUT).to(device)
    ema.copy_to(ema_model); ema_model.eval()

    from torchvision.utils import save_image

    def to01_safe(x):
        # 训练里是 [-1,1]，这里统一映射回 [0,1]
        return ((x.clamp(-1, 1) + 1) / 2).clamp(0, 1)

    psnr_total, ssim_total, n_batches = 0.0, 0.0, 0
    sample_dir = (record_path / 'samples' / f"step_{step:07d}"); sample_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(val_dl, desc="[Val]")
    for i, batch in enumerate(pbar):
        # y: 输入；x0: GT（两者都先映射到 [-1,1]）
        y  = batch["y"].to(device)  * 2 - 1
        x0 = batch["x0"].to(device) * 2 - 1

        # —— 关键：采样尺寸对齐到 y 的真实尺寸；条件直接传 y（[-1,1]）——
        H = y.shape[-1]
        x_hat = diffusion.ddim_sample(
            ema_model,
            y=y,
            image_size=H,
            steps=cfg.SCHEDULER.SAMPLE_STEPS,
            eta=cfg.SCHEDULER.SAMPLE_ETA,
            start_from_y=cfg.EVAL.USE_IMG2IMG_START
        )

        # 断言形状与数值域；一旦不满足，直接暴露采样实现的问题
        assert x_hat.shape == x0.shape == y.shape, f"shape mismatch: pred={x_hat.shape}, x0={x0.shape}, y={y.shape}"
        x_hat = x_hat.clamp(-1, 1)

        # 计算指标（都在 [0,1]）
        x_hat01, x001, y01 = to01_safe(x_hat), to01_safe(x0), to01_safe(y)
        psnr_val = psnr_metric(x_hat01, x001)
        ssim_val = ssim_metric(x_hat01, x001)
        psnr_total += psnr_val; ssim_total += ssim_val; n_batches += 1

        # —— 自检：输入对 GT 的 PSNR ——（应显著低于 pred 对 GT）
        if i == 0:
            psnr_in = psnr_metric(y01, x001)
            print(f"[Val:debug] PSNR(input,gt)={psnr_in:.2f}  PSNR(pred,gt)={psnr_val:.2f}")

        if i < cfg.EVAL.NUM_VIS_BATCH:
            grid = torch.cat([y01, x_hat01, x001], dim=0)
            save_image(grid, sample_dir / f"val_{i:03d}.png", nrow=y.size(0), padding=2)

    m_psnr = psnr_total / max(1, n_batches)
    m_ssim = ssim_total / max(1, n_batches)
    if writer is not None:
        writer.add_scalar("val/psnr", m_psnr, step)
        writer.add_scalar("val/ssim", m_ssim, step)
    print(f"[Val] step={step} PSNR={m_psnr:.3f} SSIM={m_ssim:.4f}")
    return m_psnr, m_ssim

def train_one_epoch(model: UNet, diffusion: Diffusion, train_dl: DataLoader, opt, scaler, device, cfg, ema: EMA, writer: SummaryWriter | None, global_step: int):
    model.train()
    pbar = tqdm(train_dl, desc="[Train]")
    running_loss, num_iters = 0.0, 0
    for it, batch in enumerate(pbar):
        y = batch["y"].to(device) * 2 - 1  # [-1,1]
        x0 = batch["x0"].to(device) * 2 - 1
        B = y.size(0)
        t = torch.randint(0, diffusion.steps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise)

        amp_enabled = bool(cfg.TRAIN.AMP) and (device.type == "cuda")
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            eps_pred = model(x_t, y, t)
            loss = F.mse_loss(eps_pred, noise)
        recon_lambda = float(getattr(cfg.TRAIN, "RECON_LAMBDA", 0.0) or 0.0)
        if recon_lambda > 0:
            # —— 用 FP32 + clamp，避免 /sqrt(a_bar) 下溢 → NaN ——
            with torch.amp.autocast("cuda", enabled=False):
                a_bar = diffusion.alphas_cumprod[t].float().clamp_min(1e-5)
                while a_bar.dim() < x0.dim():
                    a_bar = a_bar.unsqueeze(-1)
                x0_pred = (x_t.float() - torch.sqrt(1 - a_bar) * eps_pred.float()) / torch.sqrt(a_bar)
                # 在像素域 [0,1] 上做 L1
                loss_rec = F.l1_loss(((x0_pred + 1) / 2).clamp(0, 1), ((x0 + 1) / 2).clamp(0, 1))
            loss = loss + recon_lambda * loss_rec

        loss = loss / max(1, cfg.TRAIN.ACCUM_STEPS)

        if not torch.isfinite(loss):
            print('[warn] non-finite loss, skip batch');
            opt.zero_grad(set_to_none=True);
            continue
        scaler.scale(loss).backward()

        if (it + 1) % max(1, cfg.TRAIN.ACCUM_STEPS) == 0:
            if cfg.TRAIN.GRAD_CLIP and cfg.TRAIN.GRAD_CLIP > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            ema.update(model)
            global_step += 1

            running_loss += (loss.item() * max(1, cfg.TRAIN.ACCUM_STEPS))
            num_iters += 1

            if writer is not None and global_step % cfg.LOG.LOG_INTERVAL == 0:
                writer.add_scalar('train/loss', running_loss / max(1, num_iters), global_step)
    avg_loss = running_loss / max(1, num_iters)
    return global_step, avg_loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=str((Path(__file__).parent / "configs" / "config.yaml").resolve()),  type=str, required=False, help="YAML 配置文件路径")
    ap.add_argument("--eval_only", action="store_true", default=False, help="仅做验证/采样")
    args = ap.parse_args()

    cfg = Config.load(args.cfg)

    # 设备与随机种子
    device = torch.device(cfg.TRAIN.DEVICE if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.TRAIN.SEED)

    # 训练记录路径与日志
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"UNetDiffusion_C{cfg.MODEL.BASE_CHANNELS}_S{cfg.SCHEDULER.STEPS}"
    record_path, ckpt_dir = make_train_path(cfg.PROJECT.EXPT_RECORD_DIR, model_name, start_time)
    cfg_path = save_train_config(record_path,
                                 model=model_name,
                                 dataset_train=cfg.PROJECT.TRAIN_DIR,
                                 dataset_val=cfg.PROJECT.VAL_DIR,
                                 lr=float(cfg.TRAIN.LR),
                                 batch_size=cfg.TRAIN.BATCH_SIZE,
                                 steps=cfg.SCHEDULER.STEPS,
                                 sample_steps=cfg.SCHEDULER.SAMPLE_STEPS,
                                 seed=cfg.TRAIN.SEED)
    writer = None
    try:
        writer = SummaryWriter(log_dir=str(record_path / "logs"))
    except Exception:
        writer = None

    # 数据
    train_dl, val_dl = build_dataloaders(cfg)

    # 模型 / 调度器 / 优化器
    model = UNet(image_size=cfg.MODEL.IMAGE_SIZE, in_channels=6, out_channels=3,
                 base_channels=cfg.MODEL.BASE_CHANNELS, channel_mults=tuple(cfg.MODEL.CHANNEL_MULTS),
                 num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, attn_resolutions=tuple(cfg.MODEL.ATTN_RES),
                 time_pos_dim=cfg.MODEL.TIME_POS_DIM, time_dim=cfg.MODEL.TIME_DIM, dropout=cfg.MODEL.DROPOUT).to(device)
    ema = EMA(model, decay=cfg.TRAIN.EMA_DECAY)
    diffusion = Diffusion(steps=cfg.SCHEDULER.STEPS, schedule="cosine", device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.TRAIN.LR), weight_decay=float(cfg.TRAIN.WEIGHT_DECAY))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.TRAIN.AMP) and (device.type == "cuda"))
    # 学习率计划（与用户风格一致：CosineAnnealingLR 以 epoch 为周期也可，这里按 step 不细化，保持简洁）
    # 可按需接入 torch.optim.lr_scheduler.CosineAnnealingLR

    # 仅验证
    if args.eval_only:
        # 加载最近的 ckpt
        ckpts = sorted((record_path.parent / model_name).glob("*/ckpts/ckpt_*.pt"))
        if not ckpts:
            print("No checkpoints found under:", record_path.parent / model_name)
            return
        state = torch.load(ckpts[-1], map_location=device)
        model.load_state_dict(state["model"])
        ema.shadow = state["ema"]
        validate(model, ema, diffusion, val_dl, device, cfg, record_path, step=state.get("step", 0), writer=writer)
        return

    # 训练循环
    global_step = 0
    best_metric = -1e9 if cfg.LOG.MODE.lower() == 'max' else 1e9
    metrics_log = []
    import json, pandas as pd

    for epoch in range(cfg.TRAIN.EPOCHS):
        global_step, avg_loss = train_one_epoch(model, diffusion, train_dl, opt, scaler, device, cfg, ema, writer, global_step)

        val_psnr, val_ssim = validate(model, ema, diffusion, val_dl, device, cfg, record_path, step=global_step, writer=writer)
        lr_now = float(opt.param_groups[0]['lr'])
        rec = {
            'epoch': int(epoch+1),
            'global_step': int(global_step),
            'train_loss': float(avg_loss),
            'val_psnr': float(val_psnr),
            'val_ssim': float(val_ssim),
            'lr': lr_now,
        }
        metrics_log.append(rec)
        logs_dir = record_path / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        with (logs_dir / cfg.LOG.METRICS_JSON).open('w', encoding='utf-8') as f:
            json.dump(metrics_log, f, indent=2, ensure_ascii=False)
        try:
            pd.DataFrame(metrics_log).to_excel(logs_dir / cfg.LOG.METRICS_XLSX, index=False)
        except Exception as e:
            print('[warn] 写 Excel 失败：', e)

        metric_name = cfg.LOG.MONITOR.lower()
        cur_metric = val_psnr if metric_name == 'psnr' else val_ssim
        better = (cur_metric > best_metric) if cfg.LOG.MODE.lower() == 'max' else (cur_metric < best_metric)
        if better:
            best_metric = cur_metric
            state = {
                'step': global_step,
                'model': model.state_dict(),
                'ema': ema.shadow,
                'opt': opt.state_dict(),
                'scaler': scaler.state_dict(),
                'cfg': vars(cfg)
            }
            best_path = ckpt_dir / 'best.pt'
            torch.save(state, best_path)
            print(f"[CKPT] 更新最优 {metric_name}={cur_metric:.4f} → 保存 {best_path}")


if __name__ == "__main__":
    main()
