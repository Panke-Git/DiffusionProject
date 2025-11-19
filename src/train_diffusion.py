"""
    @Project: UnderwaterImageEnhanced-Diffusion
    @Author: ChatGPT (adapted to user's style)
    @FileName: train_diffusion.py
    @Time: 2025/11/09
    @Email: None
"""
"""
    @Project: UnderwaterImageEnhanced-Diffusion
    @Author: ChatGPT (refined)
    @FileName: train_diffusion.py
    @Time: 2025/11/19
    @Email: None
"""
# train_diffusion.py
import os
import json
import argparse
import random
import shutil
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config_utils import load_config
from datasets.pair_dataset import UnderwaterPairDataset
from models.unet_conditional_ddpm import UNetConditional
from diffusion.ddpm_scheduler import DDPMNoiseScheduler
from utils.train_utils import *


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, noise_scheduler, dataloader, optimizer, device,
                    timesteps, recon_lambda: float = 0.0,
                    epoch: int = 0):
    """
    单个 epoch 的训练：
      - 标准 DDPM 噪声预测损失
      - 可选的 L1 重建损失（recon_lambda > 0 时）
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Train {epoch:03d}", leave=False)
    for batch in pbar:
        x_inp, y_gt, _ = batch
        x_inp = x_inp.to(device)
        y_gt = y_gt.to(device)

        bsz = y_gt.size(0)
        t = torch.randint(0, timesteps, (bsz,), device=device).long()

        noisy_y, noise = noise_scheduler.q_sample(y_gt, t)
        model_input = torch.cat([noisy_y, x_inp], dim=1)  # [B,6,H,W]

        noise_pred = model(model_input, t)
        loss = F.mse_loss(noise_pred, noise)

        if recon_lambda > 0.0:
            a_bar = noise_scheduler.get_alpha_bar(t).clamp_min(1e-5)
            y0_pred = (noisy_y - torch.sqrt(1.0 - a_bar) * noise_pred) / torch.sqrt(a_bar)
            rec_loss = F.l1_loss(tensor_to_01(y0_pred), tensor_to_01(y_gt))
            loss = loss + recon_lambda * rec_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        avg_loss = total_loss / num_batches
        pbar.set_postfix(loss=f"{avg_loss:.6f}")

    return total_loss / max(1, num_batches)


@torch.no_grad()
def validate(model, noise_scheduler, dataloader, device,
             timesteps, recon_lambda: float = 0.0, epoch: int = 0):
    """
    验证：
      - 计算噪声预测 loss（+ 可选重建 loss）
      - 反推 y0_pred，并在图像域上算 PSNR / SSIM
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    psnr_list = []
    ssim_list = []

    pbar = tqdm(dataloader, desc=f"Val   {epoch:03d}", leave=False)
    for batch in pbar:
        x_inp, y_gt, _ = batch
        x_inp = x_inp.to(device)
        y_gt = y_gt.to(device)

        bsz = y_gt.size(0)
        t = torch.randint(0, timesteps, (bsz,), device=device).long()

        noisy_y, noise = noise_scheduler.q_sample(y_gt, t)
        model_input = torch.cat([noisy_y, x_inp], dim=1)

        noise_pred = model(model_input, t)
        loss = F.mse_loss(noise_pred, noise)

        a_bar = noise_scheduler.get_alpha_bar(t).clamp_min(1e-5)
        y0_pred = (noisy_y - torch.sqrt(1.0 - a_bar) * noise_pred) / torch.sqrt(a_bar)

        if recon_lambda > 0.0:
            rec_loss = F.l1_loss(tensor_to_01(y0_pred), tensor_to_01(y_gt))
            loss = loss + recon_lambda * rec_loss

        # 图像域指标
        batch_psnr = psnr_batch(y0_pred, y_gt)
        batch_ssim = ssim_batch(y0_pred, y_gt)
        psnr_list.append(batch_psnr)
        ssim_list.append(batch_ssim)

        total_loss += loss.item()
        num_batches += 1
        avg_loss = total_loss / num_batches
        pbar.set_postfix(loss=f"{avg_loss:.6f}",
                         psnr=f"{batch_psnr:.2f}",
                         ssim=f"{batch_ssim:.3f}")

    avg_loss = total_loss / max(1, num_batches)
    avg_psnr = float(sum(psnr_list) / len(psnr_list)) if psnr_list else float("nan")
    avg_ssim = float(sum(ssim_list) / len(ssim_list)) if ssim_list else float("nan")

    print(f"[Val   {epoch:03d}] Loss: {avg_loss:.6f} "
          f"PSNR: {avg_psnr:.4f}  SSIM: {avg_ssim:.4f}")

    return avg_loss, avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/config.yaml',
                        help='path to YAML config file')
    parser.add_argument('--model_name', type=str, default='BaseModel',
                        help='model name used as subfolder under SAVE_DIR')
    args = parser.parse_args()

    # 读取配置（raw dict + namespace）
    with open(args.config, 'r') as f:
        raw_cfg = yaml.safe_load(f)
    cfg = load_config(args.config)

    device_str = cfg.TRAIN.DEVICE if hasattr(cfg.TRAIN, 'DEVICE') else 'cuda:0'
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    seed_everything(9861)

    # ==== 路径与输出目录 ====
    train_dir = cfg.PROJECT.TRAIN_DIR
    val_dir = cfg.PROJECT.VAL_DIR
    save_root = cfg.PROJECT.SAVE_DIR

    model_name = args.model_name
    exp_dir = Path(save_root) / model_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 备份配置文件
    cfg_src = Path(args.config)
    cfg_dst = exp_dir / cfg_src.name
    try:
        shutil.copy(str(cfg_src), str(cfg_dst))
    except Exception:
        pass

    # ==== Dataset & DataLoader ====
    img_h = int(getattr(cfg.TRAIN, 'IMG_H', 256))
    img_w = int(getattr(cfg.TRAIN, 'IMG_W', 256))
    num_workers = int(getattr(cfg.TRAIN, 'NUM_WORKERS', 4))
    batch_size = int(cfg.TRAIN.BATCH_SIZE)

    train_set = UnderwaterPairDataset(
        root_dir=train_dir,
        input_subdir=cfg.DATASET.INPUT,
        target_subdir=cfg.DATASET.TARGET,
        img_h=img_h,
        img_w=img_w
    )
    val_set = UnderwaterPairDataset(
        root_dir=val_dir,
        input_subdir=cfg.DATASET.INPUT,
        target_subdir=cfg.DATASET.TARGET,
        img_h=img_h,
        img_w=img_w
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # ==== Model & Optimizer ====
    model = UNetConditional(
        in_channels=6,    # [noisy_y(3) + x_input(3)]
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_emb_dim=256,
        out_channels=3    # 预测噪声 eps
    ).to(device)

    # 可选预训练加载
    if getattr(cfg.TRAIN, 'WEIGHT', ''):
        weight_path = cfg.TRAIN.WEIGHT
        if weight_path:
            print(f'Loading pretrained weights from {weight_path}')
            state_dict = torch.load(weight_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)

    # LR 可能是字符串 '2e-4'，这里强制转 float
    lr = float(cfg.TRAIN.LR)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    timesteps = int(getattr(cfg.TRAIN, 'TIMESTEPS', 1000))
    noise_scheduler = DDPMNoiseScheduler(
        timesteps=timesteps,
        beta_start=1e-4,
        beta_end=0.02,
        device=device
    )

    recon_lambda = float(getattr(cfg.TRAIN, 'RECON_LAMBDA', 0.0))

    # ==== 训练循环 ====
    num_epochs = int(cfg.TRAIN.EPOCHS)
    val_every = int(getattr(cfg.TRAIN, 'VAL_AFTER_EVERY', 1))

    history = {
        "config": raw_cfg,      # 把整个 config 也写进日志里
        "epoch_metrics": []
    }
    best_val_loss = float('inf')
    best_epoch = -1
    best_model_path = exp_dir / 'best_model.pth'

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, noise_scheduler, train_loader, optimizer, device,
            timesteps=timesteps, recon_lambda=recon_lambda,
            epoch=epoch
        )

        if epoch % val_every == 0:
            val_loss, val_psnr, val_ssim = validate(
                model, noise_scheduler, val_loader, device,
                timesteps=timesteps, recon_lambda=recon_lambda,
                epoch=epoch
            )
        else:
            val_loss, val_psnr, val_ssim = float('nan'), float('nan'), float('nan')

        metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_psnr": float(val_psnr),
            "val_ssim": float(val_ssim),
        }
        history["epoch_metrics"].append(metrics)

        # 每个 epoch 覆盖写一次 JSON
        log_path = exp_dir / 'train_log.json'
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"[Epoch {epoch:03d}/{num_epochs:03d}] "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} "
              f"| PSNR: {val_psnr:.4f} | SSIM: {val_ssim:.4f}")

        # 只保存 val loss 最好的模型
        if not (val_loss != val_loss):  # 过滤 NaN
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                print(f"  >> New best model at epoch {epoch}, "
                      f"val_loss={val_loss:.6f}, saved to {best_model_path}")

    # 训练结束，补充 best 信息再写一次 JSON
    history["best_epoch"] = int(best_epoch)
    history["best_val_loss"] = float(best_val_loss)
    log_path = exp_dir / 'train_log.json'
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Training finished. Best epoch = {best_epoch}, best val_loss = {best_val_loss:.6f}")
    print(f"Best model path: {best_model_path}")


if __name__ == '__main__':
    main()
