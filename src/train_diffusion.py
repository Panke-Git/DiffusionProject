"""
    @Project: UnderwaterImageEnhanced-Diffusion
    @Author: ChatGPT (adapted to user's style)
    @FileName: train_diffusion.py
    @Time: 2025/11/09
    @Email: None
"""
# train_ddpm.py
import os
import json
import argparse
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

from config_utils import load_config
from datasets.pair_dataset import UnderwaterPairDataset
from models.unet_conditional_ddpm import UNetConditional
from diffusion.ddpm_scheduler import DDPMNoiseScheduler


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, noise_scheduler, dataloader, optimizer, device,
                    timesteps, recon_lambda=0.0):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        x_inp, y_gt, _ = batch
        x_inp = x_inp.to(device)  # 条件 input
        y_gt = y_gt.to(device)    # ground truth

        bsz = y_gt.size(0)
        t = torch.randint(0, timesteps, (bsz,), device=device).long()

        noisy_y, noise = noise_scheduler.q_sample(y_gt, t)
        model_input = torch.cat([noisy_y, x_inp], dim=1)  # [B,6,H,W]

        noise_pred = model(model_input, t)

        loss = F.mse_loss(noise_pred, noise)

        if recon_lambda > 0.0:
            a_bar = noise_scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)
            a_bar = a_bar.clamp_min(1e-5)
            # 反推 y0_pred
            y0_pred = (noisy_y - torch.sqrt(1.0 - a_bar) * noise_pred) / torch.sqrt(a_bar)
            # 映射回 [0,1] 再做 L1
            rec_loss = F.l1_loss(((y0_pred + 1.0) / 2.0).clamp(0.0, 1.0),
                                 ((y_gt + 1.0) / 2.0).clamp(0.0, 1.0))
            loss = loss + recon_lambda * rec_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def validate(model, noise_scheduler, dataloader, device,
             timesteps, recon_lambda=0.0):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        x_inp, y_gt, _ = batch
        x_inp = x_inp.to(device)
        y_gt = y_gt.to(device)

        bsz = y_gt.size(0)
        t = torch.randint(0, timesteps, (bsz,), device=device).long()

        noisy_y, noise = noise_scheduler.q_sample(y_gt, t)
        model_input = torch.cat([noisy_y, x_inp], dim=1)

        noise_pred = model(model_input, t)
        loss = F.mse_loss(noise_pred, noise)

        if recon_lambda > 0.0:
            a_bar = noise_scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)
            a_bar = a_bar.clamp_min(1e-5)
            y0_pred = (noisy_y - torch.sqrt(1.0 - a_bar) * noise_pred) / torch.sqrt(a_bar)
            rec_loss = F.l1_loss(((y0_pred + 1.0) / 2.0).clamp(0.0, 1.0),
                                 ((y_gt + 1.0) / 2.0).clamp(0.0, 1.0))
            loss = loss + recon_lambda * rec_loss

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='path to YAML config file')
    parser.add_argument('--model_name', type=str, default='model1',
                        help='model name used as subfolder under SAVE_DIR')
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = cfg.TRAIN.DEVICE if hasattr(cfg.TRAIN, 'DEVICE') else 'cuda:0'
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    set_seed(42)

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
    img_h = getattr(cfg.TRAIN, 'IMG_H', 256)
    img_w = getattr(cfg.TRAIN, 'IMG_W', 256)
    num_workers = getattr(cfg.TRAIN, 'NUM_WORKERS', 4)
    batch_size = cfg.TRAIN.BATCH_SIZE

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

    # ==== Model & Optimizer & Scheduler ====
    model = UNetConditional(
        in_channels=6,    # [noisy_y(3) + x_input(3)]
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_emb_dim=256,
        out_channels=3    # 预测噪声 eps
    ).to(device)

    # 如果需要加载预训练权重（比如预训练的 UNet），可以在 cfg.TRAIN.WEIGHT 指定
    if getattr(cfg.TRAIN, 'WEIGHT', ''):
        weight_path = cfg.TRAIN.WEIGHT
        print(f'Loading pretrained weights from {weight_path}')
        state_dict = torch.load(weight_path, map_location=device)
        # 假设是纯 state_dict
        model.load_state_dict(state_dict, strict=False)

    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    timesteps = getattr(cfg.TRAIN, 'TIMESTEPS', 1000)
    noise_scheduler = DDPMNoiseScheduler(
        timesteps=timesteps,
        beta_start=1e-4,
        beta_end=0.02,
        device=device
    )

    recon_lambda = getattr(cfg.TRAIN, 'RECON_LAMBDA', 0.0)

    # ==== 训练循环 ====
    num_epochs = cfg.TRAIN.EPOCHS
    val_every = getattr(cfg.TRAIN, 'VAL_AFTER_EVERY', 1)
    print_freq = getattr(cfg.TRAIN, 'PRINT_FREQ', 1)

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': []
    }
    best_val_loss = float('inf')
    best_epoch = -1
    best_model_path = exp_dir / 'best_model.pth'

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, noise_scheduler, train_loader, optimizer, device,
            timesteps=timesteps, recon_lambda=recon_lambda
        )

        if epoch % val_every == 0:
            val_loss = validate(
                model, noise_scheduler, val_loader, device,
                timesteps=timesteps, recon_lambda=recon_lambda
            )
        else:
            val_loss = float('nan')

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # 保存训练曲线到 JSON
        log_path = exp_dir / 'train_log.json'
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=2)

        if epoch % print_freq == 0:
            print(f"[Epoch {epoch:03d}/{num_epochs:03d}] "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # 只保存 val loss 最好的模型
        if not (val_loss != val_loss):  # 过滤 NaN
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                print(f"  >> New best model at epoch {epoch}, val_loss={val_loss:.6f}, "
                      f"saved to {best_model_path}")

    print(f"Training finished. Best epoch = {best_epoch}, best val_loss = {best_val_loss:.6f}")
    print(f"Best model path: {best_model_path}")


if __name__ == '__main__':
    main()

