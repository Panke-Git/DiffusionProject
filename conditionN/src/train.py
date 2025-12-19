"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: train.py
    @Time: 2025/12/13 22:33
    @Email: None
"""
# src/train_ddpm.py

import sys
from pathlib import Path
import argparse
import time
import math

from datetime import datetime

import copy

from .utils.generate_best_grids import auto_generate_best_grids

# 保证能找到 src 目录下的包 (models, data, diffusion, utils)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from .utils.train_utils import *

from .utils.config import load_config
from .utils.record_utils import *

@torch.no_grad()
def validate_denoise(cfg, model, diffusion, val_loader, device):

    model.eval()
    recon_lambda = float(getattr(cfg.TRAIN, "recon_lambda", 0.0) or 0.0)

    total_loss = total_psnr = total_ssim = 0.0
    count = 0

    for batch in val_loader:
        x0 = batch["gt"].to(device)        # [-1,1]
        cond = batch["input"].to(device)   # [-1,1]
        b = x0.size(0)

        t = torch.randint(0, diffusion.timesteps, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise=noise)

        eps_pred = model(x_t, cond, t)
        loss_eps = F.mse_loss(eps_pred.float(), noise.float(), reduction="mean")

        loss = loss_eps
        if recon_lambda > 0:
            x0_pred = diffusion.predict_start_from_noise(x_t.float(), t, eps_pred.float()).clamp(-1, 1)
            x0_01  = ((x0.float() + 1) / 2).clamp(0, 1)
            x0p_01 = ((x0_pred + 1) / 2).clamp(0, 1)
            loss_rec = F.l1_loss(x0p_01, x0_01, reduction="mean")
            loss = loss + recon_lambda * loss_rec
        else:
            x0_pred = diffusion.predict_start_from_noise(x_t.float(), t, eps_pred.float()).clamp(-1, 1)

        psnr = calculate_psnr(x0_pred, x0)
        ssim = calculate_ssim(x0_pred, x0)

        total_loss += float(loss.item()) * b
        total_psnr += float(psnr) * b
        total_ssim += float(ssim) * b
        count += b

    return total_loss / max(1, count), total_psnr / max(1, count), total_ssim / max(1, count)

@torch.no_grad()
def validate_sampling(cfg, model, diffusion, val_loader, device):
    model.eval()
    t_start = int(getattr(getattr(cfg, "VAL", None), "t_start", 200))   # 例如 50~300
    max_batches = int(getattr(getattr(cfg, "VAL", None), "max_batches", 1))  # 验证只抽1~2个batch即可

    total_psnr = total_ssim = 0.0
    count = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        gt = batch["gt"].to(device)         # [-1,1]
        cond = batch["input"].to(device)    # [-1,1]

        # ✅ 关键：推理阶段不使用 GT，只用 input 采样生成
        x_gen = diffusion.sample_from_input(model, cond, t_start=t_start).clamp(-1, 1)

        psnr = calculate_psnr(x_gen, gt)
        ssim = calculate_ssim(x_gen, gt)

        b = gt.size(0)
        total_psnr += float(psnr) * b
        total_ssim += float(ssim) * b
        count += b

    return total_psnr / max(1, count), total_ssim / max(1, count)

def train(cfg, device):
    set_seed(getattr(cfg.TRAIN, "seed", 42))
    start_time=datetime.now().strftime('%Y%m%d_%H%M%S')
    record_root=cfg.DATA.record_path
    train_loader, val_loader = build_dataloaders(cfg)
    model, diffusion = build_model_and_diffusion(cfg, device)
    ema_decay = float(getattr(cfg.TRAIN, "ema_decay", 0.0) or 0.0)
    ema_model = None
    if ema_decay > 0.0:
        ema_model = copy.deepcopy(model).to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        print(f"[EMA] Enabled with decay = {ema_decay}")
    else:
        print("[EMA] Disabled (ema_decay <= 0)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.TRAIN.lr,
        weight_decay=cfg.TRAIN.weight_decay,
    )

    use_amp = bool(getattr(cfg.TRAIN, "use_amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # save_dir = getattr(cfg.TRAIN, "save_dir", "checkpoints")
    # infinity 无穷大
    best_loss = math.inf
    best_psnr = -math.inf
    best_ssim = -math.inf

    num_epochs = cfg.TRAIN.epochs
    log_interval = cfg.TRAIN.log_interval
    val_interval = cfg.TRAIN.val_interval
    sample_interval = int(getattr(getattr(cfg, "VAL", None), "sample_interval", 0) or 0)
    # sample_interval=0 表示训练中不做 sampling，只在最后做一次（可选）
    do_final_sampling = bool(getattr(getattr(cfg, "VAL", None), "do_final_sampling", True))
    grad_clip = cfg.TRAIN.grad_clip

    print(f"[Train] Start training for {num_epochs} epochs")
    total_record=[]

    record_path, best_path = make_train_path(record_root, "Base_DDPM", start_time)
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    top_data = None
    top_ssim_data, top_psnr_data, top_loss_data = None, None, None

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        epoch_start_time  = time.time()

        for it, batch in enumerate(train_loader, start=1):
            x0 = batch["gt"].to(device)        # 目标增强图像 [-1,1]
            cond = batch["input"].to(device)   # 条件图像 [-1,1]
            b = x0.size(0)

            t = torch.randint(
                low=0,
                high=diffusion.timesteps,
                size=(b,),
                device=device,
                dtype=torch.long,
            )
            noise = torch.randn_like(x0)

            x_t = diffusion.q_sample(x0, t, noise=noise)

            optimizer.zero_grad(set_to_none=True)
            recon_lambda = float(getattr(cfg.TRAIN, "recon_lambda", 0.0) or 0.0)

            with torch.cuda.amp.autocast(enabled=use_amp):
                eps_pred = model(x_t, cond, t)
                loss_eps = F.mse_loss(eps_pred, noise, reduction="mean")

            loss = loss_eps
            if recon_lambda > 0.0:
                x0_pred = diffusion.predict_start_from_noise(x_t.float(), t, eps_pred.float()).clamp(-1, 1)
                x0_01 = ((x0.float() + 1) / 2).clamp(0, 1)
                x0p_01 = ((x0_pred + 1) / 2).clamp(0, 1)
                loss_rec = F.l1_loss(x0p_01, x0_01, reduction="mean")
                loss = loss + recon_lambda * loss_rec

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪（可选）
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            if ema_model is not None:
                update_ema(ema_model, model, ema_decay)

            running_loss += loss.item() * b
            running_count += b

            if it % log_interval == 0:
                avg_loss = running_loss / running_count
                elapsed = time.time() - epoch_start_time
                print(
                    f"[Epoch {epoch}/{num_epochs}] "
                    f"Iter {it}/{len(train_loader)} "
                    f"Loss: {avg_loss:.6f}  "
                    f"Time: {elapsed:.1f}s"
                )

        # 每个 epoch 的平均 train loss
        epoch_train_loss = running_loss / max(1, running_count)
        print(f"[Epoch {epoch}] Train loss: {epoch_train_loss:.6f}")

        # 是否进行验证
        if epoch % val_interval == 0:
            model_for_eval = ema_model if ema_model is not None else model

            den_loss, den_psnr, den_ssim = validate_denoise(cfg, model_for_eval, diffusion, val_loader, device)

            # ✅ 让 record 有意义：把 val_* 绑定为 denoise 指标（快速实验最实用）
            val_loss, val_psnr, val_ssim = den_loss, den_psnr, den_ssim

            # sampling：按间隔或最后一轮才跑
            do_sample = (sample_interval > 0 and epoch % sample_interval == 0) or (
                        do_final_sampling and epoch == num_epochs)

            sam_psnr = sam_ssim = None
            if do_sample:
                sam_psnr, sam_ssim = validate_sampling(cfg, model_for_eval, diffusion, val_loader, device)

            if sam_ssim is None:
                print(
                    f"[Epoch {epoch}] denoise: loss={den_loss:.6f}, psnr={den_psnr:.4f}, ssim={den_ssim:.4f} | sampling: skipped")
            else:
                print(f"[Epoch {epoch}] denoise: loss={den_loss:.6f}, psnr={den_psnr:.4f}, ssim={den_ssim:.4f} | "
                      f"sample: psnr={sam_psnr:.4f}, ssim={sam_ssim:.4f}")

            metrics = {
                "epoch": epoch,
                "den_loss": den_loss, "den_psnr": den_psnr, "den_ssim": den_ssim,
                "sam_psnr": sam_psnr, "sam_ssim": sam_ssim,
            }

            # ✅ Loss 永远用 denoise loss 保存（快且稳定）
            if den_loss < best_loss:
                best_loss = den_loss
                save_checkpoint(model, optimizer, epoch, cfg, best_path, tag="loss", metrics_dict=metrics,
                                ema_model=ema_model)
                top_loss_data = metrics

            # ✅ PSNR/SSIM：如果本轮做了 sampling，就以 sampling 为准；否则用 denoise（方便快速实验也能出 best）
            psnr_for_best = sam_psnr if sam_psnr is not None else den_psnr
            ssim_for_best = sam_ssim if sam_ssim is not None else den_ssim

            if psnr_for_best > best_psnr:
                best_psnr = psnr_for_best
                save_checkpoint(model, optimizer, epoch, cfg, best_path, tag="psnr", metrics_dict=metrics,
                                ema_model=ema_model)
                top_psnr_data= metrics

            if ssim_for_best > best_ssim:
                best_ssim = ssim_for_best
                save_checkpoint(model, optimizer, epoch, cfg, best_path, tag="ssim", metrics_dict=metrics,
                                ema_model=ema_model)
                top_ssim_data=metrics
        epoch_record = package_one_epoch(epoch=epoch,
                                         train_loss=float(epoch_train_loss),
                                         val_loss=float(val_loss),
                                         val_psnr=float(val_psnr),
                                         val_ssim=float(val_ssim)
                                         )
        total_record.append(epoch_record)
        top_data = {
            'PSNR': {
                'top_psnr': float(best_psnr),
                'top_psnr_data': top_psnr_data,
            },
            'SSIM': {
                'top_ssim': float(best_ssim),
                'top_ssim_data': top_ssim_data
            },
            'LOSS': {
                'top_sum': float(best_loss),
                'top_sum_data': top_loss_data
            }
        }
    end_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_path, json_path, top_path = save_train_data(record_path, start_time, end_time, total_record,
                                                                   top_data)
    if bool(cfg.VAL.auto_preview):
        preview_out_dir = Path(record_path)/"prview_grids"
        auto_generate_best_grids(
            cfg=cfg,
            device=device,
            ckpt_dir=best_path,
            out_dir=preview_out_dir,
            t_start=int(getattr(getattr(cfg, "VAL", None), "t_start", 200)),
            n_rows=10,
            seed=int(getattr(getattr(cfg, "VAL", None), "preview_seed", 9861)),
            use_ema=bool(getattr(getattr(cfg, "VAL", None), "preview_use_ema", True)),
        )
    print("Train record saved in excel: ", excel_path)
    print("Train record saved in json: ", json_path)
    print("Train top record saved in top: ", top_path)
    print("[Train] Finished training.")
    print(f"Best Loss: {best_loss:.6f}, Best PSNR: {best_psnr:.4f}, Best SSIM: {best_ssim:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional DDPM for underwater enhancement")
    parser.add_argument(
        "--config",
        type=str,
        default="/public/home/hnust15874739861/pro/DiffusionProject/conditionN/src/config/underwater_ddpm.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use, e.g. 'cuda', 'cuda:0', or 'cpu'",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")

    print(f"Using device: {device}")
    train(cfg, device)

