# coding=utf-8
"""
    @Project: 
    @Author: PyCharm
    @FileName： train.py
    @Date：2025/12/25 16:14
    @Email: None
"""

import os
import time
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torchvision.utils as vutils

from datasets.uie_dataset import PairedImageDataset
from models.unet_ddpm import UNetModel, UNetConfig
from diffusion.gaussian_diffusion import GaussianDiffusion, DiffusionConfig
from utils.ema import EMA, EMAConfig
from utils.metrics import denorm_to_01, psnr, ssim
from utils.misc import set_seed, ensure_dir, load_yaml, count_params

def save_ckpt(path, model, ema, optimizer, epoch, step, best_state):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_state": best_state,
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)

def _read_json(path: str):
    if (path is None) or (not os.path.exists(path)):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def append_epoch_metrics(metrics_path: str, record: dict):
    """
    Save as a JSON array:
    [
      {epoch:..., train_loss:..., ...},
      ...
    ]
    If the same epoch already exists (e.g., resume), overwrite it.
    """
    data = _read_json(metrics_path)
    if not isinstance(data, list):
        data = []

    # overwrite same epoch if exists
    updated = False
    for i, r in enumerate(data):
        if isinstance(r, dict) and r.get("epoch", None) == record.get("epoch", None):
            data[i] = record
            updated = True
            break
    if not updated:
        data.append(record)

    tmp_path = metrics_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, metrics_path)

def load_best_tag(best_state, tag):
    if tag not in best_state:
        return None
    return best_state[tag].get("path", None)

@torch.no_grad()
def run_validation(cfg, model_for_sample, diffusion, val_loader, device, out_dir, epoch):
    model_for_sample.eval()

    # 1) val loss (no sampling, cheap)
    total_loss = 0.0
    n = 0
    for batch in val_loader:
        cond = batch["cond"].to(device)
        gt = batch["gt"].to(device)
        b = gt.shape[0]
        t = torch.randint(0, diffusion.cfg.timesteps, (b,), device=gt.device).long()
        # 或 device=cond.device，二者一致即可
        loss = diffusion.p_losses(model_for_sample, gt, cond, t)
        total_loss += float(loss.item()) * b
        n += b
    val_loss = total_loss / max(n, 1)

    # 2) generate some images for PSNR/SSIM (potentially expensive)
    sampler = cfg["train"]["val_sampler"]
    steps = int(cfg["train"]["val_sampling_steps"])
    eta = float(cfg["train"].get("val_ddim_eta", 0.0))
    limit = int(cfg["train"].get("val_num_images", 64))

    total_psnr = 0.0
    total_ssim = 0.0
    m = 0

    # preview saving
    preview_dir = os.path.join(out_dir, "previews")
    ensure_dir(preview_dir)

    for batch in val_loader:
        cond = batch["cond"].to(device)
        gt = batch["gt"].to(device)
        names = batch["name"]
        b = gt.shape[0]

        if sampler == "ddpm":
            pred = diffusion.sample_ddpm(model_for_sample, cond)
        else:
            pred = diffusion.sample_ddim(model_for_sample, cond, steps=steps, eta=eta)

        pred01 = denorm_to_01(pred)
        gt01 = denorm_to_01(gt)

        total_psnr += float(psnr(pred01, gt01).item()) * b
        total_ssim += float(ssim(pred01, gt01).item()) * b
        m += b

        # save a few preview grids
        if epoch % 1 == 0:
            # first 4 samples
            k = min(4, b)
            grid = torch.cat([cond[:k], pred[:k], gt[:k]], dim=0)  # [-1,1]
            grid01 = denorm_to_01(grid)
            vutils.save_image(grid01, os.path.join(preview_dir, f"epoch{epoch:04d}.png"), nrow=k)

        if limit > 0 and m >= limit:
            break

    val_psnr = total_psnr / max(m, 1)
    val_ssim = total_ssim / max(m, 1)

    return val_loss, val_psnr, val_ssim

@torch.no_grad()
def generate_folder(cfg, model_for_sample, diffusion, input_dir, out_dir, device, tag):
    from PIL import Image
    import torchvision.transforms.functional as TF

    model_for_sample.eval()
    sampler = cfg["sample"]["sampler"]
    steps = int(cfg["sample"]["steps"])
    eta = float(cfg["sample"].get("ddim_eta", 0.0))
    bs = int(cfg["sample"]["batch_size"])
    image_size = int(cfg["data"]["image_size"])

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = []
    for p in Path(input_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    ensure_dir(out_dir)

    def to_tensor_norm(p):
        img = Image.open(str(p)).convert("RGB")
        img = TF.resize(img, [image_size, image_size], interpolation=Image.BICUBIC)
        x = TF.to_tensor(img) * 2.0 - 1.0
        return x

    idx = 0
    while idx < len(files):
        batch_paths = files[idx: idx + bs]
        cond = torch.stack([to_tensor_norm(p) for p in batch_paths], dim=0).to(device)

        if sampler == "ddpm":
            pred = diffusion.sample_ddpm(model_for_sample, cond)
        else:
            pred = diffusion.sample_ddim(model_for_sample, cond, steps=steps, eta=eta)

        pred01 = denorm_to_01(pred).clamp(0, 1)

        for pth, img_t in zip(batch_paths, pred01):
            rel = pth.relative_to(input_dir).as_posix()
            save_path = os.path.join(out_dir, rel)
            ensure_dir(os.path.dirname(save_path))
            vutils.save_image(img_t, save_path)

        idx += bs

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/uie_ddpm.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg["exp"]["seed"])
    set_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    amp_enabled = bool(cfg["train"]["amp"]) and (device.type == "cuda")
    print(f"[Info] AMP enabled: {amp_enabled} (cfg={cfg['train']['amp']}, device={device})")

    exp_name = cfg["exp"]["name"]
    run_dir = os.path.join(cfg["exp"]["output_dir"], exp_name, "")
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "checkpoints"))

    # data
    image_size = int(cfg["data"]["image_size"])
    train_set = PairedImageDataset(cfg["data"]["train_input_dir"], cfg["data"]["train_gt_dir"], image_size=image_size, is_train=True)
    val_set = PairedImageDataset(cfg["data"]["val_input_dir"], cfg["data"]["val_gt_dir"], image_size=image_size, is_train=False)

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=False,
    )

    # model
    unet_cfg = UNetConfig(
        in_channels=int(cfg["model"]["in_channels"]),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        channel_mult=list(cfg["model"]["channel_mult"]),
        num_res_blocks=int(cfg["model"]["num_res_blocks"]),
        attn_resolutions=list(cfg["model"]["attn_resolutions"]),
        dropout=float(cfg["model"]["dropout"]),
        num_heads=int(cfg["model"]["num_heads"]),
        use_scale_shift_norm=bool(cfg["model"]["use_scale_shift_norm"]),
        image_size=image_size,
    )
    model = UNetModel(unet_cfg).to(device)
    print(f"[Info] Params: {count_params(model)/1e6:.2f}M")

    # diffusion
    diff_cfg = DiffusionConfig(
        timesteps=int(cfg["diffusion"]["timesteps"]),
        beta_schedule=str(cfg["diffusion"]["beta_schedule"]),
        objective=str(cfg["diffusion"]["objective"]),
        p2_loss_weight_gamma=float(cfg["diffusion"].get("p2_loss_weight_gamma", 0.0)),
        p2_loss_weight_k=float(cfg["diffusion"].get("p2_loss_weight_k", 1.0)),
    )
    diffusion = GaussianDiffusion(diff_cfg, device=device)

    # ema
    ema = None
    if bool(cfg["ema"]["enable"]):
        ema_cfg = EMAConfig(
            enable=True,
            decay=float(cfg["ema"]["decay"]),
            update_every=int(cfg["ema"]["update_every"]),
            warmup_steps=int(cfg["ema"]["warmup_steps"]),
        )
        ema = EMA(model, ema_cfg)

    # optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    scaler = GradScaler(enabled=amp_enabled)

    # best trackers
    best_state = {
        "best_loss": {"value": float("inf"), "path": None},
        "best_psnr": {"value": -1.0, "path": None},
        "best_ssim": {"value": -1.0, "path": None},
    }

    epochs = int(cfg["train"]["epochs"])
    log_every = int(cfg["train"]["log_every"])
    val_every_epochs = int(cfg["train"]["val_every_epochs"])
    grad_clip = float(cfg["train"]["grad_clip"])

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss_sum = 0.0
        epoch_train_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for it, batch in enumerate(pbar, start=1):
            cond = batch["cond"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            b = gt.shape[0]
            t = torch.randint(0, diffusion.cfg.timesteps, (b,), device=device).long()

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                with autocast(enabled=True):
                    loss = diffusion.p_losses(model, gt, cond, t)

                # AMP path
                scaler.scale(loss).backward()

                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()

            else:
                # FP32 path
                loss = diffusion.p_losses(model, gt, cond, t)
                loss.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

            epoch_train_loss_sum += float(loss.item()) * gt.shape[0]
            epoch_train_count += gt.shape[0]

            if ema is not None:
                ema.update(model)

            global_step += 1
            if global_step % log_every == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})
        train_loss_epoch = epoch_train_loss_sum / max(epoch_train_count, 1)
        # validation & checkpointing
        if epoch % val_every_epochs == 0:
            model_for_eval = ema.ema_model if ema is not None else model
            val_loss, val_psnr, val_ssim = run_validation(cfg, model_for_eval, diffusion, val_loader, device, run_dir, epoch)
            print(f"[Val] epoch={epoch} loss={val_loss:.6f} psnr={val_psnr:.4f} ssim={val_ssim:.4f}")
            metrics_path = os.path.join(run_dir, "metrics.json")
            lr_now = float(optimizer.param_groups[0]["lr"])

            record = {
                "epoch": int(epoch),
                "train_loss": float(train_loss_epoch),
                "val_loss": float(val_loss),
                "val_psnr": float(val_psnr),
                "val_ssim": float(val_ssim),
                "lr": lr_now,
            }
            append_epoch_metrics(metrics_path, record)
            ckpt_dir = os.path.join(run_dir, "checkpoints")

            # best loss
            if val_loss < best_state["best_loss"]["value"]:
                best_state["best_loss"]["value"] = val_loss
                best_path = os.path.join(ckpt_dir, "best_loss.pt")
                best_state["best_loss"]["path"] = best_path
                save_ckpt(best_path, model, ema, optimizer, epoch, global_step, best_state)

            # best psnr
            if val_psnr > best_state["best_psnr"]["value"]:
                best_state["best_psnr"]["value"] = val_psnr
                best_path = os.path.join(ckpt_dir, "best_psnr.pt")
                best_state["best_psnr"]["path"] = best_path
                save_ckpt(best_path, model, ema, optimizer, epoch, global_step, best_state)

            # best ssim
            if val_ssim > best_state["best_ssim"]["value"]:
                best_state["best_ssim"]["value"] = val_ssim
                best_path = os.path.join(ckpt_dir, "best_ssim.pt")
                best_state["best_ssim"]["path"] = best_path
                save_ckpt(best_path, model, ema, optimizer, epoch, global_step, best_state)

    print("[Train] Finished.")
    print("[Best] ", best_state)

    # generate after training
    if bool(cfg["train"].get("generate_after_train", True)):
        print("[Gen] Generating outputs for best checkpoints on Val/input ...")
        val_input_dir = cfg["data"]["val_input_dir"]
        gen_root = os.path.join(run_dir, "generated_val")

        for tag, key in [("best_loss", "best_loss"), ("best_psnr", "best_psnr"), ("best_ssim", "best_ssim")]:
            ckpt_path = best_state[key]["path"]
            if ckpt_path is None or (not os.path.exists(ckpt_path)):
                print(f"[Gen] Skip {tag} (no checkpoint).")
                continue

            # load ckpt into model + ema
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=True)
            if ema is not None and "ema" in ckpt:
                ema.load_state_dict(ckpt["ema"])

            model_for_sample = ema.ema_model if ema is not None else model
            out_dir = os.path.join(gen_root, tag)
            ensure_dir(out_dir)
            generate_folder(cfg, model_for_sample, diffusion, val_input_dir, out_dir, device, tag)
            print(f"[Gen] Saved: {out_dir}")

if __name__ == "__main__":
    main()