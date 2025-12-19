"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: train_utils.py
    @Time: 2025/12/14 23:25
    @Email: None
"""
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import sys
import torch.nn.functional as F

# 保证能找到 src 目录下的包 (models, data, diffusion, utils)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .metrics import calculate_psnr, calculate_ssim
from data.underwater_dataset import UnderwaterImageDataset
from models.cond_unet_ddpm import UNetConditional
from models.gaussian_diffusion import GaussianDiffusion



def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def update_ema(ema_model, model, ema_decay):
    if ema_model is None:
        return
    with torch.no_grad():
        msd = model.state_dict()
        for k, v_ema in ema_model.state_dict().items():
            v = msd[k].detach()
            v_ema.mul_(ema_decay).add_(v, alpha=1.0 - ema_decay)


def build_dataloaders(cfg):
    train_ds = UnderwaterImageDataset(
        input_dir=cfg.DATA.train_input_dir,
        gt_dir=cfg.DATA.train_gt_dir,
        image_size=cfg.DATA.image_size,
        augment=cfg.DATA.augment,
    )
    val_ds = UnderwaterImageDataset(
        input_dir=cfg.DATA.val_input_dir,
        gt_dir=cfg.DATA.val_gt_dir,
        image_size=cfg.DATA.image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.DATA.batch_size,
        shuffle=True,
        num_workers=cfg.DATA.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.DATA.batch_size,
        shuffle=False,
        num_workers=cfg.DATA.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def build_model_and_diffusion(cfg, device):
    model = UNetConditional(
        in_channels=cfg.MODEL.in_channels,
        cond_channels=cfg.MODEL.cond_channels,
        out_channels=cfg.MODEL.out_channels,
        base_channels=cfg.MODEL.base_channels,
        channel_mults=tuple(cfg.MODEL.channel_mults),
        num_res_blocks=cfg.MODEL.num_res_blocks,
        time_emb_dim=cfg.MODEL.time_emb_dim,
        dropout=cfg.MODEL.dropout,
        attn_resolutions=tuple(cfg.MODEL.attn_resolutions),
        num_heads=cfg.MODEL.num_heads,
        image_size=cfg.MODEL.image_size,
    ).to(device)

    diffusion = GaussianDiffusion(
        timesteps=cfg.DIFFUSION.timesteps,
        beta_start=cfg.DIFFUSION.beta_start,
        beta_end=cfg.DIFFUSION.beta_end,
        beta_schedule=cfg.DIFFUSION.beta_schedule,
        clip_x_start=cfg.DIFFUSION.clip_x_start,
    ).to(device)

    return model, diffusion


def save_checkpoint(model, optimizer, epoch, cfg, save_dir, tag, metrics_dict, ema_model=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"best_{tag}.pt"

    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics_dict,
        "config": cfg.__dict__,
    }
    if ema_model is not None:
        state["ema_model_state"] = ema_model.state_dict()

    torch.save(state, ckpt_path)
    print(f"[Checkpoint] Saved best {tag} to {ckpt_path}")

def _strip_module_prefix(state_dict: dict) -> dict:
    # 兼容 DataParallel/Distributed 保存的 "module.xxx"
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def load_checkpoint(
    model,
    ckpt_path,
    device,
    optimizer=None,
    use_ema: bool = True,
    strict: bool = True,
):
    """
    加载 checkpoint 到 model（可选加载 optimizer）
    - use_ema=True：如果 checkpoint 里有 ema_model_state，就优先加载 EMA 权重（推荐推理/出图）
    返回：epoch, metrics, loaded_key
    """
    ckpt_path = Path(ckpt_path)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    loaded_key = "model_state"
    sd = state.get("model_state", None)

    if use_ema and ("ema_model_state" in state):
        loaded_key = "ema_model_state"
        sd = state["ema_model_state"]

    if sd is None:
        raise KeyError(f"Checkpoint missing model_state: {ckpt_path}")

    sd = _strip_module_prefix(sd)

    try:
        model.load_state_dict(sd, strict=strict)
    except RuntimeError:
        # 再尝试一次：反向加回 module. 前缀
        sd2 = {("module." + k): v for k, v in sd.items()}
        model.load_state_dict(sd2, strict=strict)

    if optimizer is not None and ("optimizer_state" in state):
        optimizer.load_state_dict(state["optimizer_state"])

    epoch = int(state.get("epoch", 0))
    metrics = state.get("metrics", {})

    print(f"[Checkpoint] Loaded {loaded_key} from {ckpt_path} (epoch={epoch})")
    return epoch, metrics, loaded_key