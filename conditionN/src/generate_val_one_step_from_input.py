"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: generate_val_one_step_from_input.py
    @Time: 2025/12/14 12:48
    @Email: None
"""
# src/generate_val_one_step_from_input.py

import os
import sys
import argparse
import random
from pathlib import Path

import torch
from torchvision.utils import save_image

# 保证能 import 到 src 下的包
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .utils.config import load_config
from .data.underwater_dataset import UnderwaterImageDataset
from .models.cond_unet_ddpm import UNetConditional
from .models.gaussian_diffusion import GaussianDiffusion


def set_seed(seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denorm_to_0_1(x):
    """
    [-1,1] -> [0,1]，并裁剪
    """
    x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


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


def load_checkpoint(model, ckpt_path, device):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    print(f"[Load] Loaded checkpoint from {ckpt_path}")
    return state.get("metrics", None)


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-step enhancement from input using conditional DDPM"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/underwater_ddpm.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to checkpoint .pt (best_loss / best_psnr / best_ssim 都可以)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="How many random val images to enhance",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="val_one_step_from_input",
        help="Directory to save images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cuda', 'cuda:0', or 'cpu'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting val indices",
    )
    parser.add_argument(
        "--t_infer",
        type=int,
        default=400,
        help="Time step used at inference (0 ~ T-1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 设备
    if "cuda" in args.device and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[Warn] CUDA not available, fallback to CPU.")
    else:
        device = torch.device(args.device)
    print(f"[Device] Using {device}")

    set_seed(args.seed)

    # 验证集
    val_ds = UnderwaterImageDataset(
        input_dir=cfg.DATA.val_input_dir,
        gt_dir=cfg.DATA.val_gt_dir,
        image_size=cfg.DATA.image_size,
        augment=False,
    )
    print(f"[Data] Val dataset size: {len(val_ds)}")

    # 模型 & 扩散器
    model, diffusion = build_model_and_diffusion(cfg, device)
    model.eval()

    # 加载 ckpt
    _ = load_checkpoint(model, args.ckpt_path, device)

    # 限制 t_infer 范围
    t_infer = int(max(0, min(args.t_infer, diffusion.timesteps - 1)))
    print(f"[Infer] t_infer = {t_infer} / {diffusion.timesteps - 1}")

    # 随机取 num_samples 张
    num = min(args.num_samples, len(val_ds))
    indices = random.sample(range(len(val_ds)), k=num)
    print(f"[Sample] Random indices (val): {indices}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Saving images to: {out_dir.resolve()}")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = val_ds[idx]
            cond = sample["input"].unsqueeze(0).to(device)  # (1,3,H,W), [-1,1]
            x0_gt = sample["gt"].unsqueeze(0).to(device)    # (1,3,H,W), [-1,1]

            in_path = Path(sample["input_path"])
            base_name = in_path.stem

            b = cond.size(0)
            t = torch.full((b,), t_infer, device=device, dtype=torch.long)

            # 关键：直接把 input 当作 x_t
            x_t = cond

            # 模型预测噪声 -> 还原 x0_pred
            eps_pred = model(x_t, cond, t)
            x0_pred = diffusion.predict_start_from_noise(x_t, t, eps_pred)  # [-1,1]

            # 反归一化保存
            cond_01 = denorm_to_0_1(cond[0]).cpu()
            gt_01   = denorm_to_0_1(x0_gt[0]).cpu()
            pred_01 = denorm_to_0_1(x0_pred[0]).cpu()

            save_image(cond_01, out_dir / f"{i:02d}_{base_name}_input.png")
            save_image(gt_01,   out_dir / f"{i:02d}_{base_name}_gt.png")
            save_image(pred_01, out_dir / f"{i:02d}_{base_name}_enhanced.png")

            print(
                f"[Saved] idx={idx}, name={base_name} -> "
                f"{i:02d}_{base_name}_input/gt/enhanced.png"
            )

    print("[Done] One-step enhancement from input finished.")


if __name__ == "__main__":
    main()
