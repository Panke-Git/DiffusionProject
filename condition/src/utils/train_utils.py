"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: train_utils.py
    @Time: 2025/12/4 23:35
    @Email: None
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F

from ..dataset.datasets import UnderwaterImageDataset


def prepare_dataloader(
        cfg: Dict[str, Any], split: str, shuffle: bool = True
) -> DataLoader:
    """根据 split 构建训练或验证 DataLoader。"""

    split_cfg = cfg["data"].get(split, {})
    if not split_cfg:
        raise ValueError(f"配置中缺少 {split} 数据集信息。")

    dataset = UnderwaterImageDataset(
        gt_dir=split_cfg["gt_dir"],
        input_dir=split_cfg.get("input_dir"),
        image_size=cfg["data"]["image_size"],
        channels=cfg["data"]["channels"],
        augmentation_cfg=cfg["data"].get("augmentation", {}) if split == "train" else {},
    )
    batch_size = (
        cfg["data"]["batch_size"] if split == "train" else cfg["data"].get("val_batch_size", cfg["data"]["batch_size"])
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
    )


def create_output_dirs(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """按照配置创建输出目录，并返回相关路径。"""

    model_name = cfg["experiment"].get("model_name", cfg["experiment"].get("name", "model"))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    root = Path(cfg["experiment"].get("output_root", "runs"))
    output_dir = root / model_name / timestamp
    samples_dir = output_dir / "samples"
    ckpt_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_dir": output_dir,
        "samples_dir": samples_dir,
        "ckpt_dir": ckpt_dir,
        "metrics_path": metrics_path,
    }


def load_config(path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """设置随机种子，确保实验可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_01(x: torch.Tensor) -> torch.Tensor:
    """将张量从 [-1,1] 映射到 [0,1]，便于可视化。"""

    return (x.clamp(-1, 1) + 1) * 0.5


def save_comparison_grid(
        model: nn.Module,
        diffusion,
        dataloader: DataLoader,
        device: torch.device,
        save_path: Path,
        num_samples: int = 5,
) -> None:
    """使用验证集生成 输入/GT/输出 的对比网格，每行 3 张图。"""

    model.eval()
    original_model = diffusion.model
    diffusion.model = model

    images_to_plot = []
    collected = 0
    with torch.no_grad():
        for batch in dataloader:
            gt = batch["image"].to(device)
            cond = batch.get("input")
            if cond is None:
                raise RuntimeError(
                    "save_comparison_grid 需要 batch['input'] 作为条件图像，"
                    "请确认验证集也配置了 input_dir。"
                )
            cond = cond.to(device)

            bsz = gt.size(0)
            noise = torch.randn_like(gt)
            t = torch.randint(0, diffusion.timesteps, (bsz,), device=device).long()
            x_noisy = diffusion.q_sample(gt, t, noise)

            # 条件尺寸对齐
            if cond.shape[2:] != x_noisy.shape[2:]:
                cond_resized = F.interpolate(
                    cond,
                    size=x_noisy.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                cond_resized = cond

            model_in = torch.cat([x_noisy, cond_resized], dim=1)
            predicted_noise = model(model_in, t)
            recon = diffusion.predict_start_from_noise(x_noisy, t, predicted_noise)

            input_vis = tensor_to_01(cond).cpu()
            gt_vis = tensor_to_01(gt).cpu()
            recon_vis = tensor_to_01(recon).cpu()

            for i in range(bsz):
                images_to_plot.extend([input_vis[i], gt_vis[i], recon_vis[i]])
                collected += 1
                if collected >= num_samples:
                    grid = make_grid(images_to_plot, nrow=3)
                    save_image(grid, save_path)
                    diffusion.model = original_model
                    return

    diffusion.model = original_model


def plot_loss_curve(metrics: List[Dict[str, float]], save_path: Path) -> None:
    """绘制训练/验证损失曲线并保存。"""

    epochs = [m["epoch"] for m in metrics]
    train_losses = [m.get("train_loss") for m in metrics]
    val_losses = [m.get("val_loss") for m in metrics]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    if any(v is not None for v in val_losses):
        plt.plot(epochs, val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def save_samples(images: torch.Tensor, save_dir: Path, epoch: int, save_grid: bool) -> None:
    """保存采样结果；同时保存单张和网格。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    # 反归一化到 [0,1]
    images = (images.clamp(-1, 1) + 1) * 0.5

    for idx, img in enumerate(images):
        save_image(img, save_dir / f"epoch{epoch:04d}_sample{idx}.png")

    if save_grid:
        grid = make_grid(images, nrow=max(1, int(len(images) ** 0.5)))
        save_image(grid, save_dir / f"epoch{epoch:04d}_grid.png")


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, save_dir / f"epoch{epoch:04d}.pt")


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """计算批次内的平均 PSNR。输入应为 [0,1] 范围。"""

    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse + 1e-8)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """使用 3x3 平均窗口计算近似 SSIM。输入应为 [0,1] 范围。"""

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(pred * pred, kernel_size=3, stride=1, padding=1) - mu_x ** 2
    sigma_y = F.avg_pool2d(target * target, kernel_size=3, stride=1, padding=1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=3, stride=1, padding=1) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-8)
    return ssim_map.mean(dim=(1, 2, 3))
