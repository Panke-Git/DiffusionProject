"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: train_utils.py
    @Time: 2025/12/4 23:35
    @Email: None
"""
import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import yaml

from ..dataset.datasets import UnderwaterImageDataset


def load_config(path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """设置随机种子，确保实验可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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




