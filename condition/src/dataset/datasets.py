"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: datasets.py
    @Time: 2025/12/4 22:28
    @Email: None
"""
"""Dataset definitions for underwater DDPM training.

现在数据集默认做「条件扩散」：
- gt_dir   ：增强后的 GT 图像
- input_dir：原始水下图像（与 gt 按文件名一一对应）

返回的 sample 字段：
- "image"      : GT 张量，范围 [-1, 1]
- "input"      : 原始水下图张量，范围 [-1, 1]
- "gt_path"    : GT 图像路径（字符串）
- "input_path" : input 图像路径（字符串）
"""

from pathlib import Path
from typing import Dict, Optional

import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


class UnderwaterImageDataset(Dataset):
    """成对的水下图像数据集：input → GT，用于条件扩散。"""

    def __init__(
        self,
        gt_dir: str,
        input_dir: Optional[str],
        image_size: int,
        channels: int,
        augmentation_cfg: Dict[str, bool],
    ) -> None:
        super().__init__()

        self.gt_dir = Path(gt_dir)
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(f"gt_dir 不存在或不是目录: {self.gt_dir}")

        if input_dir is None:
            raise ValueError(
                "当前工程已经改为『条件扩散』，必须在 config.data.<split>.input_dir 中提供原始水下图像目录。"
            )
        self.input_dir = Path(input_dir)
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"input_dir 不存在或不是目录: {self.input_dir}")

        # 收集 GT 图像列表
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        self.gt_paths = []
        for ext in exts:
            self.gt_paths.extend(sorted(self.gt_dir.glob(ext)))
        if not self.gt_paths:
            raise RuntimeError(f"在 gt_dir 中未找到任何图像文件: {self.gt_dir}")

        self.image_size = image_size
        self.channels = channels

        # 数据增强配置
        self.use_hflip = augmentation_cfg.get("horizontal_flip", False)
        self.use_color_jitter = augmentation_cfg.get("color_jitter", False)

        # ColorJitter 的参数范围（与你原来 brightness=0.1 等价）
        if self.use_color_jitter:
            self._cj_brightness = (0.9, 1.1)
            self._cj_contrast = (0.9, 1.1)
            self._cj_saturation = (0.9, 1.1)
            self._cj_hue = (-0.05, 0.05)

        # 尺寸 & 归一化：像素映射到 [-1, 1]
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5] * channels,
                    std=[0.5] * channels,
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.gt_paths)

    def _apply_paired_augment(self, gt_img: Image.Image, input_img: Image.Image):
        """对 GT 和 input 做完全一致的随机增强。"""
        # 1) 随机水平翻转
        if self.use_hflip and random.random() < 0.5:
            gt_img = TF.hflip(gt_img)
            input_img = TF.hflip(input_img)

        # 2) ColorJitter：共享同一组随机参数
        if self.use_color_jitter:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = (
                transforms.ColorJitter.get_params(
                    self._cj_brightness,
                    self._cj_contrast,
                    self._cj_saturation,
                    self._cj_hue,
                )
            )
            for fn_id in fn_idx:
                if fn_id == 0:
                    gt_img = TF.adjust_brightness(gt_img, brightness_factor)
                    input_img = TF.adjust_brightness(input_img, brightness_factor)
                elif fn_id == 1:
                    gt_img = TF.adjust_contrast(gt_img, contrast_factor)
                    input_img = TF.adjust_contrast(input_img, contrast_factor)
                elif fn_id == 2:
                    gt_img = TF.adjust_saturation(gt_img, saturation_factor)
                    input_img = TF.adjust_saturation(input_img, saturation_factor)
                elif fn_id == 3:
                    gt_img = TF.adjust_hue(gt_img, hue_factor)
                    input_img = TF.adjust_hue(input_img, hue_factor)

        return gt_img, input_img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gt_path = self.gt_paths[idx]
        gt_img = Image.open(gt_path).convert("RGB")

        input_path = self.input_dir / gt_path.name
        if not input_path.exists():
            raise FileNotFoundError(
                f"找不到与 GT 匹配的 input 图像: {input_path}（请确认文件名一一对应）"
            )
        input_img = Image.open(input_path).convert("RGB")

        # 成对数据增强
        gt_img, input_img = self._apply_paired_augment(gt_img, input_img)

        # 统一 resize + ToTensor + Normalize 到 [-1, 1]
        gt_tensor = self.base_transform(gt_img)
        input_tensor = self.base_transform(input_img)

        return {
            "image": gt_tensor,
            "input": input_tensor,
            "gt_path": str(gt_path),
            "input_path": str(input_path),
        }