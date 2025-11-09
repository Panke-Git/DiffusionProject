"""
    @Project: UnderwaterImageEnhanced-Diffusion
    @Author: ChatGPT
    @FileName: paired_dataset.py
    @Time: 2025/11/09
    @Email: None
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class PairedFolder(Dataset):
    """
    加载配对数据：root/{split}/{input,GT}，以文件名（不含扩展名）配对。
    - 返回 y（观测）、x0（GT）张量，范围 [0,1]，形状 (3,H,W)。
    - 保证 y/x0 的随机翻转等增强同步。
    """
    def __init__(self, root: str, split: str = "train", image_size: int = 256, 
                 input_dir_name: str = "input", target_dir_name: str = "GT",
                 augment: bool = True, ext: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.input_dir = self.root / split / input_dir_name
        self.gt_dir = self.root / split / target_dir_name
        assert self.input_dir.exists() and self.gt_dir.exists(), f"Expect {self.input_dir} and {self.gt_dir}"

        def index_dir(d: Path):
            idx = {}
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in ext:
                    idx[p.stem] = p
            return idx
        idx_in = index_dir(self.input_dir)
        idx_gt = index_dir(self.gt_dir)
        keys = sorted(list(set(idx_in.keys()) & set(idx_gt.keys())))
        self.pairs = [(idx_in[k], idx_gt[k]) for k in keys]
        if len(self.pairs) == 0:
            raise RuntimeError("No paired images found.")

        self.image_size = image_size
        self.augment = augment

        self.resize = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)
        self.to_tensor = T.ToTensor()
        self.color_jitter = T.ColorJitter(0.1, 0.1, 0.1, 0.05) if augment else None
        self.hflip_prob = 0.5 if augment else 0.0

    def __len__(self):
        return len(self.pairs)

    def _load_img(self, p: Path) -> Image.Image:
        return Image.open(p).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p_in, p_gt = self.pairs[idx]
        y = self._load_img(p_in)
        x0 = self._load_img(p_gt)

        # 同步 resize
        y = self.resize(y)
        x0 = self.resize(x0)

        # 同步水平翻转
        if self.augment and random.random() < self.hflip_prob:
            y = T.functional.hflip(y)
            x0 = T.functional.hflip(x0)

        # 颜色抖动（只对 y，避免破坏 GT；也可按需对两侧都用）
        if self.color_jitter is not None:
            y = self.color_jitter(y)

        y = self.to_tensor(y)      # (3,H,W) in [0,1]
        x0 = self.to_tensor(x0)    # (3,H,W) in [0,1]
        return {"y": y, "x0": x0, "name": p_in.stem}
