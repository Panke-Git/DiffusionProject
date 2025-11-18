"""
    @Project: UnderwaterImageEnhanced-Diffusion
    @Author: ChatGPT
    @FileName: pair_dataset.py
    @Time: 2025/11/09
    @Email: None
"""
# datasets/uieb_pair_dataset.py
import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class UnderwaterPairDataset(Dataset):
    """
    假设目录结构：
        root/
          input/ xxx.png
          GT/    xxx.png
    文件名一一对应。
    """

    def __init__(self, root_dir, input_subdir='input', target_subdir='GT',
                 img_h=256, img_w=256):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.input_dir = self.root_dir / input_subdir
        self.target_dir = self.root_dir / target_subdir

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input dir not found: {self.input_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target dir not found: {self.target_dir}")

        self.input_paths = sorted(
            [p for p in self.input_dir.iterdir() if p.is_file()]
        )
        if len(self.input_paths) == 0:
            raise RuntimeError(f"No input images found in {self.input_dir}")

        self.target_paths = []
        for p in self.input_paths:
            target_path = self.target_dir / p.name
            if not target_path.exists():
                raise FileNotFoundError(
                    f"Target image not found for {p.name}: {target_path}"
                )
            self.target_paths.append(target_path)

        self.transform = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),   # [0,1]
        ])

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        in_path = self.input_paths[idx]
        gt_path = self.target_paths[idx]

        inp = Image.open(in_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        inp = self.transform(inp)  # [0,1]
        gt = self.transform(gt)

        # 映射到 [-1, 1]，方便 DDPM 训练
        inp = inp * 2.0 - 1.0
        gt = gt * 2.0 - 1.0

        return inp, gt, os.path.basename(in_path)
