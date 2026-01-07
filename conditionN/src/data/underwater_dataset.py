"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: underwater_dataset.py
    @Time: 2025/12/13 10:48
    @Email: None
"""
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2


import inspect

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def _A_RandomResizedCrop(h, w, **kwargs):
    sig = inspect.signature(A.RandomResizedCrop)
    if "size" in sig.parameters:  # albumentations 2.x
        return A.RandomResizedCrop(size=(h, w), **kwargs)
    else:                         # albumentations 1.x
        return A.RandomResizedCrop(height=h, width=w, **kwargs)

class UnderwaterImageDataset(Dataset):
    def __init__(self, input_dir: str, gt_dir: str, image_size: int = 256, augment: bool = True):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.gt_dir = Path(gt_dir)
        self.image_size = image_size
        self.augment = augment

        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"input_dir not found: {self.input_dir}")
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(f"gt_dir not found: {self.gt_dir}")

        self.samples: List[Tuple[Path, Path]] = []
        input_files = sorted([p for p in self.input_dir.iterdir() if p.is_file() and is_image_file(p)])
        if len(input_files) == 0:
            raise RuntimeError(f"No image files found in {self.input_dir}")

        for in_path in input_files:
            gt_path = self.gt_dir / in_path.name
            if not gt_path.is_file():
                raise FileNotFoundError(f"GT file not found for {in_path} -> {gt_path}")
            self.samples.append((in_path, gt_path))

        # 训练增强（移植你第一个的思路），并保持 paired
        if self.augment:
            self.transform = A.Compose(
                [
                    A.OneOf(
                        [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)],
                        p=0.3
                    ),
                    A.RandomRotate90(p=0.3),
                    A.Rotate(limit=30, p=0.3),     # 你原来是 A.Rotate(p=0.3)，这里建议加 limit
                    A.Transpose(p=0.3),
                    _A_RandomResizedCrop(
                        image_size, image_size,
                        scale=(0.7, 1.0),
                        ratio=(0.75, 1.333),
                        p=1.0
                    ),
                    # 归一化到 [-1,1]（等价于 mean=0.5,std=0.5）
                    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), max_pixel_value=255.0),
                    ToTensorV2(),
                ],
                additional_targets={"gt": "image"},
            )
        else:
            # 验证/推理：稳定的 resize + center crop（或你也可以直接 Resize 到 image_size）
            self.transform = A.Compose(
                [
                    A.Resize(image_size, image_size, interpolation=1),  # 1=linear, 也可以换成 cubic(2)
                    A.CenterCrop(image_size, image_size),
                    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), max_pixel_value=255.0),
                    ToTensorV2(),
                ],
                additional_targets={"gt": "image"},
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        in_path, gt_path = self.samples[idx]

        inp = np.array(Image.open(in_path).convert("RGB"))
        gt = np.array(Image.open(gt_path).convert("RGB"))

        out = self.transform(image=inp, gt=gt)

        return {
            "input": out["image"].float(),   # (3,H,W) in [-1,1]
            "gt": out["gt"].float(),         # (3,H,W) in [-1,1]
            "input_path": str(in_path),
            "gt_path": str(gt_path),
        }


# if __name__ == "__main__":
#     # 简单自测: 你可以在本地改成自己的路径跑一下
#     import os
#     dummy_input = "E:\\PythonProject\\01_Personal\\UnderwaterImageEnhanced\\dataset\\LSUI19\\Train\\input"
#     dummy_gt = "E:\\PythonProject\\01_Personal\\UnderwaterImageEnhanced\\dataset\\LSUI19\\Train\\GT"
#     print(dummy_input)
#     if os.path.isdir(dummy_input) and os.path.isdir(dummy_gt):
#         ds = UnderwaterImageDataset(dummy_input, dummy_gt, image_size=256, augment=True)
#         print("dataset length:", len(ds))
#         item = ds[0]
#         print("input shape:", item["input"].shape,
#               "range:", (item["input"].min().item(), item["input"].max().item()))
#         print("gt shape:", item["gt"].shape)
