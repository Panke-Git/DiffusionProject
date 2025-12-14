"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: underwater_dataset.py
    @Time: 2025/12/13 10:48
    @Email: None
"""
from pathlib import Path
from typing import List, Tuple, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


class UnderwaterImageDataset(Dataset):
    """
    水下图像增强数据集:
    - input_dir:  原始水下图像目录 (条件)
    - gt_dir:     参考 GT 图像目录 (目标)
    要求: input 和 gt 文件名一一对应, 如:
        /Train/input/0001.png
        /Train/GT/0001.png
    """

    def __init__(
        self,
        input_dir: str,
        gt_dir: str,
        image_size: int = 256,
        augment: bool = False,
    ):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.gt_dir = Path(gt_dir)
        self.image_size = image_size
        self.augment = augment

        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"input_dir not found: {self.input_dir}")
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(f"gt_dir not found: {self.gt_dir}")

        # 收集所有 input 图像, 并匹配对应的 GT
        self.samples: List[Tuple[Path, Path]] = []
        input_files = sorted(
            [p for p in self.input_dir.iterdir() if p.is_file() and is_image_file(p)]
        )

        if len(input_files) == 0:
            raise RuntimeError(f"No image files found in {self.input_dir}")

        for in_path in input_files:
            rel_name = in_path.name  # 只按文件名匹配
            gt_path = self.gt_dir / rel_name
            if gt_path.is_file():
                self.samples.append((in_path, gt_path))
            else:
                # 严格模式: 找不到对应 GT 就直接报错
                raise FileNotFoundError(f"GT file not found for {in_path} -> {gt_path}")

        if len(self.samples) == 0:
            raise RuntimeError("No paired samples found!")

        # 图像变换: 统一到 image_size, 并归一到 [-1, 1]
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        in_path, gt_path = self.samples[idx]

        inp = Image.open(in_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")

        # 数据增强: 保证 input 与 GT 完全一致 (例如水平翻转)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                inp = TF.hflip(inp)
                gt = TF.hflip(gt)

        inp_t = self.base_transform(inp)  # (3, H, W), [-1,1]
        gt_t = self.base_transform(gt)    # (3, H, W), [-1,1]

        return {
            "input": inp_t,          # 条件图像 cond
            "gt": gt_t,              # 目标 x0
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
