"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: paired_dataset_old.py
    @Time: 2026/1/8 23:41
    @Email: None
"""
import os
import random
from typing import List, Dict, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

import albumentations as A


def is_image_file(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in ['jpeg', 'jpg', 'png', 'gif'])


def _sorted_image_paths(folder: str) -> List[str]:
    files = [x for x in os.listdir(folder) if is_image_file(x)]
    files.sort()
    return [os.path.join(folder, x) for x in files]


class DataReader(Dataset):
    """
    Paired dataset for conditional diffusion (input as condition, GT as target).
    ✅ Train / Val 都是整图 Resize 到 (h, w)（强制拉伸）。
    ✅ 返回 dict: {"name": filename, "cond": tensor[-1,1], "gt": tensor[-1,1]}
    """

    def __init__(
        self,
        img_dir: str,
        input: str = 'input',
        target: str = 'GT',
        mode: str = 'train',
        img_options: Optional[Dict] = None,
        # 训练增强开关（默认开一点轻增强；你也可以关掉）
        aug: bool = True,
    ):
        super().__init__()
        assert mode in ("train", "val", "test"), f"mode must be train/val/test, got {mode}"
        assert img_options is not None and "h" in img_options and "w" in img_options, \
            "img_options must contain {'h':..., 'w':...}"

        self.mode = mode
        self.img_options = img_options
        self.h = int(img_options["h"])
        self.w = int(img_options["w"])
        self.aug = bool(aug) and (mode == "train")

        input_dir = os.path.join(img_dir, input)
        target_dir = os.path.join(img_dir, target)

        self.input_filenames = _sorted_image_paths(input_dir)
        self.target_filenames = _sorted_image_paths(target_dir)

        # ✅ 强配对：用文件名做键，避免仅靠排序潜在错配
        inp_map = {os.path.basename(p): p for p in self.input_filenames}
        tar_map = {os.path.basename(p): p for p in self.target_filenames}
        common = sorted(list(set(inp_map.keys()) & set(tar_map.keys())))

        if len(common) == 0:
            raise RuntimeError("No paired files found (by identical filename) between input and GT folders.")

        self.pairs = [(inp_map[name], tar_map[name], name) for name in common]
        self.sizex = len(self.pairs)

        # ✅ 统一 Resize（train/val 都做）
        # 训练可选轻增强，但不再 crop
        if self.aug:
            self.transform = A.Compose(
                [
                    A.Resize(self.h, self.w),
                    A.OneOf(
                        [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)],
                        p=0.3
                    ),
                    A.RandomRotate90(p=0.2),
                    A.Rotate(limit=15, p=0.2),
                    A.Transpose(p=0.2),
                ],
                additional_targets={"target": "image"},
            )
        else:
            self.transform = A.Compose(
                [A.Resize(self.h, self.w)],
                additional_targets={"target": "image"},
            )

    def __len__(self):
        return self.sizex

    def load(self, index_: int):
        inp_path, tar_path, name = self.pairs[index_]

        inp_img = np.array(Image.open(inp_path).convert("RGB"))
        tar_img = np.array(Image.open(tar_path).convert("RGB"))

        transformed = self.transform(image=inp_img, target=tar_img)
        return name, transformed

    def __getitem__(self, index: int):
        index_ = index % self.sizex
        name, transformed = self.load(index_)

        inp_img = F.to_tensor(transformed["image"])      # [0,1]
        tar_img = F.to_tensor(transformed["target"])     # [0,1]

        # ✅ 扩散代码一般用 [-1,1]
        inp_img = inp_img * 2.0 - 1.0
        tar_img = tar_img * 2.0 - 1.0

        return {"name": name, "cond": inp_img, "gt": tar_img}
