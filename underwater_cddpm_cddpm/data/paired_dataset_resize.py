"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: paired_dataset_resize.py
    @Time: 2026/1/8 22:43
    @Email: None
"""
import os
from typing import List
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _list_images(folder: str) -> List[str]:
    names = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(IMG_EXTS):
            names.append(fn)
    names.sort()
    return names


class PairedImageDataset(Dataset):
    """Paired dataset: (condition input, GT). Always resize to (image_size, image_size)."""

    def __init__(
            self,
            input_dir: str,
            gt_dir: str,
            image_size: int = 256,
            random_crop: bool = False,  # 保留但不再使用（为了兼容外部传参）
            random_flip: bool = True,
            resize_only: bool = True,  # 保留但不再使用（现在永远 resize）
    ):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.image_size = int(image_size)

        # 兼容保留
        self.random_crop = bool(random_crop)
        self.random_flip = bool(random_flip)
        self.resize_only = bool(resize_only)

        self.names = _list_images(self.input_dir)
        if len(self.names) == 0:
            raise RuntimeError(f"No images found in: {self.input_dir}")

        missing = []
        for n in self.names:
            if not os.path.exists(os.path.join(self.gt_dir, n)):
                missing.append(n)
        if missing:
            raise RuntimeError(
                f"Missing {len(missing)} GT images in {self.gt_dir}. Example: {missing[0]}\n"
                "Make sure filenames match between input and GT."
            )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        inp_path = os.path.join(self.input_dir, name)
        gt_path = os.path.join(self.gt_dir, name)

        inp = Image.open(inp_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")

        # ✅ 强制：两者都直接 resize 到 (image_size, image_size)
        inp = inp.resize((self.image_size, self.image_size), resample=Image.BICUBIC)

        # 先确保 GT 与 input 原始尺寸一致（有些数据可能略有差异）
        if gt.size != Image.open(inp_path).size:
            # 更稳妥：直接对 GT 做同样 resize，不依赖原始尺寸是否一致
            pass
        gt = gt.resize((self.image_size, self.image_size), resample=Image.BICUBIC)

        # flip 仍可用（训练增强）
        if self.random_flip and random.random() < 0.5:
            inp = TF.hflip(inp)
            gt = TF.hflip(gt)

        inp_t = TF.to_tensor(inp) * 2.0 - 1.0
        gt_t = TF.to_tensor(gt) * 2.0 - 1.0

        return {"name": name, "cond": inp_t, "gt": gt_t}
