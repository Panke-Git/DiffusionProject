"""
    @Project: DiffusionProject
    @Author: Panke
    @FileName: dataset_uie.py.py
    @Time: 2025/12/23 23:29
    @Email: None
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(root):
    files = []
    for name in os.listdir(root):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMG_EXTS:
            files.append(name)
    files.sort()
    return files


class UnderwaterPairDataset(Dataset):
    def __init__(self, input_dir, gt_dir, img_size=256, augment=True):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.img_size = int(img_size)
        self.augment = bool(augment)

        input_files = list_images(input_dir)
        gt_set = set(list_images(gt_dir))

        pairs = []
        for fn in input_files:
            if fn in gt_set:
                pairs.append((fn, fn))
        if len(pairs) == 0:
            raise RuntimeError("No paired images found. Make sure input/GT filenames match 1-to-1.")

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def _load_rgb(self, path):
        return Image.open(path).convert("RGB")

    def _resize(self, img):
        return TF.resize(img, [self.img_size, self.img_size], interpolation=Image.BICUBIC)

    def _to_tensor_norm(self, img):
        x = TF.to_tensor(img)     # [0,1]
        x = x * 2.0 - 1.0         # [-1,1]
        return x

    def __getitem__(self, idx):
        in_name, gt_name = self.pairs[idx]
        in_path = os.path.join(self.input_dir, in_name)
        gt_path = os.path.join(self.gt_dir, gt_name)

        inp = self._resize(self._load_rgb(in_path))
        gt = self._resize(self._load_rgb(gt_path))

        if self.augment:
            if torch.rand(1).item() < 0.5:
                inp = TF.hflip(inp)
                gt = TF.hflip(gt)

        return {
            "input": self._to_tensor_norm(inp),
            "gt": self._to_tensor_norm(gt),
            "name": in_name
        }
