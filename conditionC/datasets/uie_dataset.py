# coding=utf-8
"""
    @Project: 
    @Author: PyCharm
    @FileName： uie_dataset.py
    @Date：2025/12/25 16:15
    @Email: None
"""

import os
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def _list_images(folder: str) -> List[Path]:
    p = Path(folder)
    files = []
    for f in p.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            files.append(f)
    files.sort()
    return files

class PairedImageDataset(Dataset):
    """
    Paired dataset:
      input_dir: degraded underwater images (condition)
      gt_dir: enhanced ground-truth images (target x0)
    pairing: by relative path (same subfolder & filename) if exists, else by basename match.
    """
    def __init__(self, input_dir: str, gt_dir: str, image_size: int, is_train: bool):
        self.input_dir = Path(input_dir)
        self.gt_dir = Path(gt_dir)
        self.image_size = int(image_size)
        self.is_train = is_train

        self.input_files = _list_images(str(self.input_dir))
        if len(self.input_files) == 0:
            raise RuntimeError(f"No images found in: {self.input_dir}")

        # Build GT mapping
        gt_files = _list_images(str(self.gt_dir))
        gt_map_rel = {}
        gt_map_base = {}
        for g in gt_files:
            rel = g.relative_to(self.gt_dir).as_posix()
            gt_map_rel[rel] = g
            gt_map_base[g.name] = g

        pairs: List[Tuple[Path, Path]] = []
        miss = 0
        for inp in self.input_files:
            rel = inp.relative_to(self.input_dir).as_posix()
            if rel in gt_map_rel:
                pairs.append((inp, gt_map_rel[rel]))
            elif inp.name in gt_map_base:
                pairs.append((inp, gt_map_base[inp.name]))
            else:
                miss += 1

        if len(pairs) == 0:
            raise RuntimeError("No paired samples found. Check Train/input vs Train/GT filenames.")
        if miss > 0:
            print(f"[Dataset] Warning: {miss} inputs have no matching GT. They were skipped.")

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def _load_rgb(self, path: Path) -> Image.Image:
        img = Image.open(str(path)).convert("RGB")
        return img

    def _augment(self, inp: Image.Image, gt: Image.Image):
        # Resize (keep it simple & stable)
        inp = TF.resize(inp, [self.image_size, self.image_size], interpolation=Image.BICUBIC)
        gt = TF.resize(gt, [self.image_size, self.image_size], interpolation=Image.BICUBIC)

        if self.is_train:
            if random.random() < 0.5:
                inp = TF.hflip(inp)
                gt = TF.hflip(gt)
            if random.random() < 0.5:
                inp = TF.vflip(inp)
                gt = TF.vflip(gt)

        return inp, gt

    def _to_tensor_norm(self, img: Image.Image) -> torch.Tensor:
        # [0,1]
        x = TF.to_tensor(img)
        # [-1,1]
        x = x * 2.0 - 1.0
        return x

    def __getitem__(self, idx: int):
        inp_path, gt_path = self.pairs[idx]
        inp = self._load_rgb(inp_path)
        gt = self._load_rgb(gt_path)
        inp, gt = self._augment(inp, gt)

        inp_t = self._to_tensor_norm(inp)
        gt_t = self._to_tensor_norm(gt)

        return {
            "cond": inp_t,     # condition: underwater input
            "gt": gt_t,        # target x0: enhanced GT
            "name": inp_path.stem,
            "relpath": inp_path.relative_to(self.input_dir).as_posix(),
        }