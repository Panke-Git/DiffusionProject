# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
# [MOD] Rewritten to be runnable + adapted for underwater RGB enhancement datasets:
#       /dataset/Train/input, /dataset/Train/GT, /dataset/Val/input, /dataset/Val/GT
# [MOD] Removed torchvision dependency to avoid torch/torchvision binary mismatch issues.

import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def _list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = [p for p in folder.iterdir() if p.is_file() and _is_image(p)]
    files.sort()
    return files


def _match_pairs(input_dir: Path, gt_dir: Path) -> List[Tuple[Path, Path]]:
    """
    [MOD] Pair by filename stem first; fallback to sorted order.
    """
    in_files = _list_images(input_dir)
    gt_files = _list_images(gt_dir)

    gt_map = {p.stem: p for p in gt_files}
    pairs = []
    missing = 0
    for ip in in_files:
        gp = gt_map.get(ip.stem, None)
        if gp is None:
            missing += 1
            continue
        pairs.append((ip, gp))

    if len(pairs) == 0:
        n = min(len(in_files), len(gt_files))
        pairs = list(zip(in_files[:n], gt_files[:n]))

    if len(pairs) == 0:
        raise RuntimeError(f"No paired images found in:\n  input={input_dir}\n  gt={gt_dir}")

    if missing > 0:
        print(f"[WARN] {missing} input images have no matched GT by stem in {gt_dir}")

    return pairs


# -----------------------------
# [MOD] minimal transform utils
# -----------------------------
class Compose:
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, img):
        for op in self.ops:
            img = op(img)
        return img


class Resize:
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, img: Image.Image):
        return img.resize((self.size, self.size), resample=Image.BICUBIC)


class RandomCrop:
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, img: Image.Image):
        w, h = img.size
        th, tw = self.size, self.size
        if w == tw and h == th:
            return img
        if w < tw or h < th:
            # pad then crop
            pad_w = max(0, tw - w)
            pad_h = max(0, th - h)
            img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)
            w, h = img.size
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return img.crop((j, i, j + tw, i + th))


class CenterCrop:
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, img: Image.Image):
        w, h = img.size
        th, tw = self.size, self.size
        i = max(0, (h - th) // 2)
        j = max(0, (w - tw) // 2)
        return img.crop((j, i, j + tw, i + th))


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, img: Image.Image):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class ToTensor:
    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.float32) / 255.0  # HWC, [0,1]
        if arr.ndim == 2:
            arr = arr[..., None]
        arr = arr.transpose(2, 0, 1)  # CHW
        return torch.from_numpy(arr)


class NormalizeToMinusOneOne:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x in [0,1] -> [-1,1]
        return x * 2.0 - 1.0


class UnderwaterPairDataset(Dataset):
    """
    [MOD] RGB paired dataset for conditional diffusion.
    Returns:
        {
          "cond": input underwater image,   shape [3,H,W], range [-1,1]
          "gt":   target enhanced/GT image, shape [3,H,W], range [-1,1]
          "name": filename stem
        }
    """
    def __init__(self, input_dir: str, gt_dir: str, image_size: int = 256, is_train: bool = True):
        self.input_dir = Path(input_dir)
        self.gt_dir = Path(gt_dir)
        self.pairs = _match_pairs(self.input_dir, self.gt_dir)
        self.is_train = is_train
        self.image_size = int(image_size)

        if is_train:
            self.tf = Compose([
                Resize(self.image_size),
                RandomCrop(self.image_size),
                RandomHorizontalFlip(0.5),
                ToTensor(),
                NormalizeToMinusOneOne(),
            ])
        else:
            self.tf = Compose([
                Resize(self.image_size),
                CenterCrop(self.image_size),
                ToTensor(),
                NormalizeToMinusOneOne(),
            ])

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_rgb(self, p: Path) -> Image.Image:
        return Image.open(p).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ip, gp = self.pairs[idx]
        cond = self.tf(self._load_rgb(ip))
        gt = self.tf(self._load_rgb(gp))
        return {"cond": cond, "gt": gt, "name": ip.stem}
