import os
from typing import List, Tuple, Optional
from PIL import Image
import torch
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
    """Paired dataset: (condition input, GT)."""

    def __init__(
        self,
        input_dir: str,
        gt_dir: str,
        image_size: int = 256,
        random_crop: bool = True,
        random_flip: bool = True,
    ):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.image_size = int(image_size)
        self.random_crop = bool(random_crop)
        self.random_flip = bool(random_flip)

        self.names = _list_images(self.input_dir)
        if len(self.names) == 0:
            raise RuntimeError(f"No images found in: {self.input_dir}")

        # ensure GT exists
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

    def _paired_resize(self, inp: Image.Image, gt: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # Resize both to at least image_size on the shorter edge, preserving aspect ratio.
        # Then crop to (image_size, image_size).
        w, h = inp.size
        # Ensure GT same size; if not, resize GT to input size first
        if gt.size != inp.size:
            gt = gt.resize((w, h), resample=Image.BICUBIC)

        # scale so that min(h,w) >= image_size
        scale = self.image_size / min(h, w)
        if scale != 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            inp = inp.resize((new_w, new_h), resample=Image.BICUBIC)
            gt = gt.resize((new_w, new_h), resample=Image.BICUBIC)
        return inp, gt

    def _paired_crop(self, inp: Image.Image, gt: Image.Image) -> Tuple[Image.Image, Image.Image]:
        w, h = inp.size
        th = tw = self.image_size
        if w == tw and h == th:
            return inp, gt

        if self.random_crop:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            i = max(0, (h - th) // 2)
            j = max(0, (w - tw) // 2)

        inp = TF.crop(inp, i, j, th, tw)
        gt = TF.crop(gt, i, j, th, tw)
        return inp, gt

    def __getitem__(self, idx: int):
        name = self.names[idx]
        inp_path = os.path.join(self.input_dir, name)
        gt_path = os.path.join(self.gt_dir, name)

        inp = Image.open(inp_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")

        inp, gt = self._paired_resize(inp, gt)
        inp, gt = self._paired_crop(inp, gt)

        if self.random_flip and random.random() < 0.5:
            inp = TF.hflip(inp)
            gt = TF.hflip(gt)

        inp_t = TF.to_tensor(inp)  # [0,1]
        gt_t = TF.to_tensor(gt)

        # Normalize to [-1,1]
        inp_t = inp_t * 2.0 - 1.0
        gt_t = gt_t * 2.0 - 1.0

        return {
            "name": name,
            "cond": inp_t,
            "gt": gt_t,
        }
