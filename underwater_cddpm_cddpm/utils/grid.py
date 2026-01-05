from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
from PIL import Image


def _to_u8(img: torch.Tensor) -> np.ndarray:
    """(3,H,W) in [-1,1] -> uint8 (H,W,3)"""
    img = img.detach().float().clamp(-1.0, 1.0)
    img = (img + 1.0) * 0.5
    img = (img * 255.0).round().to(torch.uint8)
    return img.permute(1, 2, 0).cpu().numpy()


def make_triplet_grid(inputs: List[torch.Tensor], gts: List[torch.Tensor], preds: List[torch.Tensor]) -> Image.Image:
    assert len(inputs) == len(gts) == len(preds)
    n = len(inputs)
    h, w = inputs[0].shape[-2], inputs[0].shape[-1]
    canvas = Image.new("RGB", (w * 3, h * n))
    for i in range(n):
        a = Image.fromarray(_to_u8(inputs[i]))
        b = Image.fromarray(_to_u8(gts[i]))
        c = Image.fromarray(_to_u8(preds[i]))
        canvas.paste(a, (0, i * h))
        canvas.paste(b, (w, i * h))
        canvas.paste(c, (2 * w, i * h))
    return canvas


def save_pil(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
