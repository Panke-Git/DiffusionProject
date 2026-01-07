import os
from typing import Optional
import torch
from torchvision.utils import save_image


def denorm_to_0_1(x: torch.Tensor) -> torch.Tensor:
    """x in [-1,1] -> [0,1]"""
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5


def save_tensor_image(t: torch.Tensor, path: str):
    """Save a single image tensor (C,H,W) in [-1,1] or [0,1]."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if t.min() < 0:
        t = denorm_to_0_1(t)
    save_image(t, path)
